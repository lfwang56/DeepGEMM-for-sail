"""Checkin input-layout and process-lifetime regression checks."""
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_checkin


class CheckinProcessTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'checkin uses POSIX process groups')
    def test_timeout_reaps_rank_in_a_separate_session(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / 'rank.pid'
            rank = ('import os,signal,time;from pathlib import Path;'
                    'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                    f'Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(60)')
            launcher = ('import subprocess,sys,time;'
                        f'subprocess.Popen([sys.executable,"-c",{rank!r}],start_new_session=True);time.sleep(60)')
            try:
                result = run_checkin.run_case(1, '-c ' + shlex.quote(launcher), 'script', directory, 1)
                self.assertIn('Timeout', result.error)
                self.assertTrue(pid_file.exists())
                pid = int(pid_file.read_text())
                state = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='],
                                       text=True, capture_output=True).stdout.strip()
                self.assertTrue(not state or state.startswith('Z'), f'orphaned rank {pid}: {state}')
            finally:
                if pid_file.exists():
                    try:
                        os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass


class WarmupInputsTests(unittest.TestCase):
    def test_w4_layout_matches_real_quantization_without_reference_allocation(self):
        y = torch.randn((2, 64, 128), dtype=torch.bfloat16, device='cuda')
        for mode in ('w4a16', 'w4fa16', 'w4fa16_mma', 'w4fa8_mma'):
            with self.subTest(mode=mode), mock.patch.dict(os.environ, HGGC_WARM_UP=''):
                real = utils.prepare_w4_weights(y, 32, mode)
                # Runtime treats every nonempty value, including "0", as true.
                for flag in ('1', '0'):
                    with mock.patch.dict(os.environ, HGGC_WARM_UP=flag), \
                            mock.patch.object(utils, 'quant_w4', side_effect=AssertionError('quantized during warmup')):
                        warm = utils.prepare_w4_weights(y, 32, mode)
                    self.assertIs(warm[0], y)
                    for actual, expected in zip(warm[1:3], real[1:3]):
                        self.assertEqual((actual.shape, actual.stride(), actual.dtype, actual.device),
                                         (expected.shape, expected.stride(), expected.dtype, expected.device))
                    self.assertEqual(warm[3], 1.0)

    def test_disable_accuracy_alone_keeps_real_quantization(self):
        y = torch.randn((1, 64, 128), dtype=torch.bfloat16, device='cuda')
        previous = utils.get_acc_check()
        try:
            utils.set_acc_check(False)
            with mock.patch.dict(os.environ, HGGC_WARM_UP=''), \
                    mock.patch.object(utils, 'quant_w4', wraps=utils.quant_w4) as quantize:
                result = utils.prepare_w4_weights(y, 32, 'w4fa16')
                quantize.assert_called_once()
                self.assertIsNot(result[0], y)
        finally:
            utils.set_acc_check(previous)


if __name__ == '__main__':
    unittest.main()
