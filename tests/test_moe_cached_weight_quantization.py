"""Compare cached weight quantizers with the existing E4M3 block quantizer."""
import hashlib
import json
from pathlib import Path
import statistics

import torch
import deep_gemm as dg
from deep_gemm.training.native import _launch


def fixture(shape):
    weight = (torch.randn(shape, device='cuda') * .2).bfloat16()
    elements = weight.numel()
    scale_count = shape[0] * (shape[1] // 128) * (shape[2] // 128)
    reference_q = torch.empty(shape, device='cuda', dtype=torch.float8_e4m3fn)
    reference_sf = torch.empty(scale_count, device='cuda')
    raw_q = torch.full((elements + 256,), 126, device='cuda', dtype=torch.uint8)
    actual_q = raw_q[128:-128].view(torch.float8_e4m3fn).view(shape)
    raw_sf = torch.full((scale_count + 64,), -17., device='cuda')
    actual_sf = raw_sf[32:-32]

    def launch(op):
        _launch(op, [weight, reference_q if op == 10 else actual_q,
                     reference_sf if op == 10 else actual_sf], list(shape))

    graphs = {}
    for op in (29, 30):
        launch(op)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            launch(op)
        graphs[op] = graph
    for pattern in ('random', 'zero', 'special', 'updated'):
        if pattern == 'zero':
            weight.zero_()
        elif pattern == 'special':
            # Include signed zeros, BF16 subnormals, finite extremes, Inf and
            # NaNs. Spread them over different 128x128 scale blocks.
            weight.normal_(0., .2)
            bits = torch.tensor([0, -32768, 1, -32767, 0x7f7f, -129,
                                 0x7f80, -128, 0x7fc1, -63],
                                device='cuda', dtype=torch.int16)
            for index in range(bits.numel()):
                weight.view(torch.int16).flatten()[index::4093] = bits[index]
        elif pattern == 'updated':
            weight.normal_(0., 2.)
        launch(10)
        for op in (29, 30):
            actual_q.view(torch.uint8).fill_(126)
            actual_sf.fill_(-17)
            graphs[op].replay()
            torch.cuda.synchronize()
            gates = dict(q_bits=torch.equal(actual_q.view(torch.uint8), reference_q.view(torch.uint8)),
                         scale_bits=torch.equal(actual_sf.view(torch.uint8), reference_sf.view(torch.uint8)),
                         q_canaries=bool(torch.all(raw_q[:128] == 126) & torch.all(raw_q[-128:] == 126)),
                         scale_canaries=bool(torch.all(raw_sf[:32] == -17) & torch.all(raw_sf[-32:] == -17)))
            print(json.dumps(dict(case='cached_weight_quantization_reference', shape=shape,
                                  operation=op, pattern=pattern, graph=True,
                                  every_element=True, gates=gates, passed=all(gates.values()))), flush=True)
            if not all(gates.values()):
                raise AssertionError(gates)
    if elements >= 16 * 2048 * 8192:
        for op in (10, 29, 30):
            for _ in range(5):
                launch(op)
            samples = []
            for _ in range(7):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(10):
                    launch(op)
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) / 10)
            print(json.dumps(dict(case='cached_weight_quantization_signal', shape=shape,
                                  operation=op, median_event_ms=statistics.median(samples),
                                  samples=samples, scope='isolated weight quantization only')), flush=True)
    del graphs


def main():
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(660817)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        print(json.dumps(dict(case='cached_weight_quantization_provenance',
                              device=torch.cuda.get_device_name(0), cus=dg.get_num_sms(),
                              extension_sha256=hashlib.sha256(Path(dg._C.__file__).read_bytes()).hexdigest(),
                              recipe='unchanged E4M3 128x128 block scale, FP32 scale and division')), flush=True)
        dg._C._moe_training_precompile(0, [10, 26, 27, 28, 29, 30])
        for shape in ((1, 128, 128), (3, 256, 384), (16, 4096, 8192),
                      (16, 8192, 4096), (16, 8192, 2048), (16, 2048, 8192)):
            fixture(shape)
    stream.synchronize()


if __name__ == '__main__':
    main()
