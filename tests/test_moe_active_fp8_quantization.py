"""Active FP8 quantizers: all-buffer bitwise gates, graph changes and timing."""
import hashlib
import json
from pathlib import Path
import statistics

import torch
import deep_gemm as dg
from deep_gemm.training.native import _launch


def exact(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def fixture(capacity, width, kind, mn):
    device = torch.device('cuda', 0)
    backward, row = kind == 15, kind == 9
    out_width = 2 * width if backward else width
    source = (torch.randn(capacity, width if row else 2*width, device=device)*.2).bfloat16()
    hidden = (torch.randn(capacity, width, device=device)*.1).bfloat16() if not row else None
    reference_bf = torch.empty(capacity, out_width, device=device, dtype=torch.bfloat16) if not row else None
    actual_bf = torch.empty_like(reference_bf) if not row else None
    reference_q = torch.empty(capacity, out_width, device=device, dtype=torch.float8_e4m3fn)
    reference_sf = torch.empty(capacity, out_width//128, device=device)
    actual_q = torch.empty_like(reference_q)
    actual_storage = torch.empty((out_width//128, capacity) if mn else (capacity, out_width//128), device=device)
    actual_sf = actual_storage.t() if mn else actual_storage
    count = torch.zeros((), device=device, dtype=torch.int32)
    status = torch.zeros_like(count)

    def old():
        if row:tensors = [source, reference_q, reference_sf]
        elif backward:tensors = [source, hidden, reference_bf, reference_q, reference_sf]
        else:tensors = [source, reference_bf, reference_q, reference_sf]
        _launch(kind, tensors, [capacity, width])

    def new():
        if row:tensors = [source, actual_q, actual_storage]
        elif backward:tensors = [source, hidden, actual_bf, actual_q, actual_storage]
        else:tensors = [source, actual_bf, actual_q, actual_storage]
        _launch({9:26,14:27,15:28}[kind], [*tensors, count, status], [capacity, width, int(mn)])

    def poison():
        actual_q.view(torch.uint8).fill_(126)
        actual_storage.fill_(-17)
        if actual_bf is not None:actual_bf.fill_(-7)
        status.zero_()

    old(); poison(); new()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):new()
    counts = (0, 1, 17, capacity-1, capacity) if capacity < 1000 else (0, 40960, capacity)
    for trial, rows in enumerate(counts):
        source.add_(.001*(trial+1))
        if hidden is not None:hidden.mul_(.97)
        old(); poison(); count.fill_(rows); graph.replay()
        torch.cuda.synchronize()
        gates = dict(q_active=exact(actual_q[:rows], reference_q[:rows]),
                     scales_active=torch.equal(actual_sf[:rows], reference_sf[:rows]),
                     q_tail=bool(torch.all(actual_q.view(torch.uint8)[rows:] == 126)),
                     scales_tail=bool(torch.all(actual_sf[rows:] == -17)), status=int(status)==0)
        if actual_bf is not None:
            gates['bf_active'] = torch.equal(actual_bf[:rows], reference_bf[:rows])
            gates['bf_tail'] = bool(torch.all(actual_bf[rows:] == -7))
        passed = all(gates.values())
        print(json.dumps(dict(case='active_fp8_quantization_reference', capacity=capacity,
              width=width, old_operation=kind, mn_major=mn, rows=rows, graph=True,
              every_element=True, gates=gates, passed=passed)), flush=True)
        if not passed:raise AssertionError(gates)
    for rows in (-1, capacity+1):
        poison(); count.fill_(rows); graph.replay(); torch.cuda.synchronize()
        gates = dict(status=int(status)==2, q_unchanged=bool(torch.all(actual_q.view(torch.uint8)==126)),
                     scales_unchanged=bool(torch.all(actual_sf==-17)))
        if actual_bf is not None:gates['bf_unchanged']=bool(torch.all(actual_bf==-7))
        passed=all(gates.values())
        print(json.dumps(dict(case='active_fp8_quantization_reference', capacity=capacity,
              width=width, old_operation=kind, mn_major=mn, rows=rows, graph=True,
              invalid_count=True, gates=gates, passed=passed)), flush=True)
        if not passed:raise AssertionError(gates)
    if capacity >= 1000:
        status.zero_()
        for rows in (0,40960):
            count.fill_(rows)
            for label, launch in (('full_capacity_control',old), ('active_rows',new)):
                for _ in range(5):launch()
                samples=[]
                for _ in range(5):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(20):launch()
                    end.record();end.synchronize();samples.append(begin.elapsed_time(end)/20)
                print(json.dumps(dict(case='active_fp8_quantization_signal', capacity=capacity,
                      width=width, old_operation=kind, mn_major=mn, rows=rows,
                      variant=label, median_event_ms=statistics.median(samples), samples=samples,
                      scope='isolated quantizer; no EP/full-step speedup')), flush=True)
    del graph


def main():
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(620791)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        print(json.dumps(dict(case='active_fp8_quantization_provenance', device=torch.cuda.get_device_name(0),
              torch=torch.__version__, cus=dg.get_num_sms(), extension_sha256=hashlib.sha256(Path(dg._C.__file__).read_bytes()).hexdigest(),
              arithmetic='unchanged E4M3 row/128 FP32 scales; BF16 activation boundary')), flush=True)
        dg._C._moe_training_precompile(0,[9,14,15,26,27,28])
        for kind in (9,14,15):
            for mn in (False,True):
                for capacity,width in ((129,128),(257,256)):
                    fixture(capacity,width,kind,mn)
                fixture(65536,8192 if kind==9 else 2048,kind,mn)
    stream.synchronize()


if __name__=='__main__':main()
