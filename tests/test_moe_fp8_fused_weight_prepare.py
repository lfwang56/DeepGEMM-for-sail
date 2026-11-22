"""One-read Q/QT preparation: CPU arithmetic, exact bytes, graphs and Qwen timing."""
import json
import math
import statistics

import torch
import deep_gemm as dg
from deep_gemm.training.native import _launch


def outputs(e,h,i,guard=False):
    result=[]
    guards=[]
    for n,k in ((2*i,h),(h,i),(h,2*i),(i,h)):
        for shape,dtype in (((e,n,k),torch.float8_e4m3fn),((e,n//128,k//128),torch.float32)):
            if guard:
                raw=torch.full((math.prod(shape)+256,),113 if dtype==torch.float8_e4m3fn else -17,
                               device='cuda',dtype=torch.uint8 if dtype==torch.float8_e4m3fn else torch.float32)
                result.append(raw[128:-128].view(dtype).reshape(shape))
                guards.append((raw,113 if dtype==torch.float8_e4m3fn else -17))
            else:
                result.append(torch.empty(shape,device='cuda',dtype=dtype))
    return result,guards


def same(actual,expected):
    return all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(actual,expected))


def old_prepare(weights,result,e,h,i):
    for w,q,s in ((weights[0],result[0],result[1]),(weights[1],result[2],result[3])):
        _launch(30,[w,q,s],list(w.shape))
    _launch(35,result,[e,h,i])


def unit(ctas):
    e,h,i=2,512,256
    weights=[torch.empty(e,2*i,h,device='cuda',dtype=torch.bfloat16),
             torch.empty(e,h,i,device='cuda',dtype=torch.bfloat16)]
    actual,guards=outputs(e,h,i,True)
    expected,_=outputs(e,h,i)
    def launch():_launch(36,[*weights,*actual],[e,h,i,ctas])
    launch()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):launch()
    for pattern in ('random','zero','special','updated'):
        for index,w in enumerate(weights):
            w.normal_(0.,.07 if pattern!='updated' else 2.)
            if pattern=='zero':w.zero_()
            if pattern=='special':
                bits=torch.tensor([0,-32768,1,-32767,0x7f7f,-129,0x7f80,-128,0x7fc1,-63],
                                  device='cuda',dtype=torch.int16)
                for j in range(bits.numel()):w.view(torch.int16).flatten()[j::4093]=bits[j]
                if index==0:w[:,0:128,0:128]=float('nan')
        old_prepare(weights,expected,e,h,i)
        graph.replay()
        torch.cuda.synchronize()
        assert same(actual,expected),(ctas,pattern,'legacy byte equivalence')
        for raw,value in guards:
            assert torch.all(raw[:128]==value) and torch.all(raw[-128:]==value)
        cpu_checked=pattern!='special'
        if cpu_checked:
            for j,w in enumerate(weights):
                g,n,k=w.shape
                cpu=w.cpu().float()
                blocks=cpu.reshape(g,n//128,128,k//128,128).permute(0,1,3,2,4)
                sf=blocks.abs().amax((-1,-2)).clamp_min(1e-4)/448
                q=(cpu/sf.repeat_interleave(128,1).repeat_interleave(128,2)).to(torch.float8_e4m3fn)
                assert torch.equal(q.view(torch.uint8),actual[2*j].cpu().view(torch.uint8))
                assert torch.equal(sf.view(torch.int32),actual[2*j+1].cpu().view(torch.int32))
                assert torch.equal(q.transpose(1,2).contiguous().view(torch.uint8),actual[4+2*j].cpu().view(torch.uint8))
                assert torch.equal(sf.transpose(1,2).contiguous().view(torch.int32),actual[5+2*j].cpu().view(torch.int32))
        print(json.dumps(dict(case='fp8_fused_weight_prepare_reference',ctas_per_cu=ctas,
            pattern=pattern,graph=True,mutable_weights=True,canaries=True,every_byte=True,
            legacy_q_and_qt_bits=True,independent_cpu_arithmetic=cpu_checked,
            nonfinite_and_all_nan_blocks=pattern=='special',passed=True)),flush=True)


def large():
    e,h,i=16,8192,2048
    weights=[(torch.randn(e,2*i,h,device='cuda')*.01).bfloat16(),
             (torch.randn(e,h,i,device='cuda')*.01).bfloat16()]
    actual,_=outputs(e,h,i)
    expected,_=outputs(e,h,i)
    def before():old_prepare(weights,expected,e,h,i)
    def measure(fn):
        for _ in range(5):fn()
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        times=[]
        for _ in range(5):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(20):graph.replay()
            end.record();end.synchronize();times.append(begin.elapsed_time(end)/20)
        return statistics.median(times),times
    baseline,samples_before=measure(before)
    for ctas in (1,2,4,8,16):
        def after():_launch(36,[*weights,*actual],[e,h,i,ctas])
        after();torch.cuda.synchronize()
        assert same(actual,expected)
        elapsed,samples_after=measure(after)
        print(json.dumps(dict(case='qwen_fp8_fused_weight_prepare_performance',experts=e,hidden=h,
            intermediate=i,ctas_per_cu=ctas,before_ms=baseline,after_ms=elapsed,speedup=baseline/elapsed,
            before_samples=samples_before,after_samples=samples_after,every_byte=True,passed=True,
            scope='Current BF16 weights to forward/backward E4M3 weights and FP32 scales; not a full-layer result')),flush=True)


torch.cuda.set_device(0)
torch.set_num_threads(2)
torch.manual_seed(141141)
for ctas in (1,2,4,8,16):unit(ctas)
large()
