"""Fused dHidden/SwiGLU: BF16 and FP8 byte equivalence, alias safety and timing."""
import json
import statistics

import torch
import deep_gemm as dg
from deep_gemm.training.native import _launch


def fixture(g,n,k,r,large=False):
    alias=2*n==k
    qa=(torch.randn(r,k,device='cuda')*.2).to(torch.float8_e4m3fn)
    sa=torch.ones(k//128,r,device='cuda')
    qw=(torch.randn(g,n,k,device='cuda')*.02).to(torch.float8_e4m3fn)
    sw=torch.ones(g,n//128,k//128,device='cuda')
    original=(torch.randn(r,2*n,device='cuda')*.2).bfloat16()
    raw=torch.full((r*2*n+32,),-7.,device='cuda',dtype=torch.bfloat16)
    actual=raw[16:-16].reshape(r,2*n)
    expected=torch.empty_like(actual)
    dh=torch.empty(r,n,device='cuda',dtype=torch.bfloat16)
    unused=torch.full_like(dh,17.)
    qraw=torch.full((r*2*n+256,),113,device='cuda',dtype=torch.uint8)
    qactual=qraw[128:-128].view(torch.float8_e4m3fn).reshape(r,2*n)
    sraw=torch.full((r*(2*n//128)+64,),-17.,device='cuda')
    sactual=sraw[32:-32].reshape(2*n//128,r)
    qreference=torch.empty_like(qactual)
    sreference=torch.empty_like(sactual)
    input_q,input_s=(qactual,sactual) if alias else (qa,sa)
    lengths=torch.zeros(g,device='cuda',dtype=torch.int32)
    group_ids=torch.zeros(r,device='cuda',dtype=torch.int32)
    current=torch.zeros(1,device='cuda',dtype=torch.int32)
    status=torch.zeros_like(current)
    permutation=torch.arange(r,device='cuda',dtype=torch.int32)
    destinations=torch.tensor([unused.data_ptr()],device='cuda',dtype=torch.int64)
    def before():
        expected.copy_(original)
        dg.m_grouped_fp8_gemm_nt_nopad((qa,sa.t()),(qw,sw),dh,group_ids,
                                      m_rows=lengths,disable_ue8m0_cast=True)
        _launch(28,[expected,dh,expected,qreference,sreference,current,status],[r,n,1])
    def after():
        actual.copy_(original)
        if alias:
            input_q.copy_(qa)
            input_s.copy_(sa)
        dg._C._moe_training_peer_fp8_gemm((input_q,input_s.t()),(qw,sw),unused,lengths,
            permutation,destinations,r,keep_local=False,activation_gate_up=actual)
        _launch(34,[actual,qactual,sactual,current,status],[r,2*n,1])
    patterns=('balanced','updated') if large else ('balanced','mixed','empty','skew','updated')
    balanced=2560 if large else 128
    lengths.fill_(balanced);current.fill_(balanced*g)
    group_ids[:balanced*g].copy_(torch.arange(g).repeat_interleave(balanced).to(device='cuda',dtype=torch.int32))
    before();after();torch.cuda.synchronize()
    graphs=[]
    for function in (before,after):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):function()
        graphs.append(graph)
    for index,pattern in enumerate(patterns):
        counts=[balanced]*g
        if pattern=='mixed':counts=[256,0]+[128]*(g-2)
        if pattern=='empty':counts=[0]*g
        if pattern=='skew':counts=[r]+[0]*(g-1)
        active=sum(counts)
        lengths.copy_(torch.tensor(counts,dtype=torch.int32))
        current.fill_(active)
        group_ids.zero_()
        if active:group_ids[:active].copy_(torch.repeat_interleave(torch.arange(g),torch.tensor(counts)).to(device='cuda',dtype=torch.int32))
        original.normal_(0.,.2*(1+.1*index))
        qa.copy_((torch.randn(r,k,device='cuda')*.2).to(torch.float8_e4m3fn))
        qw.copy_((torch.randn(g,n,k,device='cuda')*.02).to(torch.float8_e4m3fn))
        for graph in graphs:graph.replay()
        torch.cuda.synchronize()
        gates=dict(bf16_gradient_bits=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)),
                   fp8_gradient_bits=torch.equal(qactual[:active].view(torch.uint8),qreference[:active].view(torch.uint8)),
                   scale_bits=torch.equal(sactual[:,:active].view(torch.int32),sreference[:,:active].view(torch.int32)),
                   original_dhidden_buffer_untouched=bool(torch.all(unused==17.)),
                   canaries=bool(torch.all(raw[:16]==-7) and torch.all(raw[-16:]==-7) and
                                 torch.all(qraw[:128]==113) and torch.all(qraw[-128:]==113) and
                                 torch.all(sraw[:32]==-17) and torch.all(sraw[-32:]==-17)),
                   status=int(status)==0)
        cpu_error=None
        if not large and active:
            gu=original[:active].cpu().float();gate,up=gu.split(n,-1)
            d=dh[:active].cpu().float();sigmoid=torch.sigmoid(gate)
            want=torch.cat((d*up*sigmoid*(1+gate*(1-sigmoid)),d*(gate*sigmoid)),-1).bfloat16().float()
            got=actual[:active].cpu().float()
            cpu_error=float((got-want).norm()/want.norm().clamp_min(1e-20))
            gates['independent_cpu_swiglu']=cpu_error<.002
        print(json.dumps(dict(case='fp8_dhidden_activation_reference',groups=g,n=n,k=k,capacity=r,
            pattern=pattern,input_output_quantization_alias=alias,graph=True,every_element=True,
            cpu_relative_error=cpu_error,gates=gates,passed=all(gates.values()))),flush=True)
        assert all(gates.values()),gates
    if large:
        def measure(graph):
            for _ in range(5):graph.replay()
            torch.cuda.synchronize();samples=[]
            for _ in range(5):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):graph.replay()
                end.record();end.synchronize();samples.append(start.elapsed_time(end)/20)
            return statistics.median(samples),samples
        old,old_samples=measure(graphs[0]);new,new_samples=measure(graphs[1])
        print(json.dumps(dict(case='qwen_fp8_dhidden_activation_performance',groups=g,n=n,k=k,capacity=r,
            before_ms=old,after_ms=new,speedup=old/new,before_samples=old_samples,after_samples=new_samples,
            scope='dHidden GEMM + SwiGLU backward + row quantization; both arms include the same gate/up restoration copy; weight/input quantization excluded; not a full-layer result',
            passed=True)),flush=True)


torch.cuda.set_device(0)
torch.set_num_threads(2)
torch.manual_seed(115115)
for n,k in ((128,256),(256,512),(384,512)):fixture(3,n,k,1024)
fixture(16,2048,8192,65536,True)
