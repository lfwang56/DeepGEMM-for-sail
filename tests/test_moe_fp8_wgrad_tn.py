"""K-grouped FP8 DW: independent CPU arithmetic, mutable counts and real shapes.

This primitive test prepares panels outside the timing region. Its performance
numbers are compute-only; production panel construction is a separate gate.
"""
import gc
import json
import os

COMPACT=True
import torch
import deep_gemm as dg


def panels(left,right,counts,offsets):
    r,m=left.shape;n=right.shape[1]
    ql=torch.full((r,m),56,device=left.device,dtype=torch.uint8).view(torch.float8_e4m3fn)
    qr=torch.full((r,n),56,device=left.device,dtype=torch.uint8).view(torch.float8_e4m3fn)
    sl=torch.full((len(counts),m),.7,device=left.device)
    sr=torch.full((len(counts),n),.7,device=left.device)
    expected=[]
    for g,c in enumerate(counts):
        if not c:
            expected.append(None);continue
        start=offsets[g];padded=(c+127)//128*128
        a=torch.zeros(m,padded,device=left.device);a[:,:c]=left[start:start+c].float().t()
        b=torch.zeros(n,padded,device=left.device);b[:,:c]=right[start:start+c].float().t()
        sa=a.abs().amax(-1).clamp_min(1e-4)/448
        sb=b.abs().amax(-1).clamp_min(1e-4)/448
        aq=(a/sa[:,None]).to(torch.float8_e4m3fn)
        bq=(b/sb[:,None]).to(torch.float8_e4m3fn)
        sl[g].copy_(sa);sr[g].copy_(sb)
        ql[start:start+padded].copy_(aq.t())
        qr[start:start+padded].copy_(bq.t())
        if left.device.type=='cpu':
            ar=aq.float()*sa[:,None]
            br=bq.float()*sb[:,None]
            expected.append((ar@br.t(),ar.abs()@br.abs().t()))
    return (ql,sl),(qr,sr),expected


def rel(a,b):return float((a-b).float().norm()/b.float().norm().clamp_min(1e-20))


def unit(m,n):
    g,r=3,2048;offsets=[128,640,1152,1664]
    rng=torch.Generator().manual_seed(4101+m+n)
    left=(torch.randn(r,m,generator=rng)*.1).bfloat16()
    right=(torch.randn(r,n,generator=rng)*.1).bfloat16()
    counts=[0,129,257]
    a,b,_=panels(left,right,counts,offsets)
    a=tuple(x.cuda() for x in a);b=tuple(x.cuda() for x in b)
    lens=torch.tensor(counts,device='cuda',dtype=torch.int32);starts=torch.tensor(offsets,device='cuda',dtype=torch.int32)
    status=torch.zeros(1,device='cuda',dtype=torch.int32)
    raw=torch.full((g*m*n+32,),113.,device='cuda');out=raw[16:-16].reshape(g,m,n)
    def launch():dg._C._moe_training_fp8_wgrad_tn(a,b,out,lens,starts,status)
    out.zero_();launch();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):launch()
    errors=[]
    for change,counts in enumerate(([0,129,257],[257,7,511],[512,0,1])):
        al=(left.float()*(1+.13*change)).bfloat16();bl=(right.float()*(1-.09*change)+.001*change).bfloat16()
        ac,bc,expected=panels(al,bl,counts,offsets)
        for x,y in zip(a,ac):x.copy_(y)
        for x,y in zip(b,bc):x.copy_(y)
        lens.copy_(torch.tensor(counts,dtype=torch.int32));starts.copy_(torch.tensor(offsets,dtype=torch.int32))
        initial=torch.randn(g,m,n,generator=rng)*.01;out.copy_(initial)
        graph.replay();torch.cuda.synchronize()
        want=initial.clone()
        bounds=torch.zeros_like(want)
        for j,value in enumerate(expected):
            if value is not None:want[j]+=value[0];bounds[j]=value[1]
        actual=out.cpu();err=rel(actual,want);errors.append(err)
        scaled_error=float(((actual-want).abs()/(bounds+1e-5)).max())
        assert err<.003 and scaled_error<.001,(err,scaled_error)
        # Verify the reused Tensor Cell mainloop independently against the
        # established native FP8 GEMM, at its BF16 output boundary. This
        # separates inherited scale/accumulator rounding from layout errors.
        out.zero_();graph.replay();torch.cuda.synchronize()
        for j,count in enumerate(counts):
            if not count:
                assert not torch.count_nonzero(out[j]);continue
            start=offsets[j];k=(count+127)//128*128
            qa=a[0][start:start+k].t().contiguous()
            sa=a[1][j].reshape(m,1)
            qb=b[0][start:start+k].t().contiguous().unsqueeze(0)
            sb=b[1][j].reshape(1,n,1)
            native=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
            info=torch.zeros(m,device='cuda',dtype=torch.int32)
            length=torch.tensor([m],device='cuda',dtype=torch.int32)
            dg._C.m_grouped_int8_gemm_nt_nopad((qa,sa),(qb,sb),native,info,m_rows=length,
                disable_ue8m0_cast=True,configs=(dg.get_num_sms(),128,128,128,32,64,3,(0,0,0)))
            torch.cuda.synchronize()
            assert torch.equal(out[j].bfloat16(),native),'DW primitive/native FP8 boundary mismatch'
        assert not status.item() and torch.all(raw[:16]==113) and torch.all(raw[-16:]==113)
    before=out.clone();lens[1]=513;status.zero_();graph.replay();torch.cuda.synchronize()
    assert status.item()==2 and torch.equal(out[1],before[1])
    assert torch.all(raw[:16]==113) and torch.all(raw[-16:]==113)
    print(json.dumps(dict(case='fp8_k_grouped_dw_cpu_reference',row_scales=True,token_major_panels=True,compact_panels=False,m=m,n=n,groups=g,capacity=r,mutable_counts=True,nonzero_offsets=True,tail_zero=True,zero_expert=True,fp32_accumulation=True,guarded_output=True,invalid_metadata_guard=True,every_element=True,relative_errors=errors,per_element_sum_absolute_product_error_bound=.001,native_fp8_bf16_boundary_bitwise=True,passed=True)),flush=True)


def large(role):
    g,r,k=16,65536,2560;m,n=(4096,8192) if role=='dw1' else (8192,2048)
    torch.manual_seed(91317)
    left=(torch.randn(r,m,device='cuda')*.01).bfloat16()
    right=(torch.randn(r,n,device='cuda')*.01).bfloat16()
    counts=[k]*g;offsets=[j*k for j in range(g+1)]
    a,b,_=panels(left,right,counts,offsets)
    lengths=torch.tensor(counts,device='cuda',dtype=torch.int32);starts=torch.tensor(offsets,device='cuda',dtype=torch.int32)
    status=torch.zeros(1,device='cuda',dtype=torch.int32);out=torch.zeros(g,m,n,device='cuda');bf=torch.zeros_like(out)
    def fp():dg._C._moe_training_fp8_wgrad_tn(a,b,out,lengths,starts,status)
    def bf16():dg.k_grouped_bf16_gemm_tn_device(left,right,bf,lengths,r)
    fp();torch.cuda.synchronize();assert not status.item()
    errors=[]
    for group in (0,7,15):
        start=group*k;features=torch.tensor([0,m//3,m-1],device='cuda');columns=torch.tensor([0,n//3,n-1],device='cuda')
        ap=a[0][start:start+k].t()
        bp=b[0][start:start+k].t()
        av=ap[features].float()*a[1][group,features][:,None]
        bv=bp[columns].float()*b[1][group,columns][:,None]
        want=av.reshape(3,k)@bv.reshape(3,k).t();actual=out[group][features[:,None],columns[None,:]]
        error=rel(actual,want);errors.append(error);assert error<.003,error
    def measure(fn):
        for _ in range(2):fn()
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);times=[]
        for _ in range(7):
            begin.record()
            for _ in range(5):graph.replay()
            end.record();end.synchronize();times.append(begin.elapsed_time(end)/5)
        return sorted(times)[len(times)//2]
    before,after=measure(bf16),measure(fp)
    print(json.dumps(dict(case='qwen_fp8_dw_compute_only',row_scales=True,token_major_panels=True,compact_panels=False,role=role,m=m,n=n,groups=g,k_per_group=k,capacity=r,bf16_ms=before,fp8_ms=after,speedup=before/after,fp32_accumulation=True,sampled_reference_errors=errors,quantization="one scale per expert feature across token K",scope='Prequantized DW compute only; panel quantization/transpose excluded and not a whole-step result',passed=True)),flush=True)
    del left,right,a,b,out,bf;gc.collect();torch.cuda.synchronize()


torch.cuda.set_device(0);torch.set_num_threads(2)
for shape in ((128,128),(256,384),(384,512)):unit(*shape)
for role in ('dw1','dw2'):large(role)
