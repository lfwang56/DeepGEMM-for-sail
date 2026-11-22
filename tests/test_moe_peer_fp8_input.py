"""Tile-published FP8 input: independent packing, mutable graphs, serial control.

All source owners are local buffers on one PPU. This is a component gate,
not a distributed communication or complete MoE step benchmark.
"""
import json,math,statistics
import torch
import deep_gemm as dg
from deep_gemm.training.native import _launch


def fixture(groups, n, k, capacity, tokens, cases, benchmark=False, ctas=2, threads=128, cache_scales=False):
    world=4
    source=(torch.randn(world*tokens,k,device='cuda')*.2).bfloat16()
    qs=torch.empty_like(source,dtype=torch.float8_e4m3fn)
    ss=torch.empty(world*tokens,k//128,device='cuda')
    qtables=torch.tensor([qs[j*tokens:].data_ptr() for j in range(world)],device='cuda',dtype=torch.int64)
    stables=torch.tensor([ss[j*tokens:].data_ptr() for j in range(world)],device='cuda',dtype=torch.int64)
    btables=torch.tensor([source[j*tokens:].data_ptr() for j in range(world)],device='cuda',dtype=torch.int64)
    weight=(torch.randn(groups,n,k,device='cuda')*.02).bfloat16()
    qw=torch.empty_like(weight,dtype=torch.float8_e4m3fn)
    sw=torch.empty(groups,n//128,k//128,device='cuda')
    _launch(30,[weight,qw,sw],list(weight.shape))
    rows=torch.full((capacity,),world*tokens,device='cuda',dtype=torch.int32)
    permutation=torch.full_like(rows,-1)
    lengths=torch.zeros(groups,device='cuda',dtype=torch.int32)
    group_ids=torch.zeros(capacity,device='cuda',dtype=torch.int32)
    blocks=torch.zeros(1,device='cuda',dtype=torch.int32)
    count=torch.zeros_like(blocks);status=torch.zeros_like(blocks)
    ready=torch.zeros(capacity//128,device='cuda',dtype=torch.int32)
    q=torch.empty(capacity,k,device='cuda',dtype=torch.float8_e4m3fn)
    sf=torch.empty(k//128,capacity,device='cuda')
    bf=torch.empty(capacity,k,device='cuda',dtype=torch.bfloat16)
    rq=torch.empty_like(q);rsf=torch.empty_like(sf);rbf=torch.empty_like(bf)
    expected=torch.empty(capacity,n,device='cuda',dtype=torch.bfloat16)
    serial=torch.empty_like(expected);parallel=torch.empty_like(expected)
    aux=torch.cuda.Stream()

    def producer(copy_bf=True, compile_only=False):
        dg._C._moe_training_peer_launch(19,[qtables,stables,btables,rows,q,sf,bf,blocks,ready,status],
            [capacity,k,tokens,world,ctas,int(copy_bf),threads,int(cache_scales)],compile_only)

    def consumer(output,compile_only=False):
        dg._C._moe_training_peer_fp8_gemm((q,sf.t()),(qw,sw),output,lengths,rows,btables,tokens,
            keep_local=True,compile_only=compile_only,input_control=[ready,status],reserved_input_ctas=ctas)

    def native():
        dg.m_grouped_fp8_gemm_nt_nopad((rq,rsf.t()),(qw,sw),expected,group_ids,
                                      m_rows=lengths,disable_ue8m0_cast=True)

    def step(output, overlap, copy_bf=True):
        _launch(9,[source,qs,ss],[world*tokens,k])
        ready.zero_()
        if overlap:
            main=torch.cuda.current_stream()
            aux.wait_stream(main)
            with torch.cuda.stream(aux):producer(copy_bf)
            consumer(output)
            main.wait_stream(aux)
        else:
            producer(copy_bf)
            consumer(output)

    def baseline():
        dg._C._moe_training_peer_launch(11,[btables,permutation,rbf,status,blocks],
            [capacity,k,tokens,1,world,128])
        _launch(26,[rbf,rq,rsf,count,status],[capacity,k,1])
        native()

    # Compile/initialize the zero-work path before forking a capture stream.
    producer(compile_only=True);consumer(serial,compile_only=True)
    step(serial,False);step(parallel,True);baseline()
    graphs=[]
    for output,overlap in ((serial,False),(parallel,True)):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):step(output,overlap)
        graphs.append(graph)

    baseline_graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(baseline_graph):baseline()

    for case_index,counts in enumerate(cases):
        assert len(counts)==groups and all(v%128==0 for v in counts)
        active=sum(counts);assert active<=capacity
        counts_gpu=torch.tensor(counts,device='cuda',dtype=torch.int32)
        lengths.copy_(counts_gpu);count.fill_(active);blocks.fill_(active//128)
        indices=torch.arange(capacity,device='cuda',dtype=torch.int64)
        indices=(indices*(17+2*case_index)+case_index*11)%(world*tokens)
        indices[indices%7==0]=world*tokens
        rows.copy_(indices.int());permutation.copy_(rows);permutation.masked_fill_(rows==world*tokens,-1)
        source.mul_(.97).add_(.001)
        status.zero_();q.view(torch.uint8).fill_(117);sf.fill_(-17);bf.fill_(-19)
        serial.fill_(-23);parallel.fill_(-23)
        for graph in graphs:graph.replay()
        torch.cuda.synchronize()
        assert status.item()==0
        safe=indices.clamp(max=world*tokens-1)
        # Torch independently constructs the packed original BF16 rows.
        rbf.copy_(source[safe]);rbf[indices==world*tokens]=0
        _launch(26,[rbf,rq,rsf,count,status],[capacity,k,1]);native()
        torch.cuda.synchronize()
        assert torch.equal(q[:active].view(torch.uint8),rq[:active].view(torch.uint8))
        assert torch.equal(sf[:,:active],rsf[:,:active])
        assert torch.equal(bf[:active],rbf[:active])
        assert torch.all(q[active:].view(torch.uint8)==117) and torch.all(sf[:,active:]==-17) and torch.all(bf[active:]==-19)
        errors={}
        for mode,out in (('serial',serial),('parallel',parallel)):
            a=out[:active].float();b=expected[:active].float()
            errors[mode]=math.sqrt(((a-b)**2).sum().item()/max((b*b).sum().item(),1e-30))
            assert errors[mode]<.003,(mode,errors)
            assert torch.all(out[active:]==-23)
        assert torch.equal(serial,parallel)
        print(json.dumps(dict(case='fp8_peer_input_component_reference',shape=[groups,n,k,capacity],counts=counts,
              graph=True,source_owners_local=True,producer_ctas=ctas,producer_threads=threads,cache_scales=cache_scales,all_quantization_bytes=True,all_scales=True,
              bf16_dw_copy=True,inactive_canaries=True,serial_parallel_exact=True,errors=errors,passed=True)),flush=True)
    # Copy omission must preserve BF16 scratch, without changing quantization.
    bf.fill_(-29);step(parallel,True,False);torch.cuda.synchronize()
    assert torch.all(bf==-29) and status.item()==0
    print(json.dumps(dict(case='fp8_peer_input_no_bf16_reference',shape=[groups,n,k,capacity],producer_ctas=ctas,producer_threads=threads,cache_scales=cache_scales,passed=True)),flush=True)
    if benchmark:
        for mode,graph in zip(('staged_bf16_native','serial','parallel'),[baseline_graph,*graphs]):
            for _ in range(5):graph.replay()
            torch.cuda.synchronize();samples=[]
            for _ in range(5):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):graph.replay()
                end.record();end.synchronize();samples.append(begin.elapsed_time(end)/20)
            print(json.dumps(dict(case='fp8_peer_input_component_performance',mode=mode,shape=[groups,n,k,capacity],
                  source_owners_local=True,producer_ctas=ctas,producer_threads=threads,cache_scales=cache_scales,consumer_ctas=dg.get_num_sms()-ctas,
                  source_quantization_in_timing=mode!='staged_bf16_native',bf16_dw_copy=True,median_ms=statistics.median(samples),
                  samples_ms=samples,scope='source row quantization + input producer + FC1 GEMM; local fake source owners; not complete step')),
                  flush=True)
    del graphs
    torch.cuda.synchronize()


def main():
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(208011)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fixture(4,512,512,2048,256,[[128,256,384,512],[0,128,512,128],[0,0,0,0],[512]*4])
        for ctas,threads,cache in ((8,256,False),(8,256,True),(8,512,True),(4,512,True)):
            fixture(16,4096,8192,65536,4096,[[0]*16,[2560]*16],benchmark=True,ctas=ctas,threads=threads,cache_scales=cache)
    stream.synchronize()


if __name__=='__main__':main()
