"""Native blockwise FP8 GEMM versus the same GEMM with fused BF16 return."""
import hashlib,json,math,os,statistics
from pathlib import Path
import torch
import deep_gemm as dg
from deep_gemm.training.native import _launch


def fixture(groups, n, k, capacity):
    world=4; destination_rows=math.ceil(capacity/world)
    x=(torch.randn(capacity,k,device='cuda')*.2).bfloat16()
    w=(torch.randn(groups,n,k,device='cuda')*.02).bfloat16()
    qx=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    sx_store=torch.empty(k//128,capacity,device='cuda');sx=sx_store.t()
    qw=torch.empty_like(w,dtype=torch.float8_e4m3fn)
    sw=torch.empty(groups,n//128,k//128,device='cuda')
    full=torch.tensor(capacity,device='cuda',dtype=torch.int32)
    num_blocks=torch.zeros_like(full);status=torch.zeros_like(full)
    lengths=torch.zeros(groups,device='cuda',dtype=torch.int32)
    group_ids=torch.zeros(capacity,device='cuda',dtype=torch.int32)
    permutation=torch.full((capacity,),-1,device='cuda',dtype=torch.int32)
    reference=torch.empty(capacity,n,device='cuda',dtype=torch.bfloat16)
    actual=torch.empty_like(reference)
    reference_dest=[torch.empty(destination_rows,n,device='cuda',dtype=torch.bfloat16) for _ in range(world)]
    actual_dest=[torch.empty_like(t) for t in reference_dest]
    reference_ptrs=torch.tensor([t.data_ptr() for t in reference_dest],device='cuda',dtype=torch.int64)
    actual_ptrs=torch.tensor([t.data_ptr() for t in actual_dest],device='cuda',dtype=torch.int64)

    def quantize():
        _launch(26,[x,qx,sx_store,full,status],[capacity,k,1])
        _launch(30,[w,qw,sw],list(w.shape))

    def native():
        dg.m_grouped_fp8_gemm_nt_nopad((qx,sx),(qw,sw),reference,group_ids,
                                      m_rows=lengths,disable_ue8m0_cast=True)
        dg._C._moe_training_peer_launch(12,[reference,permutation,reference_ptrs,status,num_blocks],
                                        [capacity,n,destination_rows,world])

    def fused(keep):
        dg._C._moe_training_peer_fp8_gemm((qx,sx),(qw,sw),actual,lengths,permutation,
                                          actual_ptrs,destination_rows,keep_local=keep,
                                          return_store_bytes=int(os.getenv('DG_TEST_FP8_RETURN_STORE_BYTES','4')))

    def reset(output,destinations):
        output.fill_(-7)
        for t in destinations:t.fill_(-9)

    def metadata(pattern):
        balanced=128 if capacity<1000 else 2560
        counts=[balanced]*groups
        if pattern=='skew':counts=[capacity]+[0]*(groups-1)
        if pattern=='empty':counts=[0]*groups
        if capacity<1000 and pattern not in ('skew','empty'):counts[1]=256
        count=sum(counts)
        lengths.copy_(torch.tensor(counts,device='cuda',dtype=torch.int32));num_blocks.fill_(count//128)
        ids=torch.repeat_interleave(torch.arange(groups),torch.tensor(counts))
        group_ids.zero_();group_ids[:count].copy_(ids.to('cuda',dtype=torch.int32))
        permutation.fill_(-1)
        routes=torch.arange(count,device='cuda',dtype=torch.int32)
        if pattern=='reversed':routes=routes.flip(0)
        if pattern=='masked':routes[::11]=-1
        permutation[:count].copy_(routes)
        return count

    metadata('balanced');quantize();native()
    graphs={}
    for keep in (True,False):
        fused(keep)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):
            reset(actual,actual_dest);quantize();fused(keep)
        graphs[keep]=graph
    for index,pattern in enumerate(('balanced','masked','skew','empty','reversed','updated')):
        count=metadata(pattern);x.add_(.001*(index+1));w.mul_(.99)
        quantize();reset(reference,reference_dest);native()
        route_ids=permutation[:count].long()
        scatter_matches=True
        for owner,destination in enumerate(reference_dest):
            source_rows=((route_ids>=0)&(route_ids//destination_rows==owner)).nonzero().flatten()
            wanted=torch.full_like(destination,-9)
            wanted.index_copy_(0,route_ids[source_rows]%destination_rows,reference[source_rows])
            scatter_matches &= torch.equal(wanted.view(torch.uint8),destination.view(torch.uint8))
        print(json.dumps(dict(case='peer_fp8_scatter_oracle_reference',shape=[groups,n,k],
                              capacity=capacity,pattern=pattern,reference='Torch index_copy of native BF16 output',
                              every_element=True,passed=scatter_matches)),flush=True)
        if not scatter_matches:raise AssertionError('independent route scatter oracle mismatch')
        for keep,graph in graphs.items():
            graph.replay();torch.cuda.synchronize()
            gates=dict(peer_bits=all(torch.equal(a.view(torch.uint8),b.view(torch.uint8))
                                     for a,b in zip(actual_dest,reference_dest)),
                       local_bits=torch.equal(actual.view(torch.uint8),reference.view(torch.uint8))
                                  if keep else bool(torch.all(actual==-7)),status=int(status)==0)
            print(json.dumps(dict(case='peer_fp8_gemm_reference',shape=[groups,n,k],capacity=capacity,
                                  pattern=pattern,keep_local=keep,graph=True,every_element=True,
                                  gates=gates,passed=all(gates.values()))),flush=True)
            if not all(gates.values()):
                differences=[]
                for owner,(got,expected) in enumerate(zip(actual_dest,reference_dest)):
                    changed=(got!=expected).nonzero()
                    positions=changed[:12].cpu().tolist()
                    differences.append(dict(owner=owner,mismatched_elements=len(changed),
                        pair_swap_matches=torch.equal(got.reshape(-1,2).flip(1),expected.reshape(-1,2)),
                        samples=[dict(row=r,column=c,got=float(got[r,c]),expected=float(expected[r,c])) for r,c in positions]))
                changed=(actual!=reference).nonzero() if keep else torch.empty(0,2)
                print(json.dumps(dict(case='peer_fp8_gemm_mismatch_diagnostic',shape=[groups,n,k],
                    keep_local=keep,local_mismatches=len(changed),local_samples=changed[:12].cpu().tolist(),
                    owners=differences)),flush=True)
                raise AssertionError(gates)
    if capacity>=1000:
        metadata('balanced');quantize()
        for label,call in [('native_plus_scatter',native),('fused_return_discard',lambda:fused(False)),
                           ('fused_return_keep',lambda:fused(True))]:
            for _ in range(5):call()
            samples=[]
            for _ in range(5):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(10):call()
                end.record();end.synchronize();samples.append(begin.elapsed_time(end)/10)
            print(json.dumps(dict(case='peer_fp8_gemm_signal',shape=[groups,n,k],capacity=capacity,
                                  variant=label,median_event_ms=statistics.median(samples),samples=samples,
                                  scope='isolated GEMM and return on one GPU with four local fake owners; no actual EP/full-step claim')),flush=True)
    del graphs


def main():
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(700921)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        print(json.dumps(dict(case='peer_fp8_gemm_provenance',device=torch.cuda.get_device_name(0),
                              cus=dg.get_num_sms(),extension_sha256=hashlib.sha256(Path(dg._C.__file__).read_bytes()).hexdigest(),
                              recipe='E4M3 activation row128/W128x128 FP32 scales; BF16 peer return; local packed input')),flush=True)
        dg._C._moe_training_precompile(0,[26,30])
        for args in ((3,256,256,768),(16,8192,2048,65536),(16,8192,4096,65536)):
            fixture(*args)
    stream.synchronize()


if __name__=='__main__':main()
