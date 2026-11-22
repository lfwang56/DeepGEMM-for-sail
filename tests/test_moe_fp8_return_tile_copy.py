"""Validate READY-driven BF16 return copying against independent index_copy."""
import gc,json
import torch
import deep_gemm as dg


def fixture(width,capacity=512,stride=1024,measure=False):
    world=4
    src=(torch.randn(capacity,width,device='cuda')*.2).bfloat16()
    raw=torch.full((world,stride*width+256),-19.,device='cuda',dtype=torch.bfloat16)
    refraw=torch.full_like(raw,-19)
    dst=raw[:,128:-128].reshape(world,stride,width)
    ref=refraw[:,128:-128].reshape(world,stride,width)
    table=lambda t:torch.tensor([t[g].data_ptr() for g in range(world)],device='cuda',dtype=torch.int64)
    dt,rt=table(dst),table(ref)
    perm=torch.full((capacity,),-1,device='cuda',dtype=torch.int32)
    blocks=torch.zeros(1,device='cuda',dtype=torch.int32);status=torch.zeros_like(blocks)
    readyraw=torch.ones(capacity//128*(width//256)+16,device='cuda',dtype=torch.int32)*173
    ready=readyraw[8:-8];ready.fill_(1)
    def old():dg._C._moe_training_peer_launch(12,[src,perm,rt,status,blocks],[capacity,width,stride,world,128])
    def new(ctas=4):dg._C._moe_training_peer_launch(26,[src,perm,dt,blocks,ready,status],[capacity,width,stride,world,256,ctas])
    old();new();torch.cuda.synchronize();cg,pg=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):old()
    with torch.cuda.graph(pg):new()
    for step,active in enumerate((40960,) if measure else (0,128,256,capacity)):
        ids=(torch.arange(capacity,device='cuda',dtype=torch.int64)*37+step*13)%(world*stride)
        ids[::7]=-1;perm.copy_(ids.int());blocks.fill_(active//128);status.zero_();raw.fill_(-19);refraw.fill_(-19);src.mul_(.97).add_(.001)
        cg.replay();pg.replay();torch.cuda.synchronize()
        expected=torch.full((world*stride,width),-19.,device='cuda',dtype=torch.bfloat16)
        valid=ids[:active]>=0;expected.index_copy_(0,ids[:active][valid],src[:active][valid])
        gates=dict(torch_index_copy=torch.equal(dst.view(torch.int16),expected.reshape(world,stride,width).view(torch.int16)),legacy_copy_bits=torch.equal(dst.view(torch.int16),ref.view(torch.int16)),outer_guards=bool(torch.all(raw[:,:128]==-19)&torch.all(raw[:,-128:]==-19)),ready_read_only=bool(torch.all(ready==1)&torch.all(readyraw[:8]==173)&torch.all(readyraw[-8:]==173)),status=status.item()==0)
        print(json.dumps(dict(case='fp8_return_tile_copy_reference',width=width,capacity=capacity,active=active,graph=True,every_element=True,gates=gates,passed=all(gates.values()))),flush=True);assert all(gates.values()),gates
        del expected
    # Invalid routes and GPU counts cannot write beyond the owned outputs.
    blocks.fill_(1);perm[0]=world*stride;status.zero_();raw.fill_(-19);pg.replay();torch.cuda.synchronize()
    assert status.item()==2 and torch.all(raw[:,:128]==-19) and torch.all(raw[:,-128:]==-19)
    perm[0]=-1
    for bad in (-1,capacity//128+1):
        blocks.fill_(bad);status.zero_();raw.fill_(-19);pg.replay();torch.cuda.synchronize()
        assert status.item()==2 and torch.all(raw==-19)
    blocks.fill_(1);ready.zero_();status.fill_(64);raw.fill_(-19);pg.replay();torch.cuda.synchronize()
    assert status.item()==72 and torch.all(raw==-19)
    ready.fill_(1);status.zero_()
    if measure:
        blocks.fill_(40960//128)
        def timing(g):
            for _ in range(3):g.replay()
            torch.cuda.synchronize();samples=[];b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for _ in range(5):
                b.record()
                for _ in range(5):g.replay()
                e.record();e.synchronize();samples.append(b.elapsed_time(e)/5)
            return sorted(samples)[2]
        before=timing(cg)
        for ctas in (1,2,4,8,12,16,24,32):
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):new(ctas)
            cg.replay();g.replay();torch.cuda.synchronize();assert torch.equal(dst.view(torch.int16),ref.view(torch.int16)) and status.item()==0
            after=timing(g)
            print(json.dumps(dict(case='fp8_return_tile_copy_performance',copy_ctas=ctas,before_ms=before,after_ms=after,speedup=before/after,scope='Local BF16 scatter with READY already published; no GEMM or cross-host traffic',passed=True)),flush=True)
            del g
    del src,raw,refraw,dst,ref,cg,pg;gc.collect();torch.cuda.synchronize()


def main():
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(110116)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for width in (256,512,8192):fixture(width)
        fixture(8192,65536,40960,True)
    stream.synchronize()

if __name__=='__main__':main()
