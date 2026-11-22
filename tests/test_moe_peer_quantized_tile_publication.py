"""BF16+FP8 tile producer: byte parity, immutable boundaries and READY publication."""
import gc,json
import torch
import deep_gemm as dg


def fixture(width,capacity=512,tokens=83,measure=False):
    world,k=4,10
    x=(torch.randn(world,tokens,width,device='cuda')*.2).bfloat16()
    table=torch.tensor([x[j].data_ptr() for j in range(world)],device='cuda',dtype=torch.int64)
    perm=torch.full((capacity,),-1,device='cuda',dtype=torch.int32)
    bfraw=torch.full((capacity*width+128,),-19.,device='cuda',dtype=torch.bfloat16)
    bf=bfraw[64:-64].reshape(capacity,width);refbf=torch.empty_like(bf)
    qraw=torch.full((capacity*width+256,),117,device='cuda',dtype=torch.uint8)
    q=qraw[128:-128].view(torch.float8_e4m3fn).reshape(capacity,width);refq=torch.empty_like(q)
    sfraw=torch.full((capacity*(width//128)+64,),-17.,device='cuda');sf=sfraw[32:-32];refsf=torch.empty_like(sf)
    readyraw=torch.full((capacity//128+16,),173,device='cuda',dtype=torch.int32);ready=readyraw[8:-8]
    blocks=torch.zeros(1,device='cuda',dtype=torch.int32);status=torch.zeros_like(blocks)
    dims=[capacity,width,tokens,k,world,128]
    def control():dg._C._moe_training_peer_launch(21,[table,perm,refbf,status,blocks,refq,refsf],dims)
    def producer(ctas=8):
        ready.zero_()
        dg._C._moe_training_peer_launch(25,[table,perm,bf,status,blocks,q,sf,ready],[*dims,ctas])
    control();producer();torch.cuda.synchronize()
    cg,pg=torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):control()
    with torch.cuda.graph(pg):producer()
    for case,active in enumerate((40960,) if measure else (0,128,256,capacity)):
        ids=(torch.arange(capacity,device='cuda',dtype=torch.int64)*37+case*13)%(world*tokens*k)
        ids[::7]=-1;perm.copy_(ids.int());blocks.fill_(active//128);x.mul_(.97).add_(.001)
        bfraw.fill_(-19);qraw.fill_(117);sfraw.fill_(-17);refbf.fill_(-19);refq.view(torch.uint8).fill_(117);refsf.fill_(-17);status.zero_()
        cg.replay();pg.replay();torch.cuda.synchronize()
        safe=ids[:active].clamp(min=0);expected=x[safe//(tokens*k),(safe%(tokens*k))//k].clone();expected[ids[:active]<0]=0
        gates=dict(cpu_bf16_bits=torch.equal(bf[:active].view(torch.int16),expected.view(torch.int16)),bf16_bits=torch.equal(bf.view(torch.int16),refbf.view(torch.int16)),q_bits=torch.equal(q.view(torch.uint8),refq.view(torch.uint8)),scale_bits=torch.equal(sf.view(torch.uint8),refsf.view(torch.uint8)),bf_guards=bool(torch.all(bfraw[:64]==-19)&torch.all(bfraw[-64:]==-19)),q_guards=bool(torch.all(qraw[:128]==117)&torch.all(qraw[-128:]==117)),sf_guards=bool(torch.all(sfraw[:32]==-17)&torch.all(sfraw[-32:]==-17)),ready=torch.equal(ready, (torch.arange(capacity//128,device='cuda')<active//128).int()),ready_guards=bool(torch.all(readyraw[:8]==173)&torch.all(readyraw[-8:]==173)),status=status.item()==0)
        print(json.dumps(dict(case='quantized_tile_publication_reference',width=width,capacity=capacity,active=active,graph=True,mutable_sources=True,every_element=True,gates=gates,passed=all(gates.values()))),flush=True)
        assert all(gates.values()),gates
    blocks.fill_(1);perm[0]=world*tokens*k;status.zero_();pg.replay();torch.cuda.synchronize()
    assert status.item()==2 and not torch.count_nonzero(bf[0]) and not torch.count_nonzero(q[0].view(torch.uint8))
    perm[0]=-1
    for bad in (-1,capacity//128+1):
        blocks.fill_(bad);status.zero_();bfraw.fill_(-19);qraw.fill_(117);sfraw.fill_(-17)
        pg.replay();torch.cuda.synchronize()
        assert status.item()==2 and torch.all(bfraw==-19) and torch.all(qraw==117) and torch.all(sfraw==-17) and not torch.count_nonzero(ready)
    if measure:
        blocks.fill_(40960//128);status.zero_()
        def timing(g):
            for _ in range(3):g.replay()
            torch.cuda.synchronize();samples=[];b,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for _ in range(5):
                b.record()
                for _ in range(5):g.replay()
                e.record();e.synchronize();samples.append(b.elapsed_time(e)/5)
            return sorted(samples)[2]
        before=timing(cg)
        for ctas in (4,8,12,16,24,32):
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):producer(ctas)
            cg.replay();g.replay();torch.cuda.synchronize()
            assert torch.equal(bf.view(torch.int16),refbf.view(torch.int16)) and torch.equal(q.view(torch.uint8),refq.view(torch.uint8)) and torch.equal(sf.view(torch.uint8),refsf.view(torch.uint8)) and status.item()==0
            after=timing(g)
            print(json.dumps(dict(case='quantized_tile_publication_performance',producer_ctas=ctas,before_ms=before,after_ms=after,speedup=before/after,scope='Local producer and READY reset only; no GEMM overlap and no cross-host traffic',passed=True)),flush=True)
            del g
    del cg,pg,x,bf,q,sf,refbf,refq,refsf;gc.collect();torch.cuda.synchronize()


def main():
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(108116)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for width in (128,512,8192):fixture(width)
        fixture(8192,65536,4096,True)
    stream.synchronize()

if __name__=='__main__':main()
