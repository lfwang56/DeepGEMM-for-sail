"""Check fused staging against native staging+quantization, including dot order.

Source pointer tables refer to local buffers in this component test. Performance
is a local surrogate; real cross-host benefit requires the EP32 complete step.
"""
import gc,json,torch
import deep_gemm as dg

def fixture(width,capacity=512,tokens=83,measure=False):
    world,k=4,10
    x=(torch.randn(world,tokens,width,device='cuda')*.2).bfloat16()
    dy=(torch.randn_like(x)*.1).bfloat16()
    scores=torch.randn(world,tokens*k,device='cuda');scores[:,::11]=0
    ds=torch.full_like(scores,-23);refds=torch.full_like(scores,-23)
    tables=lambda t:torch.tensor([t[j].data_ptr() for j in range(world)],device='cuda',dtype=torch.int64)
    xt,dt,st,dst,rst=tables(x),tables(dy),tables(scores),tables(ds),tables(refds)
    perm=torch.full((capacity,),-1,device='cuda',dtype=torch.int32)
    y=(torch.randn(capacity,width,device='cuda')*.1).bfloat16()
    bfraw=torch.full((capacity*width+128,),-19.,device='cuda',dtype=torch.bfloat16)
    bf=bfraw[64:-64].reshape(capacity,width);refbf=torch.empty_like(bf)
    qraw=torch.full((capacity*width+256,),117,device='cuda',dtype=torch.uint8)
    q=qraw[128:-128].view(torch.float8_e4m3fn).reshape(capacity,width);refq=torch.empty_like(q)
    sfraw=torch.full((capacity*(width//128)+64,),-17.,device='cuda');sf=sfraw[32:-32];refsf=torch.empty_like(sf)
    blocks=torch.zeros(1,device='cuda',dtype=torch.int32);rows=torch.zeros_like(blocks);status=torch.zeros_like(blocks)
    dims=[capacity,width,tokens,k,world,128]
    def quant():dg._C._moe_training_launch(34,[refbf,refq,refsf,rows,status],[capacity,width,1])
    def separate_fwd():
        dg._C._moe_training_peer_launch(11,[xt,perm,refbf,status,blocks],dims)
        quant()
    def fused_fwd():dg._C._moe_training_peer_launch(21,[xt,perm,bf,status,blocks,q,sf],dims)
    def separate_bwd():
        dg._C._moe_training_peer_launch(13,[dt,st,rst,perm,y,refbf,blocks,status],dims)
        quant()
    def fused_bwd():dg._C._moe_training_peer_launch(22,[dt,st,dst,perm,y,bf,blocks,status,q,sf],dims)
    functions=[separate_fwd,fused_fwd,separate_bwd,fused_bwd];graphs=[]
    for fn in functions:
        fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        graphs.append(g)
    cases=(0,128,256,capacity) if not measure else (40960,)
    for case,active in enumerate(cases):
        ids=(torch.arange(capacity,device='cuda',dtype=torch.int64)*37+case*13)%(world*tokens*k)
        ids[::7]=-1;perm.copy_(ids.int());blocks.fill_(active//128);rows.fill_(active)
        x.mul_(.97).add_(.001);dy.mul_(.99).sub_(.001)
        for backward in (False,True):
            bfraw.fill_(-19);qraw.fill_(117);sfraw.fill_(-17);refbf.fill_(-19);refq.view(torch.uint8).fill_(117);refsf.fill_(-17);status.zero_();ds.fill_(-23);refds.fill_(-23)
            graphs[2 if backward else 0].replay();graphs[3 if backward else 1].replay();torch.cuda.synchronize()
            safe=ids[:active].clamp(min=0);owners=safe//(tokens*k);local=safe%(tokens*k)
            expected=(dy[owners,local//k].float()*scores[owners,local,None]).bfloat16() if backward else x[owners,local//k].clone()
            expected[ids[:active]<0]=0
            gates=dict(cpu_style_bf16_reference=torch.equal(bf[:active],expected),bf16_bits=torch.equal(bf,refbf),q_bits=torch.equal(q.view(torch.uint8),refq.view(torch.uint8)),scale_bits=torch.equal(sf.view(torch.uint8),refsf.view(torch.uint8)),bf16_guards=bool(torch.all(bfraw[:64]==-19)&torch.all(bfraw[-64:]==-19)),q_guards=bool(torch.all(qraw[:128]==117)&torch.all(qraw[-128:]==117)),sf_guards=bool(torch.all(sfraw[:32]==-17)&torch.all(sfraw[-32:]==-17)),dscore_bits=torch.equal(ds,refds),status=status.item()==0)
            print(json.dumps(dict(case='fused_peer_quantization_reference',width=width,capacity=capacity,active=active,backward=backward,graph=True,mutable_source_and_routes=True,zero_route_scores=True,source_owners_local=True,every_element=True,gates=gates,passed=all(gates.values()))),flush=True)
            assert all(gates.values()),gates
    # An invalid positive route is rejected and has zero packed data.
    blocks.fill_(1);rows.fill_(128);perm[0]=world*tokens*k
    for idx in (1,3):
        status.zero_();graphs[idx].replay();torch.cuda.synchronize()
        assert status.item()==2 and not torch.count_nonzero(bf[0]) and not torch.count_nonzero(q[0].view(torch.uint8))
    perm[0]=-1
    # Bad GPU work counts must preserve every output and canary.
    for bad in (-1,capacity//128+1):
        blocks.fill_(bad);status.zero_();bfraw.fill_(-19);qraw.fill_(117);sfraw.fill_(-17);ds.fill_(-23)
        for idx in (1,3):
            status.zero_();graphs[idx].replay();torch.cuda.synchronize()
            assert status.item()==2 and torch.all(bfraw==-19) and torch.all(qraw==117) and torch.all(sfraw==-17) and torch.all(ds==-23)
    if measure:
        blocks.fill_(40960//128);rows.fill_(40960);status.zero_()
        def timing(graph):
            for _ in range(3):graph.replay()
            torch.cuda.synchronize();samples=[];begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for _ in range(7):
                begin.record()
                for _ in range(5):graph.replay()
                end.record();end.synchronize();samples.append(begin.elapsed_time(end)/5)
            return sorted(samples)[3]
        for backward in (False,True):
            before,after=timing(graphs[2 if backward else 0]),timing(graphs[3 if backward else 1])
            print(json.dumps(dict(case='fused_peer_quantization_performance',width=width,capacity=capacity,active=40960,backward=backward,before_ms=before,after_ms=after,speedup=before/after,source_owners_local=True,scope='Local surrogate staging plus unchanged FP8 row quantization; not a cross-host or whole-step result',passed=True)),flush=True)
    del graphs,x,dy,y,bf,q,sf,refbf,refq,refsf;gc.collect();torch.cuda.synchronize()

def main():
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(101116)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for width in (128,512,8192):fixture(width)
        fixture(8192,65536,4096,True)
    stream.synchronize()
if __name__=='__main__':main()
