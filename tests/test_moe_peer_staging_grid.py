"""Change only persistent grids; require bitwise BF16/FP32 staging equivalence."""
import gc,json,os,torch
import deep_gemm as dg
KEY='DG_MOE_PEER_STAGE_CTAS_PER_CU'

def fixture(width,bm,capacity=512,tokens=83,measure=False):
    world,k=4,10
    x=(torch.randn(world,tokens,width,device='cuda')*.2).bfloat16();dy=(torch.randn_like(x)*.1).bfloat16()
    scores=torch.randn(world,tokens*k,device='cuda');scores[:,::11]=0
    table=lambda x:torch.tensor([x[j].data_ptr() for j in range(world)],device='cuda',dtype=torch.int64)
    xt,dt,st=table(x),table(dy),table(scores)
    perm=torch.full((capacity,),-1,device='cuda',dtype=torch.int32);y=(torch.randn(capacity,width,device='cuda')*.1).bfloat16()
    def guarded(n,dtype,value):
        raw=torch.full((n+128,),value,device='cuda',dtype=dtype);return raw,raw[64:-64]
    guards=[];buffers=[]
    for _ in range(2):
        raw,a=guarded(capacity*width,torch.bfloat16,-19);guards.append((raw,-19));raw,b=guarded(capacity*width,torch.bfloat16,-19);guards.append((raw,-19))
        raw,c=guarded(capacity,torch.float32,-23);guards.append((raw,-23));d=torch.full_like(scores,-29)
        buffers.append((a.reshape(capacity,width),b.reshape(capacity,width),c,d,table(d)))
    blocks=torch.zeros(1,device='cuda',dtype=torch.int32);status=torch.zeros_like(blocks);dims=[capacity,width,tokens,k,world,bm]
    def launch(op,reference):
        a,b,c,d,dst=buffers[reference]
        tensors=[xt,perm,a,status,blocks] if op==11 else [dt,st,dst,perm,y,b,blocks,status] if op==13 else [dt,st,perm,a,b,c,blocks,status]
        dg._C._moe_training_peer_launch(op,tensors,dims)
    graphs={}
    for op in (11,13,18):
        for ctas in (0,1,2,4,8,16):
            os.environ[KEY]=str(ctas);launch(op,int(ctas==0));torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):launch(op,int(ctas==0))
            graphs[op,ctas]=g
    cases=(0,bm,capacity) if not measure else (40960,)
    for case,active in enumerate(cases):
        ids=(torch.arange(capacity,device='cuda',dtype=torch.int64)*37+case*13)%(world*tokens*k);ids[::7]=-1
        perm.copy_(ids.int());blocks.fill_(active//bm);x.mul_(.97).add_(.001);dy.mul_(.99).sub_(.001)
        safe=ids[:active].clamp(min=0);owners=safe//(tokens*k);local=safe%(tokens*k);valid=ids[:active]>=0
        for op in (11,13,18):
            for ctas in (1,2,4,8,16):
                for raw,v in guards:raw.fill_(v)
                for a,b,c,d,dtbl in buffers:d.fill_(-29)
                status.zero_();graphs[op,0].replay();graphs[op,ctas].replay();torch.cuda.synchronize()
                a,b,c,d,_=buffers[0];ra,rb,rc,rd,_=buffers[1]
                expected=x[owners,local//k].clone() if op==11 else (dy[owners,local//k].float()*scores[owners,local,None]).bfloat16()
                expected[~valid]=0
                gates=dict(all_bf16_and_fp32_bits=torch.equal(a.view(torch.int16),ra.view(torch.int16)) and torch.equal(b.view(torch.int16),rb.view(torch.int16)) and torch.equal(c.view(torch.int32),rc.view(torch.int32)) and torch.equal(d.view(torch.int32),rd.view(torch.int32)),independent_bf16_reference=torch.equal((a if op==11 else b)[:active],expected),guards=all(bool(torch.all(raw[:64]==v)&torch.all(raw[-64:]==v)) for raw,v in guards),status=status.item()==0)
                if op==18:
                    raw=dy[owners,local//k].clone();raw[~valid]=0;s=scores[owners,local].clone();s[~valid]=0
                    gates['raw_dy_bits']=torch.equal(a[:active],raw);gates['row_scores_bits']=torch.equal(c[:active],s)
                print(json.dumps(dict(case='peer_staging_grid_reference',operation=op,width=width,block_m=bm,capacity=capacity,active=active,ctas_per_cu=ctas,graph=True,zero_scores=True,source_owners_local=True,every_element=True,gates=gates,passed=all(gates.values()))),flush=True)
                assert all(gates.values()),gates
    for bad in (-1,capacity//bm+1):
        blocks.fill_(bad)
        for op in (11,13,18):
            for raw,v in guards:raw.fill_(v)
            status.zero_();graphs[op,16].replay();torch.cuda.synchronize()
            assert status.item()==2 and all(bool(torch.all(raw==v)) for raw,v in guards)
    if measure:
        blocks.fill_(40960//bm);status.zero_()
        for op in (11,13,18):
            for ctas in (0,1,2,4,8,16):
                g=graphs[op,ctas]
                for _ in range(3):g.replay()
                torch.cuda.synchronize();samples=[];begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for _ in range(7):
                    begin.record()
                    for _ in range(5):g.replay()
                    end.record();end.synchronize();samples.append(begin.elapsed_time(end)/5)
                print(json.dumps(dict(case='peer_staging_grid_performance',operation=op,width=width,block_m=bm,capacity=capacity,active=40960,ctas_per_cu=ctas,median_ms=sorted(samples)[3],source_owners_local=True,scope='Pure BF16 peer staging with local surrogate owners, no EP or whole-step claim',passed=True)),flush=True)
    os.environ[KEY]='0';del graphs,x,dy,y,buffers;gc.collect();torch.cuda.synchronize()

def main():
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(105412)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for width,bm in ((128,128),(512,256),(8192,256)):fixture(width,bm)
        fixture(8192,256,65536,4096,True)
    stream.synchronize()
if __name__=='__main__':main()
