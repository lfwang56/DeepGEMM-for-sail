"""Direct peer-read gradient mover: independent collective oracle and graph timing."""
import argparse, hashlib, json, os
from pathlib import Path
from datetime import timedelta

p=argparse.ArgumentParser();p.add_argument('--ilp',type=int,required=True);a=p.parse_args()
import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as sm
import deep_gemm as dg
from moe_distributed_utils import destroy_training_process_group

rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(local);torch.set_num_threads(2);torch.manual_seed(48703+rank)
dist.init_process_group('nccl',timeout=timedelta(seconds=180))
header=Path(dg.__file__).parent/'include/deep_gemm/impls/peer_moe_training.cuh'
print(json.dumps(dict(case='dy138_source',ilp=a.ilp,rank=rank,header_sha256=hashlib.sha256(header.read_bytes()).hexdigest())),flush=True)
tokens=2048;topk=10;h=4096;rows=22016;cap=32768;block_m=256
payload=sm.empty((tokens*h*2+tokens*topk*4,),dtype=torch.uint8,device='cuda')
handle=sm.rendezvous(payload,dist.group.WORLD)
buffers=[handle.get_buffer(r,(payload.numel(),),torch.uint8) for r in range(world)]
dy=payload[:tokens*h*2].view(torch.bfloat16).view(tokens,h)
scores=payload[tokens*h*2:].view(torch.float32).view(tokens,topk)
dy.normal_(std=.1);scores.uniform_(-1,1);scores.view(-1)[::17]=0
dist.barrier();torch.cuda.synchronize()
dy_ptrs=torch.tensor([b.data_ptr() for b in buffers],dtype=torch.int64,device='cuda')
score_ptrs=torch.tensor([b.data_ptr()+tokens*h*2 for b in buffers],dtype=torch.int64,device='cuda')
permutation=torch.full((cap,),-1,dtype=torch.int32,device='cuda')
generator=torch.Generator(device='cuda');generator.manual_seed(48711+rank)
permutation[:rows-128]=torch.randint(world*tokens*topk,(rows-128,),device='cuda',dtype=torch.int32,generator=generator)
raw=torch.empty(cap,h,device='cuda',dtype=torch.bfloat16);weighted=torch.empty_like(raw)
row_scores=torch.empty(cap,device='cuda',dtype=torch.float32)
num_blocks=torch.tensor([rows//block_m],device='cuda',dtype=torch.int32)
status=torch.zeros(1,device='cuda',dtype=torch.int32)
tensors=[dy_ptrs,score_ptrs,permutation,raw,weighted,row_scores,num_blocks,status]
dims=[cap,h,tokens,topk,world,block_m]
all_dy=[torch.empty_like(dy) for _ in range(world)];all_scores=[torch.empty_like(scores) for _ in range(world)]
def expected():
    dist.all_gather(all_dy,dy);dist.all_gather(all_scores,scores)
    flat_dy=torch.cat(all_dy);flat_scores=torch.cat(all_scores).view(-1)
    route=permutation[:rows].long();valid=route>=0;safe=route.clamp_min(0)
    ref_raw=flat_dy[safe//topk].masked_fill(~valid[:,None],0)
    ref_score=flat_scores[safe].masked_fill(~valid,0)
    ref_weight=(ref_raw.float()*ref_score[:,None]).bfloat16()
    return ref_raw,ref_weight,ref_score
oracle=expected()
for ctas in (2,4,8,16):
    os.environ['DG_MOE_PEER_STAGE_CTAS_PER_CU']=str(ctas)
    raw.fill_(.125);weighted.fill_(.125);row_scores.fill_(.125)
    dg._C._moe_training_peer_launch(18,tensors,dims,False);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):dg._C._moe_training_peer_launch(18,tensors,dims,False)
    graph.replay();torch.cuda.synchronize()
    assert int(status.item())==0
    for got,ref in zip((raw,weighted,row_scores),oracle):
        assert torch.equal(got[:rows],ref),(rank,a.ilp,ctas,'independent reference')
        assert bool((got[rows:]==.125).all()),(rank,a.ilp,ctas,'inactive tail')
    # A captured mover must read changing peer payloads and changing GPU counts.
    num_blocks.zero_();raw.fill_(.25);graph.replay();torch.cuda.synchronize()
    assert bool((raw==.25).all()),'zero rows graph'
    num_blocks.fill_(rows//block_m);graph.replay();torch.cuda.synchronize()
    assert torch.equal(raw[:rows],oracle[0]),'restored rows graph'
    print(json.dumps(dict(case='dy138_independent_reference',rank=rank,ilp=a.ilp,ctas_per_cu=ctas,zero_scores=True,padding=True,zero_rows_replay=True,bitwise=True,passed=True)),flush=True)
    for _ in range(5):graph.replay()
    torch.cuda.synchronize();samples=[]
    for _ in range(5):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(20):graph.replay()
        end.record();end.synchronize();samples.append(begin.elapsed_time(end)/20)
    gathered=[None]*world;dist.all_gather_object(gathered,samples)
    if rank==0:
        maxima=[max(s[j] for s in gathered) for j in range(5)]
        print(json.dumps(dict(case='dy138_benchmark',ilp=a.ilp,ctas_per_cu=ctas,ep_size=world,hidden=h,rows=rows,scope='StageRawWeightedDy only, not MoE step',samples_max_rank_event_ms=maxima,median_max_rank_event_ms=sorted(maxima)[2],passed=True)),flush=True)
    del graph
# Read new payload values through the same pointer tables.
dy.mul_(2);scores.mul_(.5);oracle=expected()
dg._C._moe_training_peer_launch(18,tensors,dims,False);torch.cuda.synchronize()
for got,ref in zip((raw,weighted,row_scores),oracle):assert torch.equal(got[:rows],ref),'changed peer payload'
print(json.dumps(dict(case='dy138_finished',rank=rank,ilp=a.ilp,passed=True)),flush=True)
destroy_training_process_group()
