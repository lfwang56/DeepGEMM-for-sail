"""FP8 gradient pack/score kernel with a separate collective FP32 oracle."""
import argparse,hashlib,json,os
from pathlib import Path
from datetime import timedelta
p=argparse.ArgumentParser();p.add_argument('--ilp',type=int,required=True);a=p.parse_args()
import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as sm
import deep_gemm as dg
from moe_distributed_utils import destroy_training_process_group
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local);torch.set_num_threads(2);torch.manual_seed(48703+rank)
dist.init_process_group('nccl',timeout=timedelta(seconds=180))
header=Path(dg.__file__).parent/'include/deep_gemm/impls/peer_moe_training.cuh'
print(json.dumps(dict(case='grad139_source',rank=rank,ilp=a.ilp,header_sha256=hashlib.sha256(header.read_bytes()).hexdigest())),flush=True)
t=2048;k=10;h=4096;rows=22016;cap=32768;bm=128
dy_bytes=t*h*2;score_bytes=t*k*4
payload=sm.empty((dy_bytes+2*score_bytes,),dtype=torch.uint8,device='cuda')
handle=sm.rendezvous(payload,dist.group.WORLD)
buffers=[handle.get_buffer(r,(payload.numel(),),torch.uint8) for r in range(world)]
dy=payload[:dy_bytes].view(torch.bfloat16).view(t,h)
scores=payload[dy_bytes:dy_bytes+score_bytes].view(torch.float32).view(t,k)
dscores=payload[dy_bytes+score_bytes:].view(torch.float32).view(t,k)
dy.normal_(std=.1);scores.uniform_(-1,1);scores.view(-1)[::17]=0;dscores.fill_(float('nan'))
dist.barrier();torch.cuda.synchronize()
def ptr(offset):return torch.tensor([b.data_ptr()+offset for b in buffers],device='cuda',dtype=torch.int64)
dy_ptr,score_ptr,dscore_ptr=ptr(0),ptr(dy_bytes),ptr(dy_bytes+score_bytes)
g=torch.Generator(device='cuda');g.manual_seed(48711)
all_routes=torch.randperm(world*t*k,device='cuda',generator=g,dtype=torch.int32)
permutation=torch.full((cap,),-1,device='cuda',dtype=torch.int32)
permutation[:t*k]=all_routes[rank*t*k:(rank+1)*t*k]
y=torch.randn(cap,h,device='cuda',dtype=torch.bfloat16)*.1
weighted=torch.empty_like(y);q=torch.empty(cap,h,device='cuda',dtype=torch.float8_e4m3fn)
sf=torch.empty(h//128,cap,device='cuda',dtype=torch.float32)
blocks=torch.tensor([rows//bm],device='cuda',dtype=torch.int32);status=torch.zeros(1,device='cuda',dtype=torch.int32)
tensors=[dy_ptr,score_ptr,dscore_ptr,permutation,y,weighted,blocks,status,q,sf];dims=[cap,h,t,k,world,bm]
all_dy=[torch.empty_like(dy) for _ in range(world)];all_scores=[torch.empty_like(scores) for _ in range(world)]
dist.all_gather(all_dy,dy);dist.all_gather(all_scores,scores)
route=permutation[:rows].long();valid=route>=0;safe=route.clamp_min(0)
ref_raw=torch.cat(all_dy)[safe//k].masked_fill(~valid[:,None],0)
ref_scores=torch.cat(all_scores).view(-1)[safe].masked_fill(~valid,0)
ref_weight=(ref_raw.float()*ref_scores[:,None]).bfloat16()
tiles=ref_weight.float().view(rows,h//128,128)
ref_sf=tiles.abs().amax(-1).clamp_min(1e-4)/448.
ref_q=(tiles/ref_sf[:,:,None]).to(torch.float8_e4m3fn).view(rows,h)
ref_dot=(ref_raw.float()*y[:rows].float()).sum(-1)
global_dot=torch.zeros(world*t*k,device='cuda',dtype=torch.float32)
global_dot[route[valid]]=ref_dot[valid]
dist.all_reduce(global_dot)
ref_dscores=global_dot[rank*t*k:(rank+1)*t*k].view(t,k)
for ctas in (2,4,8,16):
 os.environ['DG_MOE_FP8_FUSED_QUANT_CTAS_PER_CU']=str(ctas)
 weighted.fill_(.125);q.view(torch.uint8).fill_(0x31);sf.fill_(.125)
 dg._C._moe_training_peer_launch(22,tensors,dims,False);torch.cuda.synchronize();dist.barrier()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):dg._C._moe_training_peer_launch(22,tensors,dims,False)
 graph.replay();torch.cuda.synchronize();dist.barrier()
 assert int(status.item())==0 and torch.equal(weighted[:rows],ref_weight),'weighted BF16 reference'
 assert torch.equal(q[:rows].view(torch.uint8),ref_q.view(torch.uint8)),'FP8 bits reference'
 assert torch.allclose(sf[:,:rows],ref_sf.t(),rtol=2e-6,atol=0),'FP32 scale reference'
 error=float((dscores-ref_dscores).square().sum().sqrt()/ref_dscores.square().sum().sqrt())
 assert error<2e-6 and bool(torch.isfinite(dscores).all()),('dScore FP32 reference',error)
 assert bool((weighted[rows:]==.125).all()) and bool((q[rows:].view(torch.uint8)==0x31).all()) and bool((sf[:,rows:]==.125).all()),'inactive tail'
 blocks.zero_();weighted.fill_(.25);graph.replay();torch.cuda.synchronize()
 assert bool((weighted==.25).all()),'zero rows graph'
 blocks.fill_(rows//bm);graph.replay();torch.cuda.synchronize();dist.barrier()
 assert torch.equal(weighted[:rows],ref_weight),'restored rows graph'
 print(json.dumps(dict(case='grad139_independent_reference',rank=rank,ilp=a.ilp,ctas_per_cu=ctas,zero_scores=True,padding=True,zero_rows_replay=True,weighted_and_fp8_bitwise=True,dscore_relative_rms=error,passed=True)),flush=True)
 for _ in range(5):graph.replay()
 torch.cuda.synchronize();samples=[]
 for _ in range(5):
  begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
  for _ in range(20):graph.replay()
  end.record();end.synchronize();samples.append(begin.elapsed_time(end)/20)
 gathered=[None]*world;dist.all_gather_object(gathered,samples)
 if rank==0:
  maxima=[max(s[j] for s in gathered) for j in range(5)]
  print(json.dumps(dict(case='grad139_benchmark',ilp=a.ilp,ctas_per_cu=ctas,ep_size=world,hidden=h,rows=rows,scope='ExpertLocalGradientsQuantized only, not MoE step',samples_max_rank_event_ms=maxima,median_max_rank_event_ms=sorted(maxima)[2],passed=True)),flush=True)
 del graph
print(json.dumps(dict(case='grad139_finished',rank=rank,ilp=a.ilp,passed=True)),flush=True)
destroy_training_process_group()
