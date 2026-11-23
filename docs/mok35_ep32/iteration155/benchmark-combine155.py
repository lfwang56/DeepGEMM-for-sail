"""Independent FP64 math, view/canary checks, mutable graphs and matched output hashes."""
from pathlib import Path
import argparse,hashlib,json,os
p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
import torch
import deep_gemm as dg
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);torch.manual_seed(67831+rank)
def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def fixture(c,h,k,t,start,offset,weighted):
 raw=(torch.randn(c*k*h+offset+16,device='cuda')*.2).bfloat16();x=raw[offset:offset+c*k*h].view(c*k,h)
 storage=torch.full((t*h+offset+32,),.125,device='cuda',dtype=torch.bfloat16);out=storage[offset:offset+t*h].view(t,h)
 scores=torch.randn(t,k,device='cuda')*.3
 if not weighted:scores.fill_(float('nan'))
 dims=[c,h,k,t,start,int(weighted)]
 def launch():dg._C._moe_training_launch(6,[x,scores,out],dims)
 launch();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):launch()
 for replay in range(3):
  if replay:x.mul_(.75).add_(.03125)
  storage.fill_(.125);g.replay();torch.cuda.synchronize()
  expected=torch.full_like(out,.125,dtype=torch.float64)
  n=max(0,min(c,t-start))
  if n:
   values=x[:n*k].view(n,k,h).double()
   products=values*scores[start:start+n,:,None].double() if weighted else values
   total=products.sum(1);expected[start:start+n]=total
   bound=.004*products.abs().sum(1)+2e-7
   assert bool((out[start:start+n].double().sub(total).abs()<=bound).all()),'independent FP64 bound'
  assert torch.equal(out[:start],expected[:start].bfloat16()) and torch.equal(out[start+n:],expected[start+n:].bfloat16()),'inactive rows'
  assert bool((storage[:offset]==.125).all()) and bool((storage[offset+t*h:]==.125).all()),'output guards'
  print(json.dumps(dict(case='combine155_reference',label=a.label,rank=rank,c=c,h=h,k=k,t=t,start=start,offset=offset,weighted=weighted,replay=replay,output_sha256=digest(out),independent_fp64=True,passed=True)),flush=True)
 del g,x,raw,out,storage,scores
for shape in [(5,24,10,9,2,0),(5,7,3,6,4,0),(3,256,10,4,1,1),(13,128,33,15,1,0),(0,128,10,3,0,0)]:
 for weighted in [False,True]:fixture(*shape,weighted)
for tokens in [2048,4096]:
 h=4096;k=10;x=torch.randn(tokens*k,h,device='cuda',dtype=torch.bfloat16)*.1;scores=torch.randn(tokens,k,device='cuda').softmax(-1);out=torch.empty(tokens,h,device='cuda',dtype=torch.bfloat16)
 for weighted in [False,True]:
  dims=[tokens,h,k,tokens,0,int(weighted)]
  dg._C._moe_training_launch(6,[x,scores,out],dims);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):dg._C._moe_training_launch(6,[x,scores,out],dims)
  for _ in range(5):g.replay()
  torch.cuda.synchronize();samples=[];hashes=[]
  for _ in range(5):
   x.mul_(.99).add_(.0001)
   begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
   for _ in range(20):g.replay()
   end.record();end.synchronize();samples.append(begin.elapsed_time(end)/20);hashes.append(digest(out))
  print(json.dumps(dict(case='combine155_benchmark',label=a.label,rank=rank,tokens=tokens,hidden=h,topk=k,weighted=weighted,samples_event_ms=samples,output_hashes=hashes,scope='One local top-k reduction only, not a MoE step',passed=True)),flush=True)
  del g
 del x,scores,out;torch.cuda.empty_cache()
print(json.dumps(dict(case='combine155_finished',label=a.label,rank=rank,passed=True)),flush=True)
