"""Per-launch CU budgets must retain complete additive DW on every output."""
import json,os
import torch
import deep_gemm as dg
rank=int(os.environ.get('RANK','0'));torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
torch.set_num_threads(2);torch.manual_seed(28671+rank)
initial=torch.randn(3,256,128,device='cuda');a=torch.randn(1024,256,device='cuda').bfloat16();b=torch.randn(1024,128,device='cuda').bfloat16()
lengths=torch.tensor([256,128,384],device='cuda',dtype=torch.int32)
max_cus=torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
out=initial.clone();normal=initial.clone();probe=torch.full((32768,),-1,device='cuda',dtype=torch.int64)
for bad in (-1,max_cus+1):
 try:dg.k_grouped_bf16_gemm_tn_device(a,b,out,lengths,1024,compute_cus=bad)
 except RuntimeError:pass
 else:raise AssertionError('invalid compute budget accepted')
for budget in (0,1,16,31,35,38,39):
 def run():
  out.copy_(initial);normal.copy_(initial);probe.fill_(-1)
  dg.k_grouped_bf16_gemm_tn_device(a,b,normal,lengths,1024,compute_cus=max_cus)
  dg.k_grouped_bf16_gemm_tn_device(a,b,out,lengths,1024,probe,compute_cus=budget)
 run();torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):run()
 for pattern in ([256,128,384],[0,512,128],[0,0,0]):
  a.mul_(.91);b.add_(.001);lengths.copy_(torch.tensor(pattern,device='cuda',dtype=torch.int32));graph.replay();torch.cuda.synchronize()
  torch.testing.assert_close(normal,out,rtol=0,atol=0)
  ac=a.cpu().float();bc=b.cpu().float();want=initial.cpu();start=0
  for group,k in enumerate(pattern):
   if k:want[group]+=ac[start:start+k].t()@bc[start:start+k]
   start+=k
  error=float((out.cpu()-want).norm()/want.norm().clamp_min(1e-20));assert error<2e-4,error
  clocks=probe.reshape(-1,2).cpu();active=clocks[clocks[:,0]>=0];assert len(active)>0 and bool((active[:,1]>active[:,0]).all())
  print(json.dumps(dict(case='wgrad_compute_budget_reference',rank=rank,budget=budget,pattern=pattern,relative_rms=error,bitwise_full_grid_equal=True,every_element=True,graph=True,invalid_budgets_rejected=True,passed=True)),flush=True)
 del graph
for m,n in ((4096,8192),(8192,2048)):
 left=(torch.randn(65536,m,device='cuda')*.01).bfloat16();right=(torch.randn(65536,n,device='cuda')*.01).bfloat16()
 lens=torch.full((8,),5120,device='cuda',dtype=torch.int32);initial_large=torch.randn(8,m,n,device='cuda')*.001
 base=initial_large.clone();dg.k_grouped_bf16_gemm_tn_device(left,right,base,lens,65536,compute_cus=max_cus)
 for budget in (31,35,39):
  measured=initial_large.clone();probe.fill_(-1)
  dg.k_grouped_bf16_gemm_tn_device(left,right,measured,lens,65536,probe,compute_cus=budget)
  torch.cuda.synchronize();torch.testing.assert_close(base,measured,rtol=0,atol=0)
  active=int((probe.reshape(-1,2)[:,0]>=0).sum());assert active==budget,(m,n,budget,active)
  print(json.dumps(dict(case='wgrad_compute_budget_reference',rank=rank,budget=budget,m=m,n=n,experts=8,tokens_per_expert=5120,active_ctas=active,bitwise_full_grid_equal=True,every_element=True,passed=True)),flush=True)
  del measured
 del left,right,base,initial_large
