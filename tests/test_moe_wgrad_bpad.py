"""TSM B offsets must preserve complete-K beta=1 WGrad bitwise."""
import argparse,json,os
import torch
import deep_gemm as dg
p=argparse.ArgumentParser();p.add_argument('--qwen',action='store_true');args=p.parse_args()
torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(501197)
shapes=[(3,256,128,1024)] + ([(16,4096,8192,65536),(16,8192,2048,65536)] if args.qwen else [])
for e,m,n,capacity in shapes:
 a=(torch.randn(capacity,m,device='cuda')*.01).bfloat16()
 b=(torch.randn(capacity,n,device='cuda')*.01).bfloat16()
 initial=torch.randn(e,m,n,device='cuda')*.001
 plain=initial.clone();padded=initial.clone();lengths=torch.empty(e,device='cuda',dtype=torch.int32)
 for pad in (1024,2048,3072):
  def run():
   plain.copy_(initial);padded.copy_(initial)
   os.environ['DG_BF16_WGRAD_B_PAD_BYTES']='0'
   dg.k_grouped_bf16_gemm_tn_device(a,b,plain,lengths,capacity)
   os.environ['DG_BF16_WGRAD_B_PAD_BYTES']=str(pad)
   dg.k_grouped_bf16_gemm_tn_device(a,b,padded,lengths,capacity)
  lengths.fill_(2560 if e==16 else 256)
  run();torch.cuda.synchronize();torch.testing.assert_close(plain,padded,rtol=0,atol=0)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):run()
  for pattern in ('balanced','mixed','empty'):
   a.mul_(.93);b.add_(.0001)
   lengths.fill_(2560 if e==16 else 256)
   if pattern=='mixed':lengths[::2].zero_()
   if pattern=='empty':lengths.zero_()
   graph.replay();torch.cuda.synchronize();torch.testing.assert_close(plain,padded,rtol=0,atol=0)
  print(json.dumps(dict(case='wgrad_tsm_bpad_reference',groups=e,m=m,n=n,capacity=capacity,pad=pad,
   bitwise_equal=True,beta_one=True,changed_inputs_lengths_graph=True,passed=True)),flush=True)
  del graph
 for bad in ('-1024','16','17408'):
  os.environ['DG_BF16_WGRAD_B_PAD_BYTES']=bad
  before=padded.clone()
  try:dg.k_grouped_bf16_gemm_tn_device(a,b,padded,lengths,capacity)
  except RuntimeError:pass
  else:raise AssertionError('invalid TSM padding accepted')
  torch.testing.assert_close(before,padded,rtol=0,atol=0)
 os.environ['DG_BF16_WGRAD_B_PAD_BYTES']='0'
 del a,b,initial,plain,padded,lengths
