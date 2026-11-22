"""Unprofiled local WGrad+FP32-zero signal; no EP or layer speedup claim."""
import hashlib,json,os,statistics
from pathlib import Path
import torch
import deep_gemm as dg
torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(515019)
for role,m,n in (('dw1',4096,8192),('dw2',8192,2048)):
 e,capacity=16,65536
 a=(torch.randn(capacity,m,device='cuda')*.01).bfloat16()
 b=(torch.randn(capacity,n,device='cuda')*.01).bfloat16()
 out=torch.zeros(e,m,n,device='cuda');reference=torch.zeros_like(out)
 lengths=torch.full((e,),2560,device='cuda',dtype=torch.int32)
 os.environ['DG_BF16_WGRAD_B_PAD_BYTES']='0'
 dg.k_grouped_bf16_gemm_tn_device(a,b,reference,lengths,capacity)
 torch.cuda.synchronize()
 for pad in (0,1024,2048,3072):
  os.environ['DG_BF16_WGRAD_B_PAD_BYTES']=str(pad)
  def run():
   out.zero_();dg.k_grouped_bf16_gemm_tn_device(a,b,out,lengths,capacity)
  for _ in range(5):run()
  torch.cuda.synchronize();torch.testing.assert_close(out,reference,rtol=0,atol=0)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):run()
  for _ in range(5):graph.replay()
  torch.cuda.synchronize();samples=[]
  for repeat in range(5):
   begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   begin.record()
   for _ in range(20):graph.replay()
   end.record();end.synchronize();samples.append(begin.elapsed_time(end)/20)
  torch.testing.assert_close(out,reference,rtol=0,atol=0)
  print(json.dumps(dict(case='local_wgrad_bpad_signal',role=role,groups=e,m=m,n=n,
   capacity=capacity,active_rows=40960,pad_bytes=pad,bitwise_equal=True,
   median_event_ms=statistics.median(samples),samples=samples,warmup=5,iterations=20,repeats=5,
   extension_sha256=hashlib.sha256(Path(dg._C.__file__).read_bytes()).hexdigest(),
   scope='single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup')),flush=True)
  del graph
 del a,b,out,reference,lengths
