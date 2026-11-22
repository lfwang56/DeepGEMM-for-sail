"""The optional per-CTA clock probe must preserve every wgrad output bit."""
import json
import torch
import deep_gemm as dg

torch.cuda.set_device(0)
torch.manual_seed(28671)
a=torch.randn(1024,256,device='cuda').bfloat16()
b=torch.randn(1024,128,device='cuda').bfloat16()
initial=torch.randn(3,256,128,device='cuda')
normal=initial.clone();probed=initial.clone()
lengths=torch.tensor([256,128,384],device='cuda',dtype=torch.int32)
probe=torch.full((32768,),-1,device='cuda',dtype=torch.int64)
def run():
 normal.copy_(initial);probed.copy_(initial);probe.fill_(-1)
 dg.k_grouped_bf16_gemm_tn_device(a,b,normal,lengths,1024)
 dg.k_grouped_bf16_gemm_tn_device(a,b,probed,lengths,1024,probe)
run();torch.cuda.synchronize();torch.testing.assert_close(normal,probed,rtol=0,atol=0)
for bad in [torch.empty(0,device='cuda',dtype=torch.int64),torch.empty(32768,device='cuda'),torch.empty(32768,dtype=torch.int64),probe[::2],probed.view(torch.int64)]:
 try:dg.k_grouped_bf16_gemm_tn_device(a,b,probed,lengths,1024,bad)
 except RuntimeError:pass
 else:raise AssertionError('invalid/aliased probe accepted')
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
 run();stream.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,stream=stream):run()
for pattern in ([256,128,384],[0,512,128],[0,0,0]):
 a.mul_(.91);b.add_(.001);lengths.copy_(torch.tensor(pattern,device='cuda',dtype=torch.int32));graph.replay();torch.cuda.synchronize()
 torch.testing.assert_close(normal,probed,rtol=0,atol=0)
 records=probe.reshape(-1,2).cpu();active=records[records[:,0]>=0]
 assert active.numel() and bool((active[:,1]>active[:,0]).all())
print(json.dumps(dict(case='wgrad_cta_probe_correctness',bitwise_equal=True,dynamic_lengths_and_inputs=True,graph=True,invalid_probe_cases=5,passed=True)),flush=True)
