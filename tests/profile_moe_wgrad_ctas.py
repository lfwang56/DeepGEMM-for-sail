"""Rank0-only CTA clocks paired with Asys; this is a diagnostic, not a speedup verdict."""
import argparse,gc,json,os
from datetime import timedelta
import torch
import torch.distributed as dist
import deep_gemm as dg
from deep_gemm.training import PeerMoEPlan,to_g8_weight
from moe_distributed_utils import destroy_training_process_group

class ProbePeerMoEPlan(PeerMoEPlan):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  calls=2*((self.max_tokens+self.chunk_tokens-1)//self.chunk_tokens)*self.route_window_count
  # The API checks the actual grid against this capacity before any launch.
  self.cta_clocks=torch.full((calls,32768),-1,device=self.device,dtype=torch.int64)
  self.probe_cursor=0
 def backward(self,*args,**kwargs):
  self.probe_cursor=0;self.cta_clocks.fill_(-1)
  return super().backward(*args,**kwargs)
 def _wgrad(self,left,right,output,*,asynchronous=False):
  probe=self.cta_clocks[self.probe_cursor];self.probe_cursor+=1
  if not asynchronous:
   dg.k_grouped_bf16_gemm_tn_device(left,right,output,self.lengths,self.rows,probe);return
  main=torch.cuda.current_stream(self.device);self._wgrad_stream.wait_stream(main)
  with torch.cuda.stream(self._wgrad_stream):
   for tensor in (left,right,output,self.lengths,probe):tensor.record_stream(self._wgrad_stream)
   dg.k_grouped_bf16_gemm_tn_device(left,right,output,self.lengths,self.rows,probe)
  self._wgrad_pending=True

def main():
 p=argparse.ArgumentParser();p.add_argument('--wgrad-overlap',choices=('none','both'),default='none');p.add_argument('--dgrad-compute-cus',type=int,default=0);p.add_argument('--tokens',type=int,default=4096);p.add_argument('--steps',type=int,default=3)
 args=p.parse_args();rank=int(os.environ.get('RANK',0));size=int(os.environ.get('WORLD_SIZE',1));device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)))
 torch.cuda.set_device(device);torch.set_num_threads(2)
 if size>1:dist.init_process_group('nccl',device_id=device,timeout=timedelta(seconds=600))
 group=dist.group.WORLD if size>1 else None
 torch.manual_seed(98231+rank);h,i,e,k,t=8192,2048,16,10,args.tokens
 x=(torch.randn(t,h,device=device)*.1).bfloat16();w1=to_g8_weight((torch.randn(e,2*i,h,device=device)*.02).bfloat16());w2=(torch.randn(e,h,i,device=device)*.02).bfloat16()
 ids=torch.randn(t,e*size,device=device).topk(k,-1).indices.contiguous();scores=torch.randn(t,k,device=device).softmax(-1);dy=(torch.randn_like(x)*.1).bfloat16();valid=torch.tensor(t,device=device,dtype=torch.int32)
 options=dict(group=group,device=device,chunk_tokens=t,block_m=256,ring_rows=65536,gate_up_layout='g8',fuse_fc1_activation=True,dispatch_producer=True,producer_input_aiu=True,producer_local_aiu=True,input_copy_ilp=8,expert_local_grad=True,reuse_packed_dy=True,fuse_activation_backward=True,wgrad_overlap=args.wgrad_overlap,dgrad_compute_cus=args.dgrad_compute_cus,check_errors=False,gemm_configs={r:(256,64,64,64,3) for r in ('fc1','fc2','dhidden','dx')})
 normal=PeerMoEPlan(h,i,e,k,t,**options);probed=(ProbePeerMoEPlan if rank==0 else PeerMoEPlan)(h,i,e,k,t,**options)
 accum=[(torch.zeros_like(w1,dtype=torch.float32),torch.zeros_like(w2,dtype=torch.float32)) for _ in range(2)]
 def step(plan,grads):
  for value in grads:value.zero_()
  y=plan.forward(x,ids,scores,w1,w2,valid)
  return (y,*plan.backward(dy,x,ids,scores,w1,w2,valid,grads))
 stream=torch.cuda.Stream(device);stream.wait_stream(torch.cuda.current_stream(device))
 with torch.cuda.stream(stream):
  for _ in range(3):reference=step(normal,accum[0]);actual=step(probed,accum[1])
  stream.synchronize()
  # Route packing uses atomics: two uninstrumented plans also change FP32
  # accumulation order. Keep the existing full-layer1e-5 RMS gate; the API
  # probe test separately demands bitwise equality on IDENTICAL packed inputs.
  def check_layer():
   errors=[]
   for index,(a,b) in enumerate(zip(reference,actual)):
    if index not in (2,3):torch.testing.assert_close(a,b,rtol=0,atol=0)
    av,bv=a.float(),b.float()
    error=float((av-bv).square().sum().sqrt()/av.square().sum().sqrt().clamp_min(1e-20))
    assert error<1e-5,error
    errors.append(error)
   return errors
  layer_errors=check_layer()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):actual=step(probed,accum[1])
  for _ in range(3):graph.replay()
  stream.synchronize();normal.check();probed.check()
  if size>1:dist.barrier()
  torch.cuda.synchronize();torch.cuda.profiler.start()
  for _ in range(args.steps):graph.replay()
  stream.synchronize();torch.cuda.profiler.stop()
  layer_errors=check_layer()
  probed.check()
 if rank==0:
  records=[]
  for call,clocks in enumerate(probed.cta_clocks.cpu().reshape(probed.probe_cursor,-1,2)):
   active=clocks[clocks[:,0]>=0];assert active.numel() and bool((active[:,1]>active[:,0]).all())
   delta=active[:,1]-active[:,0]
   records.append(dict(call=call,role='dw2' if call%2==0 else 'dw1',ctas=len(active),min_cycles=int(delta.min()),median_cycles=int(delta.median()),max_cycles=int(delta.max()),clocks=active.tolist()))
  print(json.dumps(dict(case='wgrad_cta_clock_diagnostic',rank=rank,ranks=size,shape=[h,i,e,k,t],wgrad_budget=os.environ.get('DG_BF16_WGRAD_COMPUTE_CUS','0'),dgrad_budget=args.dgrad_compute_cus,overlap=args.wgrad_overlap,records=records,graph_replays=args.steps,relative_rms_to_uninstrumented=layer_errors,layer_rms_gate=1e-5,identical_layout_bitwise_gate='tests/test_moe_wgrad_cta_probe.py',clock='SDK2.2 __ppu_read_clock64 per-CTA elapsed; no cross-CU timestamp comparison',scope='diagnostic only; CTA clocks include execution stalls and preemption after entry; instrumentation is not an optimization',passed=True)),flush=True)
 if size>1:dist.barrier()
 del graph,actual,reference
 del normal,probed
 gc.collect()
 destroy_training_process_group()
if __name__=='__main__':main()
