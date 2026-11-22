"""Graph replay must invalidate stale eager contexts and keep valid captures fast."""
import argparse,json,os
from datetime import timedelta
import torch
import torch.distributed as dist
from deep_gemm.training import PeerMoEPlan,DeepEPMoEPlan
from deep_gemm.training.hierarchical import HierarchicalPeerMoEPlan,make_hierarchical_groups
from deep_gemm.training.layout import to_g8_weight
from moe_distributed_utils import destroy_training_process_group
p=argparse.ArgumentParser();p.add_argument('--backend',choices=('peer','hybrid','elastic'),required=True);p.add_argument('--compute',choices=('bf16','fp8'),default='bf16');a=p.parse_args()
rank,size=int(os.environ['RANK']),int(os.environ['WORLD_SIZE']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);torch.set_num_threads(2);torch.manual_seed(27130+rank)
dist.init_process_group('nccl',device_id=device,timeout=timedelta(seconds=300));metadata=dist.new_group(backend='gloo',timeout=timedelta(seconds=300));groups=make_hierarchical_groups(size//2,metadata) if a.backend=='hybrid' else None
h,i,e,k,t=256,256,2,2,17
common=dict(device=device,compute=a.compute,check_errors=False,chunk_tokens=8,ring_rows=512,block_m=128,gate_up_layout='split')
if a.backend in ('peer','hybrid'):
 common.update(expert_local_grad=True,reuse_packed_dy=True)
 if a.compute=='bf16':common.update(gate_up_layout='g8',dispatch_producer=True,producer_input_aiu=True,producer_local_aiu=True,fuse_fc1_activation=True,fuse_dhidden_scores=True,transpose_free_dgrad=True,fuse_activation_backward=True,gemm_configs={r:(256,64,64,64,3) for r in ('fc1','fc2','dhidden','dx')})
 else:common.update(fuse_dispatch=False)
 if a.backend=='peer':
  cached=PeerMoEPlan(h,i,e,k,t,group=metadata,reuse_forward_context=True,**common);control=PeerMoEPlan(h,i,e,k,t,group=metadata,reuse_forward_context=False,**common)
 else:
  common.update(groups=groups,reuse_forward_context=True)
  cached=HierarchicalPeerMoEPlan(h,i,e,k,t,reuse_pair_context=True,source_pipeline=True,chunk_reduce=True,tail_dw_overlap=a.compute=='bf16',share_fp8_weights=a.compute=='fp8',**common)
  control=HierarchicalPeerMoEPlan(h,i,e,k,t,reuse_pair_context=False,source_pipeline=False,chunk_reduce=False,tail_dw_overlap=False,share_fp8_weights=False,**common)
else:
 os.environ['DEEPEP_NUM_MAX_ICN_PEERS']=str(size)
 common.update(group=dist.group.WORLD,elastic_hybrid=True,comm_sms=8)
 if a.compute=='fp8':common.update(fp8_active_quantization=True,fp8_mn_major_scales=True,fp8_weight_quantization=2)
 cached=DeepEPMoEPlan(h,i,e,k,t,reuse_forward_context=True,**common);control=DeepEPMoEPlan(h,i,e,k,t,reuse_forward_context=False,**common)
x=(torch.randn(t,h,device=device)*.1).bfloat16();dy=(torch.randn_like(x)*.1).bfloat16();ids=((torch.arange(t,device=device)[:,None]+rank*t)*k+torch.arange(k,device=device))%(size*e);scores=torch.randn(t,k,device=device).softmax(-1);valid=torch.tensor(t,device=device,dtype=torch.int32)
w1=(torch.randn(e,2*i,h,device=device)*.02).bfloat16();w2=(torch.randn(e,h,i,device=device)*.02).bfloat16()
if common['gate_up_layout']=='g8':w1=to_g8_weight(w1)
inputs_a=(x,ids,scores,w1,w2,valid)
def step(plan,inputs,gradient):
 y=plan.forward(*inputs);return y,*plan.backward(gradient,*inputs)
def live(plan):
 if a.backend=='hybrid':return plan.inner._has_live_forward_context() and plan.remote_inner._has_live_forward_context()
 return plan._has_live_forward_context()
def compare(name,actual,expected,want_reuse,before):
 torch.cuda.synchronize();cached.check();control.check();errors=[float((u.float()-v.float()).norm()/v.float().norm().clamp_min(1e-20)) for u,v in zip(actual,expected)]
 tolerance=.012 if a.backend=='elastic' else .0002
 assert before==want_reuse,(name,before,want_reuse)
 assert max(errors)<tolerance,(name,errors)
 print(json.dumps(dict(case='graph_context_generation_reference',rank=rank,backend=a.backend,compute=a.compute,name=name,live_before_backward=before,expected_reuse=want_reuse,relative_rms=errors,tolerance=tolerance,every_element=True,passed=True)),flush=True)
stream=torch.cuda.Stream(device);stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
 for mode in ('full_graph','forward_only_graph','positive_forward_graph','backward_only_from_eager'):
  step(cached,inputs_a,dy);step(control,inputs_a,dy);stream.synchronize()
  xb=(x*.17+.3).contiguous();ib=(size*e-1-ids).contiguous();sb=scores.flip(-1).contiguous();inputs_b=(xb,ib,sb,w1,w2,valid);dyb=(dy*.7+.02).contiguous()
  if mode=='backward_only_from_eager':
   cached.forward(*inputs_b);graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream):actual=cached.backward(dyb,*inputs_b)
   # Replaying after a different forward must still read its own supplied inputs.
   cached.forward(*inputs_a);graph.replay();stream.synchronize();before=False
   control.forward(*inputs_b);expected=control.backward(dyb,*inputs_b)
   compare(mode,actual,expected,False,before)
  else:
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream):
    if mode=='full_graph':captured=step(cached,inputs_a,dy)
    else:captured=cached.forward(*inputs_a)
   if mode!='positive_forward_graph':cached.forward(*inputs_b)
   graph.replay();stream.synchronize();before=live(cached)
   chosen,gradient=(inputs_a,dy) if mode=='positive_forward_graph' else (inputs_b,dyb)
   actual=cached.backward(gradient,*chosen);control.forward(*chosen);expected=control.backward(gradient,*chosen)
   compare(mode,actual,expected,mode=='positive_forward_graph',before)
  del graph
stream.synchronize()
for plan in (cached,control):
 if hasattr(plan,'close'):plan.close()
if groups is not None:groups.close()
destroy_training_process_group()
