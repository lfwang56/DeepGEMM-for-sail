"""Fresh-input bitwise and independent FP32 gates for in-CTA FC1 input helpers."""
import argparse,hashlib,json,statistics
from pathlib import Path
import torch
import deep_gemm as dg
p=argparse.ArgumentParser();p.add_argument('--qwen',action='store_true');p.add_argument('--benchmark',action='store_true');p.add_argument('--warp',choices=('64x64','32x128'),default='64x64');args=p.parse_args()
torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(561197)
cfg=(256,64,64,64,3)
wide_cfg=(512,64,64,64,3) if args.warp=='64x64' else (512,64,32,128,3)
execution_stream=torch.cuda.Stream()
execution_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(execution_stream):
 for h,i,e,capacity0 in [(256,128,3,0)]+([(8192,2048,16,65536)] if args.qwen else []):
  for bm in (256,):
   capacity=capacity0 or 8*bm
   source_tokens=capacity//4
   source=(torch.randn(4*source_tokens,h,device='cuda')*.02).bfloat16()
   weight=(torch.randn(e,2*i,h,device='cuda')*.02).bfloat16()
   peers=torch.tensor([source[n*source_tokens].data_ptr() for n in range(4)],device='cuda',dtype=torch.int64)
   source_rows=torch.empty(capacity,device='cuda',dtype=torch.int32)
   identity=torch.arange(capacity,device='cuda',dtype=torch.int32)
   info=torch.zeros(capacity//bm,4,device='cuda',dtype=torch.int32)
   count=torch.zeros((),device='cuda',dtype=torch.int32)
   reference=[torch.empty(capacity,w,device='cuda',dtype=torch.bfloat16) for w in (h,2*i,i)]
   candidate=[torch.empty_like(x) for x in reference]
   def layout(pattern):
    if pattern=='balanced':lengths=[2560]*e if e==16 else [2*bm,bm,0]
    elif pattern=='ragged':lengths=[bm+bm//2,bm//2]+[0]*(e-2)
    elif pattern=='skew':lengths=[capacity]+[0]*(e-1)
    elif pattern=='empty':lengths=[0]*e
    else:raise ValueError(pattern)
    records=[];offset=0;tiles=0;ranges=[]
    for group,rows in enumerate(lengths):
     ranges.append((group,offset,rows))
     prefix=tiles
     for _ in range((rows+bm-1)//bm):records.append([group,rows,prefix,offset]);tiles+=1
     offset+=((rows+bm-1)//bm)*bm
    assert offset<=capacity and tiles<=info.shape[0]
    host_rows=torch.randint(0,4*source_tokens,(capacity,),dtype=torch.int32)
    host_rows[::17]=-1
    for group,base,rows in ranges:
     tail=((rows+bm-1)//bm)*bm-rows
     if tail:host_rows[base+rows:base+rows+tail]=-1
    host_rows[offset:]=-1
    source_rows.copy_(host_rows);info.zero_()
    if records:info[:len(records)].copy_(torch.tensor(records,device='cuda',dtype=torch.int32))
    count.fill_(tiles)
    return ranges
   common=dict(config=cfg,block_m=bm,interleaved_gate_up=True,fc1_activation=True)
   def old(keep):
    a,gu,hidden=reference
    dg._C._moe_training_peer_gemm(a,weight,gu,peers,peers,source_rows,identity,info,count,
     source_tokens,source_tokens,True,False,pack_input=True,activation_gate_up=hidden,
     first_n_tile=0,num_n_tiles=1,producer_input_aiu=True,input_copy_ilp=8,
     keep_gate_up=keep,**common)
    if 2*i>cfg[0]:
     dg._C._moe_training_peer_gemm(a,weight,gu,peers,peers,identity,identity,info,count,
      source_tokens,source_tokens,False,False,activation_gate_up=hidden,
      first_n_tile=1,local_input_aiu=True,keep_gate_up=keep,**common)
   def new(keep,helpers):
    a,gu,hidden=candidate
    dg._C._moe_training_peer_gemm(a,weight,gu,peers,peers,source_rows,identity,info,count,
     source_tokens,source_tokens,True,False,pack_input=True,activation_gate_up=hidden,
     fc1_helper_threads=helpers,fc1_compute_block_m=128,input_copy_ilp=8,
     keep_gate_up=keep,**dict(common,config=wide_cfg))
   for keep in (True,False):
    for helpers in (32,64):
     for values in (reference,candidate):
      for tensor in values:tensor.fill_(-3)
     layout('balanced')
     def both():old(keep);new(keep,helpers)
     both();torch.cuda.synchronize()
     for x,y in zip(reference,candidate):torch.testing.assert_close(x,y,rtol=0,atol=0)
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):both()
     for pattern in ('balanced','ragged','skew','empty'):
      source.add_(.0001);weight.mul_(.97);ranges=layout(pattern)
      for values in (reference,candidate):
       for tensor in values:tensor.fill_(-3)
      graph.replay();torch.cuda.synchronize()
      for x,y in zip(reference,candidate):torch.testing.assert_close(x,y,rtol=0,atol=0)
      if h==256:
       for group,base,rows in ranges:
        if not rows:continue
        ids=source_rows[base:base+rows].long()
        xx=source[ids.clamp_min(0)].float();xx[ids<0]=0
        gu=(xx@weight[group].float().T).bfloat16()
        inter=gu.reshape(rows,-1,16)
        gate=inter[:,:,:8].reshape(rows,i).float()
        up=inter[:,:,8:].reshape(rows,i).float()
        hidden=((gate*torch.sigmoid(gate))*up).bfloat16()
        for expected,actual in [(hidden,candidate[2][base:base+rows])]+([(gu,candidate[1][base:base+rows])] if keep else []):
         rms=(actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)
         assert rms<.008,float(rms)
     print(json.dumps(dict(case='input_helper_wide_reference',hidden=h,intermediate=i,experts=e,
      block_m=bm,compute_block_m=128,block_n=512,warp=args.warp,helpers=helpers,keep_gate_up=keep,capacity=capacity,bitwise_all_buffers=True,
      changed_inputs_routes_lengths_graph=True,all_empty=True,passed=True)),flush=True)
     del graph
   if args.benchmark and e==16:
    layout('balanced')
    for keep in (True,False):
     for helpers in (0,32,64):
      fn=(lambda:old(keep)) if helpers==0 else (lambda:new(keep,helpers))
      for _ in range(5):fn()
      torch.cuda.synchronize()
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):fn()
      for _ in range(5):graph.replay()
      torch.cuda.synchronize();samples=[]
      for repeat in range(5):
       begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
       begin.record()
       for _ in range(20):graph.replay()
       end.record();end.synchronize();samples.append(begin.elapsed_time(end)/20)
      print(json.dumps(dict(case='local_input_helper_wide_signal',hidden=h,intermediate=i,
       experts=e,block_m=bm,compute_block_m=(128 if helpers else bm),block_n=(512 if helpers else 256),warp=args.warp,helpers=helpers,keep_gate_up=keep,capacity=capacity,active_rows=40960,
       median_event_ms=statistics.median(samples),samples=samples,
       extension_sha256=hashlib.sha256(Path(dg._C.__file__).read_bytes()).hexdigest(),
       scope='Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup')),flush=True)
      del graph
   del source,weight,peers,source_rows,identity,info,count,reference,candidate
