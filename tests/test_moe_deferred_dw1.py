"""Deferred final DW1: completion, workspace/stream guards, graph and ring windows."""
from datetime import timedelta
import json,os
import torch
import torch.distributed as dist
from deep_gemm.training import PeerMoEPlan
from deep_gemm.training.layout import to_g8_weight
from moe_distributed_utils import destroy_training_process_group


def fixture(rank,device,chunk,rows,skew):
    h,i,e,k,t=256,256,2,2,17
    opts=dict(group=dist.group.WORLD,device=device,chunk_tokens=chunk,ring_rows=rows,block_m=128,
        gate_up_layout='g8',check_errors=False,expert_local_grad=True,reuse_packed_dy=True,
        fuse_activation_backward=True,fuse_fc1_activation=True,fuse_dhidden_scores=True,
        transpose_free_dgrad=True,dispatch_producer=True,producer_input_aiu=True,producer_local_aiu=True,
        reuse_forward_context=True,gemm_configs={role:(256,64,64,64,3) for role in ('fc1','fc2','dhidden','dx')})
    pending=PeerMoEPlan(h,i,e,k,t,**opts);control=PeerMoEPlan(h,i,e,k,t,**opts)
    torch.manual_seed(12800+rank)
    x=(torch.randn(t,h,device=device)*.1).bfloat16();dy=torch.randn_like(x)*.1
    ids=((torch.arange(t,device=device)[:,None]+rank*t)*k+torch.arange(k,device=device))%(dist.get_world_size()*e)
    if skew:ids.zero_()
    scores=torch.randn(t,k,device=device).softmax(-1);valid=torch.tensor(t,device=device,dtype=torch.int32)
    w1=to_g8_weight((torch.randn(e,2*i,h,device=device)*.02).bfloat16());w2=(torch.randn(e,h,i,device=device)*.02).bfloat16()
    tensors=lambda:(x,ids,scores,w1,w2,valid)
    def compare(name,a,b):
        torch.cuda.synchronize();pending.check();control.check()
        errors=[float((aa.float()-bb.float()).norm()/bb.float().norm().clamp_min(1e-20)) for aa,bb in zip(a,b)]
        assert max(errors)<.0001,(name,errors)
        print(json.dumps(dict(case='deferred_dw1_reference',name=name,rank=rank,chunk=chunk,ring_rows=rows,
            skew=skew,every_element=True,errors=errors,passed=True)),flush=True)
    stream=torch.cuda.current_stream(device)
    pending.forward(*tensors());a=pending.backward(dy,*tensors(),_defer_last_dw1=True)
    assert pending._pending_last_dw1 is not None
    for op in (lambda:pending.forward(*tensors()),lambda:pending.backward(dy,*tensors())):
        try:op()
        except RuntimeError as error:assert 'deferred DW1' in str(error)
        else:raise AssertionError('workspace reuse accepted before deferred completion')
    wrong=torch.cuda.Stream(device=device)
    try:
        with torch.cuda.stream(wrong):pending._run_deferred_dw1()
    except RuntimeError as error:assert 'one stream' in str(error)
    else:raise AssertionError('completion accepted on another stream')
    assert pending._pending_last_dw1 is not None
    assert pending._run_deferred_dw1() and not pending._run_deferred_dw1()
    control.forward(*tensors());b=control.backward(dy,*tensors());compare('finish_and_guards',a,b)
    a=pending.backward(dy,*tensors(),_defer_last_dw1=True)
    assert pending._pending_last_dw1 is None and not pending._run_deferred_dw1()
    b=control.backward(dy,*tensors());compare('consumed_forward_fallback',a,b)
    def step(plan,defer):
        y=plan.forward(*tensors());v=plan.backward(dy,*tensors(),_defer_last_dw1=defer)
        if defer:plan._run_deferred_dw1()
        return y,*v
    step(pending,True);step(control,False);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):a=step(pending,True);b=step(control,False)
    for j in range(3):
        x.mul_(.97).add_(.001);scores.mul_(.95)
        if j==1:ids[:,0].fill_(-1)
        if j==2:ids.zero_();valid.fill_(0 if rank==0 else t-rank%3)
        graph.replay();compare('mutable_graph_'+str(j),a,b)
    del graph
    torch.cuda.synchronize()


def main():
    rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device);torch.set_num_threads(2)
    dist.init_process_group('gloo',timeout=timedelta(seconds=300))
    stream=torch.cuda.Stream(device);stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        fixture(rank,device,8,512,False)
        fixture(rank,device,17,256,True)
    stream.synchronize();destroy_training_process_group()

if __name__=='__main__':main()
