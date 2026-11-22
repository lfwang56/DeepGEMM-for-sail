"""Two logical domains on one host: live context and all-rank invalidation gates."""
from datetime import timedelta
import json,os
import torch
import torch.distributed as dist
from deep_gemm.training.hierarchical import HierarchicalPeerMoEPlan,make_hierarchical_groups
from deep_gemm.training.layout import to_g8_weight
from moe_distributed_utils import destroy_training_process_group


def main():
    rank,size=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device);torch.set_num_threads(2);torch.manual_seed(12200+rank)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(seconds=300))
    metadata=dist.new_group(backend='gloo',timeout=timedelta(seconds=300))
    groups=make_hierarchical_groups(size//2,metadata)
    h,i,e,k,t=256,256,2,2,17
    kwargs=dict(groups=groups,chunk_tokens=8,ring_rows=512,block_m=128,gate_up_layout='g8',
        expert_local_grad=True,reuse_packed_dy=True,fuse_activation_backward=True,
        fuse_fc1_activation=True,fuse_dhidden_scores=True,transpose_free_dgrad=True,
        gemm_configs={role:(256,64,64,64,3) for role in ('fc1','fc2','dhidden','dx')},
        device=device,check_errors=False,reuse_forward_context=True)
    cached=HierarchicalPeerMoEPlan(h,i,e,k,t,reuse_pair_context=True,**kwargs)
    control=HierarchicalPeerMoEPlan(h,i,e,k,t,reuse_pair_context=False,**kwargs)
    x=(torch.randn(t,h,device=device)*.1).bfloat16()
    dy=(torch.randn_like(x)*.1).bfloat16()
    ids=((torch.arange(t,device=device)[:,None]+rank*t)*k+torch.arange(k,device=device))%(size*e)
    scores=torch.randn(t,k,device=device).softmax(-1)
    valid=torch.tensor(t,device=device,dtype=torch.int32)
    w1=to_g8_weight((torch.randn(e,2*i,h,device=device)*.02).bfloat16())
    w2=(torch.randn(e,h,i,device=device)*.02).bfloat16()
    tensors=lambda:(x,ids,scores,w1,w2,valid)
    def step(plan):
        y=plan.forward(*tensors());return (y,*plan.backward(dy,*tensors()))
    def check(name,a,b,reused):
        torch.cuda.synchronize();cached.check();control.check()
        errors=[]
        for actual,want in zip(a,b):
            error=float((actual.float()-want.float()).norm()/want.float().norm().clamp_min(1e-20))
            errors.append(error)
        assert max(errors)<.0001,(name,errors)
        assert cached.hybrid_pair_forward_reused_last_backward==reused,(name,reused)
        if reused:assert cached.forward_context_reused_windows.item()>0
        print(json.dumps(dict(case='hierarchical_pair_context_reference',name=name,rank=rank,
            ranks=size,logical_domains=2,physical_scope='one-host correctness only; no cross-domain performance claim',
            reused=reused,errors=errors,every_element=True,passed=True)),flush=True)
    stream=torch.cuda.Stream(device);stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        a=step(cached);b=step(control);check('live_eager',a,b,True)
        for name in ('x','scores','ids','weight'):
            cached.forward(*tensors())
            if rank==0:
                if name=='x':x.mul_(.91).add_(.002)
                elif name=='scores':scores[:,0].mul_(.83)
                elif name=='ids':ids[:,0].fill_(size*e-1)
                else:w1.mul_(.97)
            a=cached.backward(dy,*tensors());b=control.backward(dy,*tensors())
            check('one_rank_mutation_'+name,a,b,False)
        cached.forward(*tensors())
        other=x.clone().add_(.003)
        cached.forward(other,ids,scores,w1,w2,valid)
        check('overwritten_forward',cached.backward(dy,*tensors()),control.backward(dy,*tensors()),False)
        check('consumed_context',cached.backward(dy,*tensors()),control.backward(dy,*tensors()),False)
        # A backward-only graph has no live forward and must rematerialize.
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            a=cached.backward(dy,*tensors());b=control.backward(dy,*tensors())
        graph.replay();check('backward_only_graph',a,b,False)
        del graph
        step(cached);step(control);stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):a=step(cached);b=step(control)
        for case in range(3):
            x.mul_(.99).add_(.001);scores.mul_(.98)
            if case==1:ids[:,1].fill_(-1)
            if case==2:ids[:,1].copy_(ids[:,0]);valid.fill_(max(0,t-rank%3))
            graph.replay();check('mutable_full_graph_'+str(case),a,b,True)
        del graph
    stream.synchronize()
    cached.close();control.close();groups.close()
    destroy_training_process_group()

if __name__=='__main__':main()
