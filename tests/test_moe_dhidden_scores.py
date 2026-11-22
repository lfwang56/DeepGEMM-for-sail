"""Independent FP32 and dynamic-graph gates for optional dHidden score fusion.

Scaling dHidden instead of DY changes BF16 rounding, so compare both paths
against the FP32 routed model rather than demanding their bitwise equality.
"""
import argparse
import json
import math
import torch
from deep_gemm.training import PeerMoEPlan
from deep_gemm.training.layout import to_g8_weight, from_g8_weight
from test_moe_training import reference


def rms(a, b):
    return float((a.float()-b.float()).square().sum().sqrt() /
                 b.float().square().sum().sqrt().clamp_min(1e-20))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--transpose-free-dgrad',action='store_true')
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--intermediate', type=int, default=128)
    parser.add_argument('--experts', type=int, default=4)
    parser.add_argument('--tokens', type=int, default=67)
    parser.add_argument('--wgrad-overlap', choices=('none','both'), default='none')
    args=parser.parse_args()
    torch.cuda.set_device(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(218901)
    h,i,e,t=args.hidden,args.intermediate,args.experts,args.tokens
    k=min(3,e)
    x=(torch.randn(t,h,device='cuda')*.1).bfloat16()
    dy=(torch.randn_like(x)*.1).bfloat16()
    w1=to_g8_weight((torch.randn(e,2*i,h,device='cuda')*.02).bfloat16())
    w2=(torch.randn(e,h,i,device='cuda')*.02).bfloat16()
    scores=torch.randn(t,k,device='cuda').softmax(-1)
    ids=(torch.arange(t*k,device='cuda').reshape(t,k)%e).long()
    original_ids=ids.clone();valid=torch.tensor(t,device='cuda',dtype=torch.int32)
    plan=PeerMoEPlan(h,i,e,k,t,chunk_tokens=t,ring_rows=1024,block_m=256,
        gate_up_layout='g8',fuse_fc1_activation=True,expert_local_grad=True,
        reuse_packed_dy=True,fuse_activation_backward=True,dispatch_producer=True,
        producer_input_aiu=True,producer_local_aiu=True,input_copy_ilp=8,
        fuse_dhidden_scores=True,transpose_free_dgrad=args.transpose_free_dgrad,
        wgrad_overlap=args.wgrad_overlap,check_errors=False,
        gemm_configs={r:(256,64,64,64,3) for r in ('fc1','fc2','dhidden','dx')})
    acc=(torch.zeros_like(w1,dtype=torch.float32),torch.zeros_like(w2,dtype=torch.float32))
    def step():
        plan.reset_errors()
        for a in acc:a.zero_()
        y=plan.forward(x,ids,scores,w1,w2,valid)
        return y,*plan.backward(dy,x,ids,scores,w1,w2,valid,acc)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):step();step()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):outputs=step()
    # A second capture checks backward with no preceding forward at replay.
    bgraph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(bgraph,stream=stream):
        for a in acc:a.zero_()
        backwards=plan.backward(dy,x,ids,scores,w1,w2,valid,acc)
    for number,pattern in enumerate(('balanced','duplicate','ragged','reversed','masked','empty')):
        x.copy_((torch.randn_like(x)*.1).bfloat16());dy.copy_((torch.randn_like(dy)*.1).bfloat16())
        w1.add_(torch.randn_like(w1)*.0001);w2.add_(torch.randn_like(w2)*.0001)
        scores.copy_(torch.randn_like(scores).softmax(-1));scores[0,0]=0
        scores[1,0]=1e-20;scores[2,0]=-0.25
        ids.copy_(original_ids);valid.fill_(t)
        if pattern=='duplicate':ids.fill_(0)
        if pattern=='ragged':valid.fill_(t-3);ids[1,1]=-1
        if pattern=='reversed':ids.copy_(e-1-original_ids)
        if pattern=='masked':ids.fill_(-1)
        if pattern=='empty':valid.zero_()
        graph.replay();graph.replay();torch.cuda.synchronize();plan.check()
        saved_y=outputs[0].clone()
        # Change DY after the forward graph: backward must read fresh data.
        dy.add_(.003)
        bgraph.replay();torch.cuda.synchronize();plan.check()
        rx,rp,rw1,rw2=[v.detach().float().requires_grad_() for v in (x,scores,from_g8_weight(w1),w2)]
        rids=ids.clone();rids[int(valid):]=-1
        ry=reference(rx,rids,rp,rw1,rw2)
        ry=ry+0*(rx.sum()+rp.sum()+rw1.sum()+rw2.sum())
        rg=torch.autograd.grad(ry,(rx,rw1,rw2,rp),dy.float())
        actual=(saved_y,backwards[0],from_g8_weight(backwards[1]),backwards[2],backwards[3])
        errors={n:rms(a,b) for n,a,b in zip(('y','dx','dw1','dw2','dscores'),actual,(ry,*rg))}
        passed=all(math.isfinite(v) and v<.035 for v in errors.values())
        if pattern not in ('masked','empty'):
            passed &= float(backwards[3][0,0])!=0
        print(json.dumps(dict(case='dhidden_scores_graph_reference',pattern=pattern,
            hidden=h,intermediate=i,experts=e,tokens=t,wgrad_overlap=args.wgrad_overlap,
            relative_rms=errors,limit=.035,zero_score_derivative=float(backwards[3][0,0]),
            updated_weights_and_backward_only_graph=True,passed=passed)),flush=True)
        assert passed,errors


if __name__=='__main__':main()
