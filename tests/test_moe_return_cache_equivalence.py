"""Exact output checks catch address-cache races below BF16 oracle tolerance."""

import argparse
import json
import torch
from deep_gemm.training import PeerMoEPlan


parser = argparse.ArgumentParser()
parser.add_argument('--ffn-fusion-ablation', action='store_true')
parser.add_argument('--expert-local-grad-ablation', action='store_true')
args = parser.parse_args()
if args.ffn_fusion_ablation and args.expert_local_grad_ablation:
    parser.error('choose one ablation')
torch.cuda.set_device(0)
torch.set_num_threads(2)
torch.manual_seed(19291)
h, i, e, k, t = 8192, 2048, 16, 10, 768
x = (torch.randn(t, h, device='cuda') * .1).bfloat16()
dy = (torch.randn_like(x) * .1).bfloat16()
w1 = (torch.randn(e, 2*i, h, device='cuda') * .02).bfloat16()
w2 = (torch.randn(e, h, i, device='cuda') * .02).bfloat16()
ids = torch.randn(t, e, device='cuda').topk(k, -1).indices.contiguous()
original_ids = ids.clone()
scores = torch.randn(t, k, device='cuda').softmax(-1)
scores[0, 0] = 0
valid = torch.full((), t, device='cuda', dtype=torch.int32)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
graphs, outputs, plans = [], [], []
with torch.cuda.stream(stream):
    for cache in (False, True):
        options = (dict(ffn_fusion=cache, dispatch_producer=not cache,
                        producer_input_aiu=not cache, producer_local_aiu=not cache)
                   if args.ffn_fusion_ablation else dict(inline_dispatch=True, cache_return_addresses=cache))
        if args.expert_local_grad_ablation:
            options = dict(expert_local_grad=cache, dispatch_producer=True,
                           producer_input_aiu=True, producer_local_aiu=True)
        plan = PeerMoEPlan(h, i, e, k, t, chunk_tokens=t, reuse_packed_dy=True,
                          fuse_activation_backward=True, check_errors=False,
                          gemm_configs={role:(256,64,64,64,2) for role in ('fc1','fc2','dhidden','dx')}, **options)
        plans.append(plan)
        accumulators = (torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))

        def step(plan=plan, accumulators=accumulators):
            plan.reset_errors()
            for tensor in accumulators:
                tensor.zero_()
            y = plan.forward(x, ids, scores, w1, w2, valid)
            return (y, *plan.backward(dy, x, ids, scores, w1, w2, valid, accumulators))

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        plan.check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = step()
        graphs.append(graph)
        outputs.append(result)

    for pattern in ('balanced', 'ragged', 'duplicate_skewed', 'masked'):
        x.add_(.002)
        ids.copy_(original_ids)
        valid.fill_(t - 3 if pattern == 'ragged' else t)
        if pattern == 'duplicate_skewed':
            ids.fill_(0)
        elif pattern == 'masked':
            ids.fill_(-1)
        else:
            ids[1, 1] = -1
        for _ in range(3):
            for graph in graphs:
                graph.replay()
        torch.cuda.synchronize()
        for plan in plans:
            plan.check()
        exact = {name:torch.equal(outputs[0][index],outputs[1][index])
                 for name,index in ((('y',0),('dx',1)) if args.expert_local_grad_ablation
                                    else (('y',0),('dx',1),('dscores',4)))}
        errors = {}
        for name,index in ((('dw1',2),('dw2',3),('dscores',4)) if args.expert_local_grad_ablation
                           else (('dw1',2),('dw2',3))):
            a,b = outputs[0][index],outputs[1][index]
            errors[name] = float((a-b).square().sum().sqrt() / a.square().sum().sqrt().clamp_min(1e-20))
        case = ('expert_grad_graph_equivalence' if args.expert_local_grad_ablation else
                'ffn_graph_equivalence' if args.ffn_fusion_ablation else 'return_cache_graph_equivalence')
        print(json.dumps({'case':case,'pattern':pattern,
                          'exact':exact,'wgrad_relative_rms':errors,
                          'source':('identical tensors; source-side versus expert-side router gradient' if args.expert_local_grad_ablation else
                                    'identical tensors; old producer+AIU versus depth-first fused FFN' if args.ffn_fusion_ablation
                                    else 'identical tensors and inline-dispatch GEMMs, return-address cache only')}),flush=True)
        assert all(exact.values()) and all(value < 1e-5 for value in errors.values()),(pattern,exact,errors)
torch.cuda.current_stream().wait_stream(stream)
