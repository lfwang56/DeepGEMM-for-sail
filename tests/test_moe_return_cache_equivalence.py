"""Exact output checks catch address-cache races below BF16 oracle tolerance."""

import argparse
import json
import torch
from deep_gemm.training import PeerMoEPlan
from deep_gemm.training.layout import to_g8_weight


parser = argparse.ArgumentParser()
parser.add_argument('--ffn-fusion-ablation', action='store_true')
parser.add_argument('--expert-local-grad-ablation', action='store_true')
parser.add_argument('--ring-rows-ablation', type=int)
parser.add_argument('--block-m-ablation', action='store_true')
parser.add_argument('--return-vector-ablation', action='store_true')
parser.add_argument('--dag-ablation', action='store_true')
parser.add_argument('--dag-serialization-ablation', action='store_true')
parser.add_argument('--dag-shared-pool', action='store_true')
parser.add_argument('--dag-g8', action='store_true')
parser.add_argument('--dag-input-prefetch', action='store_true')
parser.add_argument('--dag-input-ctas', type=int, default=4)
parser.add_argument('--dag-backward', action='store_true')
parser.add_argument('--dag-backward-only', action='store_true')
parser.add_argument('--dag-direct-return', action='store_true')
parser.add_argument('--dag-direct-return-ablation', action='store_true')
parser.add_argument('--input-copy-ilp-ablation', type=int, choices=(2, 4, 8))
parser.add_argument('--wgrad-overlap-ablation', choices=('dw1','both'))
parser.add_argument('--dy-prefetch-ablation', action='store_true')
parser.add_argument('--input-assist-ablation', type=int)
parser.add_argument('--input-external', action='store_true')
parser.add_argument('--input-external-serial', action='store_true')
parser.add_argument('--fc1-activation-ablation', action='store_true')
parser.add_argument('--gemm-geometry-ablation', type=int, nargs=5)
parser.add_argument('--gemm-stages', type=int, choices=(2,3), default=2)
parser.add_argument('--dgrad-compute-cus', type=int, default=0)
parser.add_argument('--replay-gradients-ablation', action='store_true')
args = parser.parse_args()
if args.dag_backward_only:
    args.dag_ablation = args.dag_backward = True
if args.dag_direct_return_ablation:
    args.dag_ablation = True
if args.dag_backward and not args.dag_g8:
    parser.error('backward DAG checks require --dag-g8')
if args.dag_g8 and not (args.dag_shared_pool and (args.dag_ablation or args.dag_serialization_ablation or args.input_copy_ilp_ablation)):
    parser.error('G8 DAG comparison requires shared-pool DAG ablation')
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
if (args.fc1_activation_ablation or args.dag_g8 or args.input_copy_ilp_ablation
        or args.wgrad_overlap_ablation or args.dy_prefetch_ablation or args.input_assist_ablation is not None or args.gemm_geometry_ablation or args.replay_gradients_ablation):
    w1 = to_g8_weight(w1)
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
        if args.ring_rows_ablation is not None:
            options = dict(expert_local_grad=True, dispatch_producer=True,
                           producer_input_aiu=True, producer_local_aiu=True,
                           ring_rows=args.ring_rows_ablation if cache else None)
        if args.block_m_ablation:
            options = dict(expert_local_grad=True, dispatch_producer=True,
                           producer_input_aiu=True, producer_local_aiu=True,
                           ring_rows=1024, block_m=256 if cache else 128)
        if args.return_vector_ablation:
            options = dict(expert_local_grad=True, dispatch_producer=True,
                           producer_input_aiu=True, producer_local_aiu=True,
                           ring_rows=1024, block_m=256, return_vector_bytes=8 if cache else 4)
        if args.dag_ablation or args.dag_serialization_ablation:
            active = True if args.dag_serialization_ablation or args.dag_direct_return_ablation else cache
            forward_dag = active and not args.dag_backward_only
            backward_dag = active and args.dag_backward
            options = dict(expert_local_grad=True, ring_rows=1024, block_m=256,
                           dag_fusion=forward_dag, dag_backward=backward_dag,
                           dispatch_producer=not forward_dag,
                           producer_input_aiu=not forward_dag,
                           producer_local_aiu=not forward_dag,
                           dag_serialize=cache if args.dag_serialization_ablation else False)
            options['dag_shared_pool'] = args.dag_shared_pool and (forward_dag or backward_dag)
            options['dag_direct_return'] = active and (cache if args.dag_direct_return_ablation else args.dag_direct_return)
        if args.fc1_activation_ablation:
            options = dict(expert_local_grad=True, dispatch_producer=True,
                           producer_input_aiu=True, producer_local_aiu=True,
                           ring_rows=1024, block_m=256, gate_up_layout='g8',
                           fuse_fc1_activation=cache)
        if args.dag_g8:
            options.update(gate_up_layout='g8', fuse_fc1_activation=True)
        if args.input_copy_ilp_ablation:
            options = dict(expert_local_grad=True, ring_rows=1024, block_m=256,
                           gate_up_layout='g8', fuse_fc1_activation=True,
                           dag_fusion=args.dag_shared_pool, dag_shared_pool=args.dag_shared_pool,
                           dag_direct_return=args.dag_direct_return and args.dag_shared_pool,
                           dispatch_producer=not args.dag_shared_pool,
                           producer_input_aiu=not args.dag_shared_pool,
                           producer_local_aiu=not args.dag_shared_pool,
                           input_copy_ilp=args.input_copy_ilp_ablation if cache else 1)
        if args.dag_input_prefetch:
            options.update(dag_input_prefetch=bool(options.get('dag_fusion') or options.get('dag_backward')),
                           dag_input_ctas=args.dag_input_ctas)
        if args.wgrad_overlap_ablation:
            options = dict(expert_local_grad=True, ring_rows=1024, block_m=256,
                           gate_up_layout='g8', fuse_fc1_activation=True,
                           dispatch_producer=True, producer_input_aiu=True,
                           producer_local_aiu=True, input_copy_ilp=8,
                           wgrad_overlap=args.wgrad_overlap_ablation if cache else 'none')
        if args.dy_prefetch_ablation:
            options = dict(expert_local_grad=True, ring_rows=1024, block_m=256,
                           gate_up_layout='g8', fuse_fc1_activation=True,
                           dispatch_producer=True, producer_input_aiu=True,
                           producer_local_aiu=True, input_copy_ilp=8,
                           overlap_expert_dy=cache)
        if args.input_assist_ablation is not None:
            options = dict(expert_local_grad=True, ring_rows=1024, block_m=256,
                           gate_up_layout='g8', fuse_fc1_activation=True, input_copy_ilp=8,
                           dispatch_producer=not cache, producer_input_aiu=not cache,
                           producer_local_aiu=not cache, inline_dispatch=cache and not args.input_external,
                           dispatch_external=cache and args.input_external,
                           dispatch_external_serial=cache and args.input_external_serial, dispatch_assist_ctas=args.input_assist_ablation if cache else 0)
        if args.gemm_geometry_ablation:
            options = dict(expert_local_grad=True, ring_rows=1024, block_m=256,
                           gate_up_layout='g8', fuse_fc1_activation=True, input_copy_ilp=8,
                           dispatch_producer=True, producer_input_aiu=True, producer_local_aiu=True)
        config = tuple(args.gemm_geometry_ablation) if cache and args.gemm_geometry_ablation else (256,64,64,64,args.gemm_stages)
        if args.replay_gradients_ablation:
            options = dict(expert_local_grad=True, ring_rows=1024, block_m=256,
                           gate_up_layout='g8', fuse_fc1_activation=True,
                           dispatch_producer=True, producer_input_aiu=True,
                           producer_local_aiu=True, input_copy_ilp=8,
                           wgrad_overlap=args.wgrad_overlap_ablation or 'none',
                           fuse_replay_gradients=cache)
        options['dgrad_compute_cus'] = args.dgrad_compute_cus if cache else 0
        plan = PeerMoEPlan(h, i, e, k, t, chunk_tokens=t, reuse_packed_dy=True,
                          fuse_activation_backward=True, check_errors=False,
                          gemm_configs={role:config for role in ('fc1','fc2','dhidden','dx')}, **options)
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
        dy.add_(.001)
        scores.copy_(torch.randn_like(scores).softmax(-1))
        scores[0, 0] = 0
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
                 for name,index in ((('y',0),('dx',1)) if args.expert_local_grad_ablation or args.ring_rows_ablation is not None or args.block_m_ablation or args.return_vector_ablation or args.replay_gradients_ablation
                                    else (('y',0),('dx',1),('dscores',4)))}
        errors = {}
        for name,index in ((('dw1',2),('dw2',3),('dscores',4)) if args.expert_local_grad_ablation or args.ring_rows_ablation is not None or args.block_m_ablation or args.return_vector_ablation or args.replay_gradients_ablation
                           else (('dw1',2),('dw2',3))):
            a,b = outputs[0][index],outputs[1][index]
            errors[name] = float((a-b).square().sum().sqrt() / a.square().sum().sqrt().clamp_min(1e-20))
        if args.replay_gradients_ablation:
            # Y and weighted DY keep the same BF16 boundaries. FP32 dScore is
            # reassociated into tile partials; bound both RMS and worst errors.
            torch.testing.assert_close(outputs[1][4], outputs[0][4], rtol=2e-5, atol=2e-6)
        case = ('replay_gradient_graph_equivalence' if args.replay_gradients_ablation else
                'gemm_geometry_graph_equivalence' if args.gemm_geometry_ablation else
                'input_assist_graph_equivalence' if args.input_assist_ablation is not None else
                'dy_prefetch_graph_equivalence' if args.dy_prefetch_ablation else
                'wgrad_stream_graph_equivalence' if args.wgrad_overlap_ablation else
                'input_copy_ilp_graph_equivalence' if args.input_copy_ilp_ablation else
                'fc1_activation_graph_equivalence' if args.fc1_activation_ablation else
                'dag_graph_equivalence' if args.dag_ablation or args.dag_serialization_ablation else
                'return_vector_graph_equivalence' if args.return_vector_ablation else
                'block_m_graph_equivalence' if args.block_m_ablation else
                'routed_row_window_graph_equivalence' if args.ring_rows_ablation is not None else
                'expert_grad_graph_equivalence' if args.expert_local_grad_ablation else
                'ffn_graph_equivalence' if args.ffn_fusion_ablation else 'return_cache_graph_equivalence')
        print(json.dumps({'case':case,'pattern':pattern,'gemm_geometry':args.gemm_geometry_ablation,'gemm_stages':args.gemm_stages,
                          'dag_shared_pool':args.dag_shared_pool,
                          'modes':[dict(forward_dag=p.dag_fusion,backward_dag=p.dag_backward,
                                        direct_return=p.dag_direct_return,input_copy_ilp=p.input_copy_ilp,
                                        wgrad_overlap=p.wgrad_overlap,
                                        overlap_expert_dy=p.overlap_expert_dy,inline_dispatch=p.inline_dispatch,
                                        dispatch_assist_ctas=p.dispatch_assist_ctas, dispatch_external=p.dispatch_external, dispatch_external_serial=p.dispatch_external_serial) for p in plans],
                          'exact':exact,'wgrad_relative_rms':errors,
                          'source':('same BF16 Y/weighted-DY boundaries; fused replay peer-DY and two-stage FP32 dScore reduction' if args.replay_gradients_ablation else
                                    'identical tensors and arithmetic; sliced producer versus inline tile owners with optional auxiliary input copy' if args.input_assist_ablation is not None else
                                    'identical tensors, GEMM tiles and dot reduction kernel; prefetched raw DY versus direct peer DY reads' if args.dy_prefetch_ablation else
                                    'identical tensors, GEMM tiles and accumulation order; auxiliary wgrad stream versus main stream' if args.wgrad_overlap_ablation else
                                    'identical tensors and GEMM tiles; only independent input vector loads per thread change' if args.input_copy_ilp_ablation else
                                    'identical G8 parameters and GEMM tiles; separate versus register-fused BF16-rounded FC1/SwiGLU' if args.fc1_activation_ablation else
                                    'identical tensors and GEMM tiles; DAG enabled versus producer control, or DAG serial versus parallel' if args.dag_ablation or args.dag_serialization_ablation else
                                    'identical tensors and BM256; four-byte versus eight-byte direct peer stores' if args.return_vector_ablation else
                                    'identical tensors; BM128 versus BM256, with matching metadata alignment' if args.block_m_ablation else
                                    'identical tensors and expert-side gradients; full packed rows versus bounded routed-row windows' if args.ring_rows_ablation is not None else
                                    'identical tensors; source-side versus expert-side router gradient' if args.expert_local_grad_ablation else
                                    'identical tensors; old producer+AIU versus depth-first fused FFN' if args.ffn_fusion_ablation
                                    else 'identical tensors and inline-dispatch GEMMs, return-address cache only')}),flush=True)
        assert all(exact.values()) and all(value < 1e-5 for value in errors.values()),(pattern,exact,errors)
torch.cuda.current_stream().wait_stream(stream)
