"""Check near-limit packed-row offsets without expanding the FP32 oracle K times.

Every top-k slot selects global expert0. The independent FP32 reference computes
that expert once per source token and sums its routing weights. This is the
same mathematical MoE, including a separate derivative for every score slot.
"""

import argparse
from datetime import timedelta
import gc
import json
import math
import os

import torch
import torch.distributed as dist

from deep_gemm.training.hierarchical import HierarchicalPeerMoEPlan, make_hierarchical_groups
from deep_gemm.training import DeepEPMoEPlan, PeerMoEPlan
from moe_distributed_utils import destroy_training_process_group
from test_moe_training_large import relative_rms
from deep_gemm.training.layout import to_g8_weight, from_g8_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--local-experts", type=int, default=16)
    parser.add_argument("--chunk-tokens", type=int, default=768)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--backends", nargs="+", choices=("deepep", "peer", "hybrid"), default=["deepep", "peer"])
    parser.add_argument("--dispatch-pipeline", action="store_true")
    parser.add_argument("--return-pipeline", action="store_true")
    parser.add_argument("--pipeline-comm-ctas", type=int, default=8)
    parser.add_argument("--pipeline-serialize", action="store_true")
    parser.add_argument("--inline-dispatch", action="store_true")
    parser.add_argument("--cache-return-addresses", action="store_true")
    parser.add_argument("--ffn-fusion", action="store_true")
    parser.add_argument("--expert-local-grad", action="store_true")
    parser.add_argument("--ring-rows", type=int)
    parser.add_argument("--block-m", type=int, choices=(128, 256), default=128)
    parser.add_argument("--return-vector-bytes", type=int, choices=(4, 8), default=4)
    parser.add_argument("--dag-fusion", action="store_true")
    parser.add_argument("--dag-input-ctas", type=int, default=4)
    parser.add_argument("--dag-return-ctas", type=int, default=4)
    parser.add_argument("--dag-serialize", action="store_true")
    parser.add_argument("--dag-trace", action="store_true")
    parser.add_argument("--dag-shared-pool", action="store_true")
    parser.add_argument("--dag-input-prefetch", action="store_true")
    parser.add_argument("--dag-backward", action="store_true")
    parser.add_argument("--dag-direct-return", action="store_true")
    parser.add_argument("--input-copy-ilp", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--dispatch-assist-ctas", type=int, default=0)
    parser.add_argument("--dispatch-external", action="store_true")
    parser.add_argument("--serialize-external-dispatch", action="store_true")
    parser.add_argument("--input-inline", action="store_true")
    parser.add_argument("--wgrad-overlap", choices=("none", "dw1", "both"), default="none")
    parser.add_argument("--dgrad-compute-cus", type=int, default=0)
    parser.add_argument("--fuse-replay-gradients", action="store_true")
    parser.add_argument("--fuse-dhidden-scores", action="store_true")
    parser.add_argument("--transpose-free-dgrad", action="store_true")
    parser.add_argument("--overlap-expert-dy", action="store_true")
    parser.add_argument("--gate-up-layout", choices=("split", "g8"), default="split")
    parser.add_argument("--fuse-fc1-activation", action="store_true")
    parser.add_argument("--gemm-config", type=int, nargs=5)
    parser.add_argument("--process-group-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--hybrid-domain-size", type=int)
    parser.add_argument("--producer-input-aiu", action="store_true")
    parser.add_argument("--producer-local-aiu", action="store_true")
    parser.add_argument("--fuse-activation-backward", action="store_true")
    args = parser.parse_args()
    if args.compute == "fp8" and args.backends != ["deepep"]:
        parser.error("FP8 full-skew validation requires the DeepEP transport")
    if args.gemm_config is None and args.compute == "bf16":
        args.gemm_config = (256,64,64,64,2)
    if "peer" in args.backends and args.process_group_backend != "nccl":
        parser.error("the peer check uses a direct-domain NCCL metadata group")
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(args.process_group_backend,
        device_id=device if args.process_group_backend == "nccl" else None,
        timeout=timedelta(seconds=600))
    reference_group = (dist.new_group(backend="gloo", timeout=timedelta(seconds=600))
                       if args.process_group_backend == "nccl" else dist.group.WORLD)
    hybrid_groups = make_hierarchical_groups(args.hybrid_domain_size, reference_group) if "hybrid" in args.backends else None
    h, i, e, k, c = args.hidden, args.intermediate, args.local_experts, args.topk, args.chunk_tokens
    total = size * c
    generator = torch.Generator().manual_seed(920173)
    global_x = (torch.randn(total, h, generator=generator) * .1).bfloat16()
    global_dy = (torch.randn(total, h, generator=generator) * .1).bfloat16()
    global_scores = torch.randn(total, k, generator=generator).softmax(-1)
    global_scores[0, 0] = 0
    section = slice(rank * c, (rank + 1) * c)
    x, dy, scores = [value[section].to(device) for value in (global_x, global_dy, global_scores)]
    ids = torch.zeros(c, k, device=device, dtype=torch.int64)
    valid = torch.full((), c, device=device, dtype=torch.int32)
    torch.manual_seed(183721 + rank)
    w1 = (torch.randn(e, 2 * i, h, device=device) * .02).bfloat16()
    w2 = (torch.randn(e, h, i, device=device) * .02).bfloat16()
    weight_targets, sources = None, None
    if rank == 0:
        rx, rp = global_x.float().to(device).requires_grad_(), global_scores.to(device).requires_grad_()
        rw1, rw2 = w1[0].float().requires_grad_(), w2[0].float().requires_grad_()
        gate, up = (rx @ rw1.t()).chunk(2, -1)
        values = (torch.nn.functional.silu(gate) * up) @ rw2.t()
        ry = values * rp.sum(-1, keepdim=True)
        gx, gw1, gw2, gp = torch.autograd.grad(ry, (rx, rw1, rw2, rp), global_dy.float().to(device))
        sources = [value.detach().cpu() for value in (ry, gx, gp)]
        weight_targets = (gw1.detach(), gw2.detach())
        del rx, rp, rw1, rw2, gate, up, values, ry, gx, gp, gw1, gw2
    targets = []
    for index, width in enumerate((h, h, k)):
        target = torch.empty(c, width, dtype=torch.float32)
        pieces = list(sources[index].split(c)) if rank == 0 else None
        dist.scatter(target, scatter_list=pieces, src=0, group=reference_group)
        targets.append(target.to(device))
    del global_x, global_dy, global_scores, sources, pieces
    gc.collect()
    torch.cuda.empty_cache()
    if args.gate_up_layout == "g8":
        w1 = to_g8_weight(w1)
    configs = None if args.gemm_config is None else {role:tuple(args.gemm_config) for role in ("fc1", "fc2", "dhidden", "dx")}
    for backend in args.backends:
        options = dict(gemm_configs=configs, block_m=args.block_m, gate_up_layout=args.gate_up_layout,
                       wgrad_overlap=args.wgrad_overlap, overlap_expert_dy=args.overlap_expert_dy)
        if args.ring_rows is not None:
            options["ring_rows"] = args.ring_rows
        if backend in ("peer", "hybrid"):
            use_producer = not (args.input_inline or args.dispatch_assist_ctas or args.dispatch_pipeline or args.inline_dispatch or args.ffn_fusion or args.dag_fusion)
            options.update(dispatch_producer=use_producer, dispatch_producer_tiles=1,
                           reuse_packed_dy=True,
                           producer_input_aiu=args.producer_input_aiu and use_producer,
                           producer_local_aiu=args.producer_local_aiu and use_producer,
                           fuse_activation_backward=args.fuse_activation_backward,
                           dispatch_pipeline=args.dispatch_pipeline, return_pipeline=args.return_pipeline,
                           pipeline_comm_ctas=args.pipeline_comm_ctas, pipeline_serialize=args.pipeline_serialize,
                           inline_dispatch=args.inline_dispatch or args.input_inline, dispatch_assist_ctas=args.dispatch_assist_ctas, dispatch_external=args.dispatch_external, dispatch_external_serial=args.serialize_external_dispatch, cache_return_addresses=args.cache_return_addresses,
                           dgrad_compute_cus=args.dgrad_compute_cus,
                           fuse_replay_gradients=args.fuse_replay_gradients,
                         fuse_dhidden_scores=args.fuse_dhidden_scores,
                         transpose_free_dgrad=args.transpose_free_dgrad,
                           ffn_fusion=args.ffn_fusion, expert_local_grad=args.expert_local_grad,
                           return_vector_bytes=args.return_vector_bytes,
                           fuse_fc1_activation=args.fuse_fc1_activation,
                           dag_fusion=args.dag_fusion, dag_input_ctas=args.dag_input_ctas,
                           dag_return_ctas=args.dag_return_ctas, dag_serialize=args.dag_serialize,
                           dag_trace=args.dag_trace, dag_shared_pool=args.dag_shared_pool,
                           dag_input_prefetch=args.dag_input_prefetch,
                           dag_backward=args.dag_backward, dag_direct_return=args.dag_direct_return)
            options["input_copy_ilp"] = args.input_copy_ilp
        else:
            options["comm_sms"] = 38
            if args.process_group_backend == "nccl":
                os.environ["DEEPEP_NUM_MAX_ICN_PEERS"] = str(size)
        cls = HierarchicalPeerMoEPlan if backend == "hybrid" else PeerMoEPlan if backend == "peer" else DeepEPMoEPlan
        plan = cls(h, i, e, k, c, chunk_tokens=c, device=device, **({"groups":hybrid_groups} if backend == "hybrid" else {"group":dist.group.WORLD}),
                   check_errors=False, compute=args.compute, **options)
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream())

        def step():
            y = plan.forward(x, ids, scores, w1, w2, valid)
            return y, *plan.backward(dy, x, ids, scores, w1, w2, valid)

        graph = None
        with torch.cuda.stream(stream):
            actual = step()
            if args.graph:
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = step()
                graph.replay()
        stream.synchronize()
        plan.check()
        y, dx, dw1, dw2, ds = actual
        errors = dict(y=relative_rms(y, targets[0]), dx=relative_rms(dx, targets[1]),
                      dscores=relative_rms(ds, targets[2]))
        unused_zero = not bool(torch.count_nonzero(dw1[1:] if rank == 0 else dw1))
        unused_zero &= not bool(torch.count_nonzero(dw2[1:] if rank == 0 else dw2))
        if rank == 0:
            errors.update(dw1=relative_rms(from_g8_weight(dw1[0]) if args.gate_up_layout == "g8" else dw1[0], weight_targets[0]),
                          dw2=relative_rms(dw2[0], weight_targets[1]))
        else:
            errors.update(dw1=0., dw2=0.)
        active_plan = plan.inner if backend == "hybrid" else plan
        active_rows = int(active_plan.current_rows if args.ring_rows is None else active_plan.global_offsets[-1])
        zero_score_derivative_preserved = rank != 0 or float(ds[0, 0]) != 0
        expected_rows = ((size * c * k + plan.alignment - 1) // plan.alignment * plan.alignment
                         if rank == 0 else 0)
        limit = .13 if args.compute == "fp8" else .04
        passed = (unused_zero and zero_score_derivative_preserved and active_rows == expected_rows
                  and all(math.isfinite(error) and error < limit for error in errors.values()))
        print(json.dumps(dict(case="skew_capacity_reference", rank=rank, ranks=size,
            backend=backend, compute=args.compute, tolerance=limit, hidden=h, intermediate=i, local_experts=e, total_experts=e * size,
            topk=k, tokens_per_rank=c, chunk_tokens=c, graph=args.graph,
            dispatch_pipeline=getattr(plan, "dispatch_pipeline", False) if backend == "peer" else None,
            return_pipeline=args.return_pipeline if backend == "peer" else None,
            pipeline_comm_ctas=args.pipeline_comm_ctas if backend == "peer" else None,
            pipeline_serialized=args.pipeline_serialize if backend == "peer" else None,
            inline_dispatch=getattr(plan, "inline_dispatch", None),
            dispatch_assist_ctas=getattr(plan, "dispatch_assist_ctas", None),
            dispatch_external=getattr(plan, "dispatch_external", None),
            dispatch_external_serial=getattr(plan, "dispatch_external_serial", None),
            cache_return_addresses=args.cache_return_addresses if backend == "peer" else None,
            ffn_fusion=args.ffn_fusion if backend == "peer" else None,
            dag_fusion=args.dag_fusion if backend == "peer" else None,
            dag_shared_pool=args.dag_shared_pool if backend == "peer" else None,
            dag_input_prefetch=args.dag_input_prefetch if backend == "peer" else None,
            dag_backward=args.dag_backward if backend == "peer" else None,
            dag_direct_return=args.dag_direct_return if backend == "peer" else None,
            input_copy_ilp=args.input_copy_ilp if backend == "peer" else None,
            wgrad_overlap=plan.wgrad_overlap,
            overlap_expert_dy=plan.overlap_expert_dy,
            gate_up_layout=plan.gate_up_layout,
            fuse_fc1_activation=args.fuse_fc1_activation if backend == "peer" else None,
            expert_local_grad=args.expert_local_grad if backend == "peer" else None,
            ring_rows=args.ring_rows, last_window_rows=int(plan.current_rows),
            block_m=plan.block_m,
            full_routed_row_capacity=plan.full_row_capacity,
            route_window_count=getattr(plan, "route_window_count", 1),
            gemm_config=args.gemm_config,
            capacity_rows=plan.rows, alignment=plan.alignment, active_rows=active_rows,
            largest_active_element_offset=max(0, min(active_rows, plan.rows) * h - 1),
            relative_rms=errors, unused_expert_gradients_exactly_zero=unused_zero,
            zero_score_derivative_preserved=zero_score_derivative_preserved,
            every_element_compared=True, reference="independent FP32 expert with summed duplicate weights",
            passed=passed)), flush=True)
        result = torch.tensor(int(passed))
        dist.all_reduce(result, op=dist.ReduceOp.MIN, group=reference_group)
        if not result.item():
            raise AssertionError("at least one rank failed the worst-skew capacity reference")
        del graph, actual, y, dx, dw1, dw2, ds
        gc.collect()
        torch.cuda.synchronize(device)
        dist.barrier(group=reference_group)
        if hasattr(plan, "close"):
            plan.close()
        del plan
        gc.collect()
        torch.cuda.empty_cache()
    if hybrid_groups is not None:
        del active_plan
        gc.collect()
        hybrid_groups.close()
    dist.barrier(group=reference_group)
    if reference_group is not dist.group.WORLD:
        dist.destroy_process_group(reference_group)
    destroy_training_process_group()


if __name__ == "__main__":
    main()
