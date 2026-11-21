"""Comparable bounded MoE forward/backward timings, including recomputation.

Both transports use the same chunk capacity, weights, routes, FP32 wgrad,
output contract and full FC1/SwiGLU/FC2 backward replay. Optimizer and JIT are
outside the measured range. Event times include every device operation in a
step; wall time additionally includes host submission and synchronization.
"""

import argparse
from datetime import timedelta
import gc
import json
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist

import deep_gemm as dg
from deep_gemm.training import NativeMoEPlan, PeerMoEPlan, PersistentMoEPlan, CollectiveFusedMoEPlan, DeepEPMoEPlan
from moe_distributed_utils import destroy_training_process_group
from deep_gemm.training.layout import to_g8_weight


def sample_outputs(values):
    return [value.detach().flatten()[::max(1, value.numel() // 8192)].float().cpu() for value in values]


def relative_rms(actual, expected):
    if not actual.numel():
        return 0.0
    return float((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12))


class ConfiguredCollectivePlan(NativeMoEPlan):
    """Benchmark control using explicit tiles in the existing nopad GEMMs."""

    def _gemm(self, a, weight, output, name, prequantized=False):
        bn, bk, wm, wn, stages = self.benchmark_gemm_config
        configs = (dg.get_num_sms(), self.block_m, bn, bk, wm, wn, stages, (0, 0, 0))
        dg.m_grouped_bf16_gemm_nt_nopad(a, weight, output, self.group_ids,
                                       m_rows=self.lengths, configs=configs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--local-experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--backend", choices=("collective", "collective_fused", "collective_both",
                                             "peer", "persistent", "both", "all", "deepep", "deepep_peer"), default="both")
    parser.add_argument("--compute", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--routing", choices=("balanced", "skewed", "masked"), default="balanced")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Profiler API brackets only timed iterations")
    parser.add_argument("--scope", choices=("step", "forward", "backward"), default="step",
                        help="Backward retains full replay and FP32 gradient zero/accumulation")
    parser.add_argument("--topology", default="unspecified")
    parser.add_argument("--gemm-config", type=int, nargs=5, metavar=("BN", "BK", "WM", "WN", "STAGES"))
    parser.add_argument("--collective-gemm-config", type=int, nargs=5,
                        metavar=("BN", "BK", "WM", "WN", "STAGES"),
                        help="Explicit BF16 nopad tiles for the collective control")
    parser.add_argument("--maximize-ctas", action="store_true")
    parser.add_argument("--m-schedule", choices=("nopad", "contiguous"), default="nopad")
    parser.add_argument("--separate-fp8-quantization", action="store_true")
    parser.add_argument("--fp8-ablation", action="store_true")
    parser.add_argument("--peer-ablation", action="store_true")
    parser.add_argument("--no-fuse-dispatch", action="store_true")
    parser.add_argument("--no-fuse-return", action="store_true")
    parser.add_argument("--capacity-pointwise", action="store_true")
    parser.add_argument("--separate-local-grad", action="store_true")
    parser.add_argument("--fuse-activation-backward", action="store_true")
    parser.add_argument("--activation-backward-ablation", action="store_true")
    parser.add_argument("--scalar-peer-copy", action="store_true")
    parser.add_argument("--collective-return-ablation", action="store_true")
    parser.add_argument("--scalar-collective-copy", action="store_true")
    parser.add_argument("--fast-weight-transpose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-transpose-ablation", action="store_true")
    parser.add_argument("--dispatch-cache-n", type=int, choices=(0, 2, 4, 8, 16), default=0)
    parser.add_argument("--reuse-packed-dy", action="store_true")
    parser.add_argument("--dispatch-cache-aiu", action="store_true")
    parser.add_argument("--staged-return", action="store_true")
    parser.add_argument("--dispatch-cache-ctas", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--dispatch-producer", action="store_true")
    parser.add_argument("--producer-local-aiu", action="store_true")
    parser.add_argument("--producer-input-aiu", action="store_true")
    parser.add_argument("--dispatch-pipeline", action="store_true")
    parser.add_argument("--return-pipeline", action="store_true")
    parser.add_argument("--pipeline-comm-ctas", type=int, default=2)
    parser.add_argument("--pipeline-statistics", action="store_true")
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
    parser.add_argument("--gate-up-layout", choices=("split", "g8"), default="split")
    parser.add_argument("--fuse-fc1-activation", action="store_true")
    parser.add_argument("--dispatch-producer-tiles", type=int, choices=(1, 2, 4), default=1)
    for role in ("fc1", "fc2", "dhidden", "dx"):
        parser.add_argument("--peer-" + role + "-config", type=int, nargs=5)
    parser.add_argument("--deepep-comm-sms", type=int, default=8)
    parser.add_argument("--deepep-scaleup", action="store_true",
                        help="Use DeepEP fabric intranode path across one verified supernode")
    parser.add_argument("--process-group-backend", choices=("nccl", "gloo"), default="nccl",
                        help="Gloo carries CPU metadata for DeepEP; payload remains in DeepEP")
    args = parser.parse_args()
    if args.process_group_backend == "gloo" and args.backend != "deepep":
        parser.error("Gloo metadata groups currently support only --backend deepep")
    if args.ring_rows is not None and args.backend not in ("peer", "deepep", "deepep_peer"):
        parser.error("routed-row windows require peer or DeepEP transports")
    if args.block_m != 128 and (args.compute != "bf16" or args.backend in ("persistent", "collective_fused", "collective_both", "all")):
        parser.error("BM256 requires BF16 peer/DeepEP or a configured collective control")
    if args.block_m == 256 and args.backend in ("collective", "both") and args.collective_gemm_config is None:
        args.collective_gemm_config = [256, 64, 64, 64, 2]
    if args.collective_gemm_config and (args.compute != "bf16" or args.m_schedule != "nopad"):
        parser.error("explicit collective tiles require BF16 nopad compute")
    if args.fp8_ablation and (args.backend != "collective" or args.compute != "fp8"):
        parser.error("FP8 ablation requires --backend collective --compute fp8")
    if args.peer_ablation and (args.backend not in ("peer", "both", "all") or args.compute != "bf16"):
        parser.error("peer ablation requires a BF16 peer backend")
    if args.activation_backward_ablation and (args.peer_ablation or args.backend not in ("peer", "both") or args.compute != "bf16"):
        parser.error("activation backward ablation requires BF16 peer/both without peer-ablation")
    if args.collective_return_ablation and (args.backend not in ("collective_fused", "collective_both") or args.compute != "bf16"):
        parser.error("collective return ablation requires BF16 collective_fused/collective_both")
    if min(args.tokens, args.hidden, args.intermediate, args.local_experts, args.topk,
           args.chunk_tokens, args.warmup, args.iterations, args.repeats) < 1:
        raise ValueError("dimensions and iteration counts must be positive")
    rank, size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if size > 1:
        dist.init_process_group(args.process_group_backend,
            device_id=device if args.process_group_backend == "nccl" else None,
            timeout=timedelta(seconds=600))
    group = dist.group.WORLD if size > 1 else None
    torch.manual_seed(98231 + rank)
    h, i, e, k, t = args.hidden, args.intermediate, args.local_experts, args.topk, args.tokens
    if k > e * size:
        raise ValueError("topk exceeds total experts")
    x = (torch.randn(t, h, device=device) * 0.1).bfloat16()
    w1 = (torch.randn(e, 2 * i, h, device=device) * 0.02).bfloat16()
    w2 = (torch.randn(e, h, i, device=device) * 0.02).bfloat16()
    if args.gate_up_layout == "g8":
        w1 = to_g8_weight(w1)
    ids = torch.randn(t, e * size, device=device).topk(k, -1).indices.contiguous()
    if args.routing == "skewed": ids.fill_(0)
    if args.routing == "masked": ids.fill_(-1)
    scores = torch.randn(t, k, device=device).softmax(-1)
    dy = (torch.randn_like(x) * 0.1).bfloat16()
    valid = torch.full((), t, device=device, dtype=torch.int32)
    accumulators = (torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    if args.deepep_scaleup:
        os.environ["DEEPEP_NUM_MAX_ICN_PEERS"] = str(size)
    backends = (("deepep", "peer") if args.backend == "deepep_peer" else
                ("collective", "peer", "persistent") if args.backend == "all" else
                ("collective", "peer") if args.backend == "both" else
                ("collective", "collective_fused") if args.backend == "collective_both" else (args.backend,))
    experiments = []
    for backend in backends:
        if args.weight_transpose_ablation:
            experiments.extend((backend, {"fast_weight_transpose": fast}) for fast in (False, True))
        elif args.fp8_ablation:
            experiments.extend((backend, {"fuse_activation_quant": fused}) for fused in (False, True))
        elif backend == "peer" and args.peer_ablation:
            experiments.extend((backend, {"fuse_dispatch": dispatch, "fuse_return": returned})
                               for dispatch, returned in ((False, False), (False, True), (True, False), (True, True)))
        elif backend == "peer" and args.activation_backward_ablation:
            experiments.extend((backend, {"fuse_activation_backward": fused}) for fused in (False, True))
        elif backend == "collective_fused" and args.collective_return_ablation:
            experiments.extend((backend, {"fuse_return": fused}) for fused in (False, True))
        else:
            experiments.append((backend, {}))
    reference_samples = None
    with torch.cuda.stream(stream):
        for backend, controls in experiments:
            graph = None
            if group is not None: dist.barrier(group=group)
            torch.cuda.synchronize(device)
            initial_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            plan_class = (DeepEPMoEPlan if backend == "deepep" else
                          PersistentMoEPlan if backend == "persistent" else PeerMoEPlan if backend == "peer"
                          else CollectiveFusedMoEPlan if backend == "collective_fused" else NativeMoEPlan)
            if backend == "collective" and args.collective_gemm_config:
                plan_class = ConfiguredCollectivePlan
            options = {"gemm_configs": {role: args.gemm_config for role in ("fc1", "fc2", "dhidden", "dx")}}
            options = options if backend == "peer" and args.gemm_config else {}
            if backend == "peer":
                options.update(fuse_dispatch=not args.no_fuse_dispatch, fuse_return=not args.no_fuse_return,
                               active_rows=not args.capacity_pointwise,
                               fuse_local_grad=not args.separate_local_grad,
                               fuse_activation_backward=args.fuse_activation_backward,
                               vectorized_copy=not args.scalar_peer_copy,
                               reuse_packed_dy=args.reuse_packed_dy)
                options["staged_return"] = args.staged_return
                options["dispatch_cache_ctas"] = args.dispatch_cache_ctas
                options["dispatch_producer"] = args.dispatch_producer
                options["return_vector_bytes"] = args.return_vector_bytes
                options["fuse_fc1_activation"] = args.fuse_fc1_activation
                options.update(dag_fusion=args.dag_fusion, dag_input_ctas=args.dag_input_ctas,
                               dag_return_ctas=args.dag_return_ctas, dag_serialize=args.dag_serialize,
                               dag_trace=args.dag_trace, dag_shared_pool=args.dag_shared_pool,
                               dag_input_prefetch=args.dag_input_prefetch,
                               dag_backward=args.dag_backward, dag_direct_return=args.dag_direct_return)
                options["producer_local_aiu"] = args.producer_local_aiu
                options["producer_input_aiu"] = args.producer_input_aiu
                options["dispatch_pipeline"] = args.dispatch_pipeline
                options["return_pipeline"] = args.return_pipeline
                options["pipeline_comm_ctas"] = args.pipeline_comm_ctas
                options["pipeline_statistics"] = args.pipeline_statistics
                options["pipeline_serialize"] = args.pipeline_serialize
                options["inline_dispatch"] = args.inline_dispatch
                options["cache_return_addresses"] = args.cache_return_addresses
                options["ffn_fusion"] = args.ffn_fusion
                options["expert_local_grad"] = args.expert_local_grad
                options["dispatch_producer_tiles"] = args.dispatch_producer_tiles
                for role in ("fc1", "fc2", "dhidden", "dx"):
                    config = getattr(args, "peer_" + role + "_config")
                    if config is not None:
                        options.setdefault("gemm_configs", {})[role] = config
            if backend == "persistent": options = {"maximize_ctas": args.maximize_ctas}
            if backend == "deepep":
                options = {"comm_sms": args.deepep_comm_sms}
                if args.gemm_config:
                    options["gemm_configs"] = {role: args.gemm_config for role in ("fc1", "fc2", "dhidden", "dx")}
            if backend == "collective":
                options = {"m_schedule": args.m_schedule,
                           "fuse_activation_quant": not args.separate_fp8_quantization,
                           "active_pointwise": not args.capacity_pointwise,
                           "vectorized_routes": not args.scalar_collective_copy}
            if backend == "collective_fused":
                options = {"active_pointwise": not args.capacity_pointwise,
                           "vectorized_routes": not args.scalar_collective_copy,
                           "fuse_return": not args.no_fuse_return,
                           "fuse_activation_backward": args.fuse_activation_backward}
                if args.gemm_config:
                    options["gemm_configs"] = {role: args.gemm_config for role in ("fc1", "fc2", "dhidden", "dx")}
            options["fast_weight_transpose"] = args.fast_weight_transpose
            options["block_m"] = args.block_m
            options["gate_up_layout"] = args.gate_up_layout
            if args.ring_rows is not None:
                options["ring_rows"] = args.ring_rows
            options.update(controls)
            if backend == "peer" and options.get("fuse_dispatch", True):
                options["dispatch_cache_n"] = args.dispatch_cache_n
                options["dispatch_cache_aiu"] = args.dispatch_cache_aiu
            plan = plan_class(h, i, e, k, t, chunk_tokens=args.chunk_tokens, group=group,
                              device=device, compute=args.compute, check_errors=False, **options)
            if isinstance(plan, ConfiguredCollectivePlan):
                plan.benchmark_gemm_config = args.collective_gemm_config

            def step():
                plan.reset_errors()
                for gradient in accumulators: gradient.zero_()
                output = plan.forward(x, ids, scores, w1, w2, valid)
                dx, dw1, dw2, ds = plan.backward(dy, x, ids, scores, w1, w2, valid, accumulators)
                return output, dx, dw1, dw2, ds

            def timed_step():
                if args.scope == "step":
                    return step()
                plan.reset_errors()
                if args.scope == "forward":
                    return (plan.forward(x, ids, scores, w1, w2, valid),)
                for gradient in accumulators:
                    gradient.zero_()
                return plan.backward(dy, x, ids, scores, w1, w2, valid, accumulators)

            for _ in range(args.warmup):
                results = step()
            torch.cuda.synchronize(device)
            plan.check()
            current_samples = sample_outputs(results)
            if reference_samples is None:
                reference_samples = current_samples
            sample_errors = dict(zip(("y", "dx", "dw1", "dw2", "dscores"),
                                     (relative_rms(a, b) for a, b in zip(current_samples, reference_samples))))
            if not all(value < 0.05 for value in sample_errors.values()):
                raise AssertionError((backend, sample_errors))
            del current_samples
            if not args.eager:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    results = timed_step()
                for _ in range(args.warmup): graph.replay()
                torch.cuda.synchronize(device)
                plan.check()
            elapsed_event, elapsed_wall = [], []
            if args.profile: torch.cuda.profiler.start()
            for _ in range(args.repeats):
                if group is not None: dist.barrier(group=group)
                torch.cuda.synchronize(device)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall_begin = time.perf_counter()
                begin.record(stream)
                for _ in range(args.iterations):
                    if graph is None: results = timed_step()
                    else: graph.replay()
                end.record(stream)
                end.synchronize()
                wall_ms = (time.perf_counter() - wall_begin) * 1000 / args.iterations
                event_ms = begin.elapsed_time(end) / args.iterations
                metrics_device = "cpu" if args.process_group_backend == "gloo" else device
                timings = torch.tensor([event_ms, wall_ms], device=metrics_device, dtype=torch.float64)
                if group is not None: dist.all_reduce(timings, op=dist.ReduceOp.MAX, group=group)
                event_ms, wall_ms = timings.cpu().tolist()
                elapsed_event.append(event_ms)
                elapsed_wall.append(wall_ms)
            if args.profile: torch.cuda.profiler.stop()
            plan.check()
            timed_indices = range(5) if args.scope == "step" else (0,) if args.scope == "forward" else range(1, 5)
            timed_errors = [relative_rms(a, reference_samples[index])
                            for a, index in zip(sample_outputs(results), timed_indices)]
            if not all(value < 0.05 for value in timed_errors):
                raise AssertionError((backend, args.scope, timed_errors))
            metrics_device = "cpu" if args.process_group_backend == "gloo" else device
            peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=metrics_device, dtype=torch.int64)
            if group is not None: dist.all_reduce(peak, op=dist.ReduceOp.MAX, group=group)
            if rank == 0:
                print(json.dumps({
                    "case": "bounded_training_benchmark", "backend": backend, "compute": args.compute,
                    "host": socket.gethostname(), "device": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                    "process_group_backend": args.process_group_backend,
                    "ep_size": size, "topology": args.topology, "graph": not args.eager,
                    "instrumented": args.profile or args.dag_trace,
                    "tokens_per_rank": t, "hidden": h, "intermediate": i, "local_experts": e,
                    "topk": k, "chunk_tokens": args.chunk_tokens, "routing": args.routing,
                    "gemm_configs": getattr(plan, "gemm_configs", None),
                    "collective_m_schedule": plan.m_schedule if backend.startswith("collective") else None,
                    "collective_active_pointwise": plan.active_pointwise if backend.startswith("collective") else None,
                    "collective_vectorized_routes": plan.vectorized_routes if backend.startswith("collective") else None,
                    "collective_fuse_return": plan.fuse_return if backend == "collective_fused" else None,
                    "collective_fuse_activation_backward": plan.fuse_activation_backward if backend == "collective_fused" else None,
                    "collective_gemm_config": args.collective_gemm_config if backend == "collective" else None,
                    "fused_activation_quantization": (plan.fuse_activation_quant
                                                       if backend == "collective" and args.compute == "fp8" else None),
                    "peer_fuse_dispatch": getattr(plan, "fuse_dispatch", None) if backend == "peer" else None,
                    "peer_fuse_return": getattr(plan, "fuse_return", None) if backend == "peer" else None,
                    "peer_active_rows": getattr(plan, "active_rows", None) if backend == "peer" else None,
                    "peer_fuse_local_grad": getattr(plan, "fuse_local_grad", None) if backend == "peer" else None,
                    "peer_fuse_activation_backward": getattr(plan, "fuse_activation_backward", None) if backend == "peer" else None,
                    "peer_vectorized_copy": getattr(plan, "vectorized_copy", None) if backend == "peer" else None,
                    "peer_dispatch_cache_n": getattr(plan, "dispatch_cache_n", None) if backend == "peer" else None,
                    "peer_dispatch_cache_ctas": getattr(plan, "dispatch_cache_ctas", None) if backend == "peer" else None,
                    "peer_dispatch_producer": getattr(plan, "dispatch_producer", None) if backend == "peer" else None,
                    "peer_dispatch_pipeline": getattr(plan, "dispatch_pipeline", None) if backend == "peer" else None,
                    "peer_return_pipeline": getattr(plan, "return_pipeline", None) if backend == "peer" else None,
                    "pipeline_comm_ctas": getattr(plan, "pipeline_comm_ctas", None) if backend == "peer" else None,
                    "dispatch_pipeline_grid": getattr(plan, "dispatch_pipeline_ctas", None) if backend == "peer" else None,
                    "return_pipeline_grid": getattr(plan, "return_pipeline_ctas", None) if backend == "peer" else None,
                    "pipeline_serialized_control": getattr(plan, "pipeline_serialize", None) if backend == "peer" else None,
                    "peer_inline_dispatch": getattr(plan, "inline_dispatch", None) if backend == "peer" else None,
                    "peer_cached_return_addresses": getattr(plan, "cache_return_addresses", None) if backend == "peer" else None,
                    "peer_return_vector_bytes": getattr(plan, "return_vector_bytes", None) if backend == "peer" else None,
                    "gate_up_layout": plan.gate_up_layout,
                    "peer_fuse_fc1_activation": getattr(plan, "fuse_fc1_activation", None) if backend == "peer" else None,
                    "weight_layout_scope": "persistent optimizer parameter storage; checkpoint conversion outside timing",
                    "dag_fusion": getattr(plan, "dag_fusion", None) if backend == "peer" else None,
                    "dag_shared_pool": getattr(plan, "dag_shared_pool", None) if backend == "peer" else None,
                    "dag_input_prefetch": getattr(plan, "dag_input_prefetch", None) if backend == "peer" else None,
                    "dag_backward": getattr(plan, "dag_backward", None) if backend == "peer" else None,
                    "dag_direct_return": getattr(plan, "dag_direct_return", None) if backend == "peer" else None,
                    "dag_direct_return_activity_scope": "return embedded in FC2/DX; task spans do not resolve internal overlap" if backend == "peer" and args.dag_direct_return else None,
                    "dag_input_prefetch_max_inflight_ctas": args.dag_input_ctas if backend == "peer" and args.dag_input_prefetch else None,
                    "dag_input_ctas": getattr(plan, "dag_input_ctas", None) if backend == "peer" and not args.dag_shared_pool else None,
                    "dag_return_ctas": getattr(plan, "dag_return_ctas", None) if backend == "peer" and not args.dag_shared_pool else None,
                    "dag_serialized_control": getattr(plan, "dag_serialize", None) if backend == "peer" else None,
                    "dag_grid": getattr(plan, "dag_grid", None) if backend == "peer" else None,
                    "dag_activity_statistics": plan.dag_monitor.cpu().tolist() if backend == "peer" and args.dag_fusion and args.dag_trace else None,
                    "dag_activity_columns": ["total_cycles", "input_active", "compute_active", "both_active", "neither_active", "fc1_active", "activation_active", "fc2_active", "return_active", "sample_count", "dropped_samples", "error"],
                    "peer_ffn_fusion": getattr(plan, "ffn_fusion", None) if backend == "peer" else None,
                    "peer_expert_local_grad": getattr(plan, "expert_local_grad", None) if backend == "peer" else None,
                    "ring_rows_requested": args.ring_rows,
                    "packed_ring_rows": plan.rows,
                    "block_m": plan.block_m,
                    "full_routed_row_capacity": plan.full_row_capacity,
                    "route_window_count": getattr(plan, "route_window_count", 1),
                    "pipeline_statistics_last_fc1": (plan.pipeline_statistics.cpu().tolist()
                        if backend == "peer" and plan.pipeline_statistics is not None else None),
                    "pipeline_statistics_last_dx": (plan.return_pipeline_statistics.cpu().tolist()
                        if backend == "peer" and plan.return_pipeline_statistics is not None else None),
                    "pipeline_statistics_columns": ["total_cycles", "wait_cycles", "copy_capacity_bytes", "role_0_input_1_compute_2_return"],
                    "peer_producer_local_aiu": getattr(plan, "producer_local_aiu", None) if backend == "peer" else None,
                    "peer_producer_input_aiu": getattr(plan, "producer_input_aiu", None) if backend == "peer" else None,
                    "peer_dispatch_producer_tiles": getattr(plan, "dispatch_producer_tiles", None) if backend == "peer" else None,
                    "peer_dispatch_cache_aiu": getattr(plan, "dispatch_cache_aiu", None) if backend == "peer" else None,
                    "peer_staged_return": getattr(plan, "staged_return", None) if backend == "peer" else None,
                    "peer_staged_return_row_padding": getattr(plan, "staged_return_row_padding", None) if backend == "peer" else None,
                    "peer_reuse_packed_dy": getattr(plan, "reuse_packed_dy", None) if backend == "peer" else None,
                    "deepep_version": getattr(plan, "deep_ep_version", None),
                    "deepep_backend": getattr(plan, "deep_ep_backend", None),
                    "deepep_rdma_ranks": getattr(plan, "deep_ep_rdma_ranks", None),
                    "deepep_comm_sms": args.deepep_comm_sms if backend == "deepep" else None,
                    "deepep_rdma_qps": getattr(plan, "deep_ep_qps", None) if backend == "deepep" else None,
                    "deepep_mnnvl_disabled": os.environ.get("SAILSHMEM_DISABLE_MNNVL") if backend == "deepep" else None,
                    "deepep_scaleup": args.deepep_scaleup if backend == "deepep" else None,
                    "persistent_ctas": getattr(plan, "num_ctas", None),
                    "persistent_resident_blocks_per_cu": getattr(plan, "resident_blocks_per_cu", None),
                    "fast_weight_transpose": plan.fast_weight_transpose,
                    "warmup": args.warmup, "iterations": args.iterations, "repeats": args.repeats,
                    "median_max_rank_event_ms": statistics.median(elapsed_event),
                    "median_max_rank_wall_ms": statistics.median(elapsed_wall),
                    "samples_max_rank_event_ms": elapsed_event, "samples_max_rank_wall_ms": elapsed_wall,
                    "workspace_bytes": plan.workspace_bytes,
                    "activation_workspace_bytes": plan.activation_workspace_bytes,
                    "max_rank_torch_allocated_bytes": int(peak),
                    "rank0_initial_torch_allocated_bytes": initial_memory,
                    "sample_relative_rms_vs_first_backend": sample_errors,
                    "timed_sample_relative_rms_vs_first_backend": timed_errors,
                    "timing_scope": args.scope,
                    "scope": ("forward; no backward, optimizer or shared expert" if args.scope == "forward"
                              else "full backward replay + FP32 wgrad zero/accumulation; no optimizer or shared expert"
                              if args.scope == "backward" else
                              "forward + full backward replay + FP32 wgrad zero/accumulation; no optimizer"),
                    "memory_scope": "logical workspace and PyTorch allocator; excludes driver/VMM reservations",
                }), flush=True)
            # Release graph references before releasing PCCL/symmetric storage.
            del graph, results
            gc.collect()
            torch.cuda.synchronize(device)
            if hasattr(plan, "close"):
                plan.close()
            del plan
    if group is not None:
        destroy_training_process_group()


if __name__ == "__main__":
    main()
