"""Comparable bounded MoE forward/backward timings, including recomputation.

Both transports use the same chunk capacity, weights, routes, FP32 wgrad,
output contract and full FC1/SwiGLU/FC2 backward replay. Optimizer and JIT are
outside the measured range. Event times include every device operation in a
step; wall time additionally includes host submission and synchronization.
"""

import argparse
import gc
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from deep_gemm.training import NativeMoEPlan, PeerMoEPlan, PersistentMoEPlan


def sample_outputs(values):
    return [value.detach().flatten()[::max(1, value.numel() // 8192)].float().cpu() for value in values]


def relative_rms(actual, expected):
    if not actual.numel():
        return 0.0
    return float((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--local-experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--backend", choices=("collective", "peer", "persistent", "both", "all"), default="both")
    parser.add_argument("--compute", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--routing", choices=("balanced", "skewed", "masked"), default="balanced")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Profiler API brackets only timed iterations")
    parser.add_argument("--topology", default="unspecified")
    parser.add_argument("--gemm-config", type=int, nargs=5, metavar=("BN", "BK", "WM", "WN", "STAGES"))
    args = parser.parse_args()
    if min(args.tokens, args.hidden, args.intermediate, args.local_experts, args.topk,
           args.chunk_tokens, args.warmup, args.iterations, args.repeats) < 1:
        raise ValueError("dimensions and iteration counts must be positive")
    rank, size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if size > 1:
        dist.init_process_group("nccl", device_id=device)
    group = dist.group.WORLD if size > 1 else None
    torch.manual_seed(98231 + rank)
    h, i, e, k, t = args.hidden, args.intermediate, args.local_experts, args.topk, args.tokens
    if k > e * size:
        raise ValueError("topk exceeds total experts")
    x = (torch.randn(t, h, device=device) * 0.1).bfloat16()
    w1 = (torch.randn(e, 2 * i, h, device=device) * 0.02).bfloat16()
    w2 = (torch.randn(e, h, i, device=device) * 0.02).bfloat16()
    ids = torch.randn(t, e * size, device=device).topk(k, -1).indices.contiguous()
    if args.routing == "skewed": ids.fill_(0)
    if args.routing == "masked": ids.fill_(-1)
    scores = torch.randn(t, k, device=device).softmax(-1)
    dy = (torch.randn_like(x) * 0.1).bfloat16()
    valid = torch.full((), t, device=device, dtype=torch.int32)
    accumulators = (torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    backends = (("collective", "peer", "persistent") if args.backend == "all" else
                ("collective", "peer") if args.backend == "both" else (args.backend,))
    reference_samples = None
    with torch.cuda.stream(stream):
        for backend in backends:
            graph = None
            if group is not None: dist.barrier(group=group)
            torch.cuda.synchronize(device)
            initial_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            plan_class = PersistentMoEPlan if backend == "persistent" else PeerMoEPlan if backend == "peer" else NativeMoEPlan
            options = {"gemm_configs": {role: args.gemm_config for role in ("fc1", "fc2", "dhidden", "dx")}}
            options = options if backend == "peer" and args.gemm_config else {}
            plan = plan_class(h, i, e, k, t, chunk_tokens=args.chunk_tokens, group=group,
                              device=device, compute=args.compute, check_errors=False, **options)

            def step():
                plan.reset_errors()
                for gradient in accumulators: gradient.zero_()
                output = plan.forward(x, ids, scores, w1, w2, valid)
                dx, dw1, dw2, ds = plan.backward(dy, x, ids, scores, w1, w2, valid, accumulators)
                return output, dx, dw1, dw2, ds

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
                    results = step()
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
                    if graph is None: results = step()
                    else: graph.replay()
                end.record(stream)
                end.synchronize()
                wall_ms = (time.perf_counter() - wall_begin) * 1000 / args.iterations
                event_ms = begin.elapsed_time(end) / args.iterations
                timings = torch.tensor([event_ms, wall_ms], device=device, dtype=torch.float64)
                if group is not None: dist.all_reduce(timings, op=dist.ReduceOp.MAX, group=group)
                event_ms, wall_ms = timings.cpu().tolist()
                elapsed_event.append(event_ms)
                elapsed_wall.append(wall_ms)
            if args.profile: torch.cuda.profiler.stop()
            plan.check()
            peak = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.int64)
            if group is not None: dist.all_reduce(peak, op=dist.ReduceOp.MAX, group=group)
            if rank == 0:
                print(json.dumps({
                    "case": "bounded_training_benchmark", "backend": backend, "compute": args.compute,
                    "ep_size": size, "topology": args.topology, "graph": not args.eager,
                    "tokens_per_rank": t, "hidden": h, "intermediate": i, "local_experts": e,
                    "topk": k, "chunk_tokens": args.chunk_tokens, "routing": args.routing,
                    "gemm_configs": getattr(plan, "gemm_configs", None),
                    "persistent_ctas": getattr(plan, "num_ctas", None),
                    "warmup": args.warmup, "iterations": args.iterations, "repeats": args.repeats,
                    "median_max_rank_event_ms": statistics.median(elapsed_event),
                    "median_max_rank_wall_ms": statistics.median(elapsed_wall),
                    "samples_max_rank_event_ms": elapsed_event, "samples_max_rank_wall_ms": elapsed_wall,
                    "workspace_bytes": plan.workspace_bytes,
                    "activation_workspace_bytes": plan.activation_workspace_bytes,
                    "max_rank_torch_allocated_bytes": int(peak),
                    "rank0_initial_torch_allocated_bytes": initial_memory,
                    "sample_relative_rms_vs_first_backend": sample_errors,
                    "scope": "forward + full backward replay + FP32 wgrad zero/accumulation; no optimizer",
                    "memory_scope": "logical workspace and PyTorch allocator; excludes driver/VMM reservations",
                }), flush=True)
            # Release graph references before releasing PCCL/symmetric storage.
            del graph, results, plan
            gc.collect()
            torch.cuda.synchronize(device)
    if group is not None:
        dist.barrier(group=group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
