"""Wall-time and memory baseline for native PPU MoE forward/backward.

PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
    tests/benchmark_moe_training.py --transport symmetric --backend both
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from deep_gemm.training import bf16_moe_backward, bf16_moe_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024, help="tokens per EP rank")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--local-experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--backend", choices=("bf16", "fp8", "both"), default="bf16")
    parser.add_argument("--transport", choices=("collective", "symmetric"), default="collective")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--grouped", action="store_true")
    args = parser.parse_args()
    if min(args.tokens, args.hidden, args.intermediate, args.local_experts,
           args.topk, args.iterations) < 1 or args.warmup < 1:
        raise ValueError("dimensions, iterations and warmup must be positive")
    size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    if size > 1:
        dist.init_process_group("nccl", device_id=device)
    group = dist.group.WORLD if size > 1 else None
    if args.topk > size * args.local_experts:
        raise ValueError("topk exceeds total experts")
    torch.manual_seed(9083 + rank)
    x = (torch.randn(args.tokens, args.hidden, device=device) * 0.1).bfloat16()
    w1 = (torch.randn(args.local_experts, 2 * args.intermediate, args.hidden,
                      device=device) * 0.03).bfloat16()
    w2 = (torch.randn(args.local_experts, args.hidden, args.intermediate,
                      device=device) * 0.03).bfloat16()
    logits = torch.randn(args.tokens, size * args.local_experts, device=device)
    values, ids = logits.topk(args.topk, dim=-1)
    scores = values.softmax(-1).contiguous()
    dy = (torch.randn_like(x) * 0.1).bfloat16()
    backends = ("deep_gemm", "deep_gemm_fp8") if args.backend == "both" else (
        "deep_gemm_fp8" if args.backend == "fp8" else "deep_gemm",)
    for backend in backends:
        samples = []
        forward_memory = []
        for iteration in range(args.warmup + args.iterations):
            if group is not None:
                dist.barrier(group=group)
            torch.cuda.synchronize(device)
            if iteration == args.warmup:
                torch.cuda.reset_peak_memory_stats(device)
            initial_memory = torch.cuda.memory_allocated(device)
            begin = time.perf_counter()
            y, state = bf16_moe_forward(
                x, ids, scores, w1, w2, group=group, backend=backend,
                transport=args.transport, recompute=args.recompute,
                grouped_gemm=args.grouped,
            )
            torch.cuda.synchronize(device)
            forward_end = time.perf_counter()
            context_memory = torch.cuda.memory_allocated(device) - initial_memory
            grads = bf16_moe_backward(dy, state)
            torch.cuda.synchronize(device)
            end = time.perf_counter()
            elapsed = torch.tensor([forward_end - begin, end - forward_end, end - begin],
                                   device=device, dtype=torch.float64)
            forward_bytes = torch.tensor(context_memory, device=device, dtype=torch.int64)
            if group is not None:
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
                dist.all_reduce(forward_bytes, op=dist.ReduceOp.MAX, group=group)
            if iteration >= args.warmup:
                samples.append(elapsed.cpu().tolist())
                forward_memory.append(int(forward_bytes.item()))
            del state, y, grads
        memory = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.int64)
        if group is not None:
            dist.all_reduce(memory, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            print(json.dumps({
                "case": "moe_training_baseline", "backend": backend,
                "ep_size": size, "transport": args.transport, "recompute": args.recompute,
                "grouped_gemm": args.grouped,
                "tokens_per_rank": args.tokens, "hidden": args.hidden,
                "intermediate": args.intermediate, "local_experts": args.local_experts,
                "topk": args.topk, "warmup": args.warmup, "iterations": args.iterations,
                "median_max_rank_ms": dict(zip(("forward", "backward", "total"),
                    (1000 * statistics.median(row[i] for row in samples) for i in range(3)))),
                "max_rank_allocated_bytes": int(memory.item()),
                "median_max_rank_forward_extra_bytes": statistics.median(forward_memory),
                "memory_scope": "PyTorch allocator reported; includes output and live forward context",
                "samples_max_rank_ms": [[1000 * value for value in row] for row in samples],
                "scope": "multi-kernel baseline; host scheduling and communication included",
            }), flush=True)
    if group is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
