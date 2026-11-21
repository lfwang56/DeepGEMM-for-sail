"""Every-element distributed DeepEP check without replicating expert weights.

Each rank independently evaluates its own experts on all source tokens in FP32.
CPU/Gloo sums their output/input/score contributions. Weight gradients already
include every source token and stay local. Token payloads use DeepEP on device.
"""

import argparse
from datetime import timedelta
import gc
import json
import math
import os

import torch
import torch.distributed as dist

from deep_gemm.training import DeepEPMoEPlan
from moe_distributed_utils import destroy_training_process_group
from test_moe_training import reference


def relative_rms(actual, expected):
    a, b = actual.detach().float(), expected.detach().float().to(actual.device)
    return float((a - b).square().sum().sqrt() / b.square().sum().sqrt().clamp_min(1e-20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--local-experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=19)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--comm-sms", type=int, default=38)
    parser.add_argument("--ring-rows", type=int)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo", timeout=timedelta(seconds=600))
    h, i, e, k, t = args.hidden, args.intermediate, args.local_experts, args.topk, args.tokens
    if min(h, i, e, k, t) < 1 or k > size * e:
        raise ValueError("invalid model or token dimensions")
    generator = torch.Generator().manual_seed(192731)
    all_x = (torch.randn(size * t, h, generator=generator) * .1).bfloat16()
    all_dy = (torch.randn(size * t, h, generator=generator) * .1).bfloat16()
    all_scores = torch.randn(size * t, k, generator=generator).softmax(-1)
    # A valid route with weight zero must retain its score derivative.
    all_scores[::t, 0] = 0
    token = torch.arange(size * t)[:, None]
    slots = torch.arange(k)[None, :]
    balanced_ids = (token * k + slots) % (size * e)
    valid_counts = [0 if source == 0 else max(0, t - source % 4) for source in range(size)]
    local_slice = slice(rank * t, (rank + 1) * t)
    x, dy = all_x[local_slice].to(device), all_dy[local_slice].to(device)
    scores = all_scores[local_slice].to(device)
    ids = balanced_ids[local_slice].to(device)
    valid = torch.tensor(valid_counts[rank], device=device, dtype=torch.int32)
    weight_rng = torch.Generator().manual_seed(88291 + rank)
    w1 = (torch.randn(e, 2 * i, h, generator=weight_rng) * .02).bfloat16().to(device)
    w2 = (torch.randn(e, h, i, generator=weight_rng) * .02).bfloat16().to(device)
    configs = {role: (256, 64, 64, 64, 2) for role in ("fc1", "fc2", "dhidden", "dx")}
    plan = DeepEPMoEPlan(h, i, e, k, t, chunk_tokens=args.chunk_tokens,
                        group=dist.group.WORLD, device=device, comm_sms=args.comm_sms,
                        ring_rows=args.ring_rows,
                        gemm_configs=configs, check_errors=False)
    print(json.dumps(dict(case="large_deepep_stage", rank=rank, stage="plan_ready",
                          rdma_ranks=plan.deep_ep_rdma_ranks)), flush=True)

    def actual():
        y = plan.forward(x, ids, scores, w1, w2, valid)
        return y, *plan.backward(dy, x, ids, scores, w1, w2, valid)

    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    with torch.cuda.stream(stream):
        outputs = actual()
        if args.graph:
            actual()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = actual()
    stream.synchronize()

    for pattern in ("balanced_masked", "duplicate_skewed", "balanced_reversed"):
        global_ids = balanced_ids.clone()
        if pattern == "duplicate_skewed":
            # Concentrates all routes on one rank, including duplicate experts.
            global_ids.fill_((size // 2) * e)
        elif pattern == "balanced_reversed":
            global_ids = (size * e - 1 - global_ids).contiguous()
        global_ids[1::t, 1] = -1
        ids.copy_(global_ids[local_slice].to(device))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if graph is None:
                outputs = actual()
            else:
                graph.replay()
        stream.synchronize()
        plan.check()
        for source, count in enumerate(valid_counts):
            global_ids[source * t + count:(source + 1) * t] = -1
        local_ids = global_ids - rank * e
        local_ids[(local_ids < 0) | (local_ids >= e)] = -1
        rx, rp = all_x.float().to(device).requires_grad_(), all_scores.to(device).requires_grad_()
        rw1, rw2 = w1.float().requires_grad_(), w2.float().requires_grad_()
        ry = reference(rx, local_ids.to(device), rp, rw1, rw2)
        # Empty experts/ranks still have explicitly defined zero gradients.
        ry = ry + 0 * (rx.sum() + rp.sum() + rw1.sum() + rw2.sum())
        gx, gw1, gw2, gp = torch.autograd.grad(ry, (rx, rw1, rw2, rp), all_dy.float().to(device))
        totals = []
        for partial in (ry, gx, gp):
            total = partial.detach().cpu().contiguous()
            dist.all_reduce(total)
            totals.append(total[local_slice].to(device))
        expected = (totals[0], totals[1], gw1, gw2, totals[2])
        errors = {name: relative_rms(a, b) for name, a, b in zip(
            ("y", "dx", "dw1", "dw2", "dscores"), outputs, expected)}
        passed = all(math.isfinite(error) and error < .04 for error in errors.values())
        print(json.dumps(dict(case="large_distributed_deepep_reference", rank=rank,
            ranks=size, hidden=h, intermediate=i, local_experts=e, total_experts=e * size,
            topk=k, tokens=t, chunk_tokens=args.chunk_tokens, pattern=pattern,
            graph=args.graph, relative_rms=errors, every_element_compared=True,
            ring_rows=args.ring_rows, packed_ring_rows=plan.rows,
            route_window_count=plan.route_window_count,
            payload_transport=plan.deep_ep_backend, rdma_ranks=plan.deep_ep_rdma_ranks,
            rdma_qps=plan.deep_ep_qps, communication_cus=plan.comm_sms,
            metadata_backend="gloo", reference="independent FP32 Torch local experts + CPU sum",
            passed=passed)), flush=True)
        passed_tensor = torch.tensor(int(passed))
        dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
        if not passed_tensor.item():
            raise AssertionError("at least one rank failed the independent reference")
        del rx, rp, rw1, rw2, ry, gx, gw1, gw2, gp, expected, totals
        gc.collect()
    del graph, outputs
    gc.collect()
    torch.cuda.synchronize(device)
    dist.barrier()
    plan.close()
    destroy_training_process_group()


if __name__ == "__main__":
    main()
