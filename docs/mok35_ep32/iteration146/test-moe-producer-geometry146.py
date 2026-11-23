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

from deep_gemm.training import DeepEPMoEPlan, PeerMoEPlan
from moe_distributed_utils import destroy_training_process_group
from test_moe_training import reference
from deep_gemm.training.layout import to_g8_weight, from_g8_weight


def relative_rms(actual, expected):
    a, b = actual.detach().float(), expected.detach().float().to(actual.device)
    return float((a - b).square().sum().sqrt() / b.square().sum().sqrt().clamp_min(1e-20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("deepep", "peer"), default="deepep")
    parser.add_argument("--dag-fusion", action="store_true")
    parser.add_argument("--dag-backward", action="store_true")
    parser.add_argument("--dag-direct-return", action="store_true")
    parser.add_argument("--fuse-fc1-activation", action="store_true")
    parser.add_argument("--input-copy-ilp", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--dispatch-producer-tiles", type=int, choices=(1,2,4), default=1)
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
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--local-experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=19)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--gemm-config", type=int, nargs=5, default=(256,64,64,64,2))
    parser.add_argument("--comm-sms", type=int, default=38)
    parser.add_argument("--ring-rows", type=int)
    parser.add_argument("--block-m", type=int, choices=(128, 256), default=128)
    parser.add_argument("--gate-up-layout", choices=("split", "g8"), default="split")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata_backend = "nccl" if args.backend == "peer" else "gloo"
    dist.init_process_group(metadata_backend, timeout=timedelta(seconds=600),
                            device_id=device if metadata_backend == "nccl" else None)
    reference_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=600)) if args.backend == "peer" else dist.group.WORLD
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
    if args.gate_up_layout == "g8":
        w1 = to_g8_weight(w1)
    configs = {role: tuple(args.gemm_config) for role in ("fc1", "fc2", "dhidden", "dx")}
    plan_options = (dict(comm_sms=args.comm_sms) if args.backend == "deepep" else
                    dict(dag_fusion=args.dag_fusion, dag_backward=args.dag_backward,
                         dag_direct_return=args.dag_direct_return,
                         dag_shared_pool=args.dag_fusion or args.dag_backward,
                         expert_local_grad=True, reuse_packed_dy=True, fuse_activation_backward=True,
                         fuse_fc1_activation=args.fuse_fc1_activation,
                         input_copy_ilp=args.input_copy_ilp,
                         dispatch_producer=not (args.dag_fusion or args.input_inline or args.dispatch_assist_ctas),
                         inline_dispatch=args.input_inline, dispatch_assist_ctas=args.dispatch_assist_ctas,
                         dispatch_external=args.dispatch_external, dispatch_external_serial=args.serialize_external_dispatch,
                         dgrad_compute_cus=args.dgrad_compute_cus,
                         fuse_replay_gradients=args.fuse_replay_gradients,
                         fuse_dhidden_scores=args.fuse_dhidden_scores,
                         transpose_free_dgrad=args.transpose_free_dgrad,
                         dispatch_producer_tiles=args.dispatch_producer_tiles,
                         producer_input_aiu=not (args.dag_fusion or args.input_inline or args.dispatch_assist_ctas),
                         producer_local_aiu=not (args.dag_fusion or args.input_inline or args.dispatch_assist_ctas)))
    plan = (DeepEPMoEPlan if args.backend == "deepep" else PeerMoEPlan)(h, i, e, k, t, chunk_tokens=args.chunk_tokens,
                        group=dist.group.WORLD, device=device, **plan_options,
                        ring_rows=args.ring_rows,
                        block_m=args.block_m,
                        gate_up_layout=args.gate_up_layout,
                        wgrad_overlap=args.wgrad_overlap,
                        overlap_expert_dy=args.overlap_expert_dy,
                        gemm_configs=configs, check_errors=False)
    print(json.dumps(dict(case="large_deepep_stage", rank=rank, stage="plan_ready",
                          backend=args.backend, rdma_ranks=getattr(plan, "deep_ep_rdma_ranks", None))), flush=True)

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
        rw1, rw2 = (from_g8_weight(w1) if args.gate_up_layout == "g8" else w1).float().requires_grad_(), w2.float().requires_grad_()
        ry = reference(rx, local_ids.to(device), rp, rw1, rw2)
        # Empty experts/ranks still have explicitly defined zero gradients.
        ry = ry + 0 * (rx.sum() + rp.sum() + rw1.sum() + rw2.sum())
        gx, gw1, gw2, gp = torch.autograd.grad(ry, (rx, rw1, rw2, rp), all_dy.float().to(device))
        totals = []
        for partial in (ry, gx, gp):
            total = partial.detach().cpu().contiguous()
            dist.all_reduce(total, group=reference_group)
            totals.append(total[local_slice].to(device))
        expected = (totals[0], totals[1], gw1, gw2, totals[2])
        compared = (outputs[0], outputs[1], from_g8_weight(outputs[2]), outputs[3], outputs[4]) if args.gate_up_layout == "g8" else outputs
        errors = {name: relative_rms(a, b) for name, a, b in zip(
            ("y", "dx", "dw1", "dw2", "dscores"), compared, expected)}
        passed = all(math.isfinite(error) and error < .04 for error in errors.values())
        print(json.dumps(dict(case="large_distributed_deepep_reference" if args.backend == "deepep" else "large_distributed_peer_reference", rank=rank,
            ranks=size, hidden=h, intermediate=i, local_experts=e, total_experts=e * size,
            topk=k, tokens=t, chunk_tokens=args.chunk_tokens, pattern=pattern,
            graph=args.graph, relative_rms=errors, every_element_compared=True,
            ring_rows=args.ring_rows, packed_ring_rows=plan.rows,
            block_m=plan.block_m,
            route_window_count=plan.route_window_count,
            backend=args.backend, gemm_config=args.gemm_config, gate_up_layout=args.gate_up_layout,
            dag_fusion=args.dag_fusion, dag_backward=args.dag_backward, dag_direct_return=args.dag_direct_return,
            input_copy_ilp=args.input_copy_ilp if args.backend == "peer" else None,
            inline_dispatch=getattr(plan, "inline_dispatch", None),
            dispatch_assist_ctas=getattr(plan, "dispatch_assist_ctas", None),
            dispatch_external=getattr(plan, "dispatch_external", None),
            dispatch_external_serial=getattr(plan, "dispatch_external_serial", None),
            wgrad_overlap=plan.wgrad_overlap,
            overlap_expert_dy=plan.overlap_expert_dy,
            payload_transport=getattr(plan, "deep_ep_backend", "verified direct peer domain"),
            rdma_ranks=getattr(plan, "deep_ep_rdma_ranks", None),
            rdma_qps=getattr(plan, "deep_ep_qps", None), communication_cus=getattr(plan, "comm_sms", None),
            metadata_backend=metadata_backend, reference="independent FP32 Torch local experts + CPU sum",
            passed=passed)), flush=True)
        passed_tensor = torch.tensor(int(passed))
        dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN, group=reference_group)
        if not passed_tensor.item():
            raise AssertionError("at least one rank failed the independent reference")
        del rx, rp, rw1, rw2, ry, gx, gw1, gw2, gp, expected, totals
        gc.collect()
    del graph, outputs
    gc.collect()
    torch.cuda.synchronize(device)
    dist.barrier()
    if hasattr(plan, "close"):
        plan.close()
    del plan
    gc.collect()
    destroy_training_process_group()


if __name__ == "__main__":
    main()
