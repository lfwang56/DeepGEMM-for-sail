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

from deep_gemm.training import DeepEPMoEPlan, PeerMoEPlan
from moe_distributed_utils import destroy_training_process_group
from test_moe_training_large import relative_rms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--local-experts", type=int, default=16)
    parser.add_argument("--chunk-tokens", type=int, default=768)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--backends", nargs="+", choices=("deepep", "peer"), default=["deepep", "peer"])
    parser.add_argument("--dispatch-pipeline", action="store_true")
    parser.add_argument("--return-pipeline", action="store_true")
    parser.add_argument("--pipeline-comm-ctas", type=int, default=8)
    parser.add_argument("--pipeline-serialize", action="store_true")
    parser.add_argument("--process-group-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--producer-input-aiu", action="store_true")
    parser.add_argument("--producer-local-aiu", action="store_true")
    parser.add_argument("--fuse-activation-backward", action="store_true")
    args = parser.parse_args()
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
    configs = {role: (256, 64, 64, 64, 2) for role in ("fc1", "fc2", "dhidden", "dx")}
    for backend in args.backends:
        options = dict(gemm_configs=configs)
        if backend == "peer":
            options.update(dispatch_producer=not args.dispatch_pipeline, dispatch_producer_tiles=1,
                           reuse_packed_dy=True,
                           producer_input_aiu=args.producer_input_aiu and not args.dispatch_pipeline,
                           producer_local_aiu=args.producer_local_aiu and not args.dispatch_pipeline,
                           fuse_activation_backward=args.fuse_activation_backward,
                           dispatch_pipeline=args.dispatch_pipeline, return_pipeline=args.return_pipeline,
                           pipeline_comm_ctas=args.pipeline_comm_ctas, pipeline_serialize=args.pipeline_serialize)
        else:
            options["comm_sms"] = 38
            if args.process_group_backend == "nccl":
                os.environ["DEEPEP_NUM_MAX_ICN_PEERS"] = str(size)
        cls = PeerMoEPlan if backend == "peer" else DeepEPMoEPlan
        plan = cls(h, i, e, k, c, chunk_tokens=c, device=device, group=dist.group.WORLD,
                   check_errors=False, **options)
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
            errors.update(dw1=relative_rms(dw1[0], weight_targets[0]),
                          dw2=relative_rms(dw2[0], weight_targets[1]))
        else:
            errors.update(dw1=0., dw2=0.)
        active_rows = int(plan.current_rows)
        zero_score_derivative_preserved = rank != 0 or float(ds[0, 0]) != 0
        expected_rows = ((size * c * k + plan.alignment - 1) // plan.alignment * plan.alignment
                         if rank == 0 else 0)
        passed = (unused_zero and zero_score_derivative_preserved and active_rows == expected_rows
                  and all(math.isfinite(error) and error < .04 for error in errors.values()))
        print(json.dumps(dict(case="skew_capacity_reference", rank=rank, ranks=size,
            backend=backend, hidden=h, intermediate=i, local_experts=e, total_experts=e * size,
            topk=k, tokens_per_rank=c, chunk_tokens=c, graph=args.graph,
            dispatch_pipeline=args.dispatch_pipeline if backend == "peer" else None,
            return_pipeline=args.return_pipeline if backend == "peer" else None,
            pipeline_comm_ctas=args.pipeline_comm_ctas if backend == "peer" else None,
            pipeline_serialized=args.pipeline_serialize if backend == "peer" else None,
            capacity_rows=plan.rows, alignment=plan.alignment, active_rows=active_rows,
            largest_active_element_offset=max(0, active_rows * h - 1),
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
    dist.barrier(group=reference_group)
    if reference_group is not dist.group.WORLD:
        dist.destroy_process_group(reference_group)
    destroy_training_process_group()


if __name__ == "__main__":
    main()
