"""Native DDP/FSDP2 integration; launch with torchrun on two or more PPUs."""

import argparse
import json
import math
import os

import torch
import torch.distributed as dist
from torch import nn

from deep_gemm.training import NativeMoEPlan, PeerMoEPlan, PersistentMoEPlan, bf16_moe, native_moe
from test_moe_training import reference, relative_rms


class Experts(nn.Module):
    def __init__(self, w1, w2, ep_group=None, transport="collective", backend="deep_gemm", grouped=False, plan=None):
        super().__init__()
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.ep_group = ep_group
        self.transport = transport
        self.backend = backend
        self.grouped = grouped
        self.plan = plan

    def forward(self, x, ids, scores):
        if self.plan is not None:
            return native_moe(x, ids, scores, self.w1, self.w2, plan=self.plan)
        return bf16_moe(x, ids, scores, self.w1, self.w2, group=self.ep_group,
                        backend=self.backend, transport=self.transport, recompute=True, grouped_gemm=self.grouped)


def full(tensor):
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ddp", "fsdp2"), required=True)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--transport", choices=("collective", "symmetric", "peer", "persistent"), default="collective")
    parser.add_argument("--native-ring", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--grouped", action="store_true")
    args = parser.parse_args()
    if args.transport in ("peer", "persistent") and not args.native_ring:
        raise ValueError("peer transport requires --native-ring")
    if args.native_ring and args.transport == "symmetric":
        raise ValueError("use peer for the native fused symmetric-memory path")
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", device_id=device)
    if size % args.ep_size or not 1 <= args.ep_size <= size:
        raise ValueError("ep-size must divide world-size")
    from torch.distributed.device_mesh import init_device_mesh

    dp_size = size // args.ep_size
    mesh = init_device_mesh("cuda", (dp_size, args.ep_size), mesh_dim_names=("dp", "ep"))
    ep_rank = rank % args.ep_size
    ep_group = mesh.get_group("ep") if args.ep_size > 1 else None
    dp_group = mesh.get_group("dp")
    generator = torch.Generator().manual_seed(8823)
    total_experts = 2 * args.ep_size
    w1_cpu = (torch.randn(total_experts, 256, 256, generator=generator) * 0.1).bfloat16()
    w2_cpu = (torch.randn(total_experts, 256, 128, generator=generator) * 0.1).bfloat16()
    xs, ids_list, ps, dys = [], [], [], []
    for peer in range(size):
        xs.append((torch.randn(5, 256, generator=generator) * 0.1).bfloat16())
        ids_list.append(torch.tensor([[(t + peer) % total_experts, (t + peer + 1) % total_experts]
                                      for t in range(5)]))
        ids_list[-1][2, 1] = -1
        ps.append(torch.softmax(torch.randn(5, 2, generator=generator), -1))
        dys.append((torch.randn(5, 256, generator=generator) * 0.1).bfloat16())
    rw1 = w1_cpu.double().requires_grad_()
    rw2 = w2_cpu.double().requires_grad_()
    expected = reference(torch.cat(xs).double(), torch.cat(ids_list), torch.cat(ps).double(), rw1, rw2)
    expected_grads = torch.autograd.grad(expected, (rw1, rw2), torch.cat(dys).double() / dp_size)
    expected_grads = tuple(t[2 * ep_rank:2 * (ep_rank + 1)] for t in expected_grads)
    parameter_dtype = torch.float32 if args.mode == "fsdp2" or args.native_ring else torch.bfloat16
    plan_class = PersistentMoEPlan if args.transport == "persistent" else PeerMoEPlan if args.transport == "peer" else NativeMoEPlan
    plan = plan_class(256, 128, 2, 2, 5, chunk_tokens=3, group=ep_group, device=device,
                      compute="fp8" if args.fp8 else "bf16") if args.native_ring else None
    local_slice = slice(2 * ep_rank, 2 * (ep_rank + 1))
    model = Experts(w1_cpu[local_slice].to(device=device, dtype=parameter_dtype),
                    w2_cpu[local_slice].to(device=device, dtype=parameter_dtype), ep_group, args.transport,
                    "deep_gemm_fp8" if args.fp8 else "deep_gemm", args.grouped, plan)
    if args.mode == "ddp":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], process_group=dp_group)
    else:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        fully_shard(model, mesh=mesh["dp"], mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False,
        ))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial = tuple(full(p).detach().clone() for p in model.parameters())
    for step in range(2):
        optimizer.zero_grad()
        y = model(xs[rank].to(device), ids_list[rank].to(device), ps[rank].to(device))
        y.backward(dys[rank].to(device))
        if step == 0:
            errors = {name: relative_rms(full(parameter.grad), target) for (name, parameter), target in zip(
                model.named_parameters(), expected_grads
            )}
            if not all(math.isfinite(error) and error < (0.12 if args.fp8 else 0.04)
                       for error in errors.values()):
                raise AssertionError((args.mode, rank, errors))
            print(json.dumps({"mode": args.mode, "rank": rank, "ep_size": args.ep_size,
                              "transport": args.transport, "fp8": args.fp8, "grouped_gemm": args.grouped,
                              "native_ring": args.native_ring,
                              "gradient_relative_rms": errors}), flush=True)
        optimizer.step()
        for parameter in model.parameters():
            if not bool(torch.isfinite(full(parameter)).all()):
                raise AssertionError("optimizer produced non-finite parameters")
    for parameter, before in zip(model.parameters(), initial):
        if torch.equal(full(parameter), before):
            raise AssertionError("optimizer did not update an expert parameter")
    dist.barrier()
    print(json.dumps({"mode": args.mode, "rank": rank, "ep_size": args.ep_size,
                      "transport": args.transport, "fp8": args.fp8, "grouped_gemm": args.grouped,
                      "native_ring": args.native_ring,
                      "optimizer_steps": 2, "passed": True}), flush=True)
    dist.destroy_process_group()
