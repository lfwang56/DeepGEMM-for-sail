"""Bounded native MoE training, changing GPU routes and full backward graphs."""

import argparse
from datetime import timedelta
import gc
import json
import math
import os
import unittest

import torch
import torch.distributed as dist

from deep_gemm.training import NativeMoEPlan, PeerMoEPlan, PersistentMoEPlan, CollectiveFusedMoEPlan, native_moe
from test_moe_training import reference, relative_rms
from moe_distributed_utils import destroy_training_process_group


def expected(x, ids, scores, w1, w2, dy, valid=None):
    rx, rp, rw1, rw2 = [value.detach().double().cpu().requires_grad_() for value in (x, scores, w1, w2)]
    ri = ids.detach().cpu().clone()
    if valid is not None:
        ri[valid:] = -1
    y = reference(rx, ri, rp, rw1, rw2)
    y = y + 0 * (rx.sum() + rp.sum() + rw1.sum() + rw2.sum())
    grads = torch.autograd.grad(y, (rx, rw1, rw2, rp), dy.detach().double().cpu())
    return (y, *grads)


def check(actual, target, compute):
    errors = {name: relative_rms(value, ref) for name, value, ref in zip(
        ("y", "dx", "dw1", "dw2", "dscores"), actual, target
    )}
    limit = 0.13 if compute == "fp8" else 0.04
    if not all(math.isfinite(value) and value < limit for value in errors.values()):
        raise AssertionError(errors)
    return errors


class NativeTests(unittest.TestCase):
    compute = "bf16"
    plan_class = NativeMoEPlan
    plan_options = {}

    def inputs(self, tokens=7):
        generator = torch.Generator().manual_seed(10391)
        x = (torch.randn(tokens, 256, generator=generator) * 0.1).bfloat16().cuda().requires_grad_()
        w1 = (torch.randn(3, 256, 256, generator=generator) * 0.1).bfloat16().cuda().requires_grad_()
        w2 = (torch.randn(3, 256, 128, generator=generator) * 0.1).bfloat16().cuda().requires_grad_()
        scores = torch.softmax(torch.randn(tokens, 2, generator=generator), -1).cuda().requires_grad_()
        ids = torch.tensor([[t % 3, (t + 1) % 3] for t in range(tokens)], dtype=torch.int64).reshape(tokens, 2).cuda()
        if tokens:
            with torch.no_grad(): scores[0, 0] = 0
        dy = (torch.randn(tokens, 256, generator=generator) * 0.1).bfloat16().cuda()
        return x, ids, scores, w1, w2, dy

    def plan(self, **options):
        return self.plan_class(256, 128, 3, 2, 7, chunk_tokens=3, compute=self.compute,
                               **(self.plan_options | options))

    def test_forward_backward_reference(self):
        x, ids, scores, w1, w2, dy = self.inputs()
        ids[2, 1] = -1
        plan = self.plan()
        y = native_moe(x, ids, scores, w1, w2, plan=plan)
        y.backward(dy)
        errors = check((y, x.grad, w1.grad, w2.grad, scores.grad), expected(x, ids, scores, w1, w2, dy), self.compute)
        self.assertNotEqual(float(scores.grad[0, 0]), 0)
        print(json.dumps({"case": "native_ring_reference", "compute": self.compute,
                          "workspace_bytes": plan.workspace_bytes, "relative_rms": errors}), flush=True)

    def test_pending_forwards_and_direct_accumulation(self):
        x, ids, scores, w1, w2, dy = self.inputs()
        plan = self.plan()
        accumulators = (torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
        y1 = native_moe(x, ids, scores, w1, w2, plan=plan, weight_grad_accumulators=accumulators)
        y2 = native_moe(x, ids, scores, w1, w2, plan=plan, weight_grad_accumulators=accumulators)
        torch.testing.assert_close(y1, y2, rtol=0, atol=0)
        y2.backward(dy)
        first = tuple(value.clone() for value in accumulators)
        y1.backward(dy)
        for value, initial in zip(accumulators, first):
            torch.testing.assert_close(value, 2 * initial, rtol=2e-5, atol=1e-7)
        self.assertIsNone(w1.grad)
        self.assertIsNone(w2.grad)

    def test_fp32_master_parameters(self):
        x, ids, scores, w1, w2, dy = self.inputs()
        w1, w2 = w1.detach().float().requires_grad_(), w2.detach().float().requires_grad_()
        plan = self.plan()
        y = native_moe(x, ids, scores, w1, w2, plan=plan)
        y.backward(dy)
        self.assertEqual(w1.grad.dtype, torch.float32)
        self.assertEqual(w2.grad.dtype, torch.float32)
        check((y, x.grad, w1.grad, w2.grad, scores.grad), expected(x, ids, scores, w1, w2, dy), self.compute)

    def test_accumulator_aliases_and_flat_bucket(self):
        x, ids, scores, w1, w2, dy = self.inputs()
        w1, w2 = w1.detach().float().requires_grad_(), w2.detach().float().requires_grad_()
        plan = self.plan()
        arena = torch.zeros(w1.numel() + w2.numel(), device="cuda")
        gradients = (arena[:w1.numel()].view_as(w1), arena[w1.numel():].view_as(w2))
        for invalid in ((w1.detach(), gradients[1]),
                        (gradients[0], arena[:w2.numel()].view_as(w2))):
            with self.assertRaisesRegex(ValueError, "overlap"):
                native_moe(x, ids, scores, w1, w2, plan=plan, weight_grad_accumulators=invalid)
        y = native_moe(x, ids, scores, w1, w2, plan=plan, weight_grad_accumulators=gradients)
        y.backward(dy)
        self.assertIsNone(w1.grad)
        self.assertIsNone(w2.grad)
        check((y, x.grad, *gradients, scores.grad), expected(x, ids, scores, w1, w2, dy), self.compute)

    def test_empty_and_dynamic_valid_tokens(self):
        for tokens in (0, 7):
            x, ids, scores, w1, w2, dy = self.inputs(tokens)
            valid = torch.tensor(0 if not tokens else 5, device="cuda", dtype=torch.int32)
            plan = self.plan()
            y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            y.backward(dy)
            check((y, x.grad, w1.grad, w2.grad, scores.grad),
                  expected(x, ids, scores, w1, w2, dy, int(valid)), self.compute)

    def test_workspace_is_bounded(self):
        a = self.plan()
        b = self.plan_class(256, 128, 3, 2, 700, chunk_tokens=3, compute=self.compute, **self.plan_options)
        self.assertEqual(a.workspace_bytes, b.workspace_bytes)
        self.assertEqual(a.activation_workspace_bytes, b.activation_workspace_bytes)

    def test_index_limits_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "indexing"):
            self.plan_class(65536, 65536, 2, 2, 7, chunk_tokens=3, compute=self.compute)
        with self.assertRaisesRegex(ValueError, "indexing"):
            self.plan_class(256, 128, 2, 2, 7, chunk_tokens=2**23, compute=self.compute)
        if self.plan_class is NativeMoEPlan:
            with self.assertRaisesRegex(ValueError, "1024"):
                self.plan_class(256, 128, 1025, 2, 7, chunk_tokens=3, compute=self.compute)

    def test_direct_backward_validation(self):
        x, ids, scores, w1, w2, dy = self.inputs()
        plan = self.plan()
        valid = torch.full((), x.shape[0], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "dtype"):
            plan.backward(dy, x, ids.int(), scores, w1, w2, valid)
        noncontiguous = torch.empty((x.shape[0], 2 * x.shape[1]), device="cuda", dtype=x.dtype)[:, ::2]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            plan.backward(noncontiguous, x, ids, scores, w1, w2, valid)

    def test_fp8_fused_and_separate_quantization(self):
        if self.compute != "fp8" or self.plan_class is not NativeMoEPlan:
            self.skipTest("FP8 collective recipe ablation")
        x, ids, scores, w1, w2, dy = self.inputs()
        valid = torch.full((), x.shape[0], device="cuda", dtype=torch.int32)
        results = []
        for fused in (True, False):
            plan = self.plan(fuse_activation_quant=fused)
            y = plan.forward(x, ids, scores, w1, w2, valid)
            results.append((y, *plan.backward(dy, x, ids, scores, w1, w2, valid)))
        for actual, reference in zip(results[0], results[1]):
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=1e-7)

    def test_peer_fusion_switches(self):
        if self.plan_class is not PeerMoEPlan:
            self.skipTest("direct peer transport ablation")
        for dispatch, returned in ((False, False), (False, True), (True, False), (True, True)):
            x, ids, scores, w1, w2, dy = self.inputs()
            ids[2].fill_(1)
            ids[3, 1] = -1
            valid = torch.full((), 5, device="cuda", dtype=torch.int32)
            plan = self.plan(fuse_dispatch=dispatch, fuse_return=returned)
            y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            y.backward(dy)
            check((y, x.grad, w1.grad, w2.grad, scores.grad),
                  expected(x, ids, scores, w1, w2, dy, 5), self.compute)

    def test_many_experts_device_schedule(self):
        if self.plan_class is not NativeMoEPlan:
            self.skipTest("checks the collective GEMM scheduler's >=128-expert path")
        x, ids, scores, _, _, dy = self.inputs()
        generator = torch.Generator().manual_seed(40501)
        experts = 129
        w1 = (torch.randn(experts, 256, 256, generator=generator) * 0.1).bfloat16().cuda().requires_grad_()
        w2 = (torch.randn(experts, 256, 128, generator=generator) * 0.1).bfloat16().cuda().requires_grad_()
        ids.copy_(torch.tensor([[0, 128], [64, -1], [128, 0], [0, 64],
                               [128, 128], [-1, -1], [64, 0]], device="cuda"))
        plan = self.plan_class(256, 128, experts, 2, 7, chunk_tokens=3, compute=self.compute)
        y = native_moe(x, ids, scores, w1, w2, plan=plan)
        y.backward(dy)
        check((y, x.grad, w1.grad, w2.grad, scores.grad), expected(x, ids, scores, w1, w2, dy), self.compute)

    def test_peer_dispatch_cache_groups(self):
        if self.plan_class is not PeerMoEPlan:
            self.skipTest("peer in-GEMM dispatch cache")
        # Six FC1 N tiles exercise reuse and the partial final N group.
        for group_n in (2, 4, 8, 16):
            torch.manual_seed(1849)
            x = (torch.randn(7, 512, device="cuda") * .1).bfloat16().requires_grad_()
            w1 = (torch.randn(3, 768, 512, device="cuda") * .03).bfloat16().requires_grad_()
            w2 = (torch.randn(3, 512, 384, device="cuda") * .03).bfloat16().requires_grad_()
            ids = torch.tensor([[0, 1], [2, 0], [0, 0], [-1, -1], [1, 2], [2, 1], [0, 2]], device="cuda")
            scores = torch.rand(7, 2, device="cuda").requires_grad_()
            dy = (torch.randn_like(x) * .1).bfloat16()
            valid = torch.full((), 5, device="cuda", dtype=torch.int32)
            plan = PeerMoEPlan(512, 384, 3, 2, 7, chunk_tokens=3,
                               dispatch_cache_n=group_n, fuse_dispatch=True, fuse_return=True,
                               reuse_packed_dy=group_n == 4)
            y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            y.backward(dy)
            check((y, x.grad, w1.grad, w2.grad, scores.grad),
                  expected(x, ids, scores, w1, w2, dy, 5), "bf16")

    def test_peer_pointwise_switches(self):
        if self.plan_class is not PeerMoEPlan:
            self.skipTest("peer active-row and local-gradient controls")
        for active, fused, vectorized in ((a, f, v) for a in (False, True)
                                         for f in (False, True) for v in (False, True)):
            x, ids, scores, w1, w2, dy = self.inputs()
            ids.fill_(1)
            ids[3].fill_(-1)
            valid = torch.full((), 5, device="cuda", dtype=torch.int32)
            plan = self.plan(active_rows=active, fuse_local_grad=fused, vectorized_copy=vectorized)
            y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            y.backward(dy)
            check((y, x.grad, w1.grad, w2.grad, scores.grad),
                  expected(x, ids, scores, w1, w2, dy, 5), self.compute)

    def test_collective_active_pointwise(self):
        if self.plan_class is not NativeMoEPlan or self.compute != "bf16":
            self.skipTest("BF16 collective active-row control")
        for active, vectorized in ((a, v) for a in (False, True) for v in (False, True)):
            x, ids, scores, w1, w2, dy = self.inputs()
            ids.fill_(2)
            ids[3].fill_(-1)
            valid = torch.full((), 5, device="cuda", dtype=torch.int32)
            plan = self.plan(active_pointwise=active, vectorized_routes=vectorized)
            y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            y.backward(dy)
            check((y, x.grad, w1.grad, w2.grad, scores.grad),
                  expected(x, ids, scores, w1, w2, dy, 5), self.compute)

    def test_graph_replay_updated_parameters(self):
        x, ids, scores, w1, w2, dy = self.inputs()
        w1, w2 = w1.detach().float().requires_grad_(), w2.detach().float().requires_grad_()
        plan = self.plan()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                for value in (x, scores, w1, w2):
                    value.grad = None
                native_moe(x, ids, scores, w1, w2, plan=plan).backward(dy)
            for value in (x, scores, w1, w2):
                value.grad = None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                y = native_moe(x, ids, scores, w1, w2, plan=plan)
                y.backward(dy)
            for updated in (False, True):
                if updated:
                    with torch.no_grad():
                        x.add_(0.02)
                        scores.mul_(0.8)
                        w1.mul_(0.75)
                        w2.mul_(-0.5)
                graph.replay()
                plan.check()
                check((y, x.grad, w1.grad, w2.grad, scores.grad),
                      expected(x, ids, scores, w1, w2, dy), self.compute)
        del graph

    def test_sticky_graph_errors(self):
        x, ids, scores, w1, w2, _ = self.inputs()
        plan = self.plan(check_errors=False)
        valid = torch.full((), 7, device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            ids[0, 0] = 3  # Invalid in the first chunk; later chunks are valid.
            graph.replay()
            with self.assertRaisesRegex(RuntimeError, "expert ID"):
                plan.check()
            ids[0, 0] = 0
            plan.reset_errors()
            graph.replay()
            plan.check()
            valid.fill_(8)
            graph.replay()
            with self.assertRaisesRegex(RuntimeError, "valid_tokens"):
                plan.check()
            valid.fill_(7)
            plan.reset_errors()
            graph.replay()
            plan.check()
            self.assertTrue(bool(torch.isfinite(y).all()))
        torch.cuda.current_stream().wait_stream(stream)

    def test_graph_replay_dynamic_routes_and_backward(self):
        x, ids, scores, w1, w2, dy = self.inputs()
        plan = self.plan()
        valid = torch.tensor(7, device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                for value in (x, scores, w1, w2): value.grad = None
                native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid).backward(dy)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for value in (x, scores, w1, w2): value.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            y.backward(dy)
        for mode in ("balanced", "skewed", "masked", "ragged"):
            if mode == "skewed": ids.fill_(0)
            elif mode == "masked": ids.fill_(-1)
            else: ids.copy_(torch.arange(14, device="cuda").reshape(7, 2) % 3)
            valid.fill_(3 if mode == "ragged" else 7)
            graph.replay()
            plan.check()
            errors = check((y, x.grad, w1.grad, w2.grad, scores.grad),
                           expected(x, ids, scores, w1, w2, dy, int(valid)), self.compute)
            print(json.dumps({"case": "native_training_graph", "compute": self.compute,
                              "routing": mode, "relative_rms": errors}), flush=True)


def ep_checks(compute, graph_mode=False, peer=False, stress=False, persistent=False, maximize=False,
              gemm_config=None, fuse_dispatch=True, fuse_return=True, fuse_activation_backward=False,
              collective_fused=False, fast_weight_transpose=False, dispatch_cache_n=0, reuse_packed_dy=False):
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=300))
    generator = torch.Generator().manual_seed(62831)
    global_w1 = (torch.randn(size * 2, 256, 256, generator=generator) * 0.1).bfloat16()
    global_w2 = (torch.randn(size * 2, 256, 128, generator=generator) * 0.1).bfloat16()
    xs, ids_all, ps, dys = [], [], [], []
    for source_rank in range(size):
        tokens = 0 if source_rank == 0 else 5 + source_rank
        xs.append((torch.randn(tokens, 256, generator=generator) * 0.1).bfloat16())
        ids_all.append(torch.tensor([[(t + source_rank) % (size * 2), (t + source_rank + 1) % (size * 2)]
                                     for t in range(tokens)], dtype=torch.int64).reshape(tokens, 2))
        ps.append(torch.randn(tokens, 2, generator=generator).softmax(-1))
        dys.append((torch.randn(tokens, 256, generator=generator) * 0.1).bfloat16())
    x = xs[rank].to(device).requires_grad_()
    ids = ids_all[rank].to(device)
    scores = ps[rank].to(device).requires_grad_()
    w1 = global_w1[rank * 2:(rank + 1) * 2].to(device).requires_grad_()
    w2 = global_w2[rank * 2:(rank + 1) * 2].to(device).requires_grad_()
    dy = dys[rank].to(device)
    plan_class = (PersistentMoEPlan if persistent else PeerMoEPlan if peer
                  else CollectiveFusedMoEPlan if collective_fused else NativeMoEPlan)
    plan_options = {"maximize_ctas": True} if persistent and maximize else {}
    plan_options["fast_weight_transpose"] = fast_weight_transpose
    if dispatch_cache_n:
        plan_options["dispatch_cache_n"] = dispatch_cache_n
    if reuse_packed_dy:
        plan_options["reuse_packed_dy"] = True
    if gemm_config:
        plan_options["gemm_configs"] = {role: gemm_config for role in ("fc1", "fc2", "dhidden", "dx")}
    if peer and not persistent:
        plan_options["fuse_activation_backward"] = fuse_activation_backward
    if peer and not persistent:
        plan_options.update(fuse_dispatch=fuse_dispatch, fuse_return=fuse_return)
    if collective_fused:
        plan_options.update(fuse_return=fuse_return, fuse_activation_backward=fuse_activation_backward)
    plan = plan_class(256, 128, 2, 2, max(value.shape[0] for value in xs),
                        chunk_tokens=3, group=dist.group.WORLD, compute=compute, device=device, **plan_options)
    valid = torch.tensor(x.shape[0], device=device, dtype=torch.int32)
    graph, stream = None, torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2 if graph_mode else 1):
            for value in (x, scores, w1, w2): value.grad = None
            y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            y.backward(dy)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if graph_mode:
        for value in (x, scores, w1, w2): value.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
            y.backward(dy)
        graph.replay()
        plan.check()
    reference_values = expected(torch.cat(xs), torch.cat(ids_all), torch.cat(ps), global_w1, global_w2, torch.cat(dys))
    begin = sum(value.shape[0] for value in xs[:rank])
    end = begin + x.shape[0]
    target = (reference_values[0][begin:end], reference_values[1][begin:end],
              reference_values[2][rank * 2:(rank + 1) * 2], reference_values[3][rank * 2:(rank + 1) * 2],
              reference_values[4][begin:end])
    errors = check((y, x.grad, w1.grad, w2.grad, scores.grad), target, compute)
    print(json.dumps({"case": "native_ep", "rank": rank, "ep_size": size, "graph": graph_mode,
                      "transport": plan.transport,
                      "compute": compute, "relative_rms": errors}), flush=True)
    if stress:
        if graph is None:
            raise ValueError("EP stress requires --graph")
        for iteration in range(128):
            mode = ("balanced", "skewed", "masked", "ragged")[iteration % 4]
            routed_ids = []
            for source_rank, original in enumerate(ids_all):
                current = original.clone()
                if mode == "skewed": current.fill_(0)
                if mode == "masked": current.fill_(-1)
                nvalid = current.shape[0] // 2 if mode == "ragged" else current.shape[0]
                current[nvalid:] = -1
                routed_ids.append(current)
            ids.copy_(routed_ids[rank])
            valid.fill_(x.shape[0] // 2 if mode == "ragged" else x.shape[0])
            plan.reset_errors()
            graph.replay()
            if iteration < 4 or iteration >= 124:
                plan.check()
                ref = expected(torch.cat(xs), torch.cat(routed_ids), torch.cat(ps), global_w1, global_w2, torch.cat(dys))
                target = (ref[0][begin:end], ref[1][begin:end], ref[2][rank * 2:(rank + 1) * 2],
                          ref[3][rank * 2:(rank + 1) * 2], ref[4][begin:end])
                check((y, x.grad, w1.grad, w2.grad, scores.grad), target, compute)
        plan.check()
        print(json.dumps({"case": "native_ep_graph_stress", "rank": rank, "ep_size": size,
                          "compute": compute, "transport": plan.transport,
                          "replays": 128, "routing_modes": 4, "passed": True}), flush=True)
    # PCCL retains communicator resources for live executable graphs. Release
    # those graph references before synchronous process-group destruction.
    del graph
    for value in (x, scores, w1, w2):
        value.grad = None
    del y, x, ids, scores, w1, w2, dy, valid, plan, stream
    gc.collect()
    destroy_training_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--ep", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--peer", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--maximize-ctas", action="store_true")
    parser.add_argument("--gemm-config", type=int, nargs=5, metavar=("BN", "BK", "WM", "WN", "STAGES"))
    parser.add_argument("--no-fuse-dispatch", action="store_true")
    parser.add_argument("--no-fuse-return", action="store_true")
    parser.add_argument("--fuse-activation-backward", action="store_true")
    parser.add_argument("--collective-fused", action="store_true")
    parser.add_argument("--fast-weight-transpose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dispatch-cache-n", type=int, choices=(0, 2, 4, 8, 16), default=0)
    parser.add_argument("--reuse-packed-dy", action="store_true")
    options, remaining = parser.parse_known_args()
    if options.dispatch_cache_n and (not options.ep or not options.peer or options.persistent or options.no_fuse_dispatch):
        parser.error("--dispatch-cache-n requires --peer --ep with fused dispatch; unit tests cover cache groups automatically")
    if options.reuse_packed_dy and (not options.ep or not options.peer or options.persistent):
        parser.error("--reuse-packed-dy requires --peer --ep; unit tests cover reuse automatically")
    if options.collective_fused and (options.peer or options.persistent or options.fp8):
        parser.error("collective-fused is a separate BF16 backend")
    if options.gemm_config and (not (options.peer or options.collective_fused) or options.persistent or options.fp8):
        parser.error("explicit GEMM tiles require a BF16 fused backend")
    if (options.no_fuse_dispatch or options.no_fuse_return) and (not (options.peer or options.collective_fused) or options.persistent):
        parser.error("peer fusion switches require --peer")
    torch.set_num_threads(2)
    recipe = "fp8" if options.fp8 else "bf16"
    if options.ep:
        ep_checks(recipe, options.graph, options.peer, options.stress, options.persistent, options.maximize_ctas,
                  options.gemm_config, not options.no_fuse_dispatch, not options.no_fuse_return,
                  options.fuse_activation_backward, options.collective_fused, options.fast_weight_transpose,
                  options.dispatch_cache_n, options.reuse_packed_dy)
    else:
        NativeTests.compute = recipe
        NativeTests.plan_class = (PersistentMoEPlan if options.persistent else PeerMoEPlan if options.peer
                                  else CollectiveFusedMoEPlan if options.collective_fused else NativeMoEPlan)
        NativeTests.plan_options = {"maximize_ctas": True} if options.persistent and options.maximize_ctas else {}
        NativeTests.plan_options["fast_weight_transpose"] = options.fast_weight_transpose
        if options.gemm_config:
            NativeTests.plan_options["gemm_configs"] = {role: options.gemm_config for role in ("fc1", "fc2", "dhidden", "dx")}
        if options.peer and not options.persistent:
            NativeTests.plan_options.update(fuse_dispatch=not options.no_fuse_dispatch,
                                            fuse_return=not options.no_fuse_return,
                                            fuse_activation_backward=options.fuse_activation_backward)
        if options.collective_fused:
            NativeTests.plan_options.update(fuse_return=not options.no_fuse_return,
                                            fuse_activation_backward=options.fuse_activation_backward)
        unittest.main(argv=[__file__] + remaining, verbosity=2)
