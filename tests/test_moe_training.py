"""CPU reference and native PPU checks for the experimental BF16 training API.

python tests/test_moe_training.py                 # CPU double reference
python tests/test_moe_training.py --native        # one PPU
torchrun --standalone --nproc-per-node=2 tests/test_moe_training.py --ep
"""

import argparse
import gc
import json
import math
import os
import unittest
import weakref

import torch
import torch.distributed as dist

from deep_gemm.training import bf16_moe, bf16_moe_backward, bf16_moe_forward


def reference(x, ids, scores, w1, w2):
    output = torch.zeros_like(x)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(ids == expert)
        gu = x[rows] @ w1[expert].t()
        gate, up = gu.chunk(2, dim=-1)
        values = (torch.nn.functional.silu(gate) * up) @ w2[expert].t()
        output.index_add_(0, rows, values * scores[rows, slots, None])
    return output


def relative_rms(actual, expected):
    actual, expected = actual.detach().double().cpu(), expected.detach().double().cpu()
    return float((actual - expected).square().sum().sqrt() / expected.square().sum().sqrt().clamp_min(1e-12))


class TrainingTests(unittest.TestCase):
    native = False
    fp8 = False
    grouped = False

    def setUp(self):
        self.device = "cuda" if self.native else "cpu"
        self.dtype = torch.bfloat16 if self.native else torch.float64
        self.backend = "deep_gemm_fp8" if self.fp8 else "deep_gemm" if self.native else "torch"
        self.options = dict(backend=self.backend, grouped_gemm=self.grouped)
        self.h, self.i = (256, 128) if self.native else (6, 4)

    def inputs(self, tokens=7, experts=3):
        generator = torch.Generator().manual_seed(20261120)
        shapes = ((tokens, self.h), (experts, 2 * self.i, self.h), (experts, self.h, self.i))
        values = [torch.randn(shape, generator=generator) * 0.1 for shape in shapes]
        scores = torch.softmax(torch.randn(tokens, 2, generator=generator), -1)
        if tokens:
            scores[0, 0] = 0  # Valid zero scores must retain a score derivative.
        ids = torch.tensor([[t % experts, (t + 1) % experts] for t in range(tokens)], dtype=torch.int64).reshape(tokens, 2)
        if tokens > 2:
            ids[2, 1] = -1
        x, w1, w2 = [v.to(device=self.device, dtype=self.dtype).requires_grad_() for v in values]
        scores = scores.to(device=self.device, dtype=torch.float32 if self.native else self.dtype).requires_grad_()
        return x, ids.to(self.device), scores, w1, w2

    def test_gradients_against_independent_reference(self):
        x, ids, scores, w1, w2 = self.inputs()
        y, context = bf16_moe_forward(x, ids, scores, w1, w2, **self.options)
        dy = torch.linspace(-0.3, 0.4, y.numel(), device=self.device).reshape(y.shape).to(self.dtype)
        actual = bf16_moe_backward(dy, context)
        rx, rp, rw1, rw2 = [t.detach().double().cpu().requires_grad_() for t in (x, scores, w1, w2)]
        expected_y = reference(rx, ids.cpu(), rp, rw1, rw2)
        expected_grads = torch.autograd.grad(expected_y, (rx, rw1, rw2, rp), dy.double().cpu())
        errors = {"y": relative_rms(y, expected_y)}
        errors.update({name: relative_rms(value, expected) for name, value, expected in zip(
            ("dx", "dw1", "dw2", "dscores"), actual, expected_grads
        )})
        limit = 0.12 if self.fp8 else 0.035 if self.native else 1e-12
        for name, error in errors.items():
            self.assertLess(error, limit, (name, errors))
        self.assertNotEqual(float(actual[3][0, 0]), 0.0)
        print(json.dumps({"case": "gradient_reference", "backend": self.backend,
                          "grouped_gemm": self.grouped, "relative_rms": errors}), flush=True)

    def test_gradcheck(self):
        if self.native:
            self.skipTest("finite differences use the CPU double implementation")
        x, ids, scores, w1, w2 = self.inputs(tokens=3, experts=2)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, p, b, c: bf16_moe(a, ids, p, b, c, backend="torch"),
            (x, scores, w1, w2), fast_mode=True,
        ))

    def test_fp8_scaling_contract(self):
        if not self.fp8:
            self.skipTest("experimental FP8 recipe")
        from deep_gemm.training.bf16 import _gemm_nt

        generator = torch.Generator().manual_seed(717)
        a = (torch.randn(5, 256, generator=generator) * 0.1).bfloat16()
        b = (torch.randn(256, 256, generator=generator) * 0.1).bfloat16()
        a[0].zero_()

        # Define the public scaling recipe independently on CPU. Distinct
        # magnitudes in adjacent blocks catch scale-axis/layout mistakes.
        b[:128, :128] *= 4
        b[128:, 128:] *= 0.125
        qa, qb = torch.empty_like(a, dtype=torch.float32), torch.empty_like(b, dtype=torch.float32)
        for row in range(a.shape[0]):
            for begin in range(0, a.shape[1], 128):
                tile = a[row, begin:begin + 128].float()
                scale = tile.abs().max().clamp_min(1e-4) / 448
                qa[row, begin:begin + 128] = (tile / scale).to(torch.float8_e4m3fn).float() * scale
        for row in range(0, b.shape[0], 128):
            for begin in range(0, b.shape[1], 128):
                tile = b[row:row + 128, begin:begin + 128].float()
                scale = tile.abs().max().clamp_min(1e-4) / 448
                qb[row:row + 128, begin:begin + 128] = (tile / scale).to(torch.float8_e4m3fn).float() * scale
        expected = (qa @ qb.t()).bfloat16()
        actual = _gemm_nt(a.cuda(), b.cuda(), self.backend)
        error = relative_rms(actual, expected)
        self.assertLess(error, 0.008)
        self.assertEqual(int(torch.count_nonzero(actual[0])), 0)
        print(json.dumps({"case": "fp8_scaling_contract", "relative_rms": error}), flush=True)

    def test_recompute_and_multiple_pending_contexts(self):
        x, ids, scores, w1, w2 = self.inputs()
        y, first = bf16_moe_forward(x, ids, scores, w1, w2, **self.options)
        ry, second = bf16_moe_forward(x, ids, scores, w1, w2, **self.options, recompute=True)
        self.assertIsNone(second.experts.hidden)
        self.assertIsNone(second.experts.x)
        dy = torch.ones_like(y)
        rg = bf16_moe_backward(dy, second)
        first_g = bf16_moe_backward(dy, first)
        self.assertTrue(torch.equal(y, ry))
        for left, right in zip(first_g, rg):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_autograd_releases_activations_and_retained_graph(self):
        x, ids, scores, w1, w2 = self.inputs()
        inputs = (x, ids, scores, w1, w2)
        saved_activations = []

        def pack(tensor):
            if tensor.is_floating_point() and not any(tensor is t for t in inputs):
                saved_activations.append(weakref.ref(tensor))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            y = bf16_moe(x, ids, scores, w1, w2, **self.options)
        y.sum().backward(retain_graph=True)
        first = tuple(t.grad.detach().clone() for t in (x, scores, w1, w2))
        self.assertTrue(any(ref() is not None for ref in saved_activations))
        y.sum().backward()
        for tensor, gradient in zip((x, scores, w1, w2), first):
            torch.testing.assert_close(tensor.grad, 2 * gradient, rtol=0, atol=0)
        gc.collect()
        self.assertTrue(saved_activations)
        self.assertTrue(all(ref() is None for ref in saved_activations))

    def test_empty_inputs_experts_and_masked_routes(self):
        for tokens in (0, 1, 5):
            with self.subTest(tokens=tokens):
                x, ids, scores, w1, w2 = self.inputs(tokens=tokens)
                ids.fill_(-1)
                y = bf16_moe(x, ids, scores, w1, w2, **self.options)
                y.sum().backward()
                for tensor in (y, x.grad, scores.grad, w1.grad, w2.grad):
                    self.assertEqual(int(torch.count_nonzero(tensor)), 0)

    def test_fp32_accumulation(self):
        x, ids, scores, w1, w2 = self.inputs()
        if not self.native:
            x, scores, w1, w2 = [t.detach().float().requires_grad_() for t in (x, scores, w1, w2)]
        accumulators = (torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
        _, functional_context = bf16_moe_forward(x, ids, scores, w1, w2, **self.options)
        expected = bf16_moe_backward(torch.ones_like(x), functional_context)
        for _ in range(2):
            bf16_moe(x, ids, scores, w1, w2, **self.options,
                     weight_grad_accumulators=accumulators).sum().backward()
        torch.testing.assert_close(accumulators[0], 2 * expected[1], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(accumulators[1], 2 * expected[2], rtol=1e-6, atol=1e-6)
        self.assertIsNone(w1.grad)
        self.assertIsNone(w2.grad)

        # Native K-grouped GEMM must accumulate directly into an existing buffer,
        # including preserving it when no expert receives a token.
        _, context = bf16_moe_forward(x, ids, scores, w1, w2, **self.options)
        returned = bf16_moe_backward(torch.ones_like(x), context,
                                     weight_grad_accumulators=accumulators)
        self.assertIs(returned[1], accumulators[0])
        self.assertIs(returned[2], accumulators[1])
        torch.testing.assert_close(accumulators[0], 3 * expected[1], rtol=2e-6, atol=1e-6)
        torch.testing.assert_close(accumulators[1], 3 * expected[2], rtol=2e-6, atol=1e-6)
        before = tuple(t.clone() for t in accumulators)
        _, context = bf16_moe_forward(x, torch.full_like(ids, -1), scores, w1, w2, **self.options)
        bf16_moe_backward(torch.ones_like(x), context, weight_grad_accumulators=accumulators)
        for actual, initial in zip(accumulators, before):
            torch.testing.assert_close(actual, initial, rtol=0, atol=0)

    def test_ragged_k_groups_and_larger_shape(self):
        if not self.native:
            self.skipTest("native layout coverage")
        self.h, self.i = 1024, 512
        x, ids, scores, w1, w2 = self.inputs(tokens=129, experts=3)
        # Expert 0 receives more than 128 rows, expert 2 is empty, and duplicate
        # routes must accumulate independently.
        ids[:, 0] = 0
        ids[:, 1] = 1
        ids[::3, 1] = 0
        y, context = bf16_moe_forward(x, ids, scores, w1, w2, **self.options, recompute=True)
        dy = torch.linspace(-0.1, 0.1, y.numel(), device=self.device).reshape(y.shape).to(self.dtype)
        actual = bf16_moe_backward(dy, context)
        rx, rp, rw1, rw2 = [t.detach().double().cpu().requires_grad_() for t in (x, scores, w1, w2)]
        expected_y = reference(rx, ids.cpu(), rp, rw1, rw2)
        expected = torch.autograd.grad(expected_y, (rx, rw1, rw2, rp), dy.double().cpu())
        errors = {"y": relative_rms(y, expected_y)}
        errors.update({name: relative_rms(value, target) for name, value, target in zip(
            ("dx", "dw1", "dw2", "dscores"), actual, expected
        )})
        self.assertLess(max(errors.values()), 0.12 if self.fp8 else 0.04, errors)
        self.assertEqual(int(torch.count_nonzero(actual[1][2])), 0)
        self.assertEqual(int(torch.count_nonzero(actual[2][2])), 0)
        print(json.dumps({"case": "ragged_k_groups", "backend": self.backend,
                          "grouped_gemm": self.grouped, "relative_rms": errors}), flush=True)

    def test_graph_capture_rejected(self):
        if not self.native:
            self.skipTest("PPU graph-capture contract")
        x, ids, scores, w1, w2 = self.inputs()
        torch.cuda.synchronize()
        with self.assertRaisesRegex(RuntimeError, "Graph capture"):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                bf16_moe(x, ids, scores, w1, w2)

    def test_optimizer_and_shared_expert_composition(self):
        x, ids, scores, w1, w2 = self.inputs()
        shared1 = w1[:1].detach().clone().requires_grad_()
        shared2 = w2[:1].detach().clone().requires_grad_()
        optimizer = torch.optim.SGD((w1, w2, shared1, shared2), lr=0.01)
        initial = w1.detach().clone()
        shared_ids = torch.zeros((x.shape[0], 1), device=self.device, dtype=torch.int64)
        shared_scores = torch.ones((x.shape[0], 1), device=self.device, dtype=scores.dtype)
        for _ in range(2):
            optimizer.zero_grad()
            routed = bf16_moe(x, ids, scores, w1, w2, **self.options)
            shared = bf16_moe(x, shared_ids, shared_scores, shared1, shared2, **self.options)
            (routed.float() + shared.float()).square().sum().backward()
            for param in (w1, w2, shared1, shared2):
                self.assertTrue(bool(torch.isfinite(param.grad).all()))
            optimizer.step()
        self.assertFalse(torch.equal(initial, w1))

    def test_mutation_and_invalid_experts(self):
        x, ids, scores, w1, w2 = self.inputs()
        _, context = bf16_moe_forward(x, ids, scores, w1, w2, **self.options)
        with torch.no_grad():
            w1.add_(1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            bf16_moe_backward(torch.ones_like(x), context)
        ids[0, 0] = w1.shape[0]
        with self.assertRaisesRegex(ValueError, "expert ID"):
            bf16_moe(x, ids, scores, w1, w2, **self.options)

        x, ids, scores, w1, w2 = self.inputs()
        y = bf16_moe(x, ids, scores, w1, w2, **self.options)
        ids.fill_(-1)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace"):
            y.sum().backward()
        with self.assertRaisesRegex(ValueError, "weight shapes"):
            bf16_moe(x[:, :0].contiguous(), ids, scores, w1[:, :, :0].contiguous(),
                     w2[:, :0].contiguous(), **self.options)


def distributed_checks(native, fp8=False, grouped=False):
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if native else torch.device("cpu")
    if native:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if native else "gloo")
    group = dist.group.WORLD
    dtype = torch.bfloat16 if native else torch.float64
    backend = "deep_gemm_fp8" if fp8 else "deep_gemm" if native else "torch"
    h, intermediate, local_experts = (256, 128, 2) if native else (6, 4, 2)
    generator = torch.Generator().manual_seed(8183)
    global_w1 = torch.randn(size * local_experts, 2 * intermediate, h, generator=generator) * 0.1
    global_w2 = torch.randn(size * local_experts, h, intermediate, generator=generator) * 0.1
    global_w1, global_w2 = global_w1.to(dtype).double(), global_w2.to(dtype).double()
    for pattern in ("balanced", "skewed", "empty_rank", "all_masked"):
        token_counts = [0 if pattern == "empty_rank" and peer == 0 else peer + 3 for peer in range(size)]
        all_x, all_ids, all_scores, all_dy = [], [], [], []
        for peer, tokens in enumerate(token_counts):
            all_x.append((torch.randn(tokens, h, generator=generator) * 0.1).to(dtype).double())
            ids = torch.tensor([[t % (size * local_experts), (t + peer + 1) % (size * local_experts)] for t in range(tokens)], dtype=torch.int64).reshape(tokens, 2)
            if pattern == "skewed": ids.fill_(0)
            if pattern == "all_masked": ids.fill_(-1)
            all_ids.append(ids)
            scores = torch.softmax(torch.randn(tokens, 2, generator=generator), -1)
            if tokens: scores[0, 0] = 0
            all_scores.append(scores.float().double() if native else scores.double())
            all_dy.append((torch.randn(tokens, h, generator=generator) * 0.1).to(dtype).double())
        rx = torch.cat(all_x).requires_grad_()
        rp = torch.cat(all_scores).requires_grad_()
        rw1, rw2 = global_w1.clone().requires_grad_(), global_w2.clone().requires_grad_()
        ref_y = reference(rx, torch.cat(all_ids), rp, rw1, rw2)
        # A completely masked reference has no parameter dependence.
        ref_y = ref_y + 0 * (rx.sum() + rp.sum() + rw1.sum() + rw2.sum())
        ref_grads = torch.autograd.grad(ref_y, (rx, rw1, rw2, rp), torch.cat(all_dy))
        start, end = sum(token_counts[:rank]), sum(token_counts[:rank + 1])
        e_start, e_end = rank * local_experts, (rank + 1) * local_experts
        for transport in (("collective", "symmetric") if native else ("collective",)):
            for recompute in (False, True):
                x = all_x[rank].to(device=device, dtype=dtype).detach().clone().requires_grad_()
                scores = all_scores[rank].to(device=device, dtype=torch.float32 if native else dtype).detach().clone().requires_grad_()
                w1 = global_w1[e_start:e_end].to(device=device, dtype=dtype).detach().clone().requires_grad_()
                w2 = global_w2[e_start:e_end].to(device=device, dtype=dtype).detach().clone().requires_grad_()
                ids = all_ids[rank].to(device)
                y = bf16_moe(x, ids, scores, w1, w2, group=group,
                             backend=backend, transport=transport, recompute=recompute, grouped_gemm=grouped)
                retain = pattern == "balanced" and recompute
                y.backward(all_dy[rank].to(device=device, dtype=dtype), retain_graph=retain)
                expected = (ref_y[start:end], ref_grads[0][start:end], ref_grads[1][e_start:e_end],
                            ref_grads[2][e_start:e_end], ref_grads[3][start:end])
                actual = (y, x.grad, w1.grad, w2.grad, scores.grad)
                errors = {key: relative_rms(value.detach(), target) for key, value, target in zip(
                    ("y", "dx", "dw1", "dw2", "dscores"), actual, expected
                )}
                limit = 0.12 if fp8 else 0.04 if native else 1e-12
                if not all(math.isfinite(error) and error < limit for error in errors.values()):
                    raise AssertionError((rank, pattern, transport, recompute, errors))
                print(json.dumps({"rank": rank, "pattern": pattern, "transport": transport,
                                  "backend": backend, "grouped_gemm": grouped,
                                  "recompute": recompute, "relative_rms": errors}), flush=True)
                if retain:
                    # Reacquire symmetric buffers after the first backward's
                    # cleanup while autograd still owns the saved activations.
                    for tensor in (x, scores, w1, w2):
                        tensor.grad = None
                    y.backward(all_dy[rank].to(device=device, dtype=dtype))
                    repeated = (y, x.grad, w1.grad, w2.grad, scores.grad)
                    errors = {key: relative_rms(value, target) for key, value, target in zip(
                        ("y", "dx", "dw1", "dw2", "dscores"), repeated, expected
                    )}
                    if not all(math.isfinite(error) and error < limit for error in errors.values()):
                        raise AssertionError(("retained_graph", rank, transport, errors))
                    print(json.dumps({"case": "retained_graph", "rank": rank, "transport": transport,
                                      "backend": backend, "grouped_gemm": grouped,
                                      "relative_rms": errors}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--fp8", action="store_true", help="native E4M3 forward/dgrad with BF16 wgrad")
    parser.add_argument("--grouped", action="store_true", help="native grouped BF16 forward/dgrad")
    parser.add_argument("--ep", action="store_true")
    args, remaining = parser.parse_known_args()
    torch.set_num_threads(2)
    if args.ep:
        distributed_checks(args.native or args.fp8 or args.grouped, args.fp8, args.grouped)
    else:
        TrainingTests.native = args.native or args.fp8 or args.grouped
        TrainingTests.fp8 = args.fp8
        TrainingTests.grouped = args.grouped
        unittest.main(argv=[__file__] + remaining, verbosity=2)
