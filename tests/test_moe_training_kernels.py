"""Native training primitives, GPU metadata and dynamic-layout graph replay."""

import unittest

import torch

import deep_gemm as dg


def launch(operation, tensors, dims):
    dg._C._moe_training_launch(operation, tensors, dims)


class KernelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9003)
        torch.set_num_threads(2)

    def test_activation_and_gradient(self):
        rows, intermediate = 7, 128
        gu = torch.randn(rows, 2 * intermediate).bfloat16()
        dh = torch.randn(rows, intermediate).bfloat16()
        a, b, out, grad = gu.cuda(), dh.cuda(), torch.empty_like(dh, device="cuda"), torch.empty_like(gu, device="cuda")
        launch(7, [a, out], [rows, intermediate])
        launch(8, [a, b, grad], [rows, intermediate])
        gate, up = gu.float().chunk(2, -1)
        expected = torch.nn.functional.silu(gate) * up
        sigmoid = gate.sigmoid()
        expected_grad = torch.cat((dh.float() * up * sigmoid * (1 + gate * (1 - sigmoid)),
                                   dh.float() * gate * sigmoid), -1).bfloat16()
        torch.testing.assert_close(out.cpu(), expected.bfloat16(), rtol=0.008, atol=0.001)
        torch.testing.assert_close(grad.cpu(), expected_grad, rtol=0.008, atol=0.002)

    def test_grouped_weight_transpose_bits(self):
        for experts, hidden, intermediate in ((2, 256, 128), (3, 768, 384), (1, 7168, 2048)):
            w1 = torch.randn(experts, 2 * intermediate, hidden, device="cuda").bfloat16()
            w2 = torch.randn(experts, hidden, intermediate, device="cuda").bfloat16()
            # Include signed zero, infinities and NaN payload bits: transpose
            # performs no floating-point arithmetic or value conversion.
            patterns = torch.tensor([0, -32768, 32767, 32640, -128, 128, 1, -1],
                                    dtype=torch.int16, device="cuda")
            w1.view(torch.int16).flatten()[:8].copy_(patterns)
            a = torch.empty((experts, hidden, 2 * intermediate), device="cuda", dtype=torch.bfloat16)
            b = torch.empty((experts, intermediate, hidden), device="cuda", dtype=torch.bfloat16)
            launch(22, [w1, w2, a, b], [experts, hidden, intermediate])
            torch.testing.assert_close(a.view(torch.int16), w1.view(torch.int16).transpose(1, 2), rtol=0, atol=0)
            torch.testing.assert_close(b.view(torch.int16), w2.view(torch.int16).transpose(1, 2), rtol=0, atol=0)

    def test_native_fp8_quantizers(self):
        a = torch.randn(5, 256).bfloat16()
        a[0].zero_()
        a[1, :128] *= 20
        a[2, 128:] *= 0.001
        out = torch.empty_like(a, device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.empty((5, 2), device="cuda")
        launch(9, [a.cuda(), out, scales], [5, 256])
        reference, reference_scales = dg.per_token_cast_to_fp8(a)
        torch.testing.assert_close(scales.cpu(), reference_scales, rtol=1e-6, atol=0)
        torch.testing.assert_close(out.float().cpu(), reference.float(), rtol=0.07, atol=0.002)
        weights = torch.randn(2, 256, 256).bfloat16()
        weights[0, :128, :128].zero_()
        weights[1, 128:, 128:] *= 0.01
        qo = torch.empty_like(weights, device="cuda", dtype=torch.float8_e4m3fn)
        qs = torch.empty((2, 2, 2), device="cuda")
        launch(10, [weights.cuda(), qo, qs], [2, 256, 256])
        for group in range(2):
            expected, expected_scales = dg.per_block_cast_to_fp8(weights[group])
            torch.testing.assert_close(qs[group].cpu(), expected_scales, rtol=1e-6, atol=0)
            torch.testing.assert_close(qo[group].float().cpu(), expected.float(), rtol=0.07, atol=0.002)

    def test_activation_quantization_fusion(self):
        for intermediate in (128, 256):
            rows = 5
            gu = torch.randn(rows, 2 * intermediate).bfloat16().cuda()
            dh = torch.randn(rows, intermediate).bfloat16().cuda()
            gu[0].zero_()
            gu[1].mul_(16)
            gu[2].mul_(0.001)
            dh[3].zero_()
            hidden = torch.empty_like(dh)
            gradient = torch.empty_like(gu)
            launch(7, [gu, hidden], [rows, intermediate])
            launch(8, [gu, dh, gradient], [rows, intermediate])
            for backward, expected in ((False, hidden), (True, gradient)):
                width = expected.shape[1]
                reference_q = torch.empty_like(expected, dtype=torch.float8_e4m3fn)
                reference_s = torch.empty((rows, width // 128), device="cuda")
                launch(9, [expected, reference_q, reference_s], [rows, width])
                q, scales = torch.empty_like(reference_q), torch.empty_like(reference_s)
                if backward:
                    output = gu.clone()
                    launch(15, [output, dh, output, q, scales], [rows, intermediate])
                else:
                    output = torch.empty_like(hidden)
                    launch(14, [gu, output, q, scales], [rows, intermediate])
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                torch.testing.assert_close(q.view(torch.uint8), reference_q.view(torch.uint8), rtol=0, atol=0)
                torch.testing.assert_close(scales, reference_s, rtol=0, atol=0)

    def test_route_pack_and_inverse(self):
        routes, hidden, topk, experts = 20, 256, 2, 3
        ids_cpu = torch.tensor([0, 1, 0, 4, -1, 5, 2, 2, 3, 0, 1, 1, 4, 5, 0, -1, 2, 0, 5, 9])
        ids = ids_cpu.cuda()
        capacity = 512
        counts = torch.zeros(experts, dtype=torch.int32, device="cuda")
        re = torch.empty(routes, dtype=torch.int32, device="cuda")
        rr = torch.empty_like(re)
        offsets = torch.empty(experts + 1, dtype=torch.int32, device="cuda")
        lengths = torch.empty_like(counts)
        group_ids = torch.empty(capacity, dtype=torch.int32, device="cuda")
        permutation = torch.empty_like(group_ids)
        status = torch.zeros(1, dtype=torch.int32, device="cuda")
        x_cpu = torch.randn(routes // topk, hidden).bfloat16()
        packed = torch.zeros(capacity, hidden, device="cuda", dtype=torch.bfloat16)
        launch(0, [ids, counts, re, rr, status], [routes, experts, 0, 2])
        launch(1, [counts, offsets, lengths], [experts, 128])
        launch(2, [offsets, group_ids, permutation], [capacity, experts])
        launch(3, [x_cpu.cuda(), re, rr, offsets, packed, permutation, status], [routes, hidden, topk, capacity])
        torch.testing.assert_close(counts.cpu(), torch.tensor([(ids_cpu == e).sum() for e in range(experts)], dtype=torch.int32))
        self.assertEqual(int(status.item()), 1)  # Bad IDs are skipped safely.
        perm = permutation.cpu()
        active = perm >= 0
        expected_routes = torch.where((ids_cpu >= 0) & (ids_cpu < experts))[0]
        torch.testing.assert_close(perm[active].sort().values.long(), expected_routes)
        torch.testing.assert_close(packed.cpu()[active], x_cpu[perm[active].long() // topk], rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(packed.cpu()[~active])), 0)
        route_output = torch.zeros(routes, hidden, device="cuda", dtype=torch.bfloat16)
        launch(4, [packed, permutation, route_output], [capacity, hidden, routes])
        torch.testing.assert_close(route_output.cpu()[expected_routes], x_cpu[expected_routes // topk], rtol=0, atol=0)

    def test_dynamic_wgrad_lengths_in_graph(self):
        groups, m, n, capacity = 3, 256, 128, 768
        a_cpu = torch.randn(capacity, m).bfloat16()
        b_cpu = torch.randn(capacity, n).bfloat16()
        initial = torch.randn(groups, m, n)
        a, b, out = a_cpu.cuda(), b_cpu.cuda(), initial.cuda()
        layout = torch.tensor([128, 0, 256], dtype=torch.int32, device="cuda")
        def run():
            dg.k_grouped_bf16_gemm_tn_device(a, b, out, layout, capacity)
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for lengths in ([128, 0, 256], [0, 384, 128], [0, 0, 0]):
            layout.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
            out.copy_(initial)
            graph.replay()
            expected = initial.clone()
            offset = 0
            for group, length in enumerate(lengths):
                expected[group].add_(a_cpu[offset:offset + length].float().t() @ b_cpu[offset:offset + length].float())
                offset += length
            torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=0.002)


if __name__ == "__main__":
    unittest.main(verbosity=2)
