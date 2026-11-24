"""Regression for BF16/F32 segmented DW, first nonempty writes and graph replay."""
import json
import torch
import deep_gemm as dg


def test_segmented_wgrad():
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(320641)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    groups, capacity, m, n = 5, 768, 128, 256
    left = [torch.randn(capacity, m, device='cuda', dtype=torch.bfloat16) * .1 for _ in range(2)]
    right = [torch.randn(capacity, n, device='cuda', dtype=torch.bfloat16) * .1 for _ in range(2)]
    offsets = [torch.zeros(groups + 1, device='cuda', dtype=torch.int32) for _ in range(2)]
    flags = torch.zeros(groups, device='cuda', dtype=torch.int32)
    storage = [torch.full((groups * m * n + 128,), 12345., device='cuda') for _ in range(3)]
    outputs = [s[64:-64].view(groups, m, n) for s in storage]

    def operation():
        outputs[0].fill_(.125)
        outputs[1].fill_(float('nan'))
        outputs[2].fill_(.125)
        outputs[2].masked_fill_(flags[:, None, None].bool(), float('nan'))
        for out, accumulate, first in [(outputs[0], True, None),
                                       (outputs[1], False, None),
                                       (outputs[2], True, flags)]:
            dg._C.k_grouped_bf16_gemm_tn_segments(
                left, right, out, offsets, capacity, accumulate=accumulate,
                max_continuous_k=10240, first_flags=first)

    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    checks = 0
    for pattern in range(3):
        for mutation in range(2):
            lengths = ([[0, 128, 0, 256, 0], [256, 0, 128, 128, 0]] if pattern == 0 else
                       [[0] * groups, [0] * groups] if pattern == 1 else
                       [[0] * groups, [128, 256, 0, 128, 0]])
            first = [int(e in (0, 2) and sum(row[e] for row in lengths) > 0) for e in range(groups)]
            flags.copy_(torch.tensor(first, device='cuda', dtype=torch.int32))
            gold = torch.zeros_like(outputs[0])
            for a, b, off, sizes in zip(left, right, offsets, lengths):
                a.mul_(.9375).add_(.001)
                b.mul_(.875).sub_(.001)
                prefix = [0]
                for size in sizes:
                    prefix.append(prefix[-1] + size)
                off.copy_(torch.tensor(prefix, device='cuda', dtype=torch.int32))
                for e, size in enumerate(sizes):
                    if size:
                        gold[e].add_(a[prefix[e]:prefix[e + 1]].float().t() @
                                     b[prefix[e]:prefix[e + 1]].float())
            graph.replay()
            torch.cuda.synchronize()
            references = [gold + .125, gold,
                          gold + torch.tensor([0. if v else .125 for v in first],
                                              device='cuda')[:, None, None]]
            errors = []
            for out, ref in zip(outputs, references):
                relative = float((out - ref).norm() / ref.norm().clamp_min(1e-30))
                assert torch.isfinite(out).all() and relative < 1e-5, (pattern, mutation, relative)
                errors.append(relative)
            assert all(bool((s[:64] == 12345.).all()) and bool((s[-64:] == 12345.).all())
                       for s in storage)
            checks += 1
            print(json.dumps(dict(case='selected_segmented_wgrad_gold',pattern=pattern,
                                  mutation=mutation,relative_rms_fp32=errors,passed=True)),flush=True)
    print(json.dumps(dict(case='selected_segmented_wgrad_finished',checks=checks,passed=True)),flush=True)


if __name__ == '__main__':
    test_segmented_wgrad()
