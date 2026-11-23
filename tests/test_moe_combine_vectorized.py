"""Exercise the aligned reduction and its scalar view fallback on one PPU.

Run with the intended installation on PYTHONPATH. No process group is required.
"""
import json
import torch
import deep_gemm


def check_combine(chunk_tokens, hidden, topk, tokens, start, input_offset, output_offset, weighted):
    # Different offsets independently exercise the two alignment conditions.
    raw = (torch.randn(chunk_tokens * topk * hidden + input_offset, device='cuda') * .2).bfloat16()
    source = raw[input_offset:].view(chunk_tokens * topk, hidden)
    storage = torch.full((tokens * hidden + output_offset + 16,), .125,
                         device='cuda', dtype=torch.bfloat16)
    output = storage[output_offset:output_offset + tokens * hidden].view(tokens, hidden)
    scores = torch.randn(tokens, topk, device='cuda') * .3
    if not weighted:
        scores.fill_(float('nan'))  # Unweighted reduction must not read scores.
    dimensions = [chunk_tokens, hidden, topk, tokens, start, int(weighted)]

    def launch():
        deep_gemm._C._moe_training_launch(6, [source, scores, output], dimensions)

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    active = max(0, min(chunk_tokens, tokens - start))
    for replay in range(3):
        if replay:
            source.mul_(.75).add_(.03125)
        storage.fill_(.125)
        graph.replay()
        torch.cuda.synchronize()
        if active:
            values = source[:active * topk].view(active, topk, hidden).double()
            products = values * scores[start:start + active, :, None].double() if weighted else values
            expected = products.sum(1)
            # BF16 conversion plus FP32 accumulation, compared to independent FP64.
            bound = .004 * products.abs().sum(1) + 2e-7
            assert bool((output[start:start + active].double().sub(expected).abs() <= bound).all())
        assert bool((output[:start] == .125).all())
        assert bool((output[start + active:] == .125).all())
        assert bool((storage[:output_offset] == .125).all())
        assert bool((storage[output_offset + tokens * hidden:] == .125).all())


def main():
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(51793)
    count = 0
    for shape in [(5, 24, 10, 9, 2), (5, 7, 3, 6, 4),
                  (3, 256, 10, 4, 1), (13, 128, 33, 15, 1),
                  (0, 128, 10, 3, 0), (2, 4096, 10, 3, 1)]:
        for input_offset, output_offset in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            for weighted in (False, True):
                check_combine(*shape, input_offset, output_offset, weighted)
                count += 1
    print(json.dumps(dict(case='combine_vectorized_reference', cases=count,
                          graph_replays=count * 3, independent_fp64=True, passed=True)))


if __name__ == '__main__':
    main()
