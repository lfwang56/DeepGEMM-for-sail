"""FP8 compute / BF16 transport: independent STE oracle and mutable graphs.

The CPU oracle explicitly implements E4M3 row/128 activations, 128x128
weight scales and BF16 intermediate/weight-gradient input boundaries.
It does not call training kernels, quantizers or NativeMoEPlan.
"""
import argparse
from datetime import timedelta
import gc
import json
import math
import os

import torch
import torch.distributed as dist
from deep_gemm.training import DeepEPMoEPlan, NativeMoEPlan
from moe_distributed_utils import destroy_training_process_group


def quant_rows(x):
    x = x.float()
    tiles = x.reshape(x.shape[0], -1, 128)
    scales = tiles.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    return ((tiles / scales).to(torch.float8_e4m3fn).float() * scales).reshape_as(x)


def quant_weight(x):
    x = x.float()
    n, k = x.shape
    tiles = x.reshape(n // 128, 128, k // 128, 128)
    scales = tiles.abs().amax((1, 3), keepdim=True).clamp_min(1e-4) / 448
    return ((tiles / scales).to(torch.float8_e4m3fn).float() * scales).reshape_as(x)


def local_oracle(x, ids, scores, dy, w1, w2, owner):
    n, h = x.shape
    e, twice_i, _ = w1.shape
    i = twice_i // 2
    y, dx = torch.zeros(n, h), torch.zeros(n, h)
    ds = torch.zeros_like(scores)
    dw1, dw2 = torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32)
    for expert in range(e):
        tok, slot = (ids == owner * e + expert).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        bx = x[tok]
        gu = (quant_rows(bx) @ quant_weight(w1[expert]).t()).bfloat16()
        gate, up = gu.float().split(i, -1)
        sigmoid = torch.sigmoid(gate)
        hidden = (gate * sigmoid * up).bfloat16()
        out = (quant_rows(hidden) @ quant_weight(w2[expert]).t()).bfloat16()
        weighted_dy = (dy[tok].float() * scores[tok, slot, None]).bfloat16()
        dh = (quant_rows(weighted_dy) @ quant_weight(w2[expert].t()).t()).bfloat16().float()
        dg = (dh * up * sigmoid * (1 + gate * (1 - sigmoid))).bfloat16()
        du = (dh * gate * sigmoid).bfloat16()
        dgu = torch.cat((dg, du), -1)
        gx = (quant_rows(dgu) @ quant_weight(w1[expert].t()).t()).bfloat16()
        y.index_add_(0, tok, out.float() * scores[tok, slot, None])
        dx.index_add_(0, tok, gx.float())
        ds[tok, slot] = (out.float() * dy[tok].float()).sum(-1)
        dw1[expert] = dgu.float().t() @ bx.float()
        dw2[expert] = weighted_dy.float().t() @ hidden.float()
    return y, dx, dw1, dw2, ds


def rms(a, b):
    a, b = a.float(), b.float()
    return float((a-b).norm() / b.norm().clamp_min(1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--early-dy', action='store_true')
    args = ap.parse_args()
    rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=600))
    cpu_group = dist.new_group(backend='gloo', timeout=timedelta(seconds=600))
    h, i, e, k, t, c = 256, 128, 2, 3, 17, 32
    if size * e < k:
        raise ValueError('test requires at least2 ranks')
    rng = torch.Generator().manual_seed(607821)
    base_x = (torch.randn(size*t, h, generator=rng)*.2).bfloat16()
    base_dy = (torch.randn(size*t, h, generator=rng)*.15).bfloat16()
    base_scores = torch.randn(size*t, k, generator=rng).softmax(-1)
    base_scores[::t, 0] = 0
    wrng = torch.Generator().manual_seed(81377+rank)
    base_w1 = (torch.randn(e, 2*i, h, generator=wrng)*.03).bfloat16()
    base_w2 = (torch.randn(e, h, i, generator=wrng)*.03).bfloat16()
    sl = slice(rank*t, (rank+1)*t)
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        x, dy = base_x[sl].to(device), base_dy[sl].to(device)
        scores = base_scores[sl].to(device)
        ids = ((torch.arange(rank*t, (rank+1)*t)[:, None]*k+torch.arange(k)) % (size*e)).to(device)
        valid = torch.tensor(t, device=device, dtype=torch.int32)
        w1, w2 = base_w1.to(device), base_w2.to(device)
        common = dict(hidden=h, intermediate=i, local_experts=e, topk=k,
                      max_tokens=t, chunk_tokens=c, group=dist.group.WORLD,
                      device=device, compute='fp8', check_errors=False)
        plans = [NativeMoEPlan(**common), DeepEPMoEPlan(**common, ring_rows=128,
                  comm_sms=38, elastic_hybrid=True, overlap_expert_dy=args.early_dy)]
        graphs, buffers = [], []
        for plan in plans:
            def step():
                y = plan.forward(x, ids, scores, w1, w2, valid)
                return y, *plan.backward(dy, x, ids, scores, w1, w2, valid)
            step(); step(); stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = step()
            graphs.append(graph); buffers.append(result)
    previous = None
    for index, pattern in enumerate(('balanced', 'skewed_unique', 'empty', 'reversed_changed_weights')):
        ax = (base_x.float() * (1 + .17*index) + .005*index).bfloat16()
        ady = (base_dy.float() * (1 - .09*index)).bfloat16()
        aw1 = (base_w1.float() * (1 + .13*index) + .001*index).bfloat16()
        aw2 = (base_w2.float() * (1 - .08*index) - .001*index).bfloat16()
        aid = (torch.arange(size*t)[:, None]*k+torch.arange(k)) % (size*e)
        if pattern == 'skewed_unique':aid.copy_(torch.arange(k).expand_as(aid))
        if pattern == 'empty':aid.fill_(-1)
        if pattern.startswith('reversed'):aid = size*e-1-aid
        counts = [0 if source == 0 else t-source%4 for source in range(size)]
        for source, count in enumerate(counts):aid[source*t+count:(source+1)*t] = -1
        aid[1::t, 1] = -1
        with torch.cuda.stream(stream):
            x.copy_(ax[sl]); dy.copy_(ady[sl]); ids.copy_(aid[sl]); scores.copy_(base_scores[sl])
            valid.fill_(counts[rank]); w1.copy_(aw1); w2.copy_(aw2)
            for graph in graphs:graph.replay()
        stream.synchronize()
        for plan in plans:plan.check()
        expected = list(local_oracle(ax, aid, base_scores, ady, aw1, aw2, rank))
        for j in (0, 1, 4):
            dist.all_reduce(expected[j], group=cpu_group)
            expected[j] = expected[j][sl]
        actual = [[v.detach().cpu() for v in result] for result in buffers]
        names = ('y', 'dx', 'dw1', 'dw2', 'dscores')
        errors = [{name:rms(v, ref) for name, v, ref in zip(names, result, expected)} for result in actual]
        pair = {name:rms(a, b) for name, a, b in zip(names, actual[1], actual[0])}
        finite = all(math.isfinite(v) for row in errors+[pair] for v in row.values())
        passed = finite and max(v for row in errors for v in row.values()) < .02 and max(pair.values()) < .012
        if pattern == 'empty':passed = passed and all(not torch.count_nonzero(v) for row in actual for v in row)
        print(json.dumps(dict(case='fp8_deepep_mutable_graph_reference', rank=rank,
              ranks=size, pattern=pattern, compute='fp8_e4m3', communication='bf16',
              activation_scales='row_128', weight_scales='128x128_fp32',
              wgrad='bf16_inputs_fp32_accumulation', early_dy=args.early_dy,
              oracle_errors=errors, native_pair_errors=pair, every_element=True, passed=passed)), flush=True)
        gate = torch.tensor(int(passed), device=device)
        dist.all_reduce(gate, op=dist.ReduceOp.MIN)
        if not gate.item():raise AssertionError('FP8 DeepEP oracle or transport mismatch')
        previous = actual
    del graphs, buffers, result, actual, previous
    gc.collect(); stream.synchronize()
    for plan in plans:
        if hasattr(plan, 'close'):plan.close()
    del plans
    gc.collect()
    destroy_training_process_group()


if __name__ == '__main__':
    main()
