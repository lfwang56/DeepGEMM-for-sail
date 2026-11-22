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
    ap.add_argument('--tokens', type=int, default=17)
    ap.add_argument('--chunk-tokens', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--intermediate', type=int, default=128)
    ap.add_argument('--early-dy', action='store_true')
    ap.add_argument('--check-zero-capacity', action='store_true')
    args = ap.parse_args()
    rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=600))
    cpu_group = dist.new_group(backend='gloo', timeout=timedelta(seconds=600))
    h, i, e, k, t, c = args.hidden, args.intermediate, 2, 3, args.tokens, args.chunk_tokens
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
              ranks=size, pattern=pattern, compute='fp8_e4m3', communication=plans[1].communication_dtype,
              fp8_source_dispatch=plans[1].fp8_source_dispatch,
              reuse_forward_context=plans[1].reuse_forward_context,
              reused_windows=int(plans[1].forward_context_reused_windows.item()) if plans[1].reuse_forward_context else 0,
              activation_scales='row_128', weight_scales='128x128_fp32',
              wgrad='bf16_inputs_fp32_accumulation', early_dy=args.early_dy,
              oracle_errors=errors, native_pair_errors=pair, every_element=True, passed=passed)), flush=True)
        gate = torch.tensor(int(passed), device=device)
        dist.all_reduce(gate, op=dist.ReduceOp.MIN)
        if not gate.item():raise AssertionError('FP8 DeepEP oracle or transport mismatch')
        previous = actual
    if plans[1].reuse_forward_context:
        # One rank changing input invalidates routed forward values everywhere.
        with torch.cuda.stream(stream):
            for plan in plans:plan.forward(x, ids, scores, w1, w2, valid)
            if rank == 0:x.add_(.017)
            fallback=[plan.backward(dy,x,ids,scores,w1,w2,valid) for plan in plans]
        stream.synchronize()
        errors={name:rms(a.cpu(),b.cpu()) for name,a,b in zip(('dx','dw1','dw2','dscores'),fallback[0],fallback[1])}
        passed=max(errors.values())<.012 and plans[1].forward_context_reused_windows.item()==0
        gate=torch.tensor(int(passed),device=device);dist.all_reduce(gate,op=dist.ReduceOp.MIN)
        print(json.dumps(dict(case='forward_context_rank_change_fallback_reference',rank=rank,ranks=size,
              pair_errors=errors,all_ranks_fell_back=True,passed=passed)),flush=True)
        if not gate.item():raise AssertionError('rank-local input change did not invalidate global context')
        del fallback
    if args.check_zero_capacity:
        with torch.cuda.stream(stream):
            zero_options = dict(common)
            zero_options['max_tokens'] = 0
            zero = DeepEPMoEPlan(**zero_options, ring_rows=128, comm_sms=38,
                                elastic_hybrid=True, overlap_expert_dy=args.early_dy)
            zx, zdy, zi, zs = x[:0], dy[:0], ids[:0], scores[:0]
            zv = torch.zeros_like(valid)
            def zero_step():
                y = zero.forward(zx, zi, zs, w1, w2, zv)
                return y, *zero.backward(zdy, zx, zi, zs, w1, w2, zv)
            zero_step(); stream.synchronize()
            zero_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(zero_graph, stream=stream):
                zero_outputs = zero_step()
            w1.mul_(1.03); w2.mul_(.97)
            zero_graph.replay()
        stream.synchronize(); zero.check()
        passed = all(value.numel() == 0 for value in (zero_outputs[0], zero_outputs[1], zero_outputs[4]))
        passed = passed and not bool(torch.count_nonzero(zero_outputs[2])) and not bool(torch.count_nonzero(zero_outputs[3]))
        passed = passed and not zero._pending_weight_events
        print(json.dumps(dict(case='fp8_zero_capacity_graph_reference', rank=rank, ranks=size,
              async_weights=zero.fp8_async_weight_quantization, graph=True, passed=passed)), flush=True)
        if not passed:raise AssertionError('zero-capacity weight stream was not joined')
        del zero_graph, zero_outputs
        stream.synchronize(); zero.close(); del zero
    del graphs, buffers, result, actual, previous
    gc.collect(); stream.synchronize()
    for plan in plans:
        if hasattr(plan, 'close'):plan.close()
    del plans
    gc.collect()
    destroy_training_process_group()


if __name__ == '__main__':
    main()
