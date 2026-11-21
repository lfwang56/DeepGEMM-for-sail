"""Full large-shape outputs and gradients against independent FP32 Torch.

This manual check uses several GiB per device. It validates every element,
whereas the distributed performance harness samples its large result tensors.
"""

import argparse
import gc
import json
import math

import torch

from deep_gemm.training import NativeMoEPlan, PeerMoEPlan, PersistentMoEPlan, CollectiveFusedMoEPlan, native_moe
from test_moe_training import reference


def relative_rms(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    return float((actual - expected).square().sum().sqrt() / expected.square().sum().sqrt().clamp_min(1e-20))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "peer", "persistent", "collective_fused"), default="native")
    parser.add_argument("--compute", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--no-fuse-dispatch", action="store_true")
    parser.add_argument("--no-fuse-return", action="store_true")
    parser.add_argument("--fuse-activation-backward", action="store_true")
    parser.add_argument("--fast-weight-transpose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dispatch-cache-n", type=int, choices=(0, 2, 4, 8, 16), default=0)
    parser.add_argument("--reuse-packed-dy", action="store_true")
    parser.add_argument("--dispatch-cache-aiu", action="store_true")
    parser.add_argument("--staged-return", action="store_true")
    parser.add_argument("--dispatch-cache-ctas", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--peer-fc1-config", type=int, nargs=5)
    parser.add_argument("--gemm-config", type=int, nargs=5)
    parser.add_argument("--dispatch-producer", action="store_true")
    parser.add_argument("--producer-local-aiu", action="store_true")
    parser.add_argument("--producer-input-aiu", action="store_true")
    parser.add_argument("--dispatch-pipeline", action="store_true")
    parser.add_argument("--return-pipeline", action="store_true")
    parser.add_argument("--pipeline-comm-ctas", type=int, default=2)
    parser.add_argument("--pipeline-serialize", action="store_true")
    parser.add_argument("--inline-dispatch", action="store_true")
    parser.add_argument("--cache-return-addresses", action="store_true")
    parser.add_argument("--ffn-fusion", action="store_true")
    parser.add_argument("--expert-local-grad", action="store_true")
    parser.add_argument("--dispatch-producer-tiles", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--topk", type=int, default=8)
    options = parser.parse_args()
    if (options.no_fuse_dispatch or options.no_fuse_return) and options.backend not in ("peer", "collective_fused"):
        parser.error("peer switches require --backend peer")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(19291)
    h, i, e, t = options.hidden, options.intermediate, options.experts, options.tokens
    if options.topk < 1:
        parser.error("--topk must be positive")
    k = min(options.topk, e)
    x = (torch.randn(t, h, device="cuda") * 0.1).bfloat16().requires_grad_()
    w1 = (torch.randn(e, 2 * i, h, device="cuda") * 0.02).bfloat16().float().requires_grad_()
    w2 = (torch.randn(e, h, i, device="cuda") * 0.02).bfloat16().float().requires_grad_()
    scores = torch.randn(t, k, device="cuda").softmax(-1).requires_grad_()
    with torch.no_grad():
        scores[0, 0] = 0
    dy = (torch.randn_like(x) * 0.1).bfloat16()
    ids = torch.randn(t, e, device="cuda").topk(k, -1).indices.contiguous()
    valid = torch.full((), max(0, t - 3), device="cuda", dtype=torch.int32)
    classes = {"native": NativeMoEPlan, "peer": PeerMoEPlan, "persistent": PersistentMoEPlan,
               "collective_fused": CollectiveFusedMoEPlan}
    if options.gemm_config and options.backend == "native":
        from benchmark_native_moe import ConfiguredCollectivePlan
        classes["native"] = ConfiguredCollectivePlan
    if options.gemm_config and options.backend == "persistent":
        parser.error("persistent kernel uses fixed GEMM tiles")
    extra = ({"gemm_configs": {role: (256, 64, 64, 64, 2)
                               for role in ("fc1", "fc2", "dhidden", "dx")}}
             if options.backend in ("peer", "collective_fused") else {"maximize_ctas": True} if options.backend == "persistent" else {})
    if options.gemm_config and options.backend in ("peer", "collective_fused"):
        extra["gemm_configs"] = {role:tuple(options.gemm_config) for role in ("fc1","fc2","dhidden","dx")}
    if options.backend == "peer":
        extra.update(fuse_dispatch=not options.no_fuse_dispatch, fuse_return=not options.no_fuse_return,
                     fuse_activation_backward=options.fuse_activation_backward)
        extra["dispatch_cache_ctas"] = options.dispatch_cache_ctas
        extra["dispatch_producer"] = options.dispatch_producer
        extra["producer_local_aiu"] = options.producer_local_aiu
        extra["producer_input_aiu"] = options.producer_input_aiu
        extra["dispatch_pipeline"] = options.dispatch_pipeline
        extra["return_pipeline"] = options.return_pipeline
        extra["pipeline_comm_ctas"] = options.pipeline_comm_ctas
        extra["pipeline_serialize"] = options.pipeline_serialize
        extra["inline_dispatch"] = options.inline_dispatch
        extra["cache_return_addresses"] = options.cache_return_addresses
        extra["ffn_fusion"] = options.ffn_fusion
        extra["expert_local_grad"] = options.expert_local_grad
        if options.ffn_fusion or options.expert_local_grad:
            extra["reuse_packed_dy"] = True
        extra["dispatch_producer_tiles"] = options.dispatch_producer_tiles
        if options.peer_fc1_config:
            extra["gemm_configs"]["fc1"] = options.peer_fc1_config
    if options.backend == "collective_fused":
        extra.update(fuse_return=not options.no_fuse_return,
                     fuse_activation_backward=options.fuse_activation_backward)
    extra["fast_weight_transpose"] = options.fast_weight_transpose
    if options.reuse_packed_dy:
        if options.backend != "peer":
            parser.error("packed gradient reuse requires peer backend")
        extra["reuse_packed_dy"] = True
    if options.dispatch_cache_n:
        if options.backend != "peer" or options.no_fuse_dispatch:
            parser.error("dispatch cache requires peer fused dispatch")
        extra["dispatch_cache_n"] = options.dispatch_cache_n
    if options.dispatch_cache_aiu:
        if options.backend != "peer" or not options.dispatch_cache_n:
            parser.error("AIU dispatch cache requires --backend peer --dispatch-cache-n > 0")
        extra["dispatch_cache_aiu"] = True
    if options.staged_return:
        if options.backend != "peer":
            parser.error("staged return requires --backend peer")
        extra["staged_return"] = True
    plan = classes[options.backend](h, i, e, k, t, chunk_tokens=t, compute=options.compute, **extra)
    if options.gemm_config and options.backend == "native":
        plan.benchmark_gemm_config = tuple(options.gemm_config)
    for pattern in ("balanced_masked", "duplicate_skewed"):
        if pattern == "duplicate_skewed":
            ids.fill_(0)
        if t > 1 and k > 1:
            ids[1, 1] = -1
        for value in (x, scores, w1, w2):
            value.grad = None
        actual_y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
        actual_y.backward(dy)
        rx, rp, rw1, rw2 = [value.detach().float().requires_grad_() for value in (x, scores, w1, w2)]
        reference_ids = ids.clone()
        reference_ids[max(0, t - 3):] = -1
        expected_y = reference(rx, reference_ids, rp, rw1, rw2)
        expected_y = expected_y + 0 * (rx.sum() + rp.sum() + rw1.sum() + rw2.sum())
        expected_grads = torch.autograd.grad(expected_y, (rx, rw1, rw2, rp), dy.float())
        actual = (actual_y, x.grad, w1.grad, w2.grad, scores.grad)
        errors = {name: relative_rms(a, b) for name, a, b in zip(
            ("y", "dx", "dw1", "dw2", "dscores"), actual, (expected_y, *expected_grads))}
        limit = 0.13 if options.compute == "fp8" else 0.04
        passed = all(math.isfinite(value) and value < limit for value in errors.values())
        print(json.dumps(dict(case="large_full_gradient_reference", backend=options.backend,
                              compute=options.compute, hidden=h, intermediate=i, experts=e, topk=k,
                              dispatch_cache_aiu=options.dispatch_cache_aiu,
                              staged_return=options.staged_return,
                              dispatch_cache_ctas=options.dispatch_cache_ctas,
                              dispatch_producer=options.dispatch_producer,
                              dispatch_pipeline=getattr(plan, "dispatch_pipeline", False),
                              return_pipeline=options.return_pipeline,
                              pipeline_comm_ctas=options.pipeline_comm_ctas,
                              pipeline_serialize=options.pipeline_serialize,
                              gemm_config=options.gemm_config,
                              inline_dispatch=options.inline_dispatch,
                              cache_return_addresses=options.cache_return_addresses,
                              ffn_fusion=options.ffn_fusion,
                              expert_local_grad=options.expert_local_grad,
                              producer_local_aiu=options.producer_local_aiu,
                              producer_input_aiu=options.producer_input_aiu,
                              dispatch_producer_tiles=options.dispatch_producer_tiles,
                              tokens=t, valid_tokens=max(0, t - 3), pattern=pattern,
                              chunk_tokens=plan.chunk_tokens, alignment=plan.alignment,
                              capacity_rows=plan.rows,
                              relative_rms=errors, every_element_compared=True,
                              fuse_dispatch=getattr(plan, "fuse_dispatch", None),
                              fuse_return=getattr(plan, "fuse_return", None),
                              fuse_activation_backward=getattr(plan, "fuse_activation_backward", None),
                              fast_weight_transpose=plan.fast_weight_transpose,
                              dispatch_cache_n=getattr(plan, "dispatch_cache_n", None),
                              reuse_packed_dy=getattr(plan, "reuse_packed_dy", None),
                              reference="independent FP32 Torch routed experts", passed=passed)), flush=True)
        if not passed:
            raise AssertionError(errors)
        del actual_y, expected_y, expected_grads, actual, rx, rp, rw1, rw2
        gc.collect()
    plan.check()


if __name__ == "__main__":
    main()
