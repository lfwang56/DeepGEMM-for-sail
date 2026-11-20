"""GPU-resident MoE schedules with a bounded activation ring and graph replay.

The caller owns the plan and parameters. A plan may be shared by sequential
layers/forwards on one stream: backward reconstructs every chunk instead of
retaining a layer's expanded expert activations in the plan.
"""

import math
import threading
from contextlib import contextmanager

import torch
import torch.distributed as dist

import deep_gemm as dg


def _launch(operation, tensors, dimensions):
    dg._C._moe_training_launch(operation, tensors, dimensions)


class NativeMoEPlan:
    """Fixed-capacity native training workspace; construct outside graph capture.

    All ranks agree on dimensions, chunk size and compute recipe. Input token
    counts may differ up to max_tokens. A plan is serialized on one CUDA/PPU
    stream; create separate plans for concurrent streams. The activation ring
    holds one chunk, independent of max_tokens and the number of pending layers.
    """

    def __init__(self, hidden, intermediate, local_experts, topk, max_tokens, *,
                 chunk_tokens=128, compute="bf16", group=None, device=None,
                 check_errors=True):
        if hidden <= 0 or hidden % 256 or intermediate <= 0 or intermediate % 128:
            raise ValueError("native training requires positive H%256=0 and I%128=0")
        if min(local_experts, topk, chunk_tokens) < 1 or max_tokens < 0:
            raise ValueError("invalid expert, top-k or token capacity")
        if compute not in ("bf16", "fp8"):
            raise ValueError("compute must be 'bf16' or 'fp8'")
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("NativeMoEPlan requires a PPU/CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if group is not None and not dist.is_initialized():
            raise RuntimeError("initialize distributed before creating an EP plan")
        self.group = group
        self.size = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0
        self.h, self.i, self.e, self.k = hidden, intermediate, local_experts, topk
        self.max_tokens, self.chunk_tokens = max_tokens, chunk_tokens
        self.compute, self.check_errors = compute, check_errors
        self.alignment = math.lcm(128, dg.get_m_alignment_for_contiguous_layout())
        self.routes = self.size * chunk_tokens * topk
        # Sum ceil(count_e/alignment)*alignment <= routes + E*(alignment-1).
        self.rows = math.ceil((self.routes + local_experts * (self.alignment - 1)) / self.alignment) * self.alignment
        if self.rows > 2**31 - 1 or self.routes > 2**31 - 1:
            raise ValueError("native route capacity exceeds int32 metadata")
        if group is not None:
            description = (hidden, intermediate, local_experts, topk, max_tokens, chunk_tokens, compute)
            descriptions = [None] * self.size
            dist.all_gather_object(descriptions, description, group=group)
            if any(other != description for other in descriptions):
                raise ValueError("EP ranks must create matching NativeMoEPlan configurations")
        self._lock = threading.RLock()
        self._stream = None
        self.buffers = {}

        def allocate(name, shape, dtype=torch.bfloat16):
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self.buffers[name] = tensor
            return tensor

        c, r, m, h, i, e, k = chunk_tokens, self.routes, self.rows, hidden, intermediate, local_experts, topk
        self.local_x = allocate("local_x", (c, h))
        self.local_ids = allocate("local_ids", (c, k), torch.int64)
        self.gathered_x = allocate("gathered_x", (self.size * c, h))
        self.gathered_ids = allocate("gathered_ids", (self.size * c, k), torch.int64)
        self.counts = allocate("counts", (e,), torch.int32)
        self.offsets = allocate("offsets", (e + 1,), torch.int32)
        self.lengths = allocate("lengths", (e,), torch.int32)
        self.group_ids = allocate("group_ids", (m,), torch.int32)
        self.permutation = allocate("permutation", (m,), torch.int32)
        self.route_experts = allocate("route_experts", (r,), torch.int32)
        self.route_rows = allocate("route_rows", (r,), torch.int32)
        self.status = allocate("status", (1,), torch.int32).zero_()
        self.packed_x = allocate("packed_x", (m, h))
        self.gate_up = allocate("gate_up", (m, 2 * i))
        self.hidden = allocate("hidden", (m, i))
        self.packed_output = allocate("packed_output", (m, h))
        self.packed_dy = allocate("packed_dy", (m, h))
        self.route_values = allocate("route_values", (r, h))
        self.local_values = allocate("local_values", (c * k, h))
        self.local_dy = allocate("local_dy", (c * k, h))
        self.w1_t = allocate("w1_t", (e, h, 2 * i))
        self.w2_t = allocate("w2_t", (e, i, h))
        self.quantized = {}
        if compute == "fp8":
            for name, shape in (("w1", (e, 2 * i, h)), ("w2", (e, h, i)),
                                ("w1_t", (e, h, 2 * i)), ("w2_t", (e, i, h))):
                data = allocate(name + "_fp8", shape, torch.float8_e4m3fn)
                scales = allocate(name + "_scale", (shape[0], shape[1] // 128, shape[2] // 128), torch.float32)
                self.quantized[name] = (data, scales)
            for width in {h, i, 2 * i}:
                data = allocate(f"activation_fp8_{width}", (m, width), torch.float8_e4m3fn)
                scales = allocate(f"activation_scale_{width}", (m, width // 128), torch.float32)
                self.quantized[width] = (data, scales)

    @property
    def workspace_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.buffers.values())

    @property
    def activation_workspace_bytes(self):
        return sum(t.numel() * t.element_size() for name, t in self.buffers.items()
                   if not name.startswith(("w1", "w2")))

    @contextmanager
    def _execution(self):
        with self._lock:
            stream = torch.cuda.current_stream(self.device).cuda_stream
            if self._stream is None:
                self._stream = stream
            elif self._stream != stream:
                raise RuntimeError("a NativeMoEPlan must remain on one stream; use a separate plan for graph capture")
            yield

    def check(self):
        """Check device validation after eager execution or a graph replay."""
        status = int(self.status.item())
        if status:
            errors = []
            if status & 1: errors.append("expert ID outside [-1, global_experts)")
            if status & 2: errors.append("route workspace capacity exceeded")
            if status & 4: errors.append("valid_tokens outside input capacity")
            raise RuntimeError("native MoE: " + "; ".join(errors))

    def _validate(self, x, ids, scores, w1, w2, valid_tokens):
        if x.ndim != 2 or x.shape[1] != self.h or x.shape[0] > self.max_tokens:
            raise ValueError("x must be [T,H] with T <= plan.max_tokens")
        if ids.shape != (x.shape[0], self.k) or scores.shape != ids.shape:
            raise ValueError("expected expert IDs and scores [T,topk]")
        if w1.shape != (self.e, 2 * self.i, self.h) or w2.shape != (self.e, self.h, self.i):
            raise ValueError("expert weights do not match the plan")
        for tensor, dtype in ((x, torch.bfloat16), (ids, torch.int64), (scores, torch.float32),
                              (w1, torch.bfloat16), (w2, torch.bfloat16), (valid_tokens, torch.int32)):
            if tensor.device != self.device or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError("native MoE tensors must have matching device, dtype and contiguous layout")
        if valid_tokens.numel() != 1:
            raise ValueError("valid_tokens must be an int32 device scalar")

    def _gather(self, output, local):
        if self.size == 1:
            output.copy_(local)
        else:
            dist.all_gather_into_tensor(output, local, group=self.group)

    def _return_routes(self):
        if self.size == 1:
            self.local_values.copy_(self.route_values)
        else:
            dist.reduce_scatter_tensor(self.local_values, self.route_values, group=self.group)

    def _quantize_weight(self, name, weight):
        data, scales = self.quantized[name]
        _launch(10, [weight, data, scales], list(weight.shape))

    def _prepare_weights(self, w1, w2, backward=False):
        if self.compute == "fp8":
            self._quantize_weight("w1", w1)
            self._quantize_weight("w2", w2)
        if backward:
            self.w1_t.copy_(w1.transpose(1, 2))
            self.w2_t.copy_(w2.transpose(1, 2))
            if self.compute == "fp8":
                self._quantize_weight("w1_t", self.w1_t)
                self._quantize_weight("w2_t", self.w2_t)

    def _gemm(self, a, weight, output, name):
        if self.compute == "bf16":
            dg.m_grouped_bf16_gemm_nt_contiguous(a, weight, output, self.group_ids)
        else:
            quantized_a = self.quantized[a.shape[1]]
            _launch(9, [a, *quantized_a], list(a.shape))
            dg.m_grouped_fp8_gemm_nt_contiguous(
                quantized_a, self.quantized[name], output, self.group_ids, disable_ue8m0_cast=True,
            )

    def _route_chunk(self, x, ids, valid_tokens, start):
        t, c, h, k = x.shape[0], self.chunk_tokens, self.h, self.k
        _launch(13, [x, ids, self.local_x, self.local_ids, valid_tokens, self.status], [t, c, h, k, start])
        self._gather(self.gathered_x, self.local_x)
        self._gather(self.gathered_ids, self.local_ids)
        self.counts.zero_()
        _launch(0, [self.gathered_ids, self.counts, self.route_experts, self.route_rows, self.status],
                [self.routes, self.e, self.rank, self.size])
        _launch(1, [self.counts, self.offsets, self.lengths], [self.e, self.alignment])
        _launch(2, [self.offsets, self.group_ids, self.permutation], [self.rows, self.e])
        self.packed_x.zero_()
        _launch(3, [self.gathered_x, self.route_experts, self.route_rows, self.offsets,
                    self.packed_x, self.permutation, self.status], [self.routes, h, k, self.rows])

    def _chunk_forward(self, w1, w2):
        self._gemm(self.packed_x, w1, self.gate_up, "w1")
        _launch(7, [self.gate_up, self.hidden], [self.rows, self.i])
        self._gemm(self.hidden, w2, self.packed_output, "w2")
        self._scatter_output()

    def _scatter_output(self):
        self.route_values.zero_()
        _launch(4, [self.packed_output, self.permutation, self.route_values], [self.rows, self.h, self.routes])
        self._return_routes()

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens):
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        with self._execution():
            self.status.zero_()
            self._prepare_weights(w1, w2)
            output = torch.empty_like(x)
            for start in range(0, self.max_tokens, self.chunk_tokens):
                self._route_chunk(x, ids, valid_tokens, start)
                self._chunk_forward(w1, w2)
                _launch(6, [self.local_values, scores, output],
                        [self.chunk_tokens, self.h, self.k, x.shape[0], start, 1])
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return output

    @torch.no_grad()
    def backward(self, dy, x, ids, scores, w1, w2, valid_tokens, accumulators=None):
        from .bf16 import _validate_accumulators

        _validate_accumulators(accumulators, w1, w2)
        if dy.shape != x.shape or dy.dtype != x.dtype or dy.device != x.device:
            raise ValueError("dy must match x")
        with self._execution():
            self.status.zero_()
            self._prepare_weights(w1, w2, backward=True)
            dx, dscores = torch.empty_like(x), torch.empty_like(scores)
            dw1, dw2 = ((torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
                        if accumulators is None else accumulators)
            for start in range(0, self.max_tokens, self.chunk_tokens):
                # Replay FC2 too: no full-batch [T,topk,H] expert output is saved.
                self._route_chunk(x, ids, valid_tokens, start)
                self._chunk_forward(w1, w2)
                dims = [self.chunk_tokens, self.h, self.k, x.shape[0], start]
                _launch(12, [self.local_values, dy, dscores], dims)
                _launch(11, [dy, scores, self.local_dy], dims)
                self._gather(self.route_values, self.local_dy)
                _launch(5, [self.route_values, self.permutation, self.packed_dy], [self.rows, self.h, self.routes])
                dg.k_grouped_bf16_gemm_tn_device(self.packed_dy, self.hidden, dw2, self.lengths, self.rows)
                # Reuse activation buffers only after their wgrad readers.
                self._gemm(self.packed_dy, self.w2_t, self.hidden, "w2_t")
                _launch(8, [self.gate_up, self.hidden, self.gate_up], [self.rows, self.i])
                dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x, dw1, self.lengths, self.rows)
                self._gemm(self.gate_up, self.w1_t, self.packed_output, "w1_t")
                self._scatter_output()
                _launch(6, [self.local_values, scores, dx], dims + [0])
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return dx, dw1, dw2, dscores


class _NativeMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ids, scores, w1, w2, valid_tokens, plan, accumulators):
        from .bf16 import _validate_accumulators

        _validate_accumulators(accumulators, w1, w2)
        output = plan.forward(x, ids, scores, w1, w2, valid_tokens)
        ctx.save_for_backward(x, ids, scores, w1, w2, valid_tokens)
        ctx.plan, ctx.accumulators = plan, accumulators
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        x, ids, scores, w1, w2, valid_tokens = ctx.saved_tensors
        dx, dw1, dw2, dscores = ctx.plan.backward(
            dy.contiguous(), x, ids, scores, w1, w2, valid_tokens, ctx.accumulators,
        )
        if ctx.accumulators is not None:
            dw1 = dw2 = None
        else:
            dw1, dw2 = dw1.to(w1.dtype), dw2.to(w2.dtype)
        return dx, None, dscores, dw1, dw2, None, None, None


def native_moe(x, expert_ids, scores, w1, w2, *, plan, valid_tokens=None, weight_grad_accumulators=None):
    """Routed SwiGLU training using GPU schedules and bounded replay storage.

    Parameters/input are BF16 and scores FP32. IDs may be int32 or int64; masked
    slots use -1. valid_tokens is an optional int32 device scalar for dynamic
    padding within a captured input shape. After graph replay, call plan.check()
    outside capture to inspect invalid-route/capacity errors.
    """
    if not isinstance(plan, NativeMoEPlan):
        raise TypeError("plan must be a NativeMoEPlan")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert IDs must be int32 or int64")
    if valid_tokens is None:
        valid_tokens = torch.tensor(x.shape[0], device=x.device, dtype=torch.int32)
    return _NativeMoE.apply(x, expert_ids.to(torch.int64), scores, w1, w2,
                            valid_tokens, plan, weight_grad_accumulators)
