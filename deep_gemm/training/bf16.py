"""Correctness-first MoE training using native PPU GEMMs and EP transport.

The public autograd operation preserves the caller's parameters. The functional
API additionally exposes FP32 weight gradients. This is a multi-kernel backend,
not a claim of a fused training megakernel or CUDA Graph support.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


class _Transport:
    def __init__(self, group, mode, input_splits, output_splits, device):
        self.group = group
        self.size = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0
        self.mode = mode
        self.device = device
        self.buffers = {}
        self.capacity = 0
        if mode not in ("collective", "symmetric"):
            raise ValueError("transport must be 'collective' or 'symmetric'")
        if mode == "symmetric" and self.size > 1:
            if device.type != "cuda":
                raise ValueError("symmetric transport requires PPU/CUDA peer memory")
            capacity = torch.tensor(
                max(max(input_splits), max(output_splits), 1), device=device, dtype=torch.int64
            )
            dist.all_reduce(capacity, op=dist.ReduceOp.MAX, group=group)
            self.capacity = int(capacity.item())

    def exchange(self, tensor, input_splits, output_splits):
        if self.size == 1:
            return tensor
        tail = tuple(tensor.shape[1:])
        if self.mode == "collective":
            output = tensor.new_empty((sum(output_splits),) + tail)
            dist.all_to_all_single(
                output, tensor.contiguous(), output_splits, input_splits, group=self.group
            )
            return output

        # Same PPU symmetric allocation/peer-pointer foundation as MegaMoE.
        # Explicit completion barriers keep this initial transport independent
        # of NVIDIA multicast and of unproven peer atomic publication contracts.
        from torch.distributed import _symmetric_memory as sm

        key = (tensor.dtype, tail)
        if key not in self.buffers:
            shape = (self.size, self.capacity) + tail
            buffer = sm.empty(shape, dtype=tensor.dtype, device=self.device)
            handle = sm.rendezvous(buffer, self.group)
            peers = [handle.get_buffer(rank, shape, tensor.dtype) for rank in range(self.size)]
            self.buffers[key] = (buffer, handle, peers)
        buffer, handle, peers = self.buffers[key]
        offset = 0
        for peer, count in enumerate(input_splits):
            if count:
                peers[peer][self.rank, :count].copy_(tensor[offset:offset + count])
            offset += count
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        parts = [buffer[peer, :count] for peer, count in enumerate(output_splits) if count]
        output = torch.cat(parts, dim=0) if parts else tensor.new_empty((0,) + tail)
        # A later exchange may overwrite the receive buffer. Complete all reads
        # before any rank is allowed to publish the next payload.
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return output


@dataclass
class _GroupedLayout:
    permutation: torch.Tensor
    padded_rows: torch.Tensor
    group_ids: torch.Tensor
    lengths: tuple
    k_layout: torch.Tensor


@dataclass
class _ExpertContext:
    rows: tuple
    counts: tuple
    x: Optional[torch.Tensor]
    gate_up: Optional[torch.Tensor]
    hidden: Optional[torch.Tensor]
    grouped: Optional[_GroupedLayout]


@dataclass
class MoEContext:
    """One forward's owned state; keep alive until backward completes.

    Operations on the same EP group must be serialized in matching rank order.
    Parameters, inputs, routing IDs and scores must not change before backward.
    """

    x: torch.Tensor
    scores: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    positions: torch.Tensor
    source_rows: torch.Tensor
    source_expert_outputs: torch.Tensor
    input_splits: tuple
    output_splits: tuple
    experts: _ExpertContext
    communication: _Transport
    backend: str
    recompute: bool
    versions: tuple


def _validate(x, ids, scores, w1, w2, group, backend):
    if backend not in ("deep_gemm", "deep_gemm_fp8", "torch"):
        raise ValueError("backend must be 'deep_gemm', 'deep_gemm_fp8' or 'torch'")
    if x.ndim != 2 or ids.ndim != 2 or scores.shape != ids.shape:
        raise ValueError("expected x[T,H], expert_ids[T,K] and scores[T,K]")
    if ids.shape[0] != x.shape[0] or ids.shape[1] < 1:
        raise ValueError("routing rows must match x, and top-k must be positive")
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert IDs must be int32 or int64")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] < 1:
        raise ValueError("expected w1[E_local,2I,H] and w2[E_local,H,I]")
    experts, two_i, hidden = w1.shape
    if (two_i <= 0 or hidden <= 0 or two_i % 2 or hidden != x.shape[1]
            or w2.shape != (experts, hidden, two_i // 2)):
        raise ValueError("incompatible expert weight shapes")
    if any(t.device != x.device for t in (ids, scores, w1, w2)):
        raise ValueError("all inputs must be on the same device")
    if any(t.dtype != x.dtype for t in (w1, w2)) or not x.is_floating_point():
        raise TypeError("x and expert weights must have the same floating dtype")
    if not scores.is_floating_point():
        raise TypeError("routing scores must be floating point")
    if any(not t.is_contiguous() for t in (x, ids, scores, w1, w2)):
        raise ValueError("inputs must be contiguous")
    if group is not None and not dist.is_initialized():
        raise RuntimeError("initialize torch.distributed before supplying an EP group")
    size = dist.get_world_size(group) if group is not None else 1
    # Validation and route scheduling intentionally synchronize in this initial
    # backend. Do not silently advertise it as graph-capturable.
    if x.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("the initial BF16 training backend does not support CUDA Graph capture")
    if bool(((ids < -1) | (ids >= experts * size)).any().item()):
        raise ValueError("expert ID outside [-1, global_experts); -1 denotes a masked slot")
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("routing scores must be finite")
    if backend != "torch":
        if x.device.type != "cuda" or x.dtype != torch.bfloat16:
            raise TypeError("native PPU training requires BF16 device inputs and weights")
        if hidden % 256 or (two_i // 2) % 128:
            raise ValueError("native BF16 training requires H divisible by 256 and I by 128")


def _gemm_nt(a, b, backend):
    if not a.shape[0]:
        return a.new_empty((0, b.shape[0]))
    if backend == "torch":
        return a @ b.t()
    import deep_gemm

    output = a.new_empty((a.shape[0], b.shape[0]))
    if backend == "deep_gemm_fp8":
        # Current-tensor E4M3 scaling supported by PPU's native NT GEMM.
        # Transposed dgrad weights are quantized after transposition; a packed
        # forward weight/scale layout cannot be reused as a transpose.
        lhs = deep_gemm.per_token_cast_to_fp8(a.contiguous(), use_ue8m0=False)
        rhs = deep_gemm.per_block_cast_to_fp8(b.contiguous(), use_ue8m0=False)
        deep_gemm.fp8_gemm_nt(lhs, rhs, output, disable_ue8m0_cast=True)
    else:
        deep_gemm.bf16_gemm_nt(a.contiguous(), b.contiguous(), output)
    return output


def _acc_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _grouped_layout(rows, counts, device):
    import deep_gemm

    alignment = max(128, deep_gemm.get_m_alignment_for_contiguous_layout())
    lengths = [((count + alignment - 1) // alignment) * alignment for count in counts]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    permutation = torch.cat(rows)
    padded_rows = torch.cat([torch.arange(count, device=device) + offset
                             for count, offset in zip(counts, offsets)])
    group_ids = torch.repeat_interleave(
        torch.arange(len(rows), device=device, dtype=torch.int32),
        torch.tensor(lengths, device=device), output_size=offsets[-1],
    )
    return _GroupedLayout(permutation, padded_rows, group_ids, tuple(lengths),
                          torch.tensor(lengths, device=device, dtype=torch.int32))


def _pack_experts(a, layout):
    padded = a.new_zeros((layout.group_ids.numel(), a.shape[1]))
    padded.index_copy_(0, layout.padded_rows, a.index_select(0, layout.permutation))
    return padded


def _expert_gemm_grouped(a, w, layout):
    import deep_gemm

    padded = _pack_experts(a, layout)
    output = a.new_empty((padded.shape[0], w.shape[1]))
    if a.shape[0]:
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            padded, w.contiguous(), output, layout.group_ids
        )
    result = a.new_empty((a.shape[0], w.shape[1]))
    result.index_copy_(0, layout.permutation, output.index_select(0, layout.padded_rows))
    return result


def _expert_forward(x, w1, w2, rows, backend, need_output=True, grouped=None):
    if grouped is not None:
        gu = _expert_gemm_grouped(x, w1, grouped)
        gate, up = gu.float().chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate) * up).to(x.dtype)
        output = _expert_gemm_grouped(activated, w2, grouped) if need_output else None
        return gu, activated, output
    gate_up = x.new_empty((x.shape[0], w1.shape[1]))
    hidden = x.new_empty((x.shape[0], w2.shape[2]))
    output = x.new_empty((x.shape[0], w2.shape[1])) if need_output else None
    for expert, indices in enumerate(rows):
        if not indices.numel():
            continue
        gu = _gemm_nt(x.index_select(0, indices), w1[expert], backend)
        gate, up = gu.to(_acc_dtype(gu)).chunk(2, dim=-1)
        activated = (torch.nn.functional.silu(gate) * up).to(x.dtype)
        gate_up.index_copy_(0, indices, gu)
        hidden.index_copy_(0, indices, activated)
        if need_output:
            output.index_copy_(0, indices, _gemm_nt(activated, w2[expert], backend))
    return gate_up, hidden, output


def _weight_grad(a, b, rows, counts, backend, out_shape, accumulator=None, grouped=None):
    # Each expert's reduction dimension is its own token count. The native
    # primitive consumes zero-padded, physically contiguous K-major groups.
    output = (torch.zeros(out_shape, device=a.device, dtype=_acc_dtype(a))
              if accumulator is None else accumulator)
    if not a.shape[0]:
        return output
    if backend == "torch":
        for expert, indices in enumerate(rows):
            if indices.numel():
                output[expert].add_(a[indices].to(output.dtype).t() @ b[indices].to(output.dtype))
        return output
    import deep_gemm

    if grouped is not None:
        deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
            _pack_experts(a, grouped), _pack_experts(b, grouped), output,
            list(grouped.lengths), grouped.k_layout, output,
        )
        return output
    lengths = [((count + 127) // 128) * 128 for count in counts]
    total = sum(lengths)
    padded_a = a.new_zeros((total, a.shape[1]))
    padded_b = b.new_zeros((total, b.shape[1]))
    offset = 0
    for indices, count, length in zip(rows, counts, lengths):
        if count:
            padded_a[offset:offset + count].copy_(a.index_select(0, indices))
            padded_b[offset:offset + count].copy_(b.index_select(0, indices))
        offset += length
    layout = torch.tensor(lengths, device=a.device, dtype=torch.int32)
    deep_gemm.k_grouped_bf16_gemm_tn_contiguous(
        padded_a, padded_b, output, lengths, layout, output
    )
    return output


@torch.no_grad()
def bf16_moe_forward(
    x, expert_ids, scores, w1, w2, *, group=None, backend="deep_gemm",
    transport="collective", recompute=False, grouped_gemm=False,
):
    """Return ``(y, context)`` for routed experts; shared experts stay external.

    Weights are ``w1[E_local,2I,H]`` in [gate|up] order and ``w2[E_local,H,I]``.
    Expert ownership is contiguous by EP rank. Scores are applied once, after
    FC2, in FP32 (FP64 for the CPU double reference). Masked IDs -1 contribute
    nothing. A valid route with score zero still has a router-score gradient.
    """
    _validate(x, expert_ids, scores, w1, w2, group, backend)
    if grouped_gemm and backend != "deep_gemm":
        raise ValueError("grouped_gemm currently supports the native BF16 backend only")
    size = dist.get_world_size(group) if group is not None else 1
    local_experts = w1.shape[0]
    flat_ids = expert_ids.reshape(-1)
    positions = torch.nonzero(flat_ids >= 0, as_tuple=False).flatten()
    selected = flat_ids.index_select(0, positions).long()
    destinations = selected // local_experts
    order = torch.argsort(destinations, stable=True)
    positions = positions.index_select(0, order)
    selected = selected.index_select(0, order)
    source_rows = positions // expert_ids.shape[1]
    counts = torch.bincount(destinations, minlength=size).to(torch.int64)
    input_splits = tuple(counts.cpu().tolist())
    if size > 1:
        received_counts = torch.empty_like(counts)
        dist.all_to_all_single(received_counts, counts, group=group)
        output_splits = tuple(received_counts.cpu().tolist())
    else:
        output_splits = input_splits
    communication = _Transport(group, transport, input_splits, output_splits, x.device)
    received_x = communication.exchange(x.index_select(0, source_rows), input_splits, output_splits)
    received_experts = communication.exchange(
        (selected % local_experts).contiguous(), input_splits, output_splits
    )
    rows = tuple(torch.nonzero(received_experts == e, as_tuple=False).flatten() for e in range(local_experts))
    expert_counts = tuple(indices.numel() for indices in rows)
    grouped = _grouped_layout(rows, expert_counts, x.device) if grouped_gemm else None
    gate_up, hidden, expert_output = _expert_forward(received_x, w1, w2, rows, backend, grouped=grouped)
    source_output = communication.exchange(expert_output, output_splits, input_splits)
    result = torch.zeros_like(x, dtype=_acc_dtype(x))
    weights = scores.reshape(-1).index_select(0, positions).to(result.dtype)
    result.index_add_(0, source_rows, source_output.to(result.dtype) * weights[:, None])
    expert_context = _ExpertContext(
        rows, expert_counts,
        None if recompute else received_x,
        None if recompute else gate_up,
        None if recompute else hidden,
        grouped,
    )
    versions = tuple((tensor, tensor._version) for tensor in (x, expert_ids, scores, w1, w2))
    context = MoEContext(
        x, scores, w1, w2, positions, source_rows, source_output,
        input_splits, output_splits, expert_context, communication, backend, recompute, versions,
    )
    return result.to(x.dtype), context


def _validate_accumulators(accumulators, w1, w2):
    if accumulators is None:
        return
    if len(accumulators) != 2:
        raise ValueError("weight_grad_accumulators must contain dW1 and dW2")
    for accumulator, weight in zip(accumulators, (w1, w2)):
        if (accumulator.shape != weight.shape or accumulator.device != weight.device
                or accumulator.dtype != torch.float32 or not accumulator.is_contiguous()
                or accumulator.requires_grad):
            raise ValueError("weight-gradient accumulators must be contiguous FP32 buffers matching the weights")

    def overlaps(a, b):
        if not a.numel() or not b.numel():
            return False
        # Accumulators are contiguous; the span also conservatively covers a
        # strided reference weight without reading any device values.
        def interval(tensor):
            span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
            start = tensor.data_ptr()
            return start, start + span * tensor.element_size()
        a_begin, a_end = interval(a)
        b_begin, b_end = interval(b)
        return a_begin < b_end and b_begin < a_end

    if overlaps(accumulators[0], accumulators[1]) or any(
            overlaps(accumulator, weight) for accumulator in accumulators for weight in (w1, w2)):
        raise ValueError("weight-gradient accumulators must not overlap each other or expert weights")


def _swiglu_backward(gate_up, dh, dtype):
    acc_dtype = _acc_dtype(gate_up)
    gate, up = gate_up.to(acc_dtype).chunk(2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    silu = gate * sigmoid
    dh = dh.to(acc_dtype)
    return torch.cat((dh * up * sigmoid * (1 + gate * (1 - sigmoid)), dh * silu), dim=-1).to(dtype)


@torch.no_grad()
def bf16_moe_backward(dy, context, *, weight_grad_accumulators=None):
    """Return ``dx, dw1, dw2, dscores``; native weight gradients are FP32.

    The recompute path re-dispatches x and rebuilds only FC1/SwiGLU context.
    It does not replay FC2 or combine. This initial implementation retains
    source-side expert outputs to compute score gradients without division by
    a possibly-zero routing score.
    """
    c = context
    _validate_accumulators(weight_grad_accumulators, c.w1, c.w2)
    if dy.shape != c.x.shape or dy.device != c.x.device or dy.dtype != c.x.dtype:
        raise ValueError("dy must match the forward input's shape, device and dtype")
    if any(tensor._version != version for tensor, version in c.versions):
        raise RuntimeError("MoE inputs or parameters changed before backward")
    dy = dy.contiguous()
    expert = c.experts
    received_x, gate_up, hidden = expert.x, expert.gate_up, expert.hidden
    if c.recompute:
        received_x = c.communication.exchange(
            c.x.index_select(0, c.source_rows), c.input_splits, c.output_splits
        )
        gate_up, hidden, _ = _expert_forward(
            received_x, c.w1, c.w2, expert.rows, c.backend, need_output=False,
            grouped=expert.grouped,
        )
    acc_dtype = _acc_dtype(c.x)
    source_dy = dy.index_select(0, c.source_rows).to(acc_dtype)
    source_scores = c.scores.reshape(-1).index_select(0, c.positions).to(acc_dtype)
    weighted_dy = (source_dy * source_scores[:, None]).to(c.x.dtype)
    received_dy = c.communication.exchange(weighted_dy, c.input_splits, c.output_splits)
    d_gate_up = torch.empty_like(gate_up)
    received_dx = torch.empty_like(received_x)
    if expert.grouped is not None:
        dh = _expert_gemm_grouped(received_dy, c.w2.transpose(1, 2).contiguous(), expert.grouped)
        d_gate_up = _swiglu_backward(gate_up, dh, c.x.dtype)
        received_dx = _expert_gemm_grouped(
            d_gate_up, c.w1.transpose(1, 2).contiguous(), expert.grouped
        )
    else:
        for e, indices in enumerate(expert.rows):
            if not indices.numel():
                continue
            # Explicit contiguous NT operands avoid depending on missing native NN
            # aliases or on an inference BLAS library's cold transposed JIT path.
            dh = _gemm_nt(received_dy[indices], c.w2[e].t().contiguous(), c.backend)
            dgu = _swiglu_backward(gate_up[indices], dh, c.x.dtype)
            d_gate_up.index_copy_(0, indices, dgu)
            received_dx.index_copy_(
                0, indices, _gemm_nt(dgu, c.w1[e].t().contiguous(), c.backend)
            )
    accum1, accum2 = (None, None) if weight_grad_accumulators is None else weight_grad_accumulators
    dw1 = _weight_grad(d_gate_up, received_x, expert.rows, expert.counts, c.backend, c.w1.shape,
                       accum1, expert.grouped)
    dw2 = _weight_grad(received_dy, hidden, expert.rows, expert.counts, c.backend, c.w2.shape,
                       accum2, expert.grouped)
    source_dx = c.communication.exchange(received_dx, c.output_splits, c.input_splits)
    dx = torch.zeros_like(c.x, dtype=acc_dtype)
    dx.index_add_(0, c.source_rows, source_dx.to(acc_dtype))
    dscores = torch.zeros_like(c.scores, dtype=acc_dtype)
    route_grad = (source_dy * c.source_expert_outputs.to(acc_dtype)).sum(dim=-1)
    dscores.reshape(-1).index_copy_(0, c.positions, route_grad)
    return dx.to(c.x.dtype), dw1, dw2, dscores.to(c.scores.dtype)


class _BF16MoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ids, scores, w1, w2, group, backend, transport, recompute, accumulators, grouped_gemm):
        _validate_accumulators(accumulators, w1, w2)
        y, state = bf16_moe_forward(
            x, ids, scores, w1, w2, group=group, backend=backend, transport=transport,
            recompute=recompute, grouped_gemm=grouped_gemm,
        )
        ctx.accumulators = accumulators
        ctx.communication = state.communication
        expert = state.experts
        ctx.moe_metadata = (state.input_splits, state.output_splits, expert.counts,
                        backend, recompute, None if expert.grouped is None else expert.grouped.lengths)
        tensors = [x, ids, scores, w1, w2, state.positions, state.source_rows,
                   state.source_expert_outputs, expert.x, expert.gate_up, expert.hidden, *expert.rows]
        if expert.grouped is not None:
            tensors.extend((expert.grouped.permutation, expert.grouped.padded_rows,
                            expert.grouped.group_ids, expert.grouped.k_layout))
        # Tensor state belongs to autograd so backward releases it even if the
        # caller retains y. Raw tensor references on ctx would defeat saved-
        # tensor hooks and keep all MoE activations alive after backward.
        ctx.save_for_backward(*tensors)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        saved = ctx.saved_tensors
        x, ids, scores, w1, w2, positions, source_rows, source_output, rx, gu, hidden = saved[:11]
        input_splits, output_splits, counts, backend, recompute, lengths = ctx.moe_metadata
        rows = tuple(saved[11:11 + len(counts)])
        grouped = None
        if lengths is not None:
            permutation, padded_rows, group_ids, k_layout = saved[11 + len(counts):]
            grouped = _GroupedLayout(permutation, padded_rows, group_ids, lengths, k_layout)
        state = MoEContext(
            x, scores, w1, w2, positions, source_rows, source_output, input_splits, output_splits,
            _ExpertContext(rows, counts, rx, gu, hidden, grouped), ctx.communication, backend, recompute, (),
        )
        dx, dw1, dw2, dscores = bf16_moe_backward(
            dy, state, weight_grad_accumulators=ctx.accumulators
        )
        # The final reverse exchange completed its peer reads before returning.
        # A retained graph may allocate fresh transport buffers on another backward.
        ctx.communication.buffers.clear()
        if ctx.accumulators is not None:
            dw1 = dw2 = None
        else:
            dw1, dw2 = dw1.to(w1.dtype), dw2.to(w2.dtype)
        return dx, None, dscores, dw1, dw2, None, None, None, None, None, None


def bf16_moe(
    x, expert_ids, scores, w1, w2, *, group=None, backend="deep_gemm",
    transport="collective", recompute=False, weight_grad_accumulators=None,
    grouped_gemm=False,
):
    """Differentiable routed MoE with caller-owned expert parameters.

    Default gradients follow parameter dtype. Optional FP32 accumulators are
    caller-owned and additive: clear them at the desired optimizer boundary.
    In that mode weight ``.grad`` is deliberately not populated; this is not
    an implicit Megatron-DDP or FSDP gradient-hook integration.

    First-order gradients only. All ranks, including empty ranks, must enter
    forward/backward collectives in the same order. Shared experts and their
    gates can be composed outside this operation.

    ``backend="deep_gemm_fp8"`` selects an experimental E4M3 forward/dgrad,
    BF16 wgrad recipe with BF16 parameters and activations. This is an
    approximate training gradient, not differentiation through FP8 rounding.
    """
    return _BF16MoE.apply(
        x, expert_ids, scores, w1, w2, group, backend, transport, recompute,
        weight_grad_accumulators, grouped_gemm,
    )
