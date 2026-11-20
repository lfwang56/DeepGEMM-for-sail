"""Native peer-read/peer-write Tensor Cell GEMMs for PPU MoE training."""

import math

import torch
import torch.distributed as dist

import deep_gemm as dg
from .bf16 import _validate_accumulators
from .native import NativeMoEPlan, _launch


def _peer(operation, tensors, dims, compile_only=False):
    dg._C._moe_training_peer_launch(operation, tensors, dims, compile_only)


class PeerMoEPlan(NativeMoEPlan):
    """Bounded BF16 training with dispatch reads and return writes inside GEMM.

    Requires a direct peer domain, including cross-host ICN within a supernode.
    PCCL/RDMA remains available through NativeMoEPlan for other topologies.
    """

    transport = "peer"

    def __init__(self, *args, **kwargs):
        if kwargs.get("compute", "bf16") != "bf16":
            raise ValueError("peer-fused GEMMs currently use BF16 compute")
        super().__init__(*args, **kwargs)
        if self.size > 64:
            raise ValueError("peer training supports at most 64 ranks per domain")
        c, h, k = self.chunk_tokens, self.h, self.k
        descriptions = (("local_x", (c, h), torch.bfloat16),
                        ("local_ids", (c, k), torch.int64),
                        ("local_values", (c * k, h), torch.bfloat16),
                        ("local_dx", (c * k, h), torch.bfloat16),
                        ("local_dy", (c * k, h), torch.bfloat16),
                        ("barrier_state", (4,), torch.int32))
        layout, size = {}, 0
        for name, shape, dtype in descriptions:
            size = (size + 15) // 16 * 16
            count = math.prod(shape)
            element = 2 if dtype == torch.bfloat16 else 8 if dtype == torch.int64 else 4
            layout[name] = (size, count * element, shape, dtype)
            size += count * element
        self.symm_handle = None
        if self.size == 1:
            self.payload = torch.empty(size, device=self.device, dtype=torch.uint8)
            self.peer_buffers = [self.payload]
        else:
            from torch.distributed import _symmetric_memory as sm

            self.payload = sm.empty((size,), dtype=torch.uint8, device=self.device)
            self.symm_handle = sm.rendezvous(self.payload, self.group)
            direct = getattr(self.symm_handle, "world_within_direct_access", True)
            direct = direct() if callable(direct) else direct
            if not direct:
                raise RuntimeError("peer training requires an EP group within one direct-access domain")
            self.peer_buffers = [self.symm_handle.get_buffer(rank, (size,), torch.uint8) for rank in range(self.size)]
        self.payload.zero_()
        self.pointer_tables = {}
        for name, (offset, length, shape, dtype) in layout.items():
            view = self.payload[offset:offset + length].view(dtype).reshape(shape)
            setattr(self, name, view)
            self.buffers[name] = view
            addresses = [buffer.data_ptr() + offset for buffer in self.peer_buffers]
            encoded = [value if value < 2**63 else value - 2**64 for value in addresses]
            table = torch.tensor(encoded, device=self.device, dtype=torch.int64)
            self.pointer_tables[name] = table
            self.buffers[name + "_peers"] = table
        for name in ("gathered_x", "gathered_ids", "route_values", "group_ids"):
            self.buffers.pop(name)
            delattr(self, name)
        self.source_rows = torch.empty(self.rows, device=self.device, dtype=torch.int32)
        self.identity_rows = torch.arange(self.rows, device=self.device, dtype=torch.int32)
        self.work_info = torch.empty((self.rows // 128, 4), device=self.device, dtype=torch.int32)
        self.num_blocks = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.buffers.update(source_rows=self.source_rows, identity_rows=self.identity_rows,
                            work_info=self.work_info, num_blocks=self.num_blocks)
        self.gate_up.zero_()
        self.hidden.zero_()
        self._prepared = False
        torch.cuda.synchronize(self.device)
        if self.group is not None:
            dist.barrier(group=self.group)

    def _barrier(self, tag, compile_only=False):
        _peer(5, [self.pointer_tables["barrier_state"], self.barrier_state, self.status],
              [self.size, self.rank, 180 * 2_000_000_000, tag], compile_only)

    def _peer_gemm(self, a, weight, output, source_rows, *, read=False, write=False,
                   source="local_x", destination="local_values", compile_only=False):
        source_stride = self.chunk_tokens if source == "local_x" else self.chunk_tokens * self.k
        dg._C._moe_training_peer_gemm(
            a, weight, output, self.pointer_tables[source], self.pointer_tables[destination],
            source_rows, self.permutation, self.work_info, self.num_blocks,
            source_stride, self.chunk_tokens * self.k, read, write, compile_only,
        )

    @torch.no_grad()
    def prepare(self, w1, w2):
        """Compile all peer GEMM variants before entering device peer barriers."""
        with self._execution():
            if not self._prepared:
                self._prepare(w1, w2)

    def _prepare(self, w1, w2):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up PeerMoEPlan before graph capture")
        w1, w2 = w1.to(torch.bfloat16), w2.to(torch.bfloat16)
        self._prepare_weights(w1, w2, backward=True)
        self._barrier(0, compile_only=True)
        _peer(0, [self.pointer_tables["local_ids"], self.counts, self.route_experts, self.route_rows, self.status],
              [self.routes, self.e, self.rank, self.size, self.chunk_tokens * self.k], True)
        _peer(1, [self.permutation, self.source_rows], [self.rows, self.size * self.chunk_tokens], True)
        _peer(2, [self.route_experts, self.route_rows, self.offsets, self.permutation, self.source_rows, self.status],
              [self.routes, self.rows, self.k], True)
        _peer(3, [self.offsets, self.lengths, self.work_info, self.num_blocks], [self.e, self.rows, 128], True)
        _peer(4, [self.pointer_tables["local_x"], self.permutation, self.packed_x, self.status],
              [self.rows, self.h, self.chunk_tokens, self.k, self.size], True)
        self._peer_gemm(self.packed_x, w1, self.gate_up, self.source_rows, read=True, compile_only=True)
        self._peer_gemm(self.hidden, w2, self.packed_output, self.identity_rows, write=True, compile_only=True)
        self._peer_gemm(self.packed_dy, self.w2_t, self.hidden, self.permutation,
                        read=True, source="local_dy", compile_only=True)
        self._peer_gemm(self.gate_up, self.w1_t, self.packed_output, self.identity_rows,
                        write=True, destination="local_dx", compile_only=True)
        self.lengths.zero_()
        dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x,
                                       torch.zeros_like(w1, dtype=torch.float32), self.lengths, self.rows)
        dg.k_grouped_bf16_gemm_tn_device(self.packed_dy, self.hidden,
                                       torch.zeros_like(w2, dtype=torch.float32), self.lengths, self.rows)
        torch.cuda.synchronize(self.device)
        if self.group is not None:
            dist.barrier(group=self.group)
        self._prepared = True

    def _route_chunk(self, x, ids, valid_tokens, start):
        _launch(13, [x, ids, self.local_x, self.local_ids, valid_tokens, self.status],
                [x.shape[0], self.chunk_tokens, self.h, self.k, start])
        self.local_values.zero_()
        self.local_dx.zero_()
        self._barrier(start * 4)
        self.counts.zero_()
        _peer(0, [self.pointer_tables["local_ids"], self.counts, self.route_experts, self.route_rows, self.status],
              [self.routes, self.e, self.rank, self.size, self.chunk_tokens * self.k])
        _launch(1, [self.counts, self.offsets, self.lengths], [self.e, self.alignment])
        _peer(1, [self.permutation, self.source_rows], [self.rows, self.size * self.chunk_tokens])
        _peer(2, [self.route_experts, self.route_rows, self.offsets, self.permutation, self.source_rows, self.status],
              [self.routes, self.rows, self.k])
        _peer(3, [self.offsets, self.lengths, self.work_info, self.num_blocks], [self.e, self.rows, 128])

    def _chunk_forward(self, w1, w2):
        self._peer_gemm(self.packed_x, w1, self.gate_up, self.source_rows, read=True)
        _launch(7, [self.gate_up, self.hidden], [self.rows, self.i])
        self._peer_gemm(self.hidden, w2, self.packed_output, self.identity_rows, write=True)
        self._barrier(1)

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens):
        with self._execution():
            self._validate(x, ids, scores, w1, w2, valid_tokens)
            self.prepare(w1, w2)
            return super().forward(x, ids, scores, w1, w2, valid_tokens)

    @torch.no_grad()
    def backward(self, dy, x, ids, scores, w1, w2, valid_tokens, accumulators=None):
        _validate_accumulators(accumulators, w1, w2)
        if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype:
            raise ValueError("dy must match x")
        with self._execution():
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            self._prepare_weights(w1, w2, backward=True)
            dx, dscores = torch.empty_like(x), torch.empty_like(scores)
            dw1, dw2 = ((torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
                        if accumulators is None else accumulators)
            for start in range(0, self.max_tokens, self.chunk_tokens):
                self._route_chunk(x, ids, valid_tokens, start)
                self._chunk_forward(w1, w2)
                dims = [self.chunk_tokens, self.h, self.k, x.shape[0], start]
                _launch(12, [self.local_values, dy, dscores], dims)
                _launch(11, [dy, scores, self.local_dy], dims)
                self._barrier(start * 4 + 2)
                _peer(4, [self.pointer_tables["local_x"], self.permutation, self.packed_x, self.status],
                      [self.rows, self.h, self.chunk_tokens, self.k, self.size])
                _peer(4, [self.pointer_tables["local_dy"], self.permutation, self.packed_dy, self.status],
                      [self.rows, self.h, self.chunk_tokens * self.k, 1, self.size])
                dg.k_grouped_bf16_gemm_tn_device(self.packed_dy, self.hidden, dw2, self.lengths, self.rows)
                self._peer_gemm(self.packed_dy, self.w2_t, self.hidden, self.permutation, read=True, source="local_dy")
                _launch(8, [self.gate_up, self.hidden, self.gate_up], [self.rows, self.i])
                dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x, dw1, self.lengths, self.rows)
                self._peer_gemm(self.gate_up, self.w1_t, self.packed_output, self.identity_rows,
                                write=True, destination="local_dx")
                self._barrier(start * 4 + 3)
                _launch(6, [self.local_dx, scores, dx], dims + [0])
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return dx, dw1, dw2, dscores
