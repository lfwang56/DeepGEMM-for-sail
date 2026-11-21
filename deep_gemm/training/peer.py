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

    def __init__(self, *args, gemm_configs=None, fuse_dispatch=True, fuse_return=True,
                 active_rows=True, fuse_local_grad=True, fuse_activation_backward=False,
                 vectorized_copy=True, dispatch_cache_n=0, reuse_packed_dy=False,
                 dispatch_cache_aiu=False, staged_return=False, dispatch_cache_ctas=2,
                 dispatch_producer=False, dispatch_producer_tiles=1, producer_local_aiu=False, **kwargs):
        if kwargs.get("compute", "bf16") != "bf16":
            raise ValueError("peer-fused GEMMs currently use BF16 compute")
        if not isinstance(fuse_dispatch, bool) or not isinstance(fuse_return, bool):
            raise TypeError("peer fusion switches must be bool")
        self.fuse_dispatch, self.fuse_return = fuse_dispatch, fuse_return
        if not isinstance(dispatch_producer, bool):
            raise TypeError("dispatch_producer must be bool")
        if dispatch_producer and (not fuse_dispatch or self.transport != "peer" or dispatch_cache_n):
            raise ValueError("producer dispatch requires fused peer dispatch without a private N cache")
        if type(dispatch_producer_tiles) is not int or dispatch_producer_tiles not in (1, 2, 4):
            raise ValueError("dispatch_producer_tiles must be 1, 2 or 4")
        self.dispatch_producer, self.dispatch_producer_tiles = dispatch_producer, dispatch_producer_tiles
        if not isinstance(producer_local_aiu, bool):
            raise TypeError("producer_local_aiu must be bool")
        if producer_local_aiu and not dispatch_producer:
            raise ValueError("local AIU input requires producer dispatch")
        self.producer_local_aiu = producer_local_aiu
        if not isinstance(active_rows, bool) or not isinstance(fuse_local_grad, bool):
            raise TypeError("peer pointwise switches must be bool")
        self.active_rows, self.fuse_local_grad = active_rows, fuse_local_grad
        if not isinstance(fuse_activation_backward, bool):
            raise TypeError("fuse_activation_backward must be bool")
        self.fuse_activation_backward = fuse_activation_backward
        if not isinstance(vectorized_copy, bool):
            raise TypeError("vectorized_copy must be bool")
        self.vectorized_copy = vectorized_copy
        if type(dispatch_cache_n) is not int or dispatch_cache_n not in (0, 2, 4, 8, 16):
            raise ValueError("dispatch_cache_n must be 0, 2, 4, 8 or 16")
        if dispatch_cache_n and (not fuse_dispatch or self.transport != "peer"):
            raise ValueError("dispatch caching requires the peer fused-dispatch path")
        self.dispatch_cache_n = dispatch_cache_n
        if type(dispatch_cache_ctas) is not int or dispatch_cache_ctas not in (1, 2, 4):
            raise ValueError("dispatch_cache_ctas must be 1, 2 or 4")
        self.dispatch_cache_ctas = dispatch_cache_ctas
        if not isinstance(dispatch_cache_aiu, bool):
            raise TypeError("dispatch_cache_aiu must be bool")
        if dispatch_cache_aiu and not dispatch_cache_n:
            raise ValueError("AIU cache reads require dispatch_cache_n > 0")
        self.dispatch_cache_aiu = dispatch_cache_aiu
        if not isinstance(staged_return, bool):
            raise TypeError("staged_return must be bool")
        self.staged_return = staged_return and fuse_return
        self.staged_return_row_padding = 16 if self.staged_return else 0
        if not isinstance(reuse_packed_dy, bool):
            raise TypeError("reuse_packed_dy must be bool")
        if reuse_packed_dy and self.transport != "peer":
            raise ValueError("packed gradient reuse is a peer-plan option")
        self.reuse_packed_dy = reuse_packed_dy and fuse_dispatch
        super().__init__(*args, **kwargs)
        self.gemm_configs = {} if gemm_configs is None else {name: tuple(value) for name, value in gemm_configs.items()}
        if set(self.gemm_configs) - {"fc1", "fc2", "dhidden", "dx"}:
            raise ValueError("unknown peer GEMM configuration role")
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
            direct = getattr(self.symm_handle, "world_within_direct_access", None)
            direct = direct() if callable(direct) else direct
            # The native PPU Torch 2.10 CUDA backend imports/maps every peer
            # allocation through the driver, but omits the newer topology flag.
            # Accept that direct-buffer backend after successful rendezvous;
            # do not infer direct pointers for an unknown/RDMA memory backend.
            if direct is None and sm.get_backend(self.device) != "CUDA":
                raise RuntimeError("symmetric-memory backend cannot supply verified direct buffers; use NativeMoEPlan")
            if direct is not None and not direct:
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
        self.dispatch_scratch = None
        if self.dispatch_cache_n:
            cus = torch.cuda.get_device_properties(self.device).multi_processor_count
            self.dispatch_scratch = torch.empty((cus * self.dispatch_cache_ctas * 128, self.h),
                                                device=self.device, dtype=torch.bfloat16)
            self.buffers["dispatch_scratch"] = self.dispatch_scratch
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
                   source="local_x", destination="local_values", compile_only=False, pack_input=False, role=None,
                   first_n_tile=0, num_n_tiles=0, local_input_aiu=False):
        source_stride = self.chunk_tokens if source == "local_x" else self.chunk_tokens * self.k
        if role is None:
            role = ("fc1" if source == "local_x" else "dhidden") if read else ("dx" if destination == "local_dx" else "fc2")
        dg._C._moe_training_peer_gemm(
            a, weight, output, self.pointer_tables[source], self.pointer_tables[destination],
            source_rows, self.permutation, self.work_info, self.num_blocks,
            source_stride, self.chunk_tokens * self.k, read, write, compile_only,
            pack_input, self.gemm_configs.get(role, ()),
            self.gate_up if role == "dhidden" and self.fuse_activation_backward else None,
            self.dispatch_cache_n if read else 0, self.dispatch_scratch if read else None,
            self.dispatch_cache_aiu if read else False,
            self.staged_return if write else False,
            self.dispatch_cache_ctas,
            first_n_tile, num_n_tiles, local_input_aiu,
        )

    def _fc1(self, w1, *, pack_input=False, compile_only=False):
        if self.dispatch_producer:
            bn = self.gemm_configs.get("fc1", (128, 64, 64, 64, 3))[0]
            ntiles = (2 * self.i + bn - 1) // bn
            produced = min(ntiles, self.dispatch_producer_tiles)
            # The first GEMM slice dispatches X and computes real FC1 columns.
            # Stream ordering then publishes packed X to a local GEMM over
            # the remaining columns. No cross-CTA wait or dispatch-only kernel.
            self._peer_gemm(self.packed_x, w1, self.gate_up, self.source_rows,
                            read=True, pack_input=True, role="fc1", compile_only=compile_only,
                            first_n_tile=0, num_n_tiles=produced)
            if produced < ntiles:
                self._peer_gemm(self.packed_x, w1, self.gate_up, self.identity_rows,
                                role="fc1", compile_only=compile_only, first_n_tile=produced,
                                local_input_aiu=self.producer_local_aiu)
        else:
            self._peer_gemm(self.packed_x, w1, self.gate_up,
                            self.source_rows if self.fuse_dispatch else self.identity_rows,
                            read=self.fuse_dispatch, pack_input=pack_input and self.fuse_dispatch,
                            role="fc1", compile_only=compile_only)

    @torch.no_grad()
    def prepare(self, w1, w2):
        """Compile all peer GEMM variants before entering device peer barriers."""
        with self._execution():
            if not self._prepared:
                error = None
                try:
                    self._prepare(w1, w2)
                except Exception as exception:
                    error = exception
                self._finish_preparation(error)

    def _prepare(self, w1, w2):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up PeerMoEPlan before graph capture")
        self._prepare_pointwise()
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
        if not self.fuse_dispatch:
            self._stage_input(compile_only=True)
        if self.active_rows:
            self._activation(compile_only=True)
            self._activation(backward=True, compile_only=True)
            self._pack_dy(compile_only=True)
        if self.fuse_local_grad:
            empty_dy = torch.empty((0, self.h), device=self.device, dtype=torch.bfloat16)
            empty_scores = torch.empty((0, self.k), device=self.device, dtype=torch.float32)
            self._local_gradients(empty_dy, empty_scores, empty_scores, 0, compile_only=True)
        self._fc1(w1, compile_only=True)
        self._peer_gemm(self.hidden, w2, self.packed_output, self.identity_rows,
                        write=self.fuse_return, compile_only=True)
        read_dy = self.fuse_dispatch and not self.reuse_packed_dy
        self._peer_gemm(self.packed_dy, self.w2_t, self.hidden,
                        self.permutation if read_dy else self.identity_rows,
                        read=read_dy, source="local_dy", compile_only=True, role="dhidden")
        self._peer_gemm(self.gate_up, self.w1_t, self.packed_output, self.identity_rows,
                        write=self.fuse_return, destination="local_dx", compile_only=True)
        if not self.fuse_return:
            self._scatter_peer("local_values", compile_only=True)
        self.lengths.zero_()
        dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x,
                                       torch.zeros_like(w1, dtype=torch.float32), self.lengths, self.rows)
        dg.k_grouped_bf16_gemm_tn_device(self.packed_dy, self.hidden,
                                       torch.zeros_like(w2, dtype=torch.float32), self.lengths, self.rows)
        torch.cuda.synchronize(self.device)

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

    def _chunk_forward(self, w1, w2, pack_input=False):
        if not self.fuse_dispatch:
            self._stage_input()
        self._fc1(w1, pack_input=pack_input)
        self._activation()
        self._peer_gemm(self.hidden, w2, self.packed_output, self.identity_rows, write=self.fuse_return)
        if not self.fuse_return:
            self._scatter_peer("local_values")
        self._barrier(1)

    def _scatter_peer(self, destination, compile_only=False):
        _peer(12 if self.vectorized_copy else 6,
              [self.packed_output, self.permutation, self.pointer_tables[destination], self.status, self.num_blocks],
              [self.rows, self.h, self.chunk_tokens * self.k, self.size], compile_only)

    def _stage_input(self, compile_only=False):
        _peer(11 if self.vectorized_copy else 7,
              [self.pointer_tables["local_x"], self.permutation, self.packed_x, self.status, self.num_blocks],
              [self.rows, self.h, self.chunk_tokens, self.k, self.size], compile_only)

    def _activation(self, backward=False, compile_only=False):
        if self.active_rows:
            _peer(9 if backward else 8, [self.gate_up, self.hidden, self.num_blocks, self.status],
                  [self.rows, self.i], compile_only)
        elif not compile_only:
            tensors = [self.gate_up, self.hidden] + ([self.gate_up] if backward else [])
            _launch(8 if backward else 7, tensors, [self.rows, self.i])

    def _pack_dy(self, compile_only=False):
        tensors = [self.pointer_tables["local_dy"], self.permutation, self.packed_dy, self.status]
        if self.active_rows:
            tensors.append(self.num_blocks)
        operation = (11 if self.vectorized_copy else 7) if self.active_rows else 4
        _peer(operation, tensors,
              [self.rows, self.h, self.chunk_tokens * self.k, 1, self.size], compile_only)

    def _local_gradients(self, dy, scores, dscores, start, compile_only=False):
        _peer(10, [self.local_values, dy, scores, dscores, self.local_dy],
              [self.chunk_tokens, self.h, self.k, dy.shape[0], start], compile_only)

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens):
        with self._execution():
            self._validate(x, ids, scores, w1, w2, valid_tokens)
            self.prepare(w1, w2)
            return super().forward(x, ids, scores, w1, w2, valid_tokens)

    @torch.no_grad()
    def backward(self, dy, x, ids, scores, w1, w2, valid_tokens, accumulators=None):
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        _validate_accumulators(accumulators, w1, w2)
        if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype or not dy.is_contiguous():
            raise ValueError("dy must be contiguous and match x")
        with self._execution():
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            self._prepare_weights(w1, w2, backward=True)
            dx, dscores = torch.empty_like(x), torch.empty_like(scores)
            dw1, dw2 = ((torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
                        if accumulators is None else accumulators)
            for start in range(0, self.max_tokens, self.chunk_tokens):
                self._route_chunk(x, ids, valid_tokens, start)
                self._chunk_forward(w1, w2, pack_input=True)
                dims = [self.chunk_tokens, self.h, self.k, x.shape[0], start]
                if self.fuse_local_grad:
                    self._local_gradients(dy, scores, dscores, start)
                else:
                    _launch(12, [self.local_values, dy, dscores], dims)
                    _launch(11, [dy, scores, self.local_dy], dims)
                self._barrier(start * 4 + 2)
                self._pack_dy()
                dg.k_grouped_bf16_gemm_tn_device(self.packed_dy, self.hidden, dw2, self.lengths, self.rows)
                # DW2 requires packed_dy above. Reuse that mandatory staging
                # result instead of fetching the same peer gradients again.
                read_dy = self.fuse_dispatch and not self.reuse_packed_dy
                self._peer_gemm(self.packed_dy, self.w2_t, self.hidden,
                                self.permutation if read_dy else self.identity_rows,
                                read=read_dy, source="local_dy", role="dhidden")
                if not self.fuse_activation_backward:
                    self._activation(backward=True)
                dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x, dw1, self.lengths, self.rows)
                self._peer_gemm(self.gate_up, self.w1_t, self.packed_output, self.identity_rows,
                                write=self.fuse_return, destination="local_dx")
                if not self.fuse_return:
                    self._scatter_peer("local_dx")
                self._barrier(start * 4 + 3)
                _launch(6, [self.local_dx, scores, dx], dims + [0])
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return dx, dw1, dw2, dscores
