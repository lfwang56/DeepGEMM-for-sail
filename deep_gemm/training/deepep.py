"""Installed DeepEP BF16 dispatch/combine with BF16 or FP8 grouped GEMMs.

DeepEP transports one copy per destination rank. Local GPU metadata expands
only the expert rows needed by the grouped GEMMs. All receive capacities are
bounded and all routing counts remain on device during graph replay.
"""

import os

import torch
import torch.distributed as dist

import deep_gemm as dg
from .native import NativeMoEPlan, _launch
from .routed_rows import allocate_route_windows, set_route_window


class DeepEPMoEPlan(NativeMoEPlan):
    transport = "deepep"

    def __init__(self, *args, comm_sms=8, gemm_configs=None, rdma_qps=None, ring_rows=None, elastic_hybrid=None,
                 fp8_active_quantization=None, fp8_mn_major_scales=None, **kwargs):
        def quant_flag(value, key):
            if value is None:
                raw = os.environ.get(key, "0")
                if raw not in ("0", "1"):
                    raise ValueError(key + " must be0or1")
                value = raw == "1"
            if type(value) is not bool:
                raise TypeError(key + " must be bool")
            return value
        self.fp8_active_quantization = quant_flag(fp8_active_quantization, "DG_MOE_FP8_ACTIVE_QUANTIZATION")
        self.fp8_mn_major_scales = quant_flag(fp8_mn_major_scales, "DG_MOE_FP8_MN_MAJOR_SCALES")
        if self.fp8_mn_major_scales and not self.fp8_active_quantization:
            raise ValueError("MN-major FP8 scales require active-row quantization")
        if self.fp8_active_quantization and kwargs.get("compute", "bf16") != "fp8":
            raise ValueError("active FP8 quantization requires FP8 expert compute")
        if kwargs.get("compute", "bf16") == "fp8" and kwargs.get("m_schedule", "nopad") != "nopad":
            raise ValueError("DeepEP FP8 GEMMs require the nopad expert schedule")
        if kwargs.get("group") is None:
            raise ValueError("DeepEP requires an initialized EP process group")
        import deep_ep

        if "packed_rows" in kwargs:
            raise ValueError("use ring_rows to enable routed-row scheduling")
        self.ring_rows = ring_rows
        self.communication_dtype = "bf16"
        super().__init__(*args, packed_rows=ring_rows, **kwargs)
        recipe = (self.fp8_active_quantization, self.fp8_mn_major_scales)
        recipes = [None] * self.size
        dist.all_gather_object(recipes, recipe, group=self.group)
        if any(other != recipe for other in recipes):
            raise ValueError("DeepEP ranks disagree on FP8 quantization layout")
        self.quantized_scale_storage = {}
        if self.compute == "fp8":
            for width in (self.h, self.i, 2 * self.i):
                if width in self.quantized_scale_storage:
                    continue
                if self.fp8_mn_major_scales:
                    storage = self._allocate("activation_scale_mn_" + str(width),
                                             (width // 128, self.rows), torch.float32)
                    data = self.quantized[width][0]
                    self.quantized[width] = (data, storage.t())
                    self.buffers.pop("activation_scale_" + str(width))
                else:
                    storage = self.quantized[width][1]
                self.quantized_scale_storage[width] = storage
        if ring_rows is not None and (self.k > 32 or self.size * self.chunk_tokens * self.h > 2**31 - 1):
            raise ValueError("DeepEP routed-row windows require topk<=32 and an int32-bounded receive matrix")
        # Preserve forward Hidden for concurrent DW2; dHidden uses a distinct
        # bounded ring and the existing activation-backward kernel consumes it.
        self.backward_hidden = (self._allocate("deepep_backward_hidden", (self.rows, self.i), torch.bfloat16)
                                if self.wgrad_overlap == "both" else None)
        self.active_pointwise = self.vectorized_routes = True
        self.gemm_configs = gemm_configs or {}
        if self.block_m == 256 and not self.gemm_configs:
            self.gemm_configs = {role: (256, 64, 64, 64, 2) for role in ("fc1", "fc2", "dhidden", "dx")}
        if type(comm_sms) is not int or comm_sms < 2 or comm_sms % 2:
            raise ValueError("DeepEP communication CU count must be an even integer >=2")
        self.comm_sms = comm_sms
        self.deep_ep_qps = max(12, (comm_sms + 1) // 2) if rdma_qps is None else rdma_qps
        if type(self.deep_ep_qps) is not int or self.deep_ep_qps < (comm_sms + 1) // 2:
            raise ValueError("DeepEP RDMA QPs must cover at least one pair of communication CUs")
        if elastic_hybrid is None:
            deep_ep.Buffer.set_num_sms(comm_sms)
            configs = (deep_ep.Buffer.get_dispatch_config(self.size),
                       deep_ep.Buffer.get_combine_config(self.size))
            nvl = max(c.get_nvl_buffer_size_hint(self.h * 2, self.size) for c in configs)
            rdma = max(c.get_rdma_buffer_size_hint(self.h * 2, self.size) for c in configs)
            self.deep_ep_buffer = deep_ep.Buffer(self.group, nvl, rdma,
                allow_mnnvl=True, explicitly_destroy=True, num_qps_per_rank=self.deep_ep_qps)
        else:
            if not isinstance(elastic_hybrid, bool):
                raise TypeError("elastic_hybrid must be bool or None for legacy Buffer")
            from .elastic import ElasticBufferAdapter
            self.deep_ep_buffer = ElasticBufferAdapter(self.group, self.chunk_tokens, self.h,
                self.k, self.e * self.size, comm_sms, elastic_hybrid)
        self.deep_ep_backend = self.deep_ep_buffer.backend
        self.deep_ep_rdma_ranks = self.deep_ep_buffer.runtime.get_num_rdma_ranks()
        self.deep_ep_version = deep_ep.__version__
        # Avoid the all-gather reference's [ranks*C,topk,H] temporary. DeepEP
        # unpermutes directly into its deduplicated [receive_capacity,H] layout.
        for name in ("gathered_x", "gathered_ids", "route_values", "local_values", "local_dy"):
            self.buffers.pop(name)
            delattr(self, name)
        n = self.size * self.chunk_tokens
        self.recv_capacity = n
        self.local_scores = self._allocate("deepep_local_scores", (self.chunk_tokens, self.k), torch.float32)
        self.local_dy_input = self._allocate("deepep_local_dy", (self.chunk_tokens, self.h), torch.bfloat16)
        self.reduced = self._allocate("deepep_reduced", (n, self.h), torch.bfloat16)
        self.recv_dscores = self._allocate("deepep_dscores", (n, self.k), torch.float32)
        self.route_window_count = 1
        if ring_rows is not None:
            allocate_route_windows(self)
            self.recv_accumulator = self._allocate("deepep_recv_accumulator", (n, self.h), torch.float32)

    def _allocate(self, name, shape, dtype):
        value = torch.empty(shape, device=self.device, dtype=dtype)
        self.buffers[name] = value
        return value

    def _prepare_local(self, w1, w2):
        if self.fp8_active_quantization:
            # Native warming zeros lengths but normally never reads offsets.
            # Active quantizers consume this GPU scalar even during warming.
            self.offsets.zero_()
        return super()._prepare_local(w1, w2)

    def _activation(self, backward=False, hidden=None):
        if not (self.fp8_active_quantization and self.fuse_activation_quant):
            return super()._activation(backward=backward, hidden=hidden)
        hidden = self.hidden if hidden is None else hidden
        width = 2 * self.i if backward else self.i
        tensors = [self.gate_up, hidden] + ([self.gate_up] if backward else [])
        tensors.extend((self.quantized[width][0], self.quantized_scale_storage[width],
                        self.current_rows, self.status))
        _launch(28 if backward else 27, tensors,
                [self.rows, self.i, int(self.fp8_mn_major_scales)])
        return True

    def _prepare_pointwise(self):
        operations = [0, 1, 2, 7, 8, 13, 16, 19, 20, 23, 24]
        if self.compute == "fp8":
            operations.extend((9, 10, 14, 15))
            if self.fp8_active_quantization:
                operations.extend((26, 27, 28))
        if self.fast_weight_transpose:
            operations.append(22)
        if self.ring_rows is not None:
            operations.append(25)
            set_route_window(self, 0, compile_only=True)
        dg._C._moe_training_precompile(self.device.index, operations)

    def _gemm(self, a, weight, output, name, prequantized=False):
        role = {"w1": "fc1", "w2": "fc2", "w2_t": "dhidden", "w1_t": "dx"}[name]
        config = self.gemm_configs.get(role)
        options = {} if config is None else {"configs": (dg.get_num_sms(), self.block_m, *config, (0, 0, 0))}
        if self.compute == "fp8":
            quantized_a = self.quantized[a.shape[1]]
            if not prequantized:
                if self.fp8_active_quantization:
                    _launch(26, [a, quantized_a[0], self.quantized_scale_storage[a.shape[1]],
                                 self.current_rows, self.status],
                            [self.rows, a.shape[1], int(self.fp8_mn_major_scales)])
                else:
                    _launch(9, [a, *quantized_a], list(a.shape))
            dg.m_grouped_fp8_gemm_nt_nopad(
                quantized_a, self.quantized[name], output, self.group_ids,
                m_rows=self.lengths, disable_ue8m0_cast=True, **options)
        else:
            dg.m_grouped_bf16_gemm_nt_nopad(a, weight, output, self.group_ids,
                                           m_rows=self.lengths, **options)

    def _dispatch_chunk(self, x, ids, scores, valid_tokens, start):
        t, c, h, k = x.shape[0], self.chunk_tokens, self.h, self.k
        _launch(13, [x, ids, self.local_x, self.local_ids, valid_tokens, self.status],
                [t, c, h, k, start])
        available = max(0, min(c, t - start))
        self.local_scores.zero_()
        self.local_scores[:available].copy_(scores[start:start + available])
        buffer = self.deep_ep_buffer
        counts, rdma_counts, expert_counts, in_rank, event = buffer.get_dispatch_layout(
            self.local_ids, self.e * self.size, async_finish=True)
        event.current_stream_wait()
        rx, ri, rw, _, handle, event = buffer.dispatch(self.local_x,
            topk_idx=self.local_ids, topk_weights=self.local_scores,
            num_tokens_per_rank=counts, num_tokens_per_rdma_rank=rdma_counts,
            num_tokens_per_expert=expert_counts, is_token_in_rank=in_rank,
            num_worst_tokens=self.recv_capacity, async_finish=True)
        event.current_stream_wait()
        if rx.shape != (self.recv_capacity, h) or ri.shape != (self.recv_capacity, k):
            raise RuntimeError("unexpected DeepEP fixed-capacity receive layout")
        # DeepEP converts global expert IDs into destination-local expert IDs.
        self.counts.zero_()
        _launch(0, [ri, self.counts, self.route_experts, self.route_rows, self.status],
                [self.routes, self.e, 0, 1])
        offsets = self.offsets if self.ring_rows is None else self.global_offsets
        lengths = self.lengths if self.ring_rows is None else self.global_lengths
        permutation = self.permutation if self.ring_rows is None else self.global_permutation
        capacity = self.rows if self.ring_rows is None else self.full_row_capacity
        _launch(1, [self.counts, offsets, lengths], [self.e, self.alignment])
        if self.ring_rows is None:
            _launch(2, [self.offsets, self.group_ids, self.permutation], [self.rows, self.e])
        else:
            dg._C._moe_training_peer_launch(1, [permutation, self.global_source_rows],
                                            [capacity, self.recv_capacity], False)
        _launch(19, [self.route_experts, self.route_rows, offsets, permutation, self.status],
                [self.routes, capacity])
        if self.ring_rows is not None:
            self.window_recv_x = rx
            return rw, handle
        _launch(20, [rx, self.permutation, self.packed_x, self.current_rows, self.status],
                [self.rows, h, self.recv_capacity, k])
        return rw, handle

    def _expert_forward(self, w1, w2):
        self._gemm(self.packed_x, w1, self.gate_up, "w1")
        fused = self._activation()
        self._gemm(self.hidden, w2, self.packed_output, "w2", prequantized=fused)

    def _combine(self, rw, handle, weighted, dscores=None):
        if self.ring_rows is not None:
            # Accumulate all local expert/window contributions in FP32 and
            # round only once at DeepEP's existing BF16 payload boundary.
            self.reduced.copy_(self.recv_accumulator)
        else:
            _launch(23, [self.packed_output, self.route_experts, self.route_rows,
                         self.offsets, rw, self.reduced, self.status],
                    [self.rows, self.h, self.recv_capacity, self.k, self.e, int(weighted)])
        output, ds, event = self.deep_ep_buffer.combine(self.reduced, handle,
            topk_weights=dscores, async_finish=True)
        event.current_stream_wait()
        return output, ds

    def _set_window(self, start):
        # This is the mutation boundary for packed X and device lengths in
        # both forward and backward. Auxiliary wgrads must finish reading them.
        self._join_wgrad()
        set_route_window(self, start)
        _launch(20, [self.window_recv_x, self.permutation, self.packed_x, self.current_rows, self.status],
                [self.rows, self.h, self.recv_capacity, self.k])

    def _reduce_window(self, rw, start, weighted):
        _launch(25, [self.packed_output, self.route_experts, self.route_rows, self.global_offsets,
                     rw, self.recv_accumulator, self.current_rows, self.status],
                [self.rows, self.h, self.recv_capacity, self.k, self.e, start,
                 int(weighted), self.full_row_capacity])

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens):
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        with self._execution():
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            # Weights may change in-place between graph replays or eager calls.
            # Match Native's dynamic FP8 recipe; quantization is inside timing.
            if self.compute == "fp8":
                self._prepare_weights(w1, w2)
            output = torch.empty_like(x)
            for start in range(0, self.max_tokens, self.chunk_tokens):
                rw, handle = self._dispatch_chunk(x, ids, scores, valid_tokens, start)
                if self.ring_rows is None:
                    self._expert_forward(w1, w2)
                else:
                    self.recv_accumulator.zero_()
                    for window in range(0, self.full_row_capacity, self.rows):
                        self._set_window(window)
                        self._expert_forward(w1, w2)
                        self._reduce_window(rw, window, weighted=True)
                local_y, _ = self._combine(rw, handle, weighted=True)
                count = max(0, min(self.chunk_tokens, x.shape[0] - start))
                output[start:start + count].copy_(local_y[:count])
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return output

    @torch.no_grad()
    def backward(self, dy, x, ids, scores, w1, w2, valid_tokens, accumulators=None):
        from .bf16 import _validate_accumulators

        self._validate(x, ids, scores, w1, w2, valid_tokens)
        _validate_accumulators(accumulators, w1, w2)
        if dy.shape != x.shape or dy.dtype != x.dtype or dy.device != x.device or not dy.is_contiguous():
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
                rw, handle = self._dispatch_chunk(x, ids, scores, valid_tokens, start)
                if self.ring_rows is None:
                    self._expert_forward(w1, w2)
                count = max(0, min(self.chunk_tokens, x.shape[0] - start))
                self.local_dy_input.zero_()
                self.local_dy_input[:count].copy_(dy[start:start + count])
                if self.overlap_expert_dy:
                    main = torch.cuda.current_stream(self.device)
                    self._gradient_stream.wait_stream(main)
                    with torch.cuda.stream(self._gradient_stream):
                        self.local_dy_input.record_stream(self._gradient_stream)
                        rdy, _, _, _, _, event = self.deep_ep_buffer.dispatch(
                            self.local_dy_input, handle=handle, async_finish=True)
                        event.current_stream_wait()
                    rdy.record_stream(main)
                    self._gradient_prefetch_pending = True
                else:
                    rdy, _, _, _, _, event = self.deep_ep_buffer.dispatch(
                        self.local_dy_input, handle=handle, async_finish=True)
                    event.current_stream_wait()
                self.recv_dscores.zero_()
                if self.ring_rows is None:
                    self._expert_backward_rows(rdy, rw, dw1, dw2)
                else:
                    self.recv_accumulator.zero_()
                    for window in range(0, self.full_row_capacity, self.rows):
                        self._set_window(window)
                        self._expert_forward(w1, w2)
                        self._expert_backward_rows(rdy, rw, dw1, dw2)
                        self._reduce_window(rw, window, weighted=False)
                local_dx, local_ds = self._combine(rw, handle, weighted=False, dscores=self.recv_dscores)
                dx[start:start + count].copy_(local_dx[:count])
                dscores[start:start + count].copy_(local_ds[:count])
                self._join_wgrad()
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return dx, dw1, dw2, dscores

    def _expert_backward_rows(self, rdy, rw, dw1, dw2):
        # Cached DY dispatch may progress during the local forward replay.
        # Join at its first actual consumer, after that replay has completed.
        self._join_gradient_prefetch()
        _launch(24, [rdy, rw, self.permutation, self.packed_output,
                     self.packed_dy, self.recv_dscores, self.current_rows, self.status],
                [self.rows, self.h, self.recv_capacity, self.k])
        self._wgrad(self.packed_dy, self.hidden, dw2,
                    asynchronous=self.wgrad_overlap == "both")
        dhidden = self.hidden if self.backward_hidden is None else self.backward_hidden
        self._gemm(self.packed_dy, self.w2_t, dhidden, "w2_t")
        fused = self._activation(backward=True, hidden=dhidden)
        self._wgrad(self.gate_up, self.packed_x, dw1,
                    asynchronous=self.wgrad_overlap != "none")
        self._gemm(self.gate_up, self.w1_t, self.packed_output, "w1_t", prequantized=fused)

    def close(self):
        if self.deep_ep_buffer is not None:
            torch.cuda.synchronize(self.device)
            self.deep_ep_buffer.destroy()
            self.deep_ep_buffer = None
