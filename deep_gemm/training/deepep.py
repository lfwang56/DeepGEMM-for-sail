"""Installed DeepEP Buffer dispatch/combine with native BF16 grouped GEMMs.

DeepEP transports one copy per destination rank. Local GPU metadata expands
only the expert rows needed by the grouped GEMMs. All receive capacities are
bounded and all routing counts remain on device during graph replay.
"""

import torch
import torch.distributed as dist

import deep_gemm as dg
from .native import NativeMoEPlan, _launch


class DeepEPMoEPlan(NativeMoEPlan):
    transport = "deepep"

    def __init__(self, *args, comm_sms=8, gemm_configs=None, rdma_qps=None, **kwargs):
        if kwargs.get("compute", "bf16") != "bf16":
            raise ValueError("the DeepEP comparison currently implements BF16")
        if kwargs.get("group") is None:
            raise ValueError("DeepEP requires an initialized EP process group")
        import deep_ep

        super().__init__(*args, **kwargs)
        self.active_pointwise = self.vectorized_routes = True
        self.gemm_configs = gemm_configs or {}
        if type(comm_sms) is not int or comm_sms < 2 or comm_sms % 2:
            raise ValueError("DeepEP communication CU count must be an even integer >=2")
        self.comm_sms = comm_sms
        self.deep_ep_qps = max(12, (comm_sms + 1) // 2) if rdma_qps is None else rdma_qps
        if type(self.deep_ep_qps) is not int or self.deep_ep_qps < (comm_sms + 1) // 2:
            raise ValueError("DeepEP RDMA QPs must cover at least one pair of communication CUs")
        deep_ep.Buffer.set_num_sms(comm_sms)
        configs = (deep_ep.Buffer.get_dispatch_config(self.size),
                   deep_ep.Buffer.get_combine_config(self.size))
        nvl = max(c.get_nvl_buffer_size_hint(self.h * 2, self.size) for c in configs)
        rdma = max(c.get_rdma_buffer_size_hint(self.h * 2, self.size) for c in configs)
        self.deep_ep_buffer = deep_ep.Buffer(self.group, nvl, rdma,
            allow_mnnvl=True, explicitly_destroy=True, num_qps_per_rank=self.deep_ep_qps)
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

    def _allocate(self, name, shape, dtype):
        value = torch.empty(shape, device=self.device, dtype=dtype)
        self.buffers[name] = value
        return value

    def _prepare_pointwise(self):
        operations = [0, 1, 2, 7, 8, 13, 16, 19, 20, 23, 24]
        if self.fast_weight_transpose:
            operations.append(22)
        dg._C._moe_training_precompile(self.device.index, operations)

    def _gemm(self, a, weight, output, name, prequantized=False):
        role = {"w1": "fc1", "w2": "fc2", "w2_t": "dhidden", "w1_t": "dx"}[name]
        config = self.gemm_configs.get(role)
        options = {} if config is None else {"configs": (dg.get_num_sms(), 128, *config, (0, 0, 0))}
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
        _launch(1, [self.counts, self.offsets, self.lengths], [self.e, self.alignment])
        _launch(2, [self.offsets, self.group_ids, self.permutation], [self.rows, self.e])
        _launch(19, [self.route_experts, self.route_rows, self.offsets, self.permutation, self.status],
                [self.routes, self.rows])
        _launch(20, [rx, self.permutation, self.packed_x, self.current_rows, self.status],
                [self.rows, h, self.recv_capacity, k])
        return rw, handle

    def _expert_forward(self, w1, w2):
        self._gemm(self.packed_x, w1, self.gate_up, "w1")
        self._activation()
        self._gemm(self.hidden, w2, self.packed_output, "w2")

    def _combine(self, rw, handle, weighted, dscores=None):
        _launch(23, [self.packed_output, self.route_experts, self.route_rows,
                     self.offsets, rw, self.reduced, self.status],
                [self.rows, self.h, self.recv_capacity, self.k, self.e, int(weighted)])
        output, ds, event = self.deep_ep_buffer.combine(self.reduced, handle,
            topk_weights=dscores, async_finish=True)
        event.current_stream_wait()
        return output, ds

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens):
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        with self._execution():
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            output = torch.empty_like(x)
            for start in range(0, self.max_tokens, self.chunk_tokens):
                rw, handle = self._dispatch_chunk(x, ids, scores, valid_tokens, start)
                self._expert_forward(w1, w2)
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
                self._expert_forward(w1, w2)
                count = max(0, min(self.chunk_tokens, x.shape[0] - start))
                self.local_dy_input.zero_()
                self.local_dy_input[:count].copy_(dy[start:start + count])
                rdy, _, _, _, _, event = self.deep_ep_buffer.dispatch(
                    self.local_dy_input, handle=handle, async_finish=True)
                event.current_stream_wait()
                self.recv_dscores.zero_()
                _launch(24, [rdy, rw, self.permutation, self.packed_output,
                             self.packed_dy, self.recv_dscores, self.current_rows, self.status],
                        [self.rows, self.h, self.recv_capacity, self.k])
                dg.k_grouped_bf16_gemm_tn_device(self.packed_dy, self.hidden, dw2, self.lengths, self.rows)
                self._gemm(self.packed_dy, self.w2_t, self.hidden, "w2_t")
                self._activation(backward=True)
                dg.k_grouped_bf16_gemm_tn_device(self.gate_up, self.packed_x, dw1, self.lengths, self.rows)
                self._gemm(self.gate_up, self.w1_t, self.packed_output, "w1_t")
                local_dx, local_ds = self._combine(rw, handle, weighted=False, dscores=self.recv_dscores)
                dx[start:start + count].copy_(local_dx[:count])
                dscores[start:start + count].copy_(local_ds[:count])
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return dx, dw1, dw2, dscores

    def close(self):
        if self.deep_ep_buffer is not None:
            torch.cuda.synchronize(self.device)
            self.deep_ep_buffer.destroy()
            self.deep_ep_buffer = None
