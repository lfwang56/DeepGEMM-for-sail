"""A persistent BF16 routed-expert forward/backward pipeline for tc02 PPUs."""

import torch

import deep_gemm as dg
from .bf16 import _validate_accumulators
from .peer import PeerMoEPlan


class PersistentMoEPlan(PeerMoEPlan):
    """One core launch per forward/backward, including GPU routes and wgrad.

    Requires a direct peer domain and enough resident resources for one CTA per
    selected CU. The plan and its signal state remain on one execution stream.
    Weight casts/transposes and optimizer operations remain outside the core.
    """

    transport = "persistent_peer"

    def __init__(self, *args, num_ctas=None, maximize_ctas=False, **kwargs):
        if kwargs.get("gemm_configs"):
            raise ValueError("persistent training currently uses a fixed native GEMM tile")
        super().__init__(*args, **kwargs)
        available = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.available_cus = available
        if maximize_ctas and num_ctas is not None:
            raise ValueError("choose either num_ctas or maximize_ctas")
        self.maximize_ctas = maximize_ctas
        # Leave one CU available to framework communication streams while a
        # persistent expert kernel waits for peers.
        self.num_ctas = max(1, available - 1) if num_ctas is None else num_ctas
        if not isinstance(self.num_ctas, int) or not 0 < self.num_ctas <= 2**31 - 1:
            raise ValueError("num_ctas must be a positive int32 value; residency is checked at preparation")
        self.grid_sync = torch.zeros(3, device=self.device, dtype=torch.int32)
        self.buffers["grid_sync"] = self.grid_sync

    def _pipeline(self, x, ids, scores, w1, w2, valid, y, *, dy=None, grads=None, compile_only=False):
        backward = dy is not None
        dx, dw1, dw2, ds = (y, w1, w2, scores) if grads is None else grads
        tables = self.pointer_tables
        tensors = [x, ids, scores, w1, w2, valid, y, x if dy is None else dy, dx, dw1, dw2, ds,
                   self.local_x, self.local_ids, self.local_values, self.local_dx, self.local_dy,
                   self.counts, self.offsets, self.lengths, self.permutation, self.route_experts, self.route_rows,
                   self.packed_x, self.gate_up, self.hidden, self.packed_dy, self.w1_t, self.w2_t,
                   self.source_rows, self.identity_rows, self.work_info, self.num_blocks,
                   self.status, self.barrier_state, tables["barrier_state"], tables["local_x"], tables["local_ids"],
                   tables["local_values"], tables["local_dx"], tables["local_dy"], self.grid_sync]
        return dg._C._moe_training_pipeline(tensors, [x.shape[0], self.max_tokens, self.rank, self.rows,
                                                    self.alignment, self.num_ctas], backward, compile_only)

    def _prepare(self, w1, w2):
        super()._prepare(w1, w2)
        w1, w2 = w1.to(torch.bfloat16), w2.to(torch.bfloat16)
        x = torch.empty((0, self.h), device=self.device, dtype=torch.bfloat16)
        ids = torch.empty((0, self.k), device=self.device, dtype=torch.int64)
        scores = torch.empty((0, self.k), device=self.device, dtype=torch.float32)
        valid = torch.zeros((), device=self.device, dtype=torch.int32)
        forward_resident = self._pipeline(x, ids, scores, w1, w2, valid, x, compile_only=True)
        gradients = (x, torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32), scores)
        backward_resident = self._pipeline(x, ids, scores, w1, w2, valid, x, dy=x, grads=gradients, compile_only=True)
        self.resident_blocks_per_cu = min(forward_resident, backward_resident)
        if self.maximize_ctas:
            self.num_ctas = max(1, self.available_cus - 1) * self.resident_blocks_per_cu
        torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens):
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        with self._execution():
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            y = torch.empty_like(x)
            self._pipeline(x, ids, scores, w1, w2, valid_tokens, y)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return y

    @torch.no_grad()
    def backward(self, dy, x, ids, scores, w1, w2, valid_tokens, accumulators=None):
        _validate_accumulators(accumulators, w1, w2)
        if dy.shape != x.shape or dy.device != x.device or dy.dtype != x.dtype or not dy.is_contiguous():
            raise ValueError("dy must be contiguous and match x")
        with self._execution():
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            self._prepare_weights(w1, w2, backward=True)
            dx, ds = torch.empty_like(x), torch.empty_like(scores)
            dw1, dw2 = ((torch.zeros_like(w1, dtype=torch.float32), torch.zeros_like(w2, dtype=torch.float32))
                        if accumulators is None else accumulators)
            self._pipeline(x, ids, scores, w1, w2, valid_tokens, dx, dy=dy, grads=(dx, dw1, dw2, ds))
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return dx, dw1, dw2, ds
