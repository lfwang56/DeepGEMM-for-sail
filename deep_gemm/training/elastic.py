"""Installed ElasticBuffer nonexpanded dispatch/combine API adapter.

All GEMM, FP32 expert accumulation and rematerialization code is shared with
DeepEPMoEPlan. The installed elastic hybrid path requires nondeterministic
routing and unique expert IDs within each token; dynamic graph and independent
gradient checks remain mandatory.
"""
import deep_ep
import torch


class _CompletedEvent:
    def current_stream_wait(self):
        pass


class ElasticBufferAdapter:
    def __init__(self, group, chunk_tokens, hidden, topk, experts, comm_sms, hybrid):
        self.buffer = deep_ep.ElasticBuffer(group, num_max_tokens_per_rank=chunk_tokens,
            hidden=hidden, num_topk=topk, deterministic=False,
            allow_hybrid_mode=hybrid, explicitly_destroy=True)
        self.experts, self.comm_sms = experts, comm_sms
        self.backend = 'elastic_hybrid' if hybrid else 'elastic_flat'
        self.runtime = self

    def get_num_rdma_ranks(self):
        return self.buffer.num_scaleout_ranks

    def get_dispatch_layout(self, ids, experts, async_finish=True):
        if experts != self.experts:
            raise ValueError('expert count must match the elastic plan')
        # ElasticBuffer constructs its routing metadata in dispatch on device.
        return None, None, None, None, _CompletedEvent()

    def dispatch(self, x, *, topk_idx=None, topk_weights=None, handle=None,
                 async_finish=True, **legacy_layout):
        options = dict(num_sms=self.comm_sms, do_cpu_sync=False, do_expand=False,
                       async_with_compute_stream=True)
        if handle is None:
            options.update(topk_idx=topk_idx, topk_weights=topk_weights,
                           num_experts=self.experts, expert_alignment=1)
        else:
            options['handle'] = handle
        rx, ri, rw, new_handle, event = self.buffer.dispatch(x, **options)
        if handle is None:
            # Without CPU synchronization Elastic allocates the worst-case
            # receive tensor but only initializes its GPU-counted prefix. The
            # legacy Buffer ABI instead guarantees -1 IDs in padding rows.
            event.current_stream_wait()
            data = rx[0] if isinstance(rx, tuple) else rx
            valid = (torch.arange(data.shape[0], device=data.device)
                     < new_handle.psum_num_recv_tokens_per_scaleup_rank[-1])
            ri.masked_fill_(~valid[:, None], -1)
            if rw is not None:
                rw.masked_fill_(~valid[:, None], 0)
            event = _CompletedEvent()
        return rx, ri, rw, None, new_handle, event

    def combine(self, x, handle, *, topk_weights=None, async_finish=True):
        return self.buffer.combine(x, handle, topk_weights=topk_weights,
                                  num_sms=self.comm_sms, async_with_compute_stream=True)

    def destroy(self):
        self.buffer.destroy()
