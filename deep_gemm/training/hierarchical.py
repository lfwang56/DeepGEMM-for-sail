"""Two-domain GPU collectives around independent domain-local bilateral MoEs.

Global ranks must be ordered as two contiguous equal physical peer domains.
The caller owns group lifetime; create groups once and share them across layers.
No token payload is transferred through the CPU metadata process group.
"""
from dataclasses import dataclass, field
from datetime import timedelta
import threading
import os

import torch
import torch.distributed as dist
import deep_gemm as dg

from .bf16 import _validate_accumulators
from .native import NativeMoEPlan, _launch
from .peer import PeerMoEPlan


@dataclass
class HierarchicalGroups:
    metadata: object
    domain: object
    pair: object
    domain_size: int
    domain_id: int
    rank: int
    size: int
    _closed: bool = False
    _devices: set = field(default_factory=set, repr=False)

    def close(self):
        """After all referencing plans and device work have finished."""
        if not self._closed:
            for device in sorted(self._devices):
                torch.cuda.synchronize(device)
            dist.barrier(group=self.metadata)
            dist.destroy_process_group(self.pair)
            dist.destroy_process_group(self.domain)
            self._closed = True


def make_hierarchical_groups(domain_size, metadata_group):
    """Collectively create two domain Gloo groups and matching PCCL pairs.

    metadata_group must contain the complete default world in global-rank order.
    Each contiguous block of domain_size ranks must be one physical peer domain;
    the caller must verify that physical topology before invoking this function.
    """
    size, rank = dist.get_world_size(metadata_group), dist.get_rank(metadata_group)
    if type(domain_size) is not int or domain_size < 1 or size != 2 * domain_size:
        raise ValueError('hierarchical MoE requires exactly two equal peer domains')
    if dist.get_process_group_ranks(metadata_group) != list(range(dist.get_world_size())):
        raise ValueError('metadata group must contain the complete ordered default world')
    if dist.get_backend(metadata_group) != 'gloo':
        raise ValueError('use a Gloo metadata group; all payload collectives remain on GPU')
    descriptions = [None] * size
    dist.all_gather_object(descriptions, domain_size, group=metadata_group)
    if any(item != domain_size for item in descriptions):
        raise ValueError('ranks disagree on domain size')
    domain_id = rank // domain_size
    domain = pair = None
    for index in range(2):
        group = dist.new_group(list(range(index * domain_size, (index + 1) * domain_size)),
                               backend='gloo', timeout=timedelta(seconds=600))
        if index == domain_id:
            domain = group
    for index in range(domain_size):
        group = dist.new_group([index, index + domain_size], backend='nccl',
                               timeout=timedelta(seconds=600))
        if index == rank % domain_size:
            pair = group
    return HierarchicalGroups(metadata_group, domain, pair, domain_size, domain_id, rank, size)


class HierarchicalPeerMoEPlan(NativeMoEPlan):
    """Pairwise inter-domain AG/RS plus an ordinary domain-local PeerMoEPlan.

    Optional live-context reuse retains the matching pair inputs and inner
    activations from forward. Otherwise backward rematerializes both dispatch
    and the inner MoE. Domain partials are BF16-rounded before reduction. This can differ from a
    flat dispatch/combine backend; validate both gradients and model quality.
    """
    transport = 'hierarchical_peer'

    def __init__(self, hidden, intermediate, local_experts, topk, max_tokens, *,
                 groups, chunk_tokens=128, device=None, check_errors=True, gradient_group=None,
                 reuse_pair_context=None, source_pipeline=None, source_serialized=None, **peer_options):
        if not isinstance(groups, HierarchicalGroups) or groups._closed:
            raise ValueError('create and retain live HierarchicalGroups before constructing a plan')
        if min(max_tokens, chunk_tokens) < 1 or 2 * max_tokens * topk > 2**31 - 1:
            raise ValueError('positive token/chunk capacities and int32-bounded pair routes are required')
        if peer_options.pop('compute', 'bf16') != 'bf16':
            raise ValueError('hierarchical peer training currently requires BF16')
        self.groups = groups
        self.group, self.size, self.rank = groups.metadata, groups.size, groups.rank
        self.gradient_group = gradient_group
        self.h, self.i, self.e, self.k = hidden, intermediate, local_experts, topk
        self.max_tokens, self.chunk_tokens = max_tokens, chunk_tokens
        self.compute, self.check_errors = 'bf16', check_errors
        self.device = torch.device('cuda', torch.cuda.current_device()) if device is None else torch.device(device)
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self._lock, self._stream, self._prepared = threading.RLock(), None, False
        if reuse_pair_context is None:
            raw=os.environ.get('DG_MOE_HYBRID_REUSE_PAIR_CONTEXT','0')
            if raw not in ('0','1'):
                raise ValueError('DG_MOE_HYBRID_REUSE_PAIR_CONTEXT must be0or1')
            reuse_pair_context=raw=='1'
        if type(reuse_pair_context) is not bool:
            raise TypeError('reuse_pair_context must be bool')
        def flag(value,key):
            if value is None:
                raw=os.environ.get(key,'0')
                if raw not in ('0','1'):raise ValueError(key+' must be0or1')
                value=raw=='1'
            if type(value) is not bool:raise TypeError(key+' must be bool')
            return value
        source_pipeline=flag(source_pipeline,'DG_MOE_HYBRID_SOURCE_PIPELINE')
        source_serialized=flag(source_serialized,'DG_MOE_HYBRID_SOURCE_SERIALIZED')
        self.hybrid_source_pipeline=source_pipeline
        self.hybrid_source_serialized=source_serialized
        self._source_forward_available=False
        self._source_forward_fingerprint=None
        self._source_forward_tensors=self._source_forward_masks=None
        self.hybrid_reuse_pair_context=reuse_pair_context
        self.hybrid_pair_forward_reused_last_backward=False
        self._pair_forward_available=False
        self._pair_forward_fingerprint=None
        self._pair_forward_mask=None
        self._pair_forward_tensors=None
        description = (hidden, intermediate, local_experts, topk, max_tokens, chunk_tokens,
                       groups.domain_size, repr(sorted(peer_options.items())),reuse_pair_context,source_pipeline,source_serialized)
        descriptions = [None] * self.size
        dist.all_gather_object(descriptions, description, group=self.group)
        if any(item != description for item in descriptions):
            raise ValueError('ranks must create matching hierarchical plans')
        if source_pipeline and not reuse_pair_context:
            raise ValueError('source pipeline requires live pair context reuse')
        if source_serialized and not source_pipeline:
            raise ValueError('source serialization requires the source pipeline')
        factor=1 if source_pipeline else 2
        self.inner = PeerMoEPlan(hidden, intermediate, local_experts, topk, factor * max_tokens,
                                 chunk_tokens=factor * chunk_tokens, group=groups.domain,
                                 device=self.device, check_errors=False, **peer_options)
        if reuse_pair_context and not self.inner.reuse_forward_context:
            self.inner.close()
            raise ValueError('pair context reuse requires domain-local forward context reuse')
        self.status = self.inner.status
        self.buffers = {'inner_' + key: value for key, value in self.inner.buffers.items()}
        self.remote_inner=None
        if source_pipeline:
            self.remote_inner=PeerMoEPlan(hidden,intermediate,local_experts,topk,max_tokens,
                chunk_tokens=chunk_tokens,group=groups.domain,device=self.device,check_errors=False,**peer_options)
            self.buffers.update({'remote_inner_'+key:value for key,value in self.remote_inner.buffers.items()})
            self._source_pair_stream=torch.cuda.Stream(device=self.device)

        def allocate(name, shape, dtype):
            tensor = torch.empty(shape, device=self.device, dtype=dtype)
            self.buffers['pair_' + name] = tensor
            return tensor

        t, h, k = max_tokens, hidden, topk
        self.pair_x = allocate('x', (2*t, h), torch.bfloat16)
        self.pair_ids = allocate('ids', (2*t, k), torch.int64)
        self.pair_scores = allocate('scores', (2*t, k), torch.float32)
        self.pair_dy = allocate('dy', (2*t, h), torch.bfloat16)
        self.send_x = allocate('send_x', (t, h), torch.bfloat16)
        self.send_ids = allocate('send_ids', (t, k), torch.int64)
        self.send_scores = allocate('send_scores', (t, k), torch.float32)
        self.send_dy = allocate('send_dy', (t, h), torch.bfloat16)
        self.validation_counts = allocate('counts', (local_experts,), torch.int32)
        self.validation_experts = allocate('route_experts', (2*t*k,), torch.int32)
        self.validation_rows = allocate('route_rows', (2*t*k,), torch.int32)
        self.inner_valid = allocate('valid', (), torch.int32).fill_(t if source_pipeline else 2*t)
        if source_pipeline:
            section=slice((1-groups.domain_id)*t,(2-groups.domain_id)*t)
            self.source_remote_x=self.pair_x[section]
            self.source_remote_ids=self.pair_ids[section]
            self.source_remote_scores=self.pair_scores[section]
            self.source_remote_dy=self.pair_dy[section]
        self.hybrid_domains, self.hybrid_domain_size = 2, groups.domain_size
        self.hybrid_pair_backend = str(dist.get_backend(groups.pair))
        groups._devices.add(self.device.index)

    @property
    def activation_workspace_bytes(self):
        return self.inner.activation_workspace_bytes + (self.remote_inner.activation_workspace_bytes if self.remote_inner is not None else 0) + sum(
            t.numel() * t.element_size() for name, t in self.buffers.items() if name.startswith("pair_"))

    def __getattr__(self, name):
        inner = self.__dict__.get('inner')
        if inner is not None:
            return getattr(inner, name)
        raise AttributeError(name)

    @torch.no_grad()
    def prepare(self, w1, w2):
        with self._execution():
            if self._prepared:
                return
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('warm hierarchical MoE before graph capture')
            error = None
            try:
                self.inner.prepare(w1, w2)
                if self.remote_inner is not None:self.remote_inner.prepare(w1,w2)
                dg._C._moe_training_precompile(self.device.index, [0, 13])
            except Exception as exception:
                error = exception
            # Both domains must finish compiling before either enters pair PCCL.
            self._finish_preparation(error)
            self._prepared = False
            error = None
            try:
                for source, output in ((self.send_x, self.pair_x), (self.send_ids, self.pair_ids),
                                       (self.send_scores, self.pair_scores)):
                    source.zero_()
                    dist.all_gather_into_tensor(output, source, group=self.groups.pair)
                for source in (self.pair_x, self.pair_scores):
                    output = torch.empty_like(source[:self.max_tokens])
                    dist.reduce_scatter_tensor(output, source, group=self.groups.pair)
                torch.cuda.current_stream(self.device).synchronize()
            except Exception as exception:
                error = exception
            # A different pair may fail its communicator warmup. Agree globally
            # before a successful pair can enter the domain-local peer barriers.
            self._finish_preparation(error)

    def _dispatch_pair(self, x, ids, scores, valid_tokens):
        # Any new dispatch overwrites pair inputs and invalidates the previous
        # live forward, even when a later launch or collective fails.
        self._pair_forward_available=False
        self._pair_forward_tensors=None
        self._pair_forward_mask=None
        _launch(13, [x, ids, self.send_x, self.send_ids, valid_tokens, self.status],
                [x.shape[0], self.max_tokens, self.h, self.k, 0])
        self.send_scores.zero_()
        self.send_scores[:x.shape[0]].copy_(scores)
        for source, output in ((self.send_x, self.pair_x), (self.send_ids, self.pair_ids),
                               (self.send_scores, self.pair_scores)):
            dist.all_gather_into_tensor(output, source, group=self.groups.pair)
        # Validate GLOBAL IDs before masking experts belonging to the other domain.
        self.validation_counts.zero_()
        _launch(0, [self.pair_ids, self.validation_counts, self.validation_experts,
                    self.validation_rows, self.status],
                [2*self.max_tokens*self.k, self.e, self.rank, self.size])
        domain_experts = self.groups.domain_size * self.e
        local = ((self.pair_ids >= self.groups.domain_id * domain_experts)
                 & (self.pair_ids < (self.groups.domain_id + 1) * domain_experts))
        return torch.where(local, self.pair_ids - self.groups.domain_id * domain_experts, -1)

    def _matching_pair_forward(self, tensors):
        if not self.hybrid_reuse_pair_context:
            return False
        fingerprint=PeerMoEPlan._context_fingerprint(*tensors)
        local=(self._pair_forward_available and self._pair_forward_fingerprint==fingerprint
               and self._pair_forward_mask is not None and self.inner._forward_context_available)
        if local:
            local=(self.inner._forward_context_fingerprint==PeerMoEPlan._context_fingerprint(
                self.pair_x,self._pair_forward_mask,self.pair_scores,tensors[3],tensors[4],self.inner_valid))
        # The metadata group is explicitly Gloo/CPU, including capture-time
        # agreement. A mutation on one rank must invalidate BOTH domains before
        # any rank skips GPU pair collectives. These decisions are not replayed.
        decisions=[None]*self.size
        dist.all_gather_object(decisions,bool(local),group=self.group)
        return all(decisions)

    def _reduce_pair(self, partial, actual_tokens):
        # Fresh allocation: later layer calls must not overwrite autograd outputs.
        output = torch.empty_like(partial[:self.max_tokens])
        dist.reduce_scatter_tensor(output, partial, group=self.groups.pair)
        return output[:actual_tokens]

    @torch.no_grad()
    def forward(self, x, ids, scores, w1, w2, valid_tokens):
        if self.hybrid_source_pipeline:
            from .hierarchical_pipeline import forward
            return forward(self,x,ids,scores,w1,w2,valid_tokens)
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        with self._execution():
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            masked = self._dispatch_pair(x, ids, scores, valid_tokens)
            partial = self.inner.forward(self.pair_x, masked, self.pair_scores, w1, w2, self.inner_valid)
            output = self._reduce_pair(partial, x.shape[0])
            if self.hybrid_reuse_pair_context:
                tensors=(x,ids,scores,w1,w2,valid_tokens)
                self._pair_forward_fingerprint=PeerMoEPlan._context_fingerprint(*tensors)
                self._pair_forward_tensors=tensors
                self._pair_forward_mask=masked
                self._pair_forward_available=True
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return output

    @torch.no_grad()
    def backward(self, dy, x, ids, scores, w1, w2, valid_tokens, accumulators=None):
        if self.hybrid_source_pipeline:
            from .hierarchical_pipeline import backward
            return backward(self,dy,x,ids,scores,w1,w2,valid_tokens,accumulators)
        self._validate(x, ids, scores, w1, w2, valid_tokens)
        _validate_accumulators(accumulators, w1, w2)
        if dy.shape != x.shape or dy.dtype != x.dtype or dy.device != x.device or not dy.is_contiguous():
            raise ValueError('dy must be contiguous and match x')
        with self._execution():
            self.prepare(w1, w2)
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.reset_errors()
            reuse=self._matching_pair_forward((x,ids,scores,w1,w2,valid_tokens))
            masked=self._pair_forward_mask if reuse else self._dispatch_pair(x,ids,scores,valid_tokens)
            self.hybrid_pair_forward_reused_last_backward=reuse
            # Consume the live pair context before any backward side effect.
            self._pair_forward_available=False
            self._pair_forward_tensors=None
            self._pair_forward_mask=None
            self.send_dy.zero_()
            self.send_dy[:x.shape[0]].copy_(dy)
            dist.all_gather_into_tensor(self.pair_dy, self.send_dy, group=self.groups.pair)
            # A fallback must rematerialize on every rank, including a
            # backward-only graph. Temporarily disable reuse so the ordinary
            # inner backward supplies that rematerialization exactly once.
            previous_reuse=self.inner.reuse_forward_context
            if not reuse:
                self.inner.reuse_forward_context=False
                self.inner._forward_context_available=False
            try:
                dx, dw1, dw2, ds = self.inner.backward(self.pair_dy, self.pair_x, masked,
                            self.pair_scores, w1, w2, self.inner_valid, accumulators)
            finally:
                self.inner.reuse_forward_context=previous_reuse
            result = self._reduce_pair(dx, x.shape[0]), dw1, dw2, self._reduce_pair(ds, x.shape[0])
            if self.check_errors and not torch.cuda.is_current_stream_capturing():
                self.check()
            return result

    def reset_errors(self):
        self.inner.reset_errors()
        if self.remote_inner is not None:self.remote_inner.reset_errors()

    def check(self):
        self.inner.check()
        if self.remote_inner is not None:self.remote_inner.check()

    def close(self):
        """Release plan peer mappings; shared group lifetime remains caller-owned."""
        if self.remote_inner is not None:
            self._source_pair_stream.synchronize()
            self.remote_inner.close()
        if hasattr(self.inner, "close"):
            self.inner.close()
