"""Two independent source-half contexts overlap pair collectives with local work."""
import torch
import torch.distributed as dist
from .native import _launch
from .bf16 import _validate_accumulators
from .peer import PeerMoEPlan


def _mask(plan,ids,valid_tokens=None):
    if valid_tokens is not None:
        active=torch.arange(ids.shape[0],device=ids.device)[:,None]<valid_tokens
        ids=torch.where(active,ids,-1)
    plan.validation_counts.zero_()
    if ids.numel():
        _launch(0,[ids,plan.validation_counts,plan.validation_experts[:ids.numel()],plan.validation_rows[:ids.numel()],plan.status],
                [ids.numel(),plan.e,plan.rank,plan.size])
    count=plan.groups.domain_size*plan.e
    local=(ids>=plan.groups.domain_id*count)&(ids<(plan.groups.domain_id+1)*count)
    return torch.where(local,ids-plan.groups.domain_id*count,-1)


def _send_inputs(plan,x,ids,scores,valid):
    _launch(13,[x,ids,plan.send_x,plan.send_ids,valid,plan.status],
            [x.shape[0],plan.max_tokens,plan.h,plan.k,0])
    plan.send_scores.zero_();plan.send_scores[:x.shape[0]].copy_(scores)


def _start(plan,payloads):
    main=torch.cuda.current_stream(plan.device);aux=plan._source_pair_stream
    aux.wait_stream(main)
    work=[]
    with torch.cuda.stream(aux):
        for source,output in payloads:
            source.record_stream(aux);output.record_stream(aux)
            work.append(dist.all_gather_into_tensor(output,source,group=plan.groups.pair,async_op=True))
    return work


def _join(plan,work):
    # Queue local compute before any possibly host-blocking Work.wait().
    with torch.cuda.stream(plan._source_pair_stream):
        for item in work:item.wait()
    torch.cuda.current_stream(plan.device).wait_stream(plan._source_pair_stream)


def _pair_result(plan,own,remote):
    padded=torch.zeros_like(remote)
    padded[:own.shape[0]].copy_(own)
    pieces=(padded,remote) if plan.groups.domain_id==0 else (remote,padded)
    return torch.cat(pieces,dim=0)


def _matching(plan,tensors):
    local=plan._source_forward_available and plan._source_forward_fingerprint==PeerMoEPlan._context_fingerprint(*tensors)
    if local:
        own_mask,remote_mask=plan._source_forward_masks
        local=(plan.inner._forward_context_available and plan.remote_inner._forward_context_available and
            plan.inner._forward_context_fingerprint==PeerMoEPlan._context_fingerprint(
                tensors[0],own_mask,tensors[2],tensors[3],tensors[4],tensors[5]) and
            plan.remote_inner._forward_context_fingerprint==PeerMoEPlan._context_fingerprint(
                plan.source_remote_x,remote_mask,plan.source_remote_scores,tensors[3],tensors[4],plan.inner_valid))
    decisions=[None]*plan.size
    dist.all_gather_object(decisions,bool(local),group=plan.group)
    return all(decisions)


def forward(plan,x,ids,scores,w1,w2,valid_tokens):
    plan._validate(x,ids,scores,w1,w2,valid_tokens)
    with plan._execution():
        plan.prepare(w1,w2)
        if plan.check_errors and not torch.cuda.is_current_stream_capturing():plan.reset_errors()
        plan._source_forward_available=False
        plan._source_forward_tensors=None
        plan._source_forward_masks=None
        _send_inputs(plan,x,ids,scores,valid_tokens)
        work=_start(plan,[(plan.send_x,plan.pair_x),(plan.send_ids,plan.pair_ids),(plan.send_scores,plan.pair_scores)])
        if plan.hybrid_source_serialized:_join(plan,work)
        own_mask=_mask(plan,ids,valid_tokens)
        own=plan.inner.forward(x,own_mask,scores,w1,w2,valid_tokens)
        if not plan.hybrid_source_serialized:_join(plan,work)
        remote_mask=_mask(plan,plan.source_remote_ids)
        remote=plan.remote_inner.forward(plan.source_remote_x,remote_mask,plan.source_remote_scores,w1,w2,plan.inner_valid)
        output=plan._reduce_pair(_pair_result(plan,own,remote),x.shape[0])
        tensors=(x,ids,scores,w1,w2,valid_tokens)
        plan._source_forward_fingerprint=PeerMoEPlan._context_fingerprint(*tensors)
        plan._source_forward_tensors=tensors
        plan._source_forward_masks=(own_mask,remote_mask)
        plan._source_forward_available=True
        if plan.check_errors and not torch.cuda.is_current_stream_capturing():plan.check()
        return output


def _backward(inner,reuse,dy,x,ids,scores,w1,w2,valid,accumulators):
    previous=inner.reuse_forward_context
    if not reuse:
        inner.reuse_forward_context=False
        inner._forward_context_available=False
    try:return inner.backward(dy,x,ids,scores,w1,w2,valid,accumulators)
    finally:inner.reuse_forward_context=previous


def backward(plan,dy,x,ids,scores,w1,w2,valid_tokens,accumulators=None):
    plan._validate(x,ids,scores,w1,w2,valid_tokens);_validate_accumulators(accumulators,w1,w2)
    if dy.shape!=x.shape or dy.dtype!=x.dtype or dy.device!=x.device or not dy.is_contiguous():
        raise ValueError('dy must be contiguous and match x')
    with plan._execution():
        plan.prepare(w1,w2)
        if plan.check_errors and not torch.cuda.is_current_stream_capturing():plan.reset_errors()
        reuse=_matching(plan,(x,ids,scores,w1,w2,valid_tokens))
        plan.hybrid_pair_forward_reused_last_backward=reuse
        payloads=[]
        if reuse:own_mask,remote_mask=plan._source_forward_masks
        else:
            _send_inputs(plan,x,ids,scores,valid_tokens)
            payloads=[(plan.send_x,plan.pair_x),(plan.send_ids,plan.pair_ids),(plan.send_scores,plan.pair_scores)]
            own_mask=_mask(plan,ids,valid_tokens)
        plan._source_forward_available=False
        plan._source_forward_tensors=None
        plan._source_forward_masks=None
        plan.send_dy.zero_();plan.send_dy[:dy.shape[0]].copy_(dy)
        work=_start(plan,[*payloads,(plan.send_dy,plan.pair_dy)])
        if plan.hybrid_source_serialized:_join(plan,work)
        own_dx,dw1,dw2,own_ds=_backward(plan.inner,reuse,dy,x,own_mask,scores,w1,w2,valid_tokens,accumulators)
        if not plan.hybrid_source_serialized:_join(plan,work)
        if not reuse:remote_mask=_mask(plan,plan.source_remote_ids)
        remote_dx,dw1,dw2,remote_ds=_backward(plan.remote_inner,reuse,plan.source_remote_dy,
            plan.source_remote_x,remote_mask,plan.source_remote_scores,w1,w2,plan.inner_valid,(dw1,dw2))
        result=(plan._reduce_pair(_pair_result(plan,own_dx,remote_dx),x.shape[0]),dw1,dw2,
                plan._reduce_pair(_pair_result(plan,own_ds,remote_ds),x.shape[0]))
        if plan.check_errors and not torch.cuda.is_current_stream_capturing():plan.check()
        return result
