"""Two independent source-half contexts overlap pair collectives with local work."""
from contextlib import contextmanager
import torch
import torch.distributed as dist
from .native import _launch
from .bf16 import _validate_accumulators
from .peer import PeerMoEPlan


@contextmanager
def _weights_from_owner(plan):
    inner=plan.remote_inner
    previous=inner._shared_fp8_weights_ready
    inner._shared_fp8_weights_ready=plan.hybrid_share_fp8_weights
    try:
        yield
    finally:
        inner._shared_fp8_weights_ready=previous


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


def _start_reduce(plan,partials):
    main=torch.cuda.current_stream(plan.device);aux=plan._source_pair_stream
    outputs=[torch.empty_like(partial[:plan.max_tokens]) for partial in partials]
    aux.wait_stream(main);work=[]
    with torch.cuda.stream(aux):
        for partial,output in zip(partials,outputs):
            partial.record_stream(aux);output.record_stream(aux)
            work.append(dist.reduce_scatter_tensor(output,partial,group=plan.groups.pair,async_op=True))
    return work,outputs


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
        reducer=None
        if plan.hybrid_chunk_reduce:
            from .hierarchical_chunk_reduce import ChunkReductions
            reducer=ChunkReductions(plan,(own,))
        with _weights_from_owner(plan):
            remote=plan.remote_inner.forward(plan.source_remote_x,remote_mask,plan.source_remote_scores,w1,w2,plan.inner_valid,
                _chunk_output_callback=reducer.forward if reducer is not None else None)
        if reducer is not None:
            output=reducer.finish(x.shape[0])[0]
            plan.hybrid_chunk_reduce_calls_last_forward=reducer.calls
        else:
            output=plan._reduce_pair(_pair_result(plan,own,remote),x.shape[0])
        tensors=(x,ids,scores,w1,w2,valid_tokens)
        plan._source_forward_fingerprint=PeerMoEPlan._context_fingerprint(*tensors)
        plan._source_forward_tensors=tensors
        plan._source_forward_masks=(own_mask,remote_mask)
        plan._source_forward_available=True
        if plan.check_errors and not torch.cuda.is_current_stream_capturing():plan.check()
        return output


def _backward(inner,reuse,dy,x,ids,scores,w1,w2,valid,accumulators,defer_dw1=False,chunk_callback=None):
    previous=inner.reuse_forward_context
    if not reuse:
        inner.reuse_forward_context=False
        inner._forward_context_available=False
    try:return inner.backward(dy,x,ids,scores,w1,w2,valid,accumulators,_defer_last_dw1=defer_dw1,_chunk_output_callback=chunk_callback)
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
        reducer=None
        if plan.hybrid_chunk_reduce:
            from .hierarchical_chunk_reduce import ChunkReductions
            reducer=ChunkReductions(plan,(own_dx,own_ds))
        with _weights_from_owner(plan):
            remote_dx,dw1,dw2,remote_ds=_backward(plan.remote_inner,reuse,plan.source_remote_dy,
                plan.source_remote_x,remote_mask,plan.source_remote_scores,w1,w2,plan.inner_valid,(dw1,dw2),
                defer_dw1=reuse and plan.hybrid_tail_dw_overlap,
                chunk_callback=reducer.backward if reducer is not None else None)
        plan.hybrid_deferred_dw1_scheduled_last_backward=plan.remote_inner._pending_last_dw1 is not None
        if reducer is not None:
            reduced_dx,reduced_ds=reducer.finish(x.shape[0])
            plan.hybrid_chunk_reduce_calls_last_backward=reducer.calls
            result=(reduced_dx,dw1,dw2,reduced_ds)
        elif plan.hybrid_tail_dw_overlap:
            partials=(_pair_result(plan,own_dx,remote_dx),_pair_result(plan,own_ds,remote_ds))
            work,outputs=_start_reduce(plan,partials)
            if plan.hybrid_tail_dw_serialized:_join(plan,work)
            plan.remote_inner._run_deferred_dw1()
            if not plan.hybrid_tail_dw_serialized:_join(plan,work)
            result=(outputs[0][:x.shape[0]],dw1,dw2,outputs[1][:x.shape[0]])
        else:
            result=(plan._reduce_pair(_pair_result(plan,own_dx,remote_dx),x.shape[0]),dw1,dw2,
                    plan._reduce_pair(_pair_result(plan,own_ds,remote_ds),x.shape[0]))
        if plan.check_errors and not torch.cuda.is_current_stream_capturing():plan.check()
        return result
