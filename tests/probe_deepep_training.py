"""Check installed ElasticBuffer expanded forward/backward and graph contracts."""

from datetime import timedelta
import argparse
import gc
import json
import os

import torch
import torch.distributed as dist
import deep_ep

from moe_distributed_utils import destroy_training_process_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--unexpanded', action='store_true')
    parser.add_argument('--allow-hybrid-mode', action='store_true')
    parser.add_argument('--nondeterministic', action='store_true')
    args = parser.parse_args()
    unexpanded = args.legacy or args.unexpanded
    rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=180))
    t, h, k, experts, alignment = 8, 256, 2, size * 2, 128
    x = ((torch.arange(t*h, device=device).reshape(t, h)
          + torch.arange(t, device=device)[:, None] * 3 + rank) % 16).bfloat16()
    ids = (torch.arange(t*k, device=device).reshape(t, k) + rank) % experts
    ids[0, 1] = -1
    scores = torch.full((t, k), .5, device=device)
    if args.legacy:
        deep_ep.Buffer.set_num_sms(8)
        configs = (deep_ep.Buffer.get_dispatch_config(size), deep_ep.Buffer.get_combine_config(size))
        nvl_bytes = max(config.get_nvl_buffer_size_hint(h * 2, size) for config in configs)
        rdma_bytes = max(config.get_rdma_buffer_size_hint(h * 2, size) for config in configs)
        buffer = deep_ep.Buffer(dist.group.WORLD, nvl_bytes, rdma_bytes, explicitly_destroy=True)
    else:
        buffer = deep_ep.ElasticBuffer(dist.group.WORLD, num_max_tokens_per_rank=t,
            hidden=h, num_topk=k, deterministic=not args.nondeterministic, allow_hybrid_mode=args.allow_hybrid_mode,
            explicitly_destroy=True)

    def combine(payload, handle, weights=None):
        options = (dict(async_finish=True) if args.legacy else
                   dict(num_sms=8, async_with_compute_stream=True))
        result, grad_scores, event = buffer.combine(payload, handle, topk_weights=weights, **options)
        event.current_stream_wait()
        return result, grad_scores

    def dispatch():
        if args.legacy:
            counts, rdma_counts, expert_counts, in_rank, event = buffer.get_dispatch_layout(
                ids, experts, async_finish=True)
            event.current_stream_wait()
            rx, ri, rw, _, handle, event = buffer.dispatch(x, topk_idx=ids, topk_weights=scores,
                num_tokens_per_rank=counts, num_tokens_per_rdma_rank=rdma_counts,
                num_tokens_per_expert=expert_counts, is_token_in_rank=in_rank,
                num_worst_tokens=size*t, async_finish=True)
            event.current_stream_wait()
            rw = rw.masked_fill(ri < 0, 0)
            return rx, rw, handle, (ri >= 0).any(-1)
        rx, ri, rw, handle, event = buffer.dispatch(x, topk_idx=ids, topk_weights=scores,
            num_experts=experts, expert_alignment=alignment, num_sms=8,
            do_cpu_sync=False, do_expand=not unexpanded, async_with_compute_stream=True)
        event.current_stream_wait()
        if unexpanded:
            return rx, rw.masked_fill(ri < 0, 0), handle, (ri >= 0).any(-1)
        positions = torch.arange(rx.shape[0], device=device)
        mask = torch.zeros(rx.shape[0], device=device, dtype=torch.bool)
        previous = torch.zeros((), device=device, dtype=torch.int32)
        for end in handle.psum_num_recv_tokens_per_expert:
            mask.logical_or_((positions >= previous) & (positions < end))
            previous = ((end + alignment - 1) // alignment) * alignment
        return rx, rw, handle, mask

    def forward():
        rx, rw, handle, mask = dispatch()
        weights = rw.sum(-1, keepdim=True) if unexpanded else rw.reshape(-1, 1)
        payload = (rx.float() * weights).bfloat16()
        payload.masked_fill_(~mask[:, None], 0)
        y, _ = combine(payload, handle)
        return y

    rx, rw, handle, mask = dispatch()
    if rank == 0 and not unexpanded:
        print(json.dumps(dict(case='deepep_expanded_layout', recv_x=list(rx.shape),
            recv_weights=list(rw.shape), psum=handle.psum_num_recv_tokens_per_expert.cpu().tolist(),
            cpu_counts=handle.num_recv_tokens_per_expert_list,
            scaleup=buffer.num_scaleup_ranks, scaleout=buffer.num_scaleout_ranks)), flush=True)
    y = forward()
    expected_y = (x.float() * (scores * (ids >= 0)).sum(-1, keepdim=True)).bfloat16()
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)

    dy = ((torch.arange(t, device=device)[:, None] % 3 + 1).expand_as(x)).bfloat16().contiguous()
    if args.legacy:
        rdy, _, _, _, _, event = buffer.dispatch(dy, handle=handle, async_finish=True)
        backward_handle = handle
    elif unexpanded:
        rdy, _, _, backward_handle, event = buffer.dispatch(dy, handle=handle,
            do_expand=False, do_cpu_sync=False, num_sms=8, async_with_compute_stream=True)
    else:
        # Installed2.2 explicitly rejects cached expanded dispatch. Recreate
        # deterministic expanded routing and use its handle for the backward
        # combine; the identity dScore check also verifies matching row order.
        rdy, _, _, backward_handle, event = buffer.dispatch(dy, topk_idx=ids,
            topk_weights=scores, num_experts=experts, expert_alignment=alignment,
            do_expand=True, do_cpu_sync=False, num_sms=8, async_with_compute_stream=True)
    event.current_stream_wait()
    rdy.masked_fill_(~mask[:, None], 0)
    weights = rw.sum(-1, keepdim=True) if unexpanded else rw.reshape(-1, 1)
    dx_rows = (rdy.float() * weights).bfloat16()
    ds_rows = (rx.float() * rdy.float()).sum(-1).contiguous()
    ds_rows.masked_fill_(~mask, 0)
    if unexpanded:
        ds_rows = ds_rows[:, None].expand_as(rw).contiguous()
        ds_rows.masked_fill_(rw == 0, 0)  # Probe scores are .5 in every valid slot.
    dx, ds = combine(dx_rows, backward_handle, ds_rows)
    expected_dx = (dy.float() * (scores * (ids >= 0)).sum(-1, keepdim=True)).bfloat16()
    expected_ds = (x.float() * dy.float()).sum(-1, keepdim=True).expand_as(scores) * (ids >= 0)
    torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
    torch.testing.assert_close(ds, expected_ds, rtol=0, atol=0)

    for _ in range(3):
        forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gy = forward()
    for _ in range(4):
        ids.copy_(ids.roll(1, 0))
        graph.replay()
        torch.cuda.synchronize()
        expected_y = (x.float() * (scores * (ids >= 0)).sum(-1, keepdim=True)).bfloat16()
        torch.testing.assert_close(gy, expected_y, rtol=0, atol=0)
    if rank == 0:
        print(json.dumps(dict(case='deepep_expanded_training_probe', ranks=size,
                             backend='legacy' if args.legacy else 'elastic_nonexpanded' if unexpanded else 'elastic_expanded',
                             deterministic=not args.nondeterministic, allow_hybrid_mode=args.allow_hybrid_mode,
                             scaleup=getattr(buffer, 'num_scaleup_ranks', None),
                             scaleout=getattr(buffer, 'num_scaleout_ranks', None),
                             forward=True, backward=True, dynamic_graph_replays=4,
                             passed=True)), flush=True)
    del graph, gy, rx, rw, handle, backward_handle, mask, rdy, dx, ds, dx_rows, ds_rows
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    buffer.destroy()
    destroy_training_process_group()


if __name__ == '__main__':
    main()
