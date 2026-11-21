"""A failed pair warmup must stop every rank before domain peer payloads."""
import gc
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from deep_gemm.training.hierarchical import HierarchicalPeerMoEPlan, make_hierarchical_groups
from moe_distributed_utils import destroy_training_process_group


def main():
    rank, size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if size != 4:
        raise ValueError('run four ranks on four PPUs for this failure-propagation check')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=120))
    metadata = dist.new_group(backend='gloo', timeout=timedelta(seconds=120))
    groups = make_hierarchical_groups(2, metadata)
    plan = HierarchicalPeerMoEPlan(256, 128, 1, 2, 8, groups=groups,
                                  device=device, chunk_tokens=8)
    torch.manual_seed(781 + rank)
    w1 = (torch.randn(1, 256, 256, device=device) * .02).bfloat16()
    w2 = (torch.randn(1, 256, 128, device=device) * .02).bfloat16()
    gather = dist.all_gather_into_tensor

    def failing_pair(output, source, *args, **kwargs):
        # Both members of pair 0 fail synchronously; pair 1 warms normally.
        # Injecting only one side would test a transport timeout instead.
        if kwargs.get('group') is groups.pair and rank % 2 == 0:
            raise RuntimeError('injected pair warmup failure')
        return gather(output, source, *args, **kwargs)

    dist.all_gather_into_tensor = failing_pair
    caught = False
    try:
        plan.prepare(w1, w2)
    except RuntimeError as error:
        caught = 'injected pair warmup failure' in str(error)
    finally:
        dist.all_gather_into_tensor = gather
    assert caught and not plan._prepared, (rank, caught, plan._prepared)
    votes = [None] * size
    dist.all_gather_object(votes, caught, group=metadata)
    assert all(votes)
    # A failed preparation must remain retryable after the host error clears.
    plan.prepare(w1, w2)
    x = torch.randn(8, 256, device=device).bfloat16() * .1
    ids = torch.arange(16, device=device).reshape(8, 2) % size
    scores = torch.full((8, 2), .5, device=device)
    valid = torch.tensor(8, dtype=torch.int32, device=device)
    y = plan.forward(x, ids, scores, w1, w2, valid)
    grads = plan.backward(x, x, ids, scores, w1, w2, valid)
    assert all(bool(torch.isfinite(t).all()) for t in (y, *grads))
    plan.check()
    assert groups._devices == {device.index}
    print(json.dumps(dict(case='hierarchical_prepare_failure', rank=rank,
                          global_failure_observed=caught, retry_passed=True, passed=True)), flush=True)
    plan.close()
    del plan, y, grads
    gc.collect()
    groups.close()
    groups.close()  # Caller-owned groups close idempotently.
    destroy_training_process_group()


if __name__ == '__main__':
    main()
