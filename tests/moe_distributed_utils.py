"""Keep the rendezvous server alive until all ranks finish PG destruction."""

from datetime import timedelta

import torch
import torch.distributed as dist


def destroy_training_process_group():
    if not dist.is_initialized():
        return
    rank, size = dist.get_rank(), dist.get_world_size()
    # Retain the TCPStore server on rank 0 even after the default group drops
    # its reference. Other hosts can still be completing communicator cleanup.
    store = dist.distributed_c10d._get_default_store()
    torch.cuda.synchronize()
    # Device barriers can complete locally before another host finishes its
    # receive-side work. Keep every communicator alive until all devices are
    # quiescent; a CPU rendezvous introduces no new GPU collective to drain.
    ready = [f"deep_gemm_training_device_ready_{peer}" for peer in range(size)]
    store.set(ready[rank], "done")
    store.wait(ready, timedelta(seconds=300))
    dist.destroy_process_group()
    keys = [f"deep_gemm_training_shutdown_{peer}" for peer in range(size)]
    store.set(keys[rank], "done")
    store.wait(keys, timedelta(seconds=300))
    # Once nonzero ranks acknowledge the completed wait they make no more
    # requests. Rank 0 must not exit just because every rank published `done`:
    # another client may still be entering its final store.wait().
    acknowledgements = [f"deep_gemm_training_shutdown_ack_{peer}" for peer in range(1, size)]
    if rank:
        store.set(acknowledgements[rank - 1], "done")
    elif acknowledgements:
        store.wait(acknowledgements, timedelta(seconds=300))
