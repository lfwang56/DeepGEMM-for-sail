"""Check PPU symmetric peer mapping within an explicitly selected EP domain."""

from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist
from torch.distributed import _symmetric_memory as sm


rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=120))
print(json.dumps({"case": "symmetric_probe_start", "rank": rank, "world": size}), flush=True)
try:
    local = sm.empty((128,), dtype=torch.float32, device=device)
    handle = sm.rendezvous(local, dist.group.WORLD)
except RuntimeError as error:
    print(json.dumps({"case": "symmetric_probe_unsupported", "rank": rank, "error": str(error)}), flush=True)
    dist.destroy_process_group()
else:
    direct = getattr(handle, "world_within_direct_access", True)
    direct = direct() if callable(direct) else direct
    if not direct:
        print(json.dumps({"case": "symmetric_probe_unsupported", "rank": rank, "error": "world outside direct-access domain"}), flush=True)
    else:
        local.fill_(rank + 1)
        torch.cuda.synchronize()
        dist.barrier()
        peers = [handle.get_buffer(peer, (128,), torch.float32) for peer in range(size)]
        for peer, buffer in enumerate(peers):
            torch.testing.assert_close(buffer.cpu(), torch.full((128,), peer + 1.), rtol=0, atol=0)
        for buffer in peers:
            buffer[rank] = 40 + rank
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(local[:size].cpu(), torch.arange(size, dtype=torch.float32) + 40, rtol=0, atol=0)
        print(json.dumps({"case": "symmetric_probe_passed", "rank": rank, "world": size,
                          "peer_read": True, "peer_write": True, "direct_access": bool(direct)}), flush=True)
        del peers, handle, local
    dist.barrier()
    dist.destroy_process_group()
