"""Topology-labelled PCCL transport checks across hosts/supernodes."""

import argparse
from datetime import timedelta
import json
import os
import socket
import time

import torch
import torch.distributed as dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("same-supernode", "cross-supernode"), required=True)
    options = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    print(json.dumps({"case": "transport_start", "rank": rank, "host": socket.gethostname(),
                      "topology": options.topology, "torch": torch.__version__}), flush=True)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180),
                            device_id=torch.device("cuda", local_rank))
    values = torch.full((1024 * 1024,), rank + 1, device="cuda", dtype=torch.float32)
    dist.all_reduce(values)
    expected = size * (size + 1) / 2
    torch.testing.assert_close(values, torch.full_like(values, expected), rtol=0, atol=0)
    local = torch.full((128, 256), rank + 1, device="cuda", dtype=torch.bfloat16)
    gathered = torch.empty((size * 128, 256), device="cuda", dtype=torch.bfloat16)
    dist.all_gather_into_tensor(gathered, local)
    for peer in range(size):
        torch.testing.assert_close(gathered[peer * 128:(peer + 1) * 128],
                                   torch.full_like(local, peer + 1), rtol=0, atol=0)
    reduced = torch.empty_like(local)
    dist.reduce_scatter_tensor(reduced, gathered)
    torch.testing.assert_close(reduced, torch.full_like(local, size * (rank + 1)), rtol=0, atol=0)
    dist.barrier()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(10):
        local.fill_(rank + 1)
        dist.all_gather_into_tensor(gathered, local)
        dist.reduce_scatter_tensor(reduced, gathered)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - begin) * 1000 / 10
    print(json.dumps({"case": "transport_passed", "rank": rank, "topology": options.topology,
                      "ep_size": size, "gather_reduce_pair_ms": elapsed}), flush=True)
    dist.destroy_process_group()
