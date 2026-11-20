"""A failed compiler must be reported on every EP rank before peer execution."""

import argparse
import json
import os
from unittest.mock import patch

import torch
import torch.distributed as dist

from deep_gemm.training import NativeMoEPlan, PeerMoEPlan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer", action="store_true")
    options = parser.parse_args()
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl", device_id=device)
    plan_class = PeerMoEPlan if options.peer else NativeMoEPlan
    plan = plan_class(256, 128, 2, 2, 5, chunk_tokens=3, device=device, group=dist.group.WORLD)
    w1 = torch.zeros((2, 256, 256), device=device, dtype=torch.bfloat16)
    w2 = torch.zeros((2, 256, 128), device=device, dtype=torch.bfloat16)
    method = "_prepare" if options.peer else "_prepare_local"
    target = getattr(plan, method)

    def fail(*_):
        raise RuntimeError("injected compiler failure")

    replacement = target if rank != 1 else fail
    with patch.object(plan, method, replacement):
        try:
            plan.prepare(w1, w2)
        except RuntimeError as error:
            assert "injected compiler failure" in str(error), error
        else:
            raise AssertionError("a rank entered execution despite another compiler failing")
    assert not plan._prepared
    plan.prepare(w1, w2)
    assert plan._prepared
    dist.barrier()
    print(json.dumps({"case": "moe_startup_failure_agreement", "rank": rank,
                      "transport": plan.transport, "error_observed": True,
                      "retry_after_fix_passed": True}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
