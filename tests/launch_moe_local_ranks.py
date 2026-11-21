"""Launch this host's distributed test ranks without an elastic agent.

Run one instance on every host with matching rendezvous options and distinct
node ranks. Each child gets its own log; the parent returns after every child
has exited. There is no additional TCPStore exit barrier after the test closes
its process group.
"""

import argparse
import json
import os
import resource
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--local-ranks", type=int, default=8)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--log-directory", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    # Preserve per-rank diagnostics without writing a process-sized core for
    # every rank when a distributed runtime aborts after a timeout.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    if not command or not 0 <= options.node_rank < options.nodes or options.local_ranks < 1:
        parser.error("command, valid node rank and positive local rank count are required")
    directory = Path(options.log_directory)
    directory.mkdir(parents=True, exist_ok=True)
    children, logs = [], []
    for local_rank in range(options.local_ranks):
        rank = options.node_rank * options.local_ranks + local_rank
        environment = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(local_rank),
                           WORLD_SIZE=str(options.nodes * options.local_ranks),
                           LOCAL_WORLD_SIZE=str(options.local_ranks), MASTER_ADDR=options.master_addr,
                           MASTER_PORT=str(options.master_port), PYTHONUNBUFFERED="1")
        output = (directory / f"rank{rank:03d}.log").open("w")
        logs.append(output)
        children.append(subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT))
    print(json.dumps({"case": "local_ranks_started", "node_rank": options.node_rank,
                      "pids": [child.pid for child in children], "log_directory": str(directory)}), flush=True)
    while True:
        codes = [child.poll() for child in children]
        if all(code is not None for code in codes):
            break
        if any(code is not None and code != 0 for code in codes):
            # A failed rank cannot participate in later collectives. Stop only
            # siblings owned by this launcher instead of waiting forever on
            # an earlier rank that is blocked inside such a collective.
            for child, code in zip(children, codes):
                if code is None:
                    child.terminate()
            deadline = time.monotonic() + 3
            for child in children:
                try:
                    child.wait(timeout=max(.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    child.kill()
            break
        time.sleep(.05)
    codes = [child.wait() for child in children]
    for output in logs:
        output.close()
    print(json.dumps({"case": "local_ranks_finished", "node_rank": options.node_rank,
                      "return_codes": codes, "passed": all(code == 0 for code in codes)}), flush=True)
    sys.exit(0 if all(code == 0 for code in codes) else 1)


if __name__ == "__main__":
    main()
