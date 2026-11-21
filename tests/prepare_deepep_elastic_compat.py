"""Prepare an isolated installed-DeepEP copy for native PPU JIT headers."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
args = parser.parse_args()
root = Path(args.root)
original = Path(importlib.util.find_spec("deep_ep").origin).parent
parent = root / "build/campaign/deepep-native-scope"
destination = parent / "deep_ep"
if not destination.exists():
    shutil.copytree(original, destination)
records = []
for name in ("handle.cuh", "comm.cuh"):
    source = original / "include/deep_ep/common" / name
    target = destination / "include/deep_ep/common" / name
    contents = source.read_text()
    if name == "handle.cuh":
        contents = contents.replace("cuda::thread_scope_thread", "hggc::thread_scope_thread")
        contents = contents.replace("cuda::thread_scope_device", "hggc::thread_scope_device")
    else:
        # Native half aliases/guards must precede cooperative_groups' CUDA
        # half headers, including the standalone teardown barrier kernel.
        contents = contents.replace("#include <cooperative_groups.h>\n#include <pccl.h>",
                                    "#include <pccl.h>\n#include <cooperative_groups.h>")
    target.write_text(contents)
    records.append(dict(path=str(target.relative_to(parent)),
                        original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        patched_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
bridge = root / "build/campaign/compat/deepep/pccl.h"
bridge.parent.mkdir(parents=True, exist_ok=True)
bridge.write_text("""#pragma once
// Native PCCL declarations before CUDA-compatible JIT headers.
#include <hggc_runtime_api.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#define __CUDA_FP16_H__
#define __CUDA_FP16_HPP__
#define __CUDA_BF16_H__
#define __CUDA_BF16_HPP__
using __nv_bfloat16 = __ppu_bfloat16;
using __nv_bfloat162 = __ppu_bfloat162;
using __nv_bfloat16_raw = __ppu_bfloat16_raw;
using __nv_bfloat162_raw = __ppu_bfloat162_raw;
using nv_bfloat16 = __ppu_bfloat16;
using nv_bfloat162 = __ppu_bfloat162;
#include_next <pccl.h>
""")
result = dict(original_package=str(original), shadow_package=str(destination),
              patches=records, installed_package_modified=False,
              pccl_bridge=str(bridge), pccl_bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest())
(parent / "compat-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
