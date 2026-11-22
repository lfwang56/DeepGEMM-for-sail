"""Use an idle eight-device host for three independent matched LM seeds."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=("collective", "peer", "hybrid", "elastic"), required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--compute", choices=("bf16", "fp8"), default="bf16")
parser.add_argument("--fp8-async-weights", action="store_true")
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--peer-recipe", choices=("cache", "producer", "producer-input", "producer-expert", "producer-window"), default="cache")
parser.add_argument("--source-chunk-tokens", type=int, default=128)
parser.add_argument("--gemm-stages", type=int, choices=(2,3), default=2)
parser.add_argument("--gemm-config", type=int, nargs=5,
                    help="BN BK WM WN stages; overrides --gemm-stages")
parser.add_argument("--block-m", type=int, choices=(128, 256), default=128)
parser.add_argument("--gate-up-layout", choices=("split", "g8"), default="split")
parser.add_argument("--fuse-fc1-activation", action="store_true")
parser.add_argument("--fuse-dhidden-scores", action="store_true")
parser.add_argument("--transpose-free-dgrad", action="store_true")
parser.add_argument("--dag-fusion", action="store_true")
parser.add_argument("--dag-backward", action="store_true")
parser.add_argument("--dag-direct-return", action="store_true")
parser.add_argument("--input-copy-ilp", type=int, choices=(1, 2, 4, 8), default=1)
parser.add_argument("--fixed-route-seeds", default="")
parser.add_argument("--freeze-library", action="store_true")
args = parser.parse_args()
if args.input_copy_ilp != 1 and args.backend not in ("peer", "hybrid"):
    parser.error("input copy ILP requires the peer backend")
if (args.dag_fusion or args.dag_backward or args.dag_direct_return) and args.backend not in ("peer", "hybrid"):
    parser.error("DAG options require the peer backend")
root = Path.cwd()
directory = root / "build/campaign" / args.name
directory.parent.mkdir(parents=True, exist_ok=True)
directory.mkdir(exist_ok=False)
source_paths = [root / p for p in (
    "deep_gemm/training/native.py", "deep_gemm/training/layout.py", "deep_gemm/training/deepep.py", "deep_gemm/training/routed_rows.py",
    "deep_gemm/training/peer.py", "deep_gemm/training/hierarchical.py", "deep_gemm/include/deep_gemm/impls/peer_moe_gemm.cuh",
    "deep_gemm/include/deep_gemm/impls/peer_moe_training.cuh",
    "deep_gemm/include/deep_gemm/common/peer_moe_training.cuh",
    "deep_gemm/include/deep_gemm/impls/moe_training.cuh",
    "deep_gemm/include/deep_gemm/common/moe_training.cuh",
    "deep_gemm/include/deep_gemm/impls/peer_moe_dag.cuh",
    "deep_gemm/include/deep_gemm/common/peer_moe_dag.cuh",
    "csrc/apis/peer_moe_dag.hpp", "csrc/apis/peer_moe_training.hpp", "tests/train_moe_lm.py", "tests/evaluate_moe_lm.py",
    "tests/train_moe_lm_distributed.py", "tests/moe_distributed_utils.py", "tests/run_moe_lm_seeds.py")]
source_paths.extend((root / "deep_gemm").glob("_C*.so"))
# Editable installs expose vendor headers through directory symlinks. Hash the
# same dereferenced tree that copytree freezes, including those dependencies.
source_paths = sorted(set(source_paths) | {
    Path(directory) / name
    for directory, _, files in os.walk(root / 'deep_gemm', followlinks=True)
    for name in files if Path(name).suffix in ('.py', '.cuh', '.hpp', '.h')
})
try:
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    dirty_files = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
except (subprocess.CalledProcessError, FileNotFoundError):
    source_commit = None  # The source archive still records every file hash.
    dirty_files = None
metadata = dict(commit=source_commit, dirty_tracked_files=dirty_files,
                files={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
lut_directory = os.environ.get('SAIL_DEEPGEMM_LUT_PATH')
if lut_directory:
    lut_directory = Path(lut_directory).resolve(strict=True)
    lut_hashes = {str(p.relative_to(lut_directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in lut_directory.rglob('*') if p.is_file()}
    if not lut_hashes:
        raise RuntimeError('the requested runtime LUT directory is empty')
    metadata['runtime_lut'] = dict(source_directory=str(lut_directory), files=lut_hashes)
else:
    metadata['runtime_lut'] = None
library_parent = root
if args.freeze_library:
    library_parent = directory / 'frozen-library'
    shutil.copytree(root / 'deep_gemm', library_parent / 'deep_gemm',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in source_paths:
        if path.is_relative_to(root / 'deep_gemm'):
            copied = library_parent / path.relative_to(root)
            if hashlib.sha256(copied.read_bytes()).hexdigest() != metadata['files'][str(path.relative_to(root))]:
                raise RuntimeError('library changed while freezing: '+str(path))
    frozen_tests = library_parent / 'tests'
    frozen_tests.mkdir()
    for test_name in ('train_moe_lm.py', 'train_moe_lm_distributed.py', 'evaluate_moe_lm.py', 'moe_distributed_utils.py'):
        shutil.copy2(root / 'tests' / test_name, frozen_tests / test_name)
    if lut_directory:
        frozen_lut = library_parent / 'runtime-lut'
        shutil.copytree(lut_directory, frozen_lut)
        if {str(p.relative_to(frozen_lut)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in frozen_lut.rglob('*') if p.is_file()} != lut_hashes:
            raise RuntimeError('runtime LUT changed while freezing')
        lut_directory = frozen_lut
if lut_directory:
    metadata['runtime_lut']['execution_directory'] = str(lut_directory)
(directory / "source_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
gemm_config = list(map(str, args.gemm_config or (256, 64, 64, 64, args.gemm_stages)))
recipe = (["--dispatch-cache-n", "16", "--dispatch-cache-aiu", "--reuse-packed-dy",
           "--gemm-config", *gemm_config] if args.backend in ("peer", "hybrid") else [])
if args.backend in ("peer", "hybrid") and args.peer_recipe in ("producer", "producer-input", "producer-expert", "producer-window"):
    recipe = ["--dispatch-producer", "--dispatch-producer-tiles", "1", "--reuse-packed-dy",
              "--gemm-config", *gemm_config]
    if args.peer_recipe in ("producer-input", "producer-expert", "producer-window"):
        recipe.extend(["--producer-input-aiu", "--producer-local-aiu", "--fuse-activation-backward"])
    if args.peer_recipe in ("producer-expert", "producer-window"):
        recipe.append("--expert-local-grad")
    if args.peer_recipe == "producer-window":
        recipe.extend(["--ring-rows", "65536"])
if args.backend == "elastic":
    recipe = ["--ring-rows", "65536", "--deepep-comm-sms", "38"]
    if args.compute == "fp8":
        recipe += ["--fp8-active-quantization", "--fp8-mn-major-scales"]
        if args.fp8_async_weights:
            recipe.append("--fp8-async-weights")
fixed_route_seeds = {int(value) for value in args.fixed_route_seeds.split(",") if value}
if args.dag_fusion:
    recipe = [flag for flag in recipe if flag not in ("--dispatch-producer", "--producer-input-aiu", "--producer-local-aiu")]
for enabled, flag in ((args.dag_fusion, "--dag-fusion"), (args.dag_backward, "--dag-backward"),
                      (args.dag_direct_return, "--dag-direct-return")):
    if enabled:
        recipe.append(flag)
children = []
for index, seed in enumerate((4827, 4891, 6113)):
    command = ["torchrun", "--standalone", "--nproc-per-node", "2", "--max-restarts", "0",
        str(library_parent / "tests/train_moe_lm_distributed.py") if args.freeze_library else "tests/train_moe_lm_distributed.py", "--backend", args.backend, "--compute", args.compute,
        "--graph", "--reference-check", "--steps", str(args.steps), "--seed", str(seed),
        "--eval-interval", "200", "--eval-batches", "20", "--hidden", "512", "--layers", "4",
        "--experts", "8", "--global-batch", "8", "--sequence", "128", "--chunk-tokens", str(args.source_chunk_tokens),
        "--block-m", str(args.block_m), "--gate-up-layout", args.gate_up_layout,
        "--input-copy-ilp", str(args.input_copy_ilp),
        "--checkpoint", str(directory / f"seed{seed}.pt"), *recipe]
    if args.fuse_fc1_activation:
        command.append("--fuse-fc1-activation")
    if args.fuse_dhidden_scores:
        command.append("--fuse-dhidden-scores")
    if args.transpose_free_dgrad:
        command.append("--transpose-free-dgrad")
    if seed in fixed_route_seeds:
        command.append("--reference-fixed-routes")
    output = (directory / f"seed{seed}.log").open("w")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=f"{2*index},{2*index+1}", OMP_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(library_parent)+":"+str(library_parent / "tests")+":.:tests")
    if args.backend == "elastic":
        environment['PYTHONPATH'] = str(library_parent)+":"+str(library_parent / "tests")+":build/campaign/deepep-native-scope:.:tests"
        environment['DEEPEP_NUM_MAX_ICN_PEERS'] = "2"
    if lut_directory:
        environment['SAIL_DEEPGEMM_LUT_PATH'] = str(lut_directory)
    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=environment)
    children.append((seed, child, output))
    print(json.dumps(dict(case="lm_seed_started", seed=seed, pid=child.pid,
                          devices=environment["CUDA_VISIBLE_DEVICES"], command=command)), flush=True)
results = []
for seed, child, output in children:
    code = child.wait()
    output.close()
    results.append(dict(seed=seed, returncode=code))
    print(json.dumps(dict(case="lm_seed_finished", seed=seed, returncode=code)), flush=True)
print(json.dumps(dict(case="lm_seed_matrix_finished", backend=args.backend,
                      results=results, passed=all(r["returncode"] == 0 for r in results))), flush=True)
sys.exit(0 if all(r["returncode"] == 0 for r in results) else 1)
