# Qwen3.8 routed-layer BF16 training on PPU

Target: `wlf_sgl_2.2`, Torch2.11.0, SAIL SDK2.2.0, ZW-M890P/ppu0015.
The build manifest identifies the exact source commit, extension/wheel SHA256,
compiler and pinned vendor revisions. Install the matching wheel into an isolated
`pip --target` directory; do not combine a new Python package with an older extension.
The source archive contains vendor sources and standard PKG-INFO for Gitless builds.

## Topology and scope

Each host has 8 PPUs; each designated block of 8 hosts is one physical supernode.
Domain EP32 uses four hosts from ONE supernode. Domain EP64 needs all eight.
Cross-domain EP64 uses four hosts in each of TWO supernodes and is a separate
experiment. Check the whole host before/during/after execution; exclude timings
if another job appears. Never stop other users' processes. Recovered idle hosts
can be reused.

H8192/I2048/totalE512/top10 are the pinned Qwen3.8 routed-expert dimensions.
EP32 has16local experts, EP64 has8. Performance includes FWD, complete backward
rematerialization, DX/DW1/DW2/dScore and FP32 DW zero/accumulation. Attention,
router, shared expert and optimizer are excluded. This is not full2.4T training.
Unprofiled maximum-rank GPU-event medians use20iterations ×5repeats after5warmups.
Profiler rank0 kernel intervals are diagnostic and have a different scope.

## Recommended configuration

Use G8 gate/up weight layout, BM256, producer input/local AIU, input ILP8,
fused FC1/SwiGLU and dHidden/activation backward, packed DY reuse and expert-local
gradients. Use stage3 for all four M-group GEMMs AND an explicit stage3 WGrad LUT:

```bash
PPU_MOE_SOURCE_ROOT="$PWD"
export PYTHONPATH="$PPU_MOE_SOURCE_ROOT:$PPU_MOE_SOURCE_ROOT/tests"
export SAIL_DEEPGEMM_LUT_PATH="$PPU_MOE_SOURCE_ROOT/docs/qwen38_bilateral/wgrad_luts40/stage3"
export DG_JIT_CACHE_DIR="$PPU_MOE_SOURCE_ROOT/build/reproduce/jit-cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
```

When using the isolated wheel, put its target directory first in PYTHONPATH.
M-group `--gemm-config ... 3` does not select the WGrad pipeline; the LUT does.
The directory's preliminary name `wgrad_luts40` is retained in the measured41–44
recipes. Keep auxiliary WG/DY streams, input assistants, DAG/FFN task pools,
replay-gradient fusion and its warp reduction OFF in the recommended recipe.
They passed numerical checks but have not beaten the selected complete-layer
latency. Optional grid budgets default0; they cap persistent CTA counts using
measured occupancy, not physical CU affinity or resource isolation.

```python
import torch
from deep_gemm.training import PeerMoEPlan, native_moe, to_g8_weight

plan = PeerMoEPlan(
    8192, 2048, 16, 10, 4096,
    group=ep_group, device=device, chunk_tokens=4096,
    ring_rows=65536, block_m=256, gate_up_layout="g8",
    dispatch_producer=True, producer_input_aiu=True, producer_local_aiu=True,
    input_copy_ilp=8, fuse_fc1_activation=True,
    expert_local_grad=True, reuse_packed_dy=True,
    fuse_activation_backward=True,
    gemm_configs={r: (256,64,64,64,3) for r in ("fc1","fc2","dhidden","dx")},
)
# Convert once before constructing the optimizer, not on every forward.
w1 = torch.nn.Parameter(to_g8_weight(canonical_w1))
w2 = torch.nn.Parameter(canonical_w2)
y = native_moe(x, ids, scores, w1, w2, plan=plan, valid_tokens=valid)
y.backward(dy)
plan.check()  # Outside graph capture.
```

X/DY are contiguous BF16, scores FP32, IDs int32/int64 with masked slots -1.
`valid` is an int32 device scalar. BF16 or FP32 master parameters receive FP32
expert accumulation. `from_g8_weight` restores canonical checkpoint layout.
`native_moe(..., weight_grad_accumulators=(dw1,dw2))` writes caller-owned FP32
accumulators and leaves parameter `.grad` unset; the caller clears/reduces them
at its intended boundary. No optimizer or Megatron/DP gradient hook is installed
automatically. Higher derivatives are unsupported. Warm the complete FWD/BWD
before graph capture. Replay can change X/DY/scores/IDs/valid at fixed capacities.
Check device status outside capture. Per-window joins protect buffer reuse.

## Actual domain-EP32 launch

Run one launcher on each of four otherwise idle hosts in the same supernode.
Use the same master/port/log directory and distinct node ranks0..3:

```bash
PPU_MOE_NODE_RANK=0
PPU_MOE_MASTER=172.17.0.219
python tests/launch_moe_local_ranks.py \
  --node-rank "$PPU_MOE_NODE_RANK" --nodes 4 --local-ranks 8 \
  --master-addr "$PPU_MOE_MASTER" --master-port 29874 \
  --log-directory "$PPU_MOE_SOURCE_ROOT/build/reproduce/peer-t4096" -- \
  python tests/benchmark_native_moe.py --backend peer --deepep-scaleup \
    --hidden 8192 --intermediate 2048 --local-experts 16 --topk 10 \
    --tokens 4096 --chunk-tokens 4096 --ring-rows 65536 --block-m 256 \
    --gate-up-layout g8 --gemm-config 256 64 64 64 3 \
    --dispatch-producer --producer-input-aiu --producer-local-aiu \
    --expert-local-grad --reuse-packed-dy --fuse-activation-backward \
    --fuse-fc1-activation --input-copy-ilp 8 --scope step
```

For the strongest installed DeepEP baseline, prepare the isolated native-HGGC
header adapter (the installed package remains unchanged):

```bash
python tests/prepare_deepep_elastic_compat.py --root "$PPU_MOE_SOURCE_ROOT"
export PYTHONPATH="$PPU_MOE_SOURCE_ROOT/build/campaign/deepep-native-scope:$PYTHONPATH"
export CPATH="$PPU_MOE_SOURCE_ROOT/build/campaign/compat/deepep:/usr/local/pccl/include:/usr/local/pccl/nccl_wrapper/include:/usr/local/PPU_SDK/include"
```

Use the same launcher, dimensions, BF16, G8, MGroup/WGrad stage3, ring and timing
scope, with `--backend elastic --deepep-scaleup --deepep-comm-sms 38
--overlap-expert-dy`. Keep the original Buffer `--backend deepep` and serial-DY
controls in the evidence; the fastest accepted dispatch/combine result is the
denominator. AG/RS collectives are auxiliary correctness/performance references.
Elastic requires unique top-k IDs within each token and nondeterministic routing.
The benchmark's balanced case generates random top-k IDs, not adjacent expert IDs.

Repeat `--scope forward`, `backward`, `step`. T1024 uses C1024; T8192 uses C4096.
Backward includes replay and DW zeroing. Separately measured FWD+BWD can differ
slightly from the complete-step measurement. Audit every host and preserve all
rank logs; invalid/partial/interfered phases contribute no accepted latency.

## Numerical and lifecycle checks

```bash
python tests/test_setup_source_metadata.py
python tests/test_moe_training_large.py --backend peer \
  --hidden 8192 --intermediate 2048 --experts 16 --topk 10 --tokens 80 \
  --ring-rows 65536 --block-m 256 --gate-up-layout g8 \
  --gemm-config 256 64 64 64 3 --dispatch-producer \
  --producer-input-aiu --producer-local-aiu --fuse-fc1-activation \
  --input-copy-ilp 8 --expert-local-grad --reuse-packed-dy --fuse-activation-backward
python tests/test_moe_return_cache_equivalence.py --gemm-stages 3 --input-copy-ilp-ablation 8
```

On all 32 ranks use `tests/test_deepep_moe_large_distributed.py --backend peer`
with localE16, ring65536, BM256, G8, stage3 and graph; add FC1 activation fusion
and ILP8. The test compares every Y/DX/DW1/DW2/dScore element to independent FP32,
including empty, masked, reversed and duplicate/skew routing. For Elastic use
`tests/test_moe_elastic_distributed.py --no-elastic-hybrid-mode --comm-sms 38`
with matching ring/layout/tiles/graph. Its skew uses unique expert IDs.
`tests/test_moe_training_skew_capacity.py --backends peer --chunk-tokens 4096`
with the same expert-local/G8/stage3 flags separately validates full dropless
capacity. That script selects producer mode automatically; it has no
`--dispatch-producer` CLI flag. Do not change numerical thresholds to accept a
failed candidate. Sampled benchmark outputs do not replace these references.

## Two-domain EP64

Global ranks must be ordered as two contiguous physical domains. Initialize the
default GPU process group and a complete ordered Gloo metadata group, then call
`make_hierarchical_groups(domain_size=32, metadata_group=metadata)` collectively.
Pass the returned caller-owned `groups` into `HierarchicalPeerMoEPlan` with H8192,
I2048, localE8, top10 and the same peer options. Close all plans/device work before
`groups.close()`. Groups can be shared by layers. Pair payload AG/RS runs on GPU;
Gloo carries metadata only. Backward repeats dispatch and full expert replay,
reduces DX/dScore, and keeps each expert's FP32 DW local.

Use `tests/test_moe_hierarchical_distributed.py --hybrid-domain-size 32` on all 64
ranks for the independent reference. Domain partials are BF16-rounded, so flat
DeepEP rounding differs. The four-rank
`tests/test_moe_hierarchical_prepare_failure.py` injects a pair warmup failure,
checks global failure propagation and validates a successful retry.

Benchmark `--backend hybrid --hybrid-domain-size 32`; the cross-domain Elastic
baseline uses `--backend elastic --elastic-hybrid-mode` without `--deepep-scaleup`.
Match C2048 first. Separately tuned Hybrid C4096 versus Elastic C2048 must be
labeled as different capacities. Cross64 DeepEP C4096 exceeds its int32 receive
indexing guard; C3840 is valid but was slower than C2048. Cross-domain64 results
cannot be relabeled as same-domain64.

## Public-text whole-model validation

The 57.8M four-layer H512/I1024/E8/top2 model trains all parameters on public Tiny
Shakespeare with dynamic routing. Three paired seeds4827/4891/6113 run2000updates,
using globalB8, sequence128 and C256. To exercise the selected WGrad stage3 even
at these small shapes, use the quality LUT with explicit small DW keys:

```bash
export SAIL_DEEPGEMM_LUT_PATH="$PPU_MOE_SOURCE_ROOT/docs/qwen38_bilateral/quality_luts/wgrad-stage3"
python tests/run_moe_lm_seeds.py --backend peer --name lm-peer-wg3 \
  --steps 2000 --source-chunk-tokens 256 --block-m 256 --gate-up-layout g8 \
  --gemm-stages 3 --fuse-fc1-activation --input-copy-ilp 8 \
  --peer-recipe producer-window --freeze-library --fixed-route-seeds 4827,4891,6113
```

Repeat with backend `hybrid` and a different name. Each seed uses two PPUs;
Hybrid quality uses two logical one-rank domains on that host, while actual
cross-domain64 correctness is validated separately. The fixed-route option
freezes branch choices ONLY for the initial continuous FP32 gradient reference;
training stays dynamic. Unfrozen BF16/FP32 branch-tie failures are preserved.
The runner records parameter/data/source/extension hashes, dirty tracked files,
and freezes the dereferenced package headers plus the exact runtime LUT.

Evaluate each saved seed on all 111539 heldout causal targets, using its exact
frozen package and LUT (the Hybrid checkpoint explicitly evaluates via Peer):

```bash
PPU_MOE_RUN="$PPU_MOE_SOURCE_ROOT/build/campaign/lm-peer-wg3"
PYTHONPATH="$PPU_MOE_RUN/frozen-library:$PPU_MOE_SOURCE_ROOT/tests" \
SAIL_DEEPGEMM_LUT_PATH="$PPU_MOE_RUN/frozen-library/runtime-lut" \
python tests/evaluate_moe_lm.py --checkpoint "$PPU_MOE_RUN/seed4827.pt"
```

Repeat for4891/6113 and the paired baseline. Do not use train-prefix loss or
partial-batch evaluation as final quality. This is small-model validation,
not a statistical noninferiority proof or full2.4T convergence.

## Profiling interpretation

`analyze_overlap.py --steps 3 --chunks 1 --output result.json trace.sqlite`
partitions actual kernel intervals. Mixed kernels include peer traffic and
compute; local-operand external consumers can include ready waits. Kernel
intersection is not pure wire/MMA overlap or saved latency. Use complete-step
unprofiled timings to select a recipe, and report computation slowdowns alongside
hidden intervals. `plot_stream_overlap.py` renders the middle of three captured
rank0 graph replays; retain source traces and all-host audit provenance.
