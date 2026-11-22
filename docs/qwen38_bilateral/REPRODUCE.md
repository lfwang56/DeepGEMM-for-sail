# Qwen3.8 routed MoE training on PPU

Target: `wlf_sgl_2.2`, Torch 2.11.0, SAIL SDK 2.2.0, ZW-M890P / ppu0015.
Use the delivery manifest's exact source revision, wheel and vendor hashes.
Install the wheel into an isolated target directory; put that directory first in
`PYTHONPATH`. The standalone source archive includes the pinned Actlize and fmt
sources and `PKG-INFO` so that rebuilding does not require Git history.

## Scope and topology

Each host has eight PPUs. Each listed block of eight hosts is one physical
supernode. Domain EP32 uses four hosts in ONE supernode; domain EP64 requires all
eight. Cross-domain EP64 uses four hosts in each of TWO supernodes. Record actual
host identities; a benchmark's topology label alone does not establish topology.
Gate every device on every host before, during and after each job. Exclude a phase
if a foreign job appears. Never stop another user's processes. A recovered idle
host may be reused, with a replacement preferred after an interruption.

The pinned routed-expert dimensions are H=8192, I=2048, E=512, top-k=10. EP32 has
16 local experts; EP64 has eight. These are the user-requested Qwen3.8 expert
layer dimensions. Measurements do not run the full 2.4T model. The timed scope is
one routed layer's forward and complete backward, including FP32 DW1/DW2 zeroing
and accumulation, DX, dScore and current-weight preparation. Attention, shared
experts, router projection/top-k and optimizer updates are excluded.

Select recipes using unprofiled maximum-rank GPU-event medians: five warmups,
20 iterations and five repeats. Profiled rank-zero spans are diagnostic.
Keep identical live-forward-context policies on DeepEP and peer comparisons.
Historical 105.75 ms DeepEP data predates that common optimization.

## Build and common runtime

```bash
PPU_MOE_SOURCE_ROOT="$PWD"
export PYTHONPATH="$PPU_MOE_SOURCE_ROOT:$PPU_MOE_SOURCE_ROOT/tests"
export SAIL_DEEPGEMM_LUT_PATH="$PPU_MOE_SOURCE_ROOT/docs/qwen38_bilateral/wgrad_luts40/stage3"
export DG_JIT_CACHE_DIR="$PPU_MOE_SOURCE_ROOT/build/reproduce/jit-cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export DG_MOE_REUSE_FORWARD_CONTEXT=1
export DG_BF16_WGRAD_COMPUTE_CUS=0 DG_BF16_WGRAD_TILE_CHUNK=0
```

An explicit M-group GEMM stage count does not select the weight-gradient pipeline;
the runtime LUT does. Both recommended precision paths retain BF16 operands and
FP32 accumulation for DW. Experimental FP8 DW primitives are not integrated into
these plans: their measured panel preparation erased the compute-only gains.
The unsupported experimental FP8 transposed-load API was removed.

## BF16 recipe

Use G8 gate/up layout, BM256, input producer/local AIU, ILP8, fused FC1/SwiGLU,
fused dHidden/activation/dScore, KN weight reads and live forward context. DW and
dGrad use full resources serially. Splitting their CU budgets or launching them
concurrently increased contention in the measured Qwen shape.

```bash
export DG_MOE_PEER_STAGE_CTAS_PER_CU=8
export DG_MOE_FP8_INPUT_PIPELINE=0 DG_MOE_FP8_BF16_INPUT_PIPELINE=0
export DG_MOE_FP8_INPUT_SERIALIZE=0 DG_MOE_FP8_INPUT_SERIAL_FULL_GRID=0
export DG_MOE_FP8_TRANSPOSE_QUANTIZED_WEIGHTS=0
export DG_MOE_FP8_WEIGHT_TRANSPOSE_SWIZZLE=0
export DG_MOE_FP8_VECTOR_ROW_QUANT=0 DG_MOE_FP8_FUSED_STAGE_QUANT=0
export DG_MOE_FP8_TILED_GRAD_QUANT=0

python tests/benchmark_native_moe.py --backend peer --compute bf16 \
  --hidden 8192 --intermediate 2048 --local-experts 16 --topk 10 \
  --tokens 4096 --chunk-tokens 4096 --ring-rows 65536 --block-m 256 \
  --gate-up-layout g8 --gemm-config 256 64 64 64 3 \
  --dispatch-producer --producer-input-aiu --producer-local-aiu \
  --input-copy-ilp 8 --fuse-fc1-activation --fuse-activation-backward \
  --fuse-dhidden-scores --transpose-free-dgrad \
  --expert-local-grad --reuse-packed-dy --wgrad-overlap none --scope step
```

Run the command through the distributed launcher below; a direct invocation is
only a one-rank check. For T16384 retain C4096. The clean EP32 measurements from
iteration105 are 86.65→63.96 ms at T4096 and 334.32→252.00 ms at T16384. Each
baseline/peer pair uses the same four hosts. The interrupted earlier T16384
phase is excluded from these numbers.

## FP8 recipe and optional pipelines

Use split gate/up layout and M128 routing windows. The recommended base recipe
fuses BF16 input packing with unchanged 128-element E4M3 row quantization, fuses
weighted DY staging with its quantization, uses direct BF16 peer output writes,
and reuses quantized weights for backward transposes. Swizzled weight transpose
is bit preserving. Weight quantization remains inside the timed layer.

```bash
export DG_MOE_PEER_STAGE_CTAS_PER_CU=0
export DG_MOE_FP8_WEIGHT_QUANTIZER=2 DG_MOE_FP8_ASYNC_WEIGHTS=0
export DG_MOE_FP8_TRANSPOSE_QUANTIZED_WEIGHTS=1
export DG_MOE_FP8_WEIGHT_TRANSPOSE_SWIZZLE=1
export DG_MOE_FP8_VECTOR_ROW_QUANT=1
export DG_MOE_FP8_FUSED_STAGE_QUANT=3
export DG_MOE_FP8_FUSED_QUANT_CTAS_PER_CU=16
export DG_MOE_FP8_TILED_GRAD_QUANT=2
export DG_MOE_FP8_INPUT_PIPELINE=0 DG_MOE_FP8_BF16_INPUT_PIPELINE=0
export DG_MOE_FP8_INPUT_SERIALIZE=0 DG_MOE_FP8_INPUT_SERIAL_FULL_GRID=0

python tests/benchmark_native_moe.py --backend peer --compute fp8 \
  --hidden 8192 --intermediate 2048 --local-experts 16 --topk 10 \
  --tokens 4096 --chunk-tokens 4096 --ring-rows 65536 --block-m 128 \
  --gate-up-layout split --no-fuse-dispatch --expert-local-grad \
  --reuse-packed-dy --wgrad-overlap none --scope step
```

Iteration107's accepted EP32 recipe is 79.20→62.45 ms at T4096 and
299.50→232.96 ms at T16384, using cached tiled DY gradients and16CTAs/CU.
The earlier iteration104 base uses tiled mode0/grid4. Local component timings
alone do not promote a recipe.

`DG_MOE_FP8_TILED_GRAD_QUANT=1` splits dScore into 128-feature partial reductions;
`2` additionally caches route scores within each CTA iteration. Both require
fused DY quantization (stage-quant bit 2). BF16 DY, FP8 bytes and scale bits match
the base recipe. The changed FP32 dScore summation order has independent FP64
and small-model quality checks. Mode2 is bitwise equal to mode1 for dScore in
component tests. Select the CTA grid with an actual EP32 complete-step sweep.

`DG_MOE_FP8_BF16_INPUT_PIPELINE=1` is an optional producer/consumer path. A producer
pulls original BF16 rows, retains BF16 X for DW, quantizes them and publishes each
M128 tile. FC1 acquires covered tiles on another stream. It requires fused input
quantization (stage-quant bit1), bounded ring rows and H<=8192. Set
`DG_MOE_FP8_INPUT_CTAS` to reserve a producer CTA budget; this does not pin physical
CUs. `DG_MOE_FP8_INPUT_SERIALIZE=1` supplies a serialized control and
`DG_MOE_FP8_INPUT_SERIAL_FULL_GRID=1` restores full FC1 grid after that join.
A numerically valid pipeline can still lose performance because of bandwidth,
shared-memory or register contention. Keep it off unless the audited complete
step improves. Serialized full-grid mode may use up to8producerCTAs/CU; the
concurrent budget must remain below the CU count. The older `DG_MOE_FP8_INPUT_PIPELINE=1` pulls source FP8 and stages
BF16 separately for DW; the two input pipelines are mutually exclusive.

The base FP8 input path stages before FC1. Its return GEMM fuses peer writes.
Do not label that base path a fully fused dispatch-and-GEMM megakernel.

## Distributed launcher and DeepEP baseline

On each of four idle hosts in one supernode, use node ranks0..3 and the same
master, port and log directory:

```bash
PPU_MOE_NODE_RANK=0
PPU_MOE_MASTER=172.17.0.215
python tests/launch_moe_local_ranks.py \
  --node-rank "$PPU_MOE_NODE_RANK" --nodes 4 --local-ranks 8 \
  --master-addr "$PPU_MOE_MASTER" --master-port 29874 \
  --log-directory "$PPU_MOE_SOURCE_ROOT/build/reproduce/domain32" -- \
  python tests/benchmark_native_moe.py YOUR_PRECISION_AND_RECIPE_ARGUMENTS
```

Prepare the installed DeepEP's isolated native-HGGC header adapter:

```bash
python tests/prepare_deepep_elastic_compat.py --root "$PPU_MOE_SOURCE_ROOT"
export PYTHONPATH="$PPU_MOE_SOURCE_ROOT/build/campaign/deepep-native-scope:$PYTHONPATH"
export CPATH="$PPU_MOE_SOURCE_ROOT/build/campaign/compat/deepep:/usr/local/pccl/include:/usr/local/pccl/nccl_wrapper/include:/usr/local/PPU_SDK/include"
```

The tuned denominator is `--backend elastic --deepep-scaleup
--deepep-comm-sms 38 --overlap-expert-dy`, using installed DeepEP dispatch/combine.
BF16 uses the same G8/MGroup/WGrad-stage3 and live-context policy. FP8 uses split
layout, current-weight quantization, vector row quantization, quantized swizzled
weight transpose, and `DG_MOE_FP8_SOURCE_DISPATCH=1`. Leave peer input/DY fusion
flags off for this control. DeepEP's FP8 source dispatch communicates FP8 forward
input; its original BF16 backward input and BF16 returns remain timed.
Keep Buffer/serial-DY controls in the evidence. AG/RS is an auxiliary reference.
Elastic requires unique top-k expert IDs per token.

## Numerical, autograd and quality checks

Use `tests/test_moe_elastic_distributed.py`,
`tests/test_deepep_moe_large_distributed.py`,
`tests/test_moe_fp8_peer_distributed.py` and
`tests/test_moe_training_skew_capacity.py` on all participating ranks with matching
precision, layout, context and ring configuration. Validate every Y/DX/DW1/DW2/
dScore element, mutable graphs, masked/empty/skew routes and capacity overflow.
The skew script accepts `--gemm-config`; it chooses producer mode itself and has
no `--dispatch-producer` argument. Keep unchanged numerical thresholds.

`native_moe` supports FP32 master expert weights, returns FP32 expert gradients
and requires a separate plan per outstanding captured layer. Each source chunk
retains its last live window; earlier windows under skew are rematerialized.
Warm full forward/backward before capture. Fixed capacities permit mutable
X/DY/IDs/scores/valid counts. Call `plan.check()` outside capture. Caller-owned
`weight_grad_accumulators=(dw1,dw2)` suppress parameter `.grad`; the caller owns
clearing/reduction. Optimizer and Megatron/DP hooks are not installed implicitly.

Full-model quality uses public Tiny Shakespeare: 57.8M parameters, four MoE layers,
H512/I1024/E8/top2, global batch8, length128, 2000 updates, seeds4827/4891/6113.
Use `tests/run_moe_lm_seeds.py --freeze-library` with the quality runtime LUT and
`tests/evaluate_moe_lm.py` on all111539 heldout causal targets per checkpoint.
The fixed-route option only stabilizes the initial continuous gradient reference;
training routing stays dynamic. Compare canonical initial-state hashes and each
run's immutable package/extension/LUT metadata. Retain losses per seed.
This checks a complete small model, not full Qwen2.4T convergence.

## Cross-domain EP64 and profiling

Order global ranks as two contiguous physical domains. Initialize GPU and Gloo
metadata groups, then call `make_hierarchical_groups(domain_size=32,
metadata_group=metadata)` collectively. `HierarchicalPeerMoEPlan` uses peer fusion
within each domain and GPU collectives across domains. Gloo carries metadata.
Close plans after device completion before closing caller-owned groups. Validate
with `tests/test_moe_hierarchical_distributed.py`; benchmark with `--backend hybrid
--hybrid-domain-size 32`. Cross-domain Elastic uses `--backend elastic
--elastic-hybrid-mode` without `--deepep-scaleup`. Earlier cross64 results do not
validate later context/FP8 changes or establish domain64 performance.

For profiles, partition the exact timed graph's GPU activity union and report
idle gaps separately. GEMM kernels that read/write peers contain arithmetic,
traffic and waits; expert gradient kernels also contain pointwise reductions.
Their full durations are not pure wire time. Cumulative kernel durations can
overlap and must not be added as wall time. State the rank, source hash, profiled
span and unprofiled maximum-rank median separately.
