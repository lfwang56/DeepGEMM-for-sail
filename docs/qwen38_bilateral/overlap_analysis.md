# Communication/computation decomposition at Qwen3.8 dimensions

The latest user request targets MoK-like 2–3x gains and prioritizes measured
overlap. Existing ~12% reduction is an intermediate result, not proof of an
effective persistent communication/compute pipeline.

## Scope and source alignment

Pinned MoK paper: https://arxiv.org/html/2609.36070v1 . The reported maxima are
BF16 FWD1.92x/BWD1.58x, MXFP8 FWD2.37x/BWD1.78x, and full-model throughput1.41x
on512GB300. These are different metrics. MoK includes a shared expert and its
DeepEP+Torch reference retains forward activations for autograd backward.
The current PPU benchmark is BF16 routed-expert-only FWD plus full backward
replay/rematerialization, including FP32 weight-gradient zero/accumulation,
without optimizer. Its denominator is tuned installed PPU DeepEP Buffer with
38 communication CUs/19QPs and native grouped GEMMs. The same replay contract
applies to both PPU paths. This is not yet MoK's exact training/memory contract.

Source reviewed: cursor/mixture-of-kittens at
f3a8bd2432457c5a2a2ea13915f0dcd0224581e4, specifically
csrc/megakernel/{forward,dispatch_combine}.cuh and
benchmarks/{bench_mok,bench_deepep_torch}.py. Communication CTAs pull inputs and
push outputs; compute tasks consume ready minibatches with GPU dependencies.
The communication and compute roles execute concurrently in one kernel.

## Measured kernel decomposition

Same supernode EP32,32 physical PPUs on225/226/227/223, H8192/I2048/E512/K10,
T4096 per rank,C768,BN256/BK64/WM64/WN64/S2. Asys captures rank0, three complete
graph steps. All ranks and exclusive-node audits completed cleanly in each of
the three profile phases. Values below are sums of kernel intervals divided by
three; they are instrumented rank0 measurements, not max-rank benchmark times.

| Component (ms/step) | DeepEP | Explicit peer copy + local GEMM | Current bilateral fusion |
|---|---:|---:|---:|
| Local GEMM kernels |96.013|93.971|71.490|
| GEMM with embedded peer traffic |0|0|44.506|
| DeepEP communication + notifications |27.301|0|0|
| Explicit peer payload copies |0|26.479|5.680|
| Explicit peer rank barriers |0|6.186|6.572|
| Layout metadata |0.552|included below|included below|
| Routing, activation, local copies, zeroing |37.677|13.830|13.867|
| Weight transpose |2.585|2.582|2.585|
| Kernel union, harness collectives excluded |164.129|143.049|144.700|

All explicit kernels are on stream15. The measured intersection between separate
communication kernels and local compute kernels is **0ms** for all three.
The fused kernels contain compute and peer access together; Asys kernel names
cannot resolve their internal overlap. Their entire duration must not be called
pure compute. The current FC1 producer-slice and local-remainder kernels have a
full stream dependency, so they cannot hide the whole dispatch behind FC1.

DeepEP proportions: local GEMMs58.50%, communication+notify16.63%, layout0.34%,
other routing/local work22.96%, transpose1.58%. Pure baseline GEMM role sums:
FC1 FWD14.078ms,FC2 FWD6.927ms, backward FC1 replay14.094ms,FC2 replay6.920ms,
dHidden6.914ms,DX13.844ms,DW1 **22.076ms**,DW2 **11.160ms**.
Forward GEMMs total21.005ms; backward GEMMs
including replay75.008ms.

At unchanged local GEMM cost, hiding just DeepEP's communication+notification
time would reduce164.129ms to136.828ms (~1.20x), not2x. Removing every non-GEMM
cost while retaining those GEMMs would be ~1.71x. These are conditional
arithmetic estimates, not physical hardware limits. A2x–3x result requires
faster computation and/or avoided intermediate work in addition to overlap.

Explicit-copy peer is as fast as current fusion in the instrumented comparison.
Consequently, the current gain over DeepEP cannot be attributed wholly to
overlap. Fresh uninstrumented controls are required. The first such rerun
finished all32 ranks but foreign GPU work appeared on227 and223; exclude that
entire phase's timings and move the next run to an idle domain window.

## Resident pipeline follow-up (completed)

Opt-in resident tile pipeline: reserve communication CTAs, acquire/release one
flag per input M tile, and feed local AIU GEMM loads immediately when each tile
is ready. Return communication CTAs wait for individual completed M/N GEMM
tiles and issue vector peer stores while other compute CTAs continue. Prove
residency with actual occupancy, retain timeouts/status, and preserve all BF16
rounding and gradient contracts. Measure serialized-copy, direct-scatter,
input-pipeline, return-pipeline and bilateral-pipeline controls. Record per-CTA
wait/total cycles as diagnostics; do not infer device-wide overlap percentage
by summing CTA cycles or treating clock64 values across CUs as synchronized.

Artifacts in the campaign workspace: outputs/qwen38/overlap-decomposition.json,
work/reference/qwen38-{deepep,unfused,iter10}-profile.sqlite, and the replayable
work/analyze-moe-overlap.py. Profile runs are diagnostic, not candidate timing.


## Routed-row windows and larger source chunks (iteration18)

The source-token dispatch chunk is no longer also the physical local GEMM
capacity. Global routing metadata covers the worst skew, while a65536-row
local activation ring processes every routed row in GPU-selected windows.
This permits EP32 C4096 at H8192 without overflowing the existing int32
physical-matrix indexing. Independent all-element FP32 extreme-skew checks
passed on every rank for both DeepEP and peer paths, including all1,310,720
routes directed at one expert. No token or route is dropped.

Clean uninstrumented max-rank event medians at T4096: old C768 DeepEP164.589ms,
peer140.078ms; C4096/S65536 DeepEP129.608ms, peer112.326ms. A matched explicit
peer-copy control is113.331ms. The net bilateral speedup over equally tuned
DeepEP is1.154x; the difference from explicit peer copy is only0.9%. The
larger old-baseline ratio includes source chunking, fewer repeated FP32
weight-gradient updates and other scheduling changes, so is not pure overlap.

The new fused rank0 Asys trace has112.259ms kernel union: local GEMMs56.547ms,
mixed GEMM+peer traffic33.852ms, expert-gradient communication/pointwise4.701ms,
rank barriers4.905ms, other routing/activation/copies/zeroing9.673ms, weight
transpose2.581ms. DW1 is10.862ms and DW2 is5.669ms, down from22.076/11.160ms
in the old small-source-chunk reference. Standalone-kernel communication/
compute intersection remains0ms. Internal mixed-kernel overlap has not yet
been measured directly; mixed duration is not pure compute time. Matching
large-chunk DeepEP and explicit-copy traces are being collected before
reporting a new communication fraction.

Public-text full-model validation is complete for this recipe: a57.8M MoE LM,
2000steps, three paired seeds, full111539-token validation. Matched initial
states and all train/eval node audits passed. BF16 native mean loss1.589085,
peer1.588399 (relative mean difference-0.0431%). This is small-model numerical
quality evidence, not full2.4T-model convergence or a statistical superiority
claim. Raw concatenated JSON logging was decoded with JSONDecoder, preserving
both rank-start records.

Local evidence: outputs/qwen38/iter18-routed-row-window-controls.json,
outputs/qwen38/ep32-route-window-skew-c4096-reference.json,
outputs/qwen38/iter18-fused-overlap-decomposition.json and
outputs/qwen38/public-text-lm-window18-paired.json.
