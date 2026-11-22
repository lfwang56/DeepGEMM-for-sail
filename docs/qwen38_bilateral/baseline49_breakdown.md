# Strong Elastic baseline decomposition and fusion estimates

Measurement on2026-11-22: wlf_sgl_2.2, Torch2.11, ZW-M890P/ppu0015,
domainEP32 on four whole idle hosts, Qwen H8192/I2048/E512/top10,
T4096/C4096, ring65536, G8, stage3, Elastic38 communication CUs,
early cached DY dispatch, serial weight-gradient GEMMs. Complete routed
BF16 FWD plus rematerialized BWD, FP32 DW zero/accumulation, DX and dScore.
Shared expert, attention and optimizer are excluded.

The package is frozen49 (source567fde5c088e914a4ccd3158724b98641eb9ac3c,
nativeedd5ab3ecf756bec60d30a469fcbac22b964ca77df3e9e3a4686fd1913dcad0c).
It is independent of optional52/53 experiments in the working tree.
Both phases passed all four whole-host monitoring audits.

Unprofiled median maximum-rank event latency105.84309082ms
(wall105.84348385), consistent with the previous105.7501709ms denominator.
Profiler maximum-rank event108.08320109ms includes instrumentation effects.
The table below instead uses rank0's three timed graph launches:
mean first-to-last GPU activity span106.02788533ms. Do not normalize it
to105.75 or treat these distinct metrics as identical.

## Graph-correlated decomposition

Select the three cudaGraphLaunch correlation IDs and their KERNEL, MEMCPY
and MEMSET records. Exclude harness activity before/after the three graphs.
The cumulative rows overlap; subtract the measured intersection once.

| Work | Cumulative ms | Percent of rank0 graph span |
| --- | ---: | ---: |
| Ordinary forward/input-gradient GEMMs |38.782306|36.5775%|
| FP32 weight-gradient GEMMs |16.312750|15.3853%|
| Local data/routing-gradient handling |26.447249|24.9440%|
| Activation and activation backward |7.169292|6.7617%|
| Weight transpose |2.580705|2.4340%|
| DeepEP dispatch/combine main kernels |14.648921|13.8161%|
| DeepEP local epilogues |2.938373|2.7713%|
| Subtract communication/local-data overlap |-2.979790|-2.8104%|
| Unrecorded GPU activity gaps |0.128079|0.1208%|
| Total graph span |106.027885|100%|

Local data/routing-gradient handling consists of: zero6.087796ms,
expert-output reduce6.895255, FP32-to-BF16 conversion4.925889,
DY weighting and dScore4.340939, expert-input permute2.576952,
source-chunk copy1.086941, metadata/other0.347612 and local DMA0.185866.
These include our training adapter and useful gradient arithmetic; they
are not all DeepEP library overhead or cross-card communication.

Main dispatch/combine kernel durations include notification, waiting,
local accesses and reductions. They are not pure link-transfer time.
The entire DeepEP family occupies17.587294ms cumulatively, with
2.979790ms overlapping other local work and14.607504ms exposed.
Main kernels alone expose12.474126ms and overlap2.174794ms.
There was zero interval overlap between this communication family and
ordinary/WG GEMMs in this rank0 capture. Intent to prefetch DY does not
establish measured compute/communication concurrency.

## Conditional Amdahl estimates

Holding every other kernel's duration fixed:

- Remove only exposed main dispatch/combine intervals:
  106.027885/(106.027885-12.474126)=1.13334x.
- Remove exposed intervals of the whole DeepEP family:
  106.027885/(106.027885-14.607504)=1.15978x.
- Extreme assumption that every non-GEMM cost disappears:
  106.027885/55.095056=1.92445x.

The last estimate is deliberately optimistic: activation and gradient
arithmetic must execute after fusion. None is a formal hardware ceiling,
a prediction of an implemented fused kernel, or proof that twofold gains
are impossible. A new algorithm can remove replay, change GEMM efficiency,
and avoid memory traffic, invalidating the unchanged-GEMM assumption.

The current peer49 gain1.3687x includes more than overlapping network
transfer: it removes adapter work, FC2 replay and weight transposes,
while adding a dHidden/score epilogue. Its mixed kernel durations cannot
be called pure GEMM time and substituted into this baseline calculation.

## MoK comparison

The primary paper is https://arxiv.org/html/2609.36070v1 and source is pinned
to f3a8bd2432457c5a2a2ea13915f0dcd0224581e4.

At Qwen3.5 MXFP8 EP64/T2048, Table1 reports baseline/MoK forward
2.527/1.068ms and backward2.267/1.273ms. Adding the separately measured
passes gives4.794/2.341=2.04784x; this is an arithmetic combination, not
a separately measured complete-model training step. Table2's512-GPU
production training throughput improvement is1.41x.

Thus a pure-communication estimate of1.16x must not be presented as the
ceiling for MoK's combined algorithmic changes. Model, precision, shared
expert inclusion and timing scope also differ from the current PPU case.

## Evidence and analysis

Controller artifacts: outputs/qwen38-round2/baseline49-profile-provenance.json,
baseline49-profile-coarse.json, baseline49-graph-breakdown.json.
SQLite: work/qwen38-r2-elastic49-profile.sqlite.
Reproduction: work/qwen38-r2-elastic49-profile-cases.json,
work/analyze-r2-elastic49.py, work/analyze-elastic-baseline-r2.py.

The graph analyzer verifies that exclusive activity sums equal the GPU
activity union. Older kernel-only summaries include a small amount of
harness work and omit DMA; retain them as earlier artifacts, not an exact
accounting of the graph span.
