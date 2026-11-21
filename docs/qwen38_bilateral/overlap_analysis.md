# Qwen3.8 routed-layer communication and compute analysis

Updated2026-11-21UTC, through iteration35. Iteration36 tests the native PPU yield hint in ready
polls; no benefit is claimed before controlled numerical and latency checks.

## Measurement contract

The pinned Qwen3.8-2.4T-A95B configuration has H8192,I2048,E512,top10:
https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/912e40d903bfc891be8fe3c51faa0a09d182ce95/config.json .
Tests execute one routed-expert layer with random tensors, excluding attention,
router, shared expert and optimizer. All performance paths use BF16 operands,
FP32 weight-gradient zero/accumulation and full expert replay in backward.
They do not run the full2.4T model. Domain EP32 means32 physical PPUs on four
hosts from one eight-host supernode; it does not mean eight hosts form one GPU.

Recommended peer recipe: BM256, all four GEMMs BN256/BK64/WM64/WN64/stages2,
C4096, routed activation ring65536, G8 gate/up layout, fused FC1 activation,
producer input/local AIU, input-copy ILP8, expert-local gradients, fused
activation backward, packed DY reuse. Keep wgrad_overlap=none and
 overlap_expert_dy=False. WG warp map0/AIU-before-TSM1 remain the defaults.

The strong baseline is the installed PPU DeepEP2.2 Buffer dispatch/combine,
38 communication CUs/19QPs, native grouped GEMMs, same shape and replay
contract. Serial and early cached-DY dispatch are both measured; the strongest
observed setting is early DY at119.972473ms, versus peer97.358685ms in the
same matched experiment (~1.232x). A2–3x gain has not been achieved.
Collective all-gather/reduce-scatter is an auxiliary baseline and cannot be
substituted for DeepEP to inflate the ratio.

## What “local GEMM” means

It labels a kernel whose GEMM operands and output are in local device memory.
It is not an isolated compute-only benchmark, and the label does not establish
whether it overlaps another kernel. Only the intervals establish concurrency.
The current fastest schedule launches these GEMMs serially on its dependency
path. FC1 first computes one N slice while materializing X, then launches the
remaining slices using that local X. This whole-producer boundary is a concrete
remaining opportunity, not a mathematical requirement.

Backward also has data dependencies: weighted DY precedes dHidden/DW2;
activation derivatives precede DX/DW1. DW1 and DW2 can overlap independent
work, but share the same compute and memory resources. Communication of a
future tile can overlap a ready tile's GEMM if the protocol, resources and
buffer lifetimes allow it. CUDA graphs may assign internal work to different
device streams; plotted lane IDs do not imply fixed host-level main/aux roles.

The earlier56.547ms local /33.852ms mixed /4.701ms gradient work /4.905ms
wait figures are from the112.259ms kernel-union version, not a12.3ms step.
They are sums of kernel intervals and were not measured as isolated executions.
A mixed GEMM's whole interval must not be called pure compute or pure wire time.

## Current serial and raw-DY overlap profiles

Asys captures rank0, three whole graph steps, T4096/rank, C4096 and ring65536.
Each accepted phase has all32 ranks and four clean exclusive-host audits.
The serial capture's first interrupted/interfered attempt is excluded; only
its clean attempt1 supplies the accepted trace. Timing ratios use separate
uninstrumented maximum-rank medians, not these diagnostic kernel sums.

| Rank0 category, ms/step | Peer serial | Peer DY prefetch | DeepEP early DY |
| --- | ---: | ---: | ---: |
| Local GEMM interval sum |53.834|58.471|67.166|
| Mixed GEMM and peer traffic |27.614|27.426|0|
| Explicit DY copy |0|5.576|0|
| Expert gradient communication/pointwise |4.506|1.595|included below|
| DeepEP communication/notify |0|0|19.504|
| DeepEP layout |0|0|0.466|
| Rank barrier kernels |2.945|3.861|included in DeepEP|
| Routing, pointwise, copies, zeroing |6.570|6.566|33.983|
| Weight transpose |2.583|2.581|2.590|
| Explicit communication intersecting local GEMM |0|5.571|0|
| Kernel union, harness collectives excluded |98.052|100.505|120.379|

Category sums double-count simultaneous activity; the analyzer also exports a
disjoint partition whose sum equals each kernel union. In peer prefetch,
5.571ms of copy runs concurrently with local compute and only0.005ms is
copy-only. Nevertheless, replay FC1 remaining columns stretch from11.784ms to
17.176ms. Reducing the gradient kernel from4.506ms to1.595ms does not cover
this slowdown plus the changed waits. Unprofiled complete-step medians likewise
regress97.358685->99.848474ms. The default stays serial.

DeepEP's early dispatch overlaps about3.329ms of routing/copy/zero work in
this capture, with zero measured overlap against local GEMM kernels. Its
~1.27ms improvement must not be described as hidden GEMM communication.
These observations are kernel-span measurements, not physical ICN wire/MMA
activity or device-wide utilization percentages.

## Legal weight-gradient concurrency also regressed

Iteration30 independently tested legal stream dependencies for both DWs:

| Rank0 diagnostic, ms/step | Serial DW | Concurrent DW |
| --- | ---: | ---: |
| DW2 |5.869|8.043|
| DW1 |10.832|15.505|
| Mixed GEMM/traffic interval sum |27.246|37.322|
| Local GEMM interval sum |52.684|63.566|
| DW intersecting mixed GEMM |0|13.287|
| DW intersecting other local GEMM |0|6.137|
| Kernel union |98.185|102.194|

The13.287+6.137ms are concurrent kernel intervals, not19.424ms of wire
latency saved. Real unprofiled EP32 medians were97.575024ms(serial),
98.918170ms(DW1 overlap),100.497021ms(both). DeepEP also regressed under
coarse DW concurrency. Native per-kernel Acu resource records show216
registers/thread and512-thread compute CTAs for the relevant large WG kernels;
measured contention is plausible, but this is not proof of a physical floor.

The earlier missing window join (iteration29) failed distributed DW reference
checks. Its timings are invalid and retained only as a failed experiment.
Iteration30 centralizes DeepEP's WG join in every _set_window before ring reuse.

## Comparison with MoK

Pinned paper https://arxiv.org/html/2609.36070v1 and public source
cursor/mixture-of-kittens f3a8bd2432457c5a2a2ea13915f0dcd0224581e4 were
reviewed. Reported maxima are BF16 FWD1.92x/BWD1.58x, MXFP8
FWD2.37x/BWD1.78x, full-model throughput1.41x on512GB300. These are distinct
metrics. Its source uses communication and compute roles with tile/task
readiness and a DeepEP+Torch dispatch/combine reference. NVIDIA Blackwell
instructions and resource assumptions cannot be copied to ZW-M890P.

PPU tests have full backward replay and routed-only scope; MoK includes a
shared expert and saved activations in its reference. The current result
cannot be presented as reproducing MoK's exact protocol or reported speedup.
Hiding communication alone at unchanged local GEMM cost is a conditional
arithmetic estimate, not a guarantee or hardware lower bound. Improving the
critical path requires useful supply overlap, cheaper computation and/or
avoiding redundant dataflow, validated by complete-step latency.

## Evidence

- outputs/qwen38/iter32-dy-prefetch-controls.json: unprofiled timings.
- outputs/qwen38/iter32-asys-provenance.json: all host/rank audits.
- outputs/qwen38/iter32-dy-overlap-decomposition.json: categories, kernels,
  intersections and disjoint partitions.
- outputs/qwen38/iter30-wgrad-overlap-decomposition.json: DW concurrency.
- outputs/qwen38/profile30-weight-gradient-streams.{png,svg,json} and
  profile32-dy-prefetch-streams.{png,svg,json}: measured sampled graph lanes.
- work/analyze-moe-overlap.py and work/plot_moe_stream_overlap.py: reproducible
  analyzers; raw32SQLite files are work/qwen38-wlf-ep32-profile32-*.sqlite.
- docs/qwen38_bilateral/ITERATIONS.md: all positive and negative experiments.


## Input-ready consumers: measured concurrency and a serialization control

Iterations33–35 keep producer97.4–97.5ms as the recommendation. Linear input
assistants with inline fallback are100–105ms; compiling separate lower-register
consumers did not make the operator faster. All candidate paths passed exact
mutable graph and independent all-rank FP32 checks; all timings below have four
clean same-domain hosts and32 ranks.

Compiled-code classification matters. `_external_dispatch` has neither inline
peer-copy code nor resident communication-CTA code: it is a **local-operand
GEMM with input-ready waits**. Its interval includes waiting and is not active
MMA time. `_owned_dispatch` still contains peer copies and is mixed. The
analyzer exports `local_compute_with_ready_wait` separately, while including it
in communication/GEMM intersections. It does not mislabel external consumers as
payload communication merely because the kernel name contains pipeline.

Iteration34 rank0 Asys producer union97.454ms; static inline8 union102.000ms.
External8 union226.081ms: mover117.510ms, external FC1 consumer127.156ms,
copy/consumer intersection117.500ms, rank barriers37.570ms. External32 union
231.785ms: mover134.057ms, external FC1 consumer140.787ms, intersection
134.004ms, barriers29.771ms. Genuine concurrent spans coexist with severe
slowdown/imbalance. No wire bandwidth or kernel active-MMA fraction is inferred.

The local-pointer resource diagnostics identify inline GEMM240regs/thread,
external224 (both512threads); generic linear mover64regs, static linear80,
128threads. At8CTAs, isolated local static mover improves15.957->9.416ms
with zero stack. That does not prove a corresponding distributed gain. Two
local Asys attempts had ownership-gate anomalies and remain excluded; clean
r2 explicitly passes the application job tag and records PID/birth/cgroup.

Iteration35 preserves the same mover/consumer code, grids, flags and data and
moves only the auxiliary event join before or after GEMM. This supplies a
controlled test of simultaneous work, separate from the strong DeepEP baseline:

| Full step, ms | External parallel | External serialized |
| --- | ---: | ---: |
| Warp-row mover,8CTAs |162.378015|113.376074|
| Warp-row mover,32CTAs |181.066919|99.994977|

The warp-row mover with inline fallback is100.911017ms versus linear
inline102.278467ms. Producer control97.532886ms remains faster than both.
Strongest matched DeepEP120.096179ms gives1.2313x. These comparisons establish
large concurrent-scheduling cost; they do not isolate a particular physical
cache/queue mechanism. Native yield polling is tested next without changing
release/acquire ordering, payload or numeric precision.

Evidence: iter33/34/35-input-assist-controls.json,
ep32-input-assist-reference.json (iteration-specific collection prefixes),
iter34-asys-provenance.json,iter34-input-overlap-decomposition.json and
work/stall-review-before34.md in the campaign workspace.
