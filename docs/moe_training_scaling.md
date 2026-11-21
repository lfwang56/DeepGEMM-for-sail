# PPU MoE scaling optimization record

Measurements use M890 / ppu0015, SDK2.2.0 and PyTorch2.10. Eight devices per
host, eight hosts per supernode. Times include forward, full backward replay
and FP32 wgrad zero/accumulation, excluding JIT and optimizer. Each result is
the median of five repeats of twenty graph steps, taking the maximum rank
event time within each repeat. Foreign-job interference excludes timings.

## EP32 baseline and active-row scheduling

H7168, I2048, E_local8, top8, T_per_rank1024. Peer tiles are
(BN256,BK64,WM64,WN64,stages2). Four hosts .222/.225/.226/.227 in one supernode;
all 32 ranks exited normally, with clean availability postflights.

| Mode | C512 baseline / ms | C1024 baseline / ms | C1024 active rows / ms |
| --- | ---: | ---: | ---: |
| Native collective, NoPad | 183.202 | 179.456 | 179.438 |
| Peer packed dispatch, separate return | 101.484 | 97.869 | 55.912 |
| Peer packed dispatch, fused return | 94.490 | 91.057 | 49.119 |
| Peer direct dispatch, separate return | 104.791 | 102.597 | 60.529 |
| Peer direct dispatch, fused return | 97.856 | 95.668 | 53.591 |

The active-row candidate bounds activation and backward input packing by the
GPU work count and fuses local score/weighted gradients. Ring allocation is
unchanged; this removes worst-capacity scans rather than changing semantics.
`active_rows=False` and `fuse_local_grad=False` retain controls. The same-peer
return-only contrast improves from 6.96% to 12.15% latency reduction at C1024.
The historical 3.65x comparison to that fixed-capacity collective includes
these scheduling improvements, and must not be described as isolated GEMM
communication fusion. Stronger collective baselines follow below.

Correctness: 15-method peer suite passed (two unrelated methods skipped),
including graph/input updates, masks, skew, empty inputs and four new control
combinations. Independent FP32 Torch full-element H7168/I2048/E8/T128 checks
passed for balanced/masked and duplicate/skewed routing; maximum relative RMS
0.004075. Logs: `scale-active-pointwise-unit.log`,
`scale-active-pointwise-full-reference.log`,
`scale-peer-ablation-large-ep32-c1024-active/`.

## Stronger collective baseline and dHidden epilogue fusion

BF16 collective NoPad now also uses GPU row counts for activations, route
transforms and packed-input clearing. `active_pointwise=False` retains the
capacity-scanning control; FP8 and the contiguous schedule are unchanged.
Its 16-method suite passed (three unrelated skips), together with the full
large-shape independent reference checks (`scale-native-active-*.log`).

EP16 C1024 on .225/.226, identical large shape and timing protocol:

| Path | Step / ms |
| --- | ---: |
| Stronger collective, active pointwise | 53.769 |
| Peer hybrid, separate activation backward | 49.252 |
| Peer hybrid, dHidden/activation-backward epilogue fusion | 49.143 |

All16 ranks exited0 without detected interference. The epilogue preserves
BF16 rounding of dHidden before differentiating SwiGLU. Its full independent
large reference and 16-method suite passed (`scale-fused-activation-*.log`).
The measured 0.2% difference is not a material speedup claim; this fusion is
an opt-in control (`fuse_activation_backward=True`), not the default.
The old 101.58ms collective denominator must not be used as the strongest
current EP16 baseline. Profiler evidence directs the next experiment toward
scalar peer packing rather than asserting that every additional fusion wins.

## EP64 cross-supernode measurement

Eight hosts .63/.213/.219/.220/.223/.225/.226/.227 completed the large C512
collective benchmark with all64 rank parents exiting0 and clean postflights:
`scale-collective-large-ep64-c512-clean/`. This predates collective active
pointwise optimization. A prior attempt involving .222 was rejected for
postflight foreign-job interference despite all64 rank processes exiting0.
Direct-peer EP64 still requires eight simultaneously idle hosts in one domain;
cross-supernode collective capability is not a direct-peer fusion claim.

## Vectorized peer copies and local collective return fusion

Peer staging/scatter now uses aligned BF16x8 transfers. Scalar controls remain
available through `vectorized_copy=False`. Large full-reference checks and the
16-method peer suite passed, including all eight active/local-gradient/vector
control combinations. On three available hosts (.225/.226/.227), EP24 C1024:

| Path | Step / ms |
| --- | ---: |
| Native active-row collective | 67.158 |
| Vector peer packed dispatch, separate return | 30.273 |
| Vector peer packed dispatch, fused return | 29.072 |
| Vector peer direct dispatch, separate return | 47.788 |
| Vector peer direct dispatch, fused return | 46.957 |

All24 rank processes exited0 and postflights were clean. Overall hybrid versus
this baseline is2.31x; isolated return fusion is now only3.97% latency reduction,
because the independent-copy control was also accelerated. Do not carry the
older scalar control's12.15% figure over to the vector result.

`CollectiveFusedMoEPlan` adds a different BF16 path: PCCL handles transport,
including RDMA across domains, and the local GEMM epilogue writes directly into
the collective return ring. It requires no symmetric-memory peer mappings and
does not put RDMA inside GEMM. `fuse_return=False` provides the same-GEMM control.
Optional dHidden/activation-backward fusion preserves its BF16 rounding.

Its 16-method suite (five unrelated skips), full large-shape reference with
activation fusion, and four-rank DDP two-step optimizer check passed. The
Native FP8 suite also passed after factoring the activation helper. EP16 C1024
on .225/.226, all16 rank exits0/clean:

| Path | Step / ms |
| --- | ---: |
| Native active collective | 53.697 |
| Collective-fused GEMMs, separate return copy | 56.022 |
| Same GEMMs with return epilogue fusion | 53.198 |

This is about5% for the controlled return contrast and about1% versus native
auto-selected GEMMs. Cross-domain communication remains fixed-capacity; neither
number establishes large RDMA/compute overlap gains.

## Final EP32 baseline: vectorized collective routes and weight transposes

Collective BF16 NoPad now separates permutation construction from input packing
and uses BF16x8 copies for current padded expert rows and route transforms.
`vectorized_routes=False` retains the scalar active-row control. All plans now
default to a combined grouped weight-transpose kernel with BF16 bit preservation;
`fast_weight_transpose=False` retains Torch transpose/copy. The transpose falls
back for unaligned input pointers. FP8 compute/quantization semantics are unchanged.

EP8 same-peer transpose-only ablation, H7168/I2048/T1024/C1024, all8 normal exits
and clean periodic monitoring: 28.833ms with Torch copies, 22.929ms with the new
transpose, a 20.48% latency reduction. The six-method primitive suite includes
bitwise checks for signed zero, infinity and NaN payloads across three shapes.

Final intra-supernode EP32 uses .225/.226/.227/.223, H7168/I2048/E_local8/top8,
T_per_rank1024/C1024. Every path enables fast weight transposes. Peer tiles are
(256,64,64,64,2), BM128. All32 ranks exited0 without detected foreign jobs:
`scale-ep32-final-fast-attempt0/`.

| Path | Full forward/backward / ms |
| --- | ---: |
| Optimized Native collective, automatic GEMM tiles | 69.491 |
| Peer separate dispatch and return copies | 24.274 |
| Peer hybrid: separate dispatch, fused return | 23.252 |
| Peer fused dispatch, separate return | 42.415 |
| Peer bilateral dispatch/return fusion | 41.688 |

The implementation comparison is2.989x. It includes transport organization,
pointwise scheduling and GEMM differences; the isolated same-peer return fusion
reduces latency by4.21%. An additional clean32-rank measurement explicitly sets
Native to the peer tiles:72.045ms (`scale-ep32-final-matched-native-r1-attempt0/`).
Use the faster69.491ms automatic baseline. Do not reuse the historical179.456ms
capacity-scanning denominator as the current best baseline.

EP8 H7168/I8192/T1024/C256 is a useful counterexample: optimized Native111.944ms
and Peer hybrid112.956ms, all8 normal exits and clean monitoring. Larger compute
shapes do not inherit the I2048 speedup. Independent full-element I8192 reference
checks passed, maximum relative RMS about0.0041.

## Cross-domain EP64 and final validation scope

Eight hosts .63/.213/.219/.220/.223/.225/.226/.227 span three supernodes.
H7168/I2048/E_local8/top8/T1024/C512, with fast transpose disabled consistently
for this group. All64 ranks exited0 and availability postflights were clean:
`scale-ep64-collective-v2-return-ablation/`.

| Path | Full forward/backward / ms |
| --- | ---: |
| Optimized Native collective, automatic tiles | 208.727 |
| Collective local-fusion GEMMs, separate return copy | 211.054 |
| Same GEMMs with local return-ring epilogue fusion | 210.598 |

There is no material controlled return-fusion gain and no overall gain against
Native automatic tiles. PCCL transport is still fixed-capacity all-gather and
reduce-scatter. This is not a sparse hierarchical dispatcher or RDMA inside GEMM.
Intra-domain direct-peer EP64 remains unmeasured; cross-domain64 must not be
presented as that result.

Final Peer EP32 and collective-local-fused EP64 passed independent global CPU
FP64 references and128 dynamic-routing graph replays, with all ranks exiting0
and clean periodic monitoring. The first Peer32 attempt failed at SSH startup
(24/32 clients joined), and its failed logs are retained; the complete retry
passed. Selected new transpose, collective vector-route and peer vector-copy
kernels passed hggc-memcheck with0 errors. This filtered check does not cover
every GEMM or all distributed race behavior.

Collective local fusion passed DDP and FSDP2 DP2xEP2 two-optimizer-step checks
with FP32 parameter policy. Its complete57.8M public-text MoE model ran2000steps
with seed4827; full111,539-successor held-out CE is1.585704793 nats. This is a
single-seed small-model validation, separate from the earlier three-seed BF16,
FP8 and Torch comparison; it does not establish large-model convergence.
