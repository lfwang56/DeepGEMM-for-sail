
## Iteration 136: shape-specific geometry, buffers and DW tuning

Baseline: EP32 BF16 T2048 12.9846 ms vs tuned DeepEP 17.8734 ms; FP8 13.1612 vs 17.5790 ms. Existing Asys traces identify two active DW kernels taking 2.63 ms, plus 0.16 ms empty DW calls. DW uses 256 KiB TSM and builtin stage4 because prior stage3 LUT has no H4096/I1024 entries. Search same precision DeepEP and fusion symmetrically; no speedup claim until independent references and clean complete-step reruns.

Operational exclusions: DW node63 was occupied at preflight and no work launched; moved to freshly idle node80. Ring initial cohort had foreign activity on225/226 in phase2; preserve/exclude and replace hosts. Geometry matrix continues independently on215–218. No kernel implementation modified in this iteration.

### Completed DW isolated search

All eight candidates passed two independent FP32 reference patterns for both DW roles, graph replay beta=1 accumulation, empty-group preservation, library hash, and whole-host pre/post audits on node80. Tile256x256x64/warp64x64/stage3 provisionally wins: DW1 1.6253/2.4903 ms and DW2 0.8703/1.3337 ms at T2048/4096 equivalents, versus builtin 1.7052/2.6081 and 0.8934/1.4043 ms. Smaller tiles and stage2 were slower. Candidates were assigned different devices; these are search signals only and must be confirmed in paired EP32 complete-step tests on the same cohort. No complete-step improvement claimed.

## Iteration 137: DW warp mapping and AIU issue order

All eight stage3 candidates passed independent FP32 role/pattern checks, graph beta=1 and empty-group checks, immutable-library validation and whole-host audit. AIU-before-TSM=1 consistently beats0 on the observed cards (T2048 DW1 about1.62–1.64 versus1.69–1.73 ms). Warp mapping0–3 gives no robust large gain; retain native map0 pending full-step tests. Best isolated sums are2.4723/3.7880 ms at T2048/4096 equivalents, within a few percent of the default-map stage3 variant. No changed math or full-step speedup claim.

## Iteration 138: prefetch independent gradient peer reads

Implemented header-only ILP2/4/8 StageRawWeightedDy candidates in isolated library copies, preserving immutable135 baseline. One-host eight-GPU mover benchmark includes real symmetric-memory peer reads, a separate collective oracle, bitwise raw/weighted DY and score checks, valid zero scores, padded routes, inactive tail, zero-row graph replay and changed peer payload. All32 ranks across four phases passed source/library hashes and whole-host audits. Best baseline CTA4 1.19842 ms; ILP2 CTA4 0.93183; ILP4 CTA4 0.88279; ILP8 CTA2 0.88086. About26.5% lower mover latency at best versus the tuned baseline mover; scope is eight-GPU mover only. Must validate cross-host EP32 full FWD+BWD and both T before promotion. StageRawWeightedDy only affects BF16 dHidden-score recipe; FP8 uses another kernel and receives no speedup claim.
