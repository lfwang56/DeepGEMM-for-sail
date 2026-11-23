
## Iteration 136: shape-specific geometry, buffers and DW tuning

Baseline: EP32 BF16 T2048 12.9846 ms vs tuned DeepEP 17.8734 ms; FP8 13.1612 vs 17.5790 ms. Existing Asys traces identify two active DW kernels taking 2.63 ms, plus 0.16 ms empty DW calls. DW uses 256 KiB TSM and builtin stage4 because prior stage3 LUT has no H4096/I1024 entries. Search same precision DeepEP and fusion symmetrically; no speedup claim until independent references and clean complete-step reruns.

Operational exclusions: DW node63 was occupied at preflight and no work launched; moved to freshly idle node80. Ring initial cohort had foreign activity on225/226 in phase2; preserve/exclude and replace hosts. Geometry matrix continues independently on215–218. No kernel implementation modified in this iteration.

### Completed DW isolated search

All eight candidates passed two independent FP32 reference patterns for both DW roles, graph replay beta=1 accumulation, empty-group preservation, library hash, and whole-host pre/post audits on node80. Tile256x256x64/warp64x64/stage3 provisionally wins: DW1 1.6253/2.4903 ms and DW2 0.8703/1.3337 ms at T2048/4096 equivalents, versus builtin 1.7052/2.6081 and 0.8934/1.4043 ms. Smaller tiles and stage2 were slower. Candidates were assigned different devices; these are search signals only and must be confirmed in paired EP32 complete-step tests on the same cohort. No complete-step improvement claimed.

## Iteration 137: DW warp mapping and AIU issue order

All eight stage3 candidates passed independent FP32 role/pattern checks, graph beta=1 and empty-group checks, immutable-library validation and whole-host audit. AIU-before-TSM=1 consistently beats0 on the observed cards (T2048 DW1 about1.62–1.64 versus1.69–1.73 ms). Warp mapping0–3 gives no robust large gain; retain native map0 pending full-step tests. Best isolated sums are2.4723/3.7880 ms at T2048/4096 equivalents, within a few percent of the default-map stage3 variant. No changed math or full-step speedup claim.

## Iteration 138: prefetch independent gradient peer reads

Implemented header-only ILP2/4/8 StageRawWeightedDy candidates in isolated library copies, preserving immutable135 baseline. One-host eight-GPU mover benchmark includes real symmetric-memory peer reads, a separate collective oracle, bitwise raw/weighted DY and score checks, valid zero scores, padded routes, inactive tail, zero-row graph replay and changed peer payload. All32 ranks across four phases passed source/library hashes and whole-host audits. Best baseline CTA4 1.19842 ms; ILP2 CTA4 0.93183; ILP4 CTA4 0.88279; ILP8 CTA2 0.88086. About26.5% lower mover latency at best versus the tuned baseline mover; scope is eight-GPU mover only. Must validate cross-host EP32 full FWD+BWD and both T before promotion. StageRawWeightedDy only affects BF16 dHidden-score recipe; FP8 uses another kernel and receives no speedup claim.

## Iteration 139: FP8 gradient prefetch candidate (initial gate incomplete)

Generated isolated ILP2/4/8 ExpertLocalGradients/Quantized headers, preserving sequential per-lane dot accumulation and quantization recipe. The initial mover matrix returned nonzero in its baseline phase before candidate performance was accepted. Preserve the attempt and inspect its independent-reference failure before continuing. No FP8 speedup or complete-step improvement claimed; candidate headers are not promoted to the active implementation.

## Iteration 140: FP8 gradient prefetch evaluation

Initial139 baseline failure came from requiring torch.to(float8) bytes rather than the established PPU native quantization contract. Retained failed logs. New gate uses independent collective gather for BF16 weighted data (bitwise), an independent FP32 dScore oracle (<2e-6 relative RMS), independent FP8 dequantization (<3% RMS), and exact bytes against unchanged native row quantizer34, including zero scores, padding, inactive tails and zero-row graph replay. All32 ranks and host audits passed. At CTA2/4/8/16, original mover:2.43585/1.36199/1.33581/1.09749 ms; ILP2:1.62218/1.12556/1.07651/1.09370; ILP4:1.93184/1.85145/1.44791/1.43960; ILP8:2.09205/2.09532/2.16584/2.22783. Retain ILP2/CTA8 candidate and original/CTA16 control for full EP32; extra live registers make ILP4/8 unattractive in this quantizing kernel. These are one-host eight-GPU component timings only.

## Iteration 141: real EP32 cross-host gradient mover

Fresh whole-host gates on221/222/223/224 permitted an isolated direct-peer test after earlier DeepEP failure; all64 rank checks across two phases passed separate collective oracles, source hashes and four-host audits. EP32 original CTA2/4/8/16:1.16581/1.01727/1.22262/1.22330 ms; ILP4:0.95083/0.96191/0.98470/0.98425 ms. Tuning the original CTA count already explains most of the earlier8-GPU gain. New mover improves about6.5% against its strongest measured EP32 control; no26% claim applies to the complete step. SN2 direct-peer access worked; earlier failed phase used DeepEP dispatch and has no accepted timing. Next gate is complete routed MoE FWD+BWD.

### Iteration136 completed geometry matrix

All20 reference/timing phases passed whole-host audits. Each configuration has independent32-rank FP32 references for both DeepEP and fused paths at full H/I/E and19 source tokens. Full2048-token step comparisons:

- mok32-opt136-geometry-base-timing-elastic: 17.91384 ms
- mok32-opt136-geometry-base-timing-peer: 12.93788 ms
- mok32-opt136-geometry-m128n128w32x32s3-timing-elastic: 20.05481 ms
- mok32-opt136-geometry-m128n128w32x32s3-timing-peer: 14.10852 ms
- mok32-opt136-geometry-m128n128w32x64s3-timing-elastic: 19.85384 ms
- mok32-opt136-geometry-m128n128w32x64s3-timing-peer: 14.11463 ms
- mok32-opt136-geometry-m128n256s2-timing-elastic: 18.82905 ms
- mok32-opt136-geometry-m128n256s2-timing-peer: 13.86454 ms
- mok32-opt136-geometry-s2-timing-elastic: 18.75039 ms
- mok32-opt136-geometry-s2-timing-peer: 13.99161 ms

Keep original BM256/BN256/BK64/warp64x64/stage3. All four alternative geometries were slower for complete FWD+BWD. Reassessed existing profile and PPU source-backed guidance rather than extending a losing tile search; subsequent directions are peer gradient prefetch and controlled tc02 compiler issue reuse.
