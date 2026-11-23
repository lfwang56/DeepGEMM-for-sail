
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

## Iteration143: opt-in native prefetch and tc02 MMA-reuse issue policy

Built isolated native runtime143 successfully (SHA5f47106c...), preserving baseline library. Added validated1/2/4/8 prefetch parameters and tested a process-stable optional tc02 peer-GEMM compiler reuse flag. All four EP8 diagnostic reference/timing phases passed. Same per-rank H4096/I1024/E_local16/top10/T2048 compute shape, global E128/EP8 rather than target E512/EP32: reuse0 12.27136 ms, reuse1 12.26493 ms, no meaningful improvement. Do not promote MMA policy or claim EP32 speedup from this diagnostic. Keep prototype source and reproducible gate; await complete EP32 gradient-path controls and ring search before selecting a final implementation.

## Iteration142: complete EP32 gradient-path prototype evaluation

All10 phases passed: BF16/FP8 independent32-rank FP32 references at full H4096/I1024/E512/top10, plus eight unprofiled full2048-token FWD+BWD timings on221–224. Candidate128-bit vector prefetch uses identical BF16 weights/operands and FP32 DW accumulation; no precision change. Results:

- mok32-opt142-prototype-timing-bf16-new2: 13.20137 ms
- mok32-opt142-prototype-timing-bf16-new4: 13.17517 ms
- mok32-opt142-prototype-timing-bf16-old4: 12.93368 ms
- mok32-opt142-prototype-timing-bf16-old8: 12.97655 ms
- mok32-opt142-prototype-timing-fp8-new16: 13.15677 ms
- mok32-opt142-prototype-timing-fp8-new8: 13.09093 ms
- mok32-opt142-prototype-timing-fp8-old16: 13.15994 ms
- mok32-opt142-prototype-timing-fp8-old4: 13.48318 ms

BF16 prefetch lost against the tuned original mover despite winning the isolated component benchmark. Do not promote that kernel on isolated timing. Use the strongest original FP8 CTA control before attributing a gain to new prefetch. Any selected recipe still requires T4096 confirmation, same-cohort DeepEP control and full-step profile.

## Iteration144: DW tile/resource interaction

All eight variants passed independent FP32 DW role/pattern checks and whole-host audits. Tested stage2 smaller accumulator tiles intended to permit additional resident CTAs, two1024-thread alternatives and wider-K tiles. Control256x256x64/warp64x64/stage3 remains fastest: DW1 1.6246/2.4984 ms, DW2 0.8637/1.3202 ms at T2048/4096 equivalents. Every candidate was slower (DW1 T2048 1.96–2.98 ms). Runtime occupancy and function resources were queried by the existing native launcher, not inferred from shared-memory size:

- variant0: ['num_sms:39, max_active_tb_num:1, threadblock_count:39, threads:512']; ['dynamic_smem_bytes:196608, vreg:216, local_size_bytes:0']
- variant1: ['num_sms:39, max_active_tb_num:1, threadblock_count:39, threads:512']; ['dynamic_smem_bytes:98304, vreg:144, local_size_bytes:0']
- variant2: ['num_sms:39, max_active_tb_num:1, threadblock_count:39, threads:512']; ['dynamic_smem_bytes:98304, vreg:160, local_size_bytes:0']
- variant3: ['num_sms:39, max_active_tb_num:1, threadblock_count:39, threads:1024']; ['dynamic_smem_bytes:131072, vreg:128, local_size_bytes:0']
- variant4: ['num_sms:39, max_active_tb_num:1, threadblock_count:39, threads:1024']; ['dynamic_smem_bytes:131072, vreg:128, local_size_bytes:0']
- variant5: ['num_sms:39, max_active_tb_num:1, threadblock_count:39, threads:512']; ['dynamic_smem_bytes:196608, vreg:144, local_size_bytes:0']
- variant6: ['num_sms:39, max_active_tb_num:1, threadblock_count:39, threads:512']; ['dynamic_smem_bytes:196608, vreg:160, local_size_bytes:0']
- variant7: ['num_sms:39, max_active_tb_num:1, threadblock_count:39, threads:512']; ['dynamic_smem_bytes:196608, vreg:144, local_size_bytes:0']

Retain stage3 tile for complete-step verification. All variants still report one resident CTA per CU, including the smaller shared-memory cases. The1024-thread variants also lost. No claim that reducing TSM alone increases occupancy or useful overlap.

## Iteration146: FC1 producer/compute interleaving (first gate failed)

Prepared unchanged-precision BF16 producer1/2/4 and32/64-thread helper recipes. Extended only the existing independent FP32 reference CLI to pass producer_tiles; oracle inputs, comparison and gold math are unchanged. EP8 diagnostic baseline reference/timing completed, but producer2 failed its independent-reference phase, so its performance and subsequent variants are not accepted. Preserve logs and diagnose before continuing; no target EP32 gain claimed.

### Iteration136 completed ring-capacity matrix

All16 same-cohort EP32 precision/backend/capacity timing phases passed whole-host audits on215–218. BF16/FP8 both tested R65536/131072/262144/520192; same precision DeepEP dispatch/combine also received all four capacities. Full T2048 step results:

- mok32-opt136-ring-r3-r131072-bf16-elastic: 17.57444 ms, workspace 6088311280 bytes
- mok32-opt136-ring-r3-r131072-bf16-peer: 12.70555 ms, workspace 4578978560 bytes
- mok32-opt136-ring-r3-r131072-fp8-elastic: 17.25404 ms, workspace 7745405424 bytes
- mok32-opt136-ring-r3-r131072-fp8-peer: 12.81907 ms, workspace 6367415040 bytes
- mok32-opt136-ring-r3-r262144-bf16-elastic: 17.36554 ms, workspace 10116415984 bytes
- mok32-opt136-ring-r3-r262144-bf16-peer: 12.56177 ms, workspace 8610237184 bytes
- mok32-opt136-ring-r3-r262144-fp8-elastic: 17.01120 ms, workspace 12742394352 bytes
- mok32-opt136-ring-r3-r262144-fp8-peer: 12.61463 ms, workspace 11381721856 bytes
- mok32-opt136-ring-r3-r520192-bf16-elastic: 17.29758 ms, workspace 18046747120 bytes
- mok32-opt136-ring-r3-r520192-bf16-peer: 12.50676 ms, workspace 16546777600 bytes
- mok32-opt136-ring-r3-r520192-fp8-elastic: 16.94144 ms, workspace 22580216304 bytes
- mok32-opt136-ring-r3-r520192-fp8-peer: 12.55118 ms, workspace 21253638400 bytes
- mok32-opt136-ring-r3-r65536-bf16-elastic: 17.89484 ms, workspace 4074258928 bytes
- mok32-opt136-ring-r3-r65536-bf16-peer: 12.97947 ms, workspace 2563349248 bytes
- mok32-opt136-ring-r3-r65536-fp8-elastic: 17.59144 ms, workspace 5246910960 bytes
- mok32-opt136-ring-r3-r65536-fp8-peer: 13.15956 ms, workspace 3860261632 bytes

Large-R winners are provisional until full-skew capacity gates and matched DeepEP+fusion stage3-DW/t4096 reruns pass. Increased workspace is a material tradeoff. Earlier SN2 foreign-job and launch-failure attempts remain excluded.

## Iteration145: completed dw-full145 gate

All10 phases passed whole-host audits. Target EP32 H4096/I1024/E512/top10; stage3 DW references and T2048/T4096 complete-step controls.

- mok32-opt145-dw-full-bf16-t2048-builtin: 12.91855 ms
- mok32-opt145-dw-full-bf16-t2048-stage3: 12.80921 ms
- mok32-opt145-dw-full-bf16-t4096-builtin: 23.42980 ms
- mok32-opt145-dw-full-bf16-t4096-stage3: 23.21907 ms
- mok32-opt145-dw-full-fp8-t2048-builtin: 13.15521 ms
- mok32-opt145-dw-full-fp8-t2048-stage3: 13.06761 ms
- mok32-opt145-dw-full-fp8-t4096-builtin: 23.23394 ms
- mok32-opt145-dw-full-fp8-t4096-stage3: 23.03077 ms

Select only after same-cohort end-to-end confirmation; retain all rejected candidates and immutable baseline.

## Iteration147: completed helper147 gate

All6 phases passed whole-host audits. EP8 component cohort on node88, same per-rank H/I/E_local16. Helpers pass the same unchanged independent FP32 oracle; no EP32 performance claim yet. Prior producer2 attempt failed a documented AIU one-N-tile validation, before numerical execution, not a numerical mismatch.

- mok32-opt147-fc1-base-timing: 12.45612 ms
- mok32-opt147-fc1-h32-timing: 15.39129 ms
- mok32-opt147-fc1-h64-timing: 13.07593 ms

Select only after same-cohort end-to-end confirmation; retain all rejected candidates and immutable baseline.

## Iteration148: paired final DeepEP controls (initial cohort excluded)

First reference phase was rejected by whole-host postflight/launch gates on215/216. No timing accepted. The controller canceled only verified tagged ranks. Preserve phase and replace215/216 with213/219, subject to fresh idle gates; final fusion timings must use the replacement cohort too.

### Iteration148 replacement-cohort preflight excluded

The replacement213/217/218/219 cohort returned75 before any accepted reference/timing. Preserve its controller record and reselect from a fresh whole-host census. No rejected timings enter comparison.

## Iteration149: full-source-token skew capacity gate

Both BF16 and FP8 completed clean EP32 gates on221–224 at H4096/I1024/E512/top10, C2048 and R520192, with the stage3 DW lookup enabled. All32 ranks received the complete2048 source tokens per rank and all routes concentrated on one expert. Independent FP32 reference evaluates that expert once per source token and sums routing weights, preserving individual score derivatives. This crosses ring windows at near-limit packed strides and includes zero score/empty experts. These are correctness gates, not speed measurements.

## Iteration152: source FP8 transport with fused gradient quantization (prerequisite failed)

Created an isolated Python compatibility prototype allowing source-quantized X transport with fused DY quantization only (bit2); no native kernel/precision change. Baseline reference/timing passed, but the legacy source-pipeline prerequisite returned1 before candidate cases ran. All candidate performance remains unaccepted. Iteration153 cached-copy scheduling prototype was prepared in another isolated library, but its guarded runner refused to launch GPU work because152 did not complete all prerequisites. Preserve both prototypes and inspect the legacy failure before further testing.
