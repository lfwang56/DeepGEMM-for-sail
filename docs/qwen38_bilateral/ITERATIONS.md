# Qwen3.8 bilateral MoE optimization iterations

## Objective and baseline

Optimize bilateral dispatch/return fusion at the pinned Qwen3.8-2.4T-A95B
dimensions recorded in HINTS.md. Initial source is cbd10be75229 (full SHA from
the source audit), before kernel changes for this objective. Earlier H7168,
I2048, top8 results are historical and do not satisfy the new target dimensions.

Baseline preparation in progress: inspect the PPU dispatch/combine library,
measure optimized standalone dispatch/grouped-GEMM/combine and current bilateral
fusion, profile the current bilateral hot kernels, then iterate on the bottleneck.

Each iteration records source/patch, chip/toolchain, shape/topology, correctness,
timing protocol, results, exclusions and next direction. Commit after each
candidate's benchmark and log. Preserve the best verified candidate at completion.

## Baseline: Qwen3.8 EP32, before kernel changes

Source cbd10be75229. ZW-M890P/ppu0015, SDK2.2.0-663c22, Torch2.10.0.
H8192/I2048, 512 total experts (16/rank), top10, T1024/rank, C512.
Four same-domain hosts .225/.226/.227/.223; all32 ranks exited0 and all
idle-monitor gates returned0. Routed experts only; shared expert excluded.
Routing is random top-k without replacement (the CLI labels this balanced).
Five repeats x20 graph steps, per-repeat maximum rank event time then median;
full forward/backward replay and FP32 wgrads, no optimizer/JIT.

| Backend | fuse_dispatch | fuse_return | Step ms |
| --- | --- | --- | ---: |
| collective | null | null | 106.7143 |
| peer | false | false | 42.3923 |
| peer | false | true | 40.8655 |
| peer | true | false | 68.0199 |
| peer | true | true | 67.1540 |

Logs: qwen38-ep32-bilateral-baseline-attempt0/. The collective is an auxiliary
all-gather/reduce-scatter reference. PPU DeepEP baseline integration is pending.
No kernel optimization or industry-baseline speedup claim is made by this run.

## Iteration 1: CTA-private input cache across N tiles

Direction: load peer rows into CTA-private HBM scratch inside the GEMM, reuse
them across four N tiles, and use cache-global async copies into TSM. No separate
dispatch kernel and no cross-CTA synchronization. Existing return fusion stays
enabled. Scratch is bounded to two CTAs per CU, about156MiB for H8192.

Correctness preparation: the new unit method passed all cache-group sizes
2/4/8/16, six-N-tile tails, duplicates, masks and ragged chunks, against independent
CPU FP64 outputs and all gradients. One method,122.991s including cold JIT.

Performance shape/protocol matches the Qwen EP32 baseline above. Torch2.10
native container is the iteration signal; the final DeepEP comparison will use
the user's wlf_sgl_2.2 (Torch2.11, installed DeepEP2.2) consistently.
Whole-phase return code: 1. Nodes: 172.17.0.225, 172.17.0.226, 172.17.0.227, 172.17.0.222.
Logs: qwen38-ep32-bilateral-iter1-cache4-attempt0/. Failed/interfered phases are excluded from speedup claims.

| Backend | dispatch | return | cache N | Step ms |
| --- | --- | --- | --- | ---: |
| collective | null | null | 0 | 106.8472 |
| peer | false | false | 0 | 42.3934 |
| peer | false | true | 0 | 40.7894 |
| peer | true | false | 4 | 62.6312 |
| peer | true | true | 4 | 61.5644 |

This phase is not accepted for performance claims; inspect launcher/rank logs before the next iteration.

### Iteration1 retry in the user container

`wlf_sgl_2.2`, Torch2.11.0 runtime with installed DeepEP2.2.0. The existing
native extension compiled against Torch2.10 imports here and passed3 GPU
reference/updated-parameter graph methods before this run; an independent
Torch2.11 rebuild is still being resolved. Do not claim that rebuild passed.

Same Qwen EP32 dimensions and5x20graph protocol; .219/.220/.213/.215.
All32 rank parents exited0 and all periodic availability gates exited0.
Logs: qwen38-wlf-ep32-cache4-attempt0/. Routed experts only; shared expert excluded.

| Backend | dispatch | return | cache N | Step ms |
| --- | --- | --- | --- | ---: |
| collective | null | null | 0 | 106.7633 |
| peer | false | false | 0 | 41.8626 |
| peer | false | true | 0 | 40.2097 |
| peer | true | false | 4 | 58.9652 |
| peer | true | true | 4 | 57.4816 |

These are accepted same-runtime measurements. The noncached bilateral
control and installed DeepEP baseline still need measurement in this environment.

### Noncached control excluded

The first same-runtime noncached control on .225/.226/.227/.223 completed
all32 ranks, but .227 reported foreign GPU work during the phase (gate75).
`qwen38-wlf-ep32-cache0-control-attempt0/` is excluded. The queued cache8/16
controls were not launched. Retry on a different supernode after fresh checks.

The SN1 retry (.219/.220/.213/.215) also completed all32 ranks, but .213
reported foreign GPU work (gate75). `qwen38-wlf-ep32-cache0-control-sn1-attempt0/`
is excluded. Next selection prioritizes .215/.216 ahead of the contested .213.


### Iteration 1 controls: wlf SN1 stable-node matrix (2026-11-21 04:05 UTC)

- Direction: controlled dispatch-cache grouping sweep; unchanged iteration-1 kernels.
- Runtime: wlf_sgl_2.2, EP32 within SN1 (.219/.220/.215/.216), Qwen H8192/I2048/E512/K10, T1024/rank, C512.
- Results: cache0 phase completed with all gates passing; cache8 completed all rank parents, but .220 returned gate75; cache16 was not started.
- Status: cache8 quarantined pending PID-ownership audit. A monitor exit race was discovered: an exiting process can lose its /proc/environ before disappearing from the driver process table. Do not label this real foreign interference without matching the recorded PID against rank receipts. Raw logs are preserved. Timings will be appended after audit.
- Next: resolve gate classification, then continue cache grouping and backward packed-DY reuse. No source change is accepted on these unclassified measurements.

### Iteration 1 completed grouping sweep and PID audit (2026-11-21 04:11 UTC)

- The cache16 SN1 phase completed: all 32 rank parents and all four ownership gates passed after the monitor fix. TT latency 53.2545166 ms, samples [53.4234985,53.2545166,53.2108704,53.2622314,53.2236877] ms.
- Same-runtime same-node SN1 cache0: 64.4701233 ms; cache8:55.2007019 ms. Cache16 reduces TT step time 17.40% (1.2106x) relative to cache0. Prior cache4 57.4815613 ms ran .219/.220/.213/.215; distinguish node sets.
- Audit correction: the earlier cache0 SN2 (.227), cache0 SN1 (.213) and cache8 stable SN1 (.220) gate75 records contain only PIDs listed in the respective local_ranks_started receipts. Container PidMode=host, so these are the same PIDs. All parents returned zero and postflight process tables were empty. This was an ownership monitor exit race, not evidence of foreign GPU work. Preserve raw gate results; accept timings under explicit post-hoc ownership audit. Earlier exclusion descriptions are superseded by this audit.
- Additional audited cache0 timing: SN2 67.0972290 ms; original SN1 64.3218140 ms. Do not mix SN1 and SN2 denominators.
- Ordinary installed DeepEP Buffer two-rank independent probe now passes forward, DX, router-score gradients and four dynamic-route graph replays, with clean teardown and ownership gate. ElasticBuffer JIT remains a separate compatibility issue.
- Next direction: reuse the mandatory packed backward-DY buffer in dHidden; retain fused forward dispatch and fused forward/backward return.

### Iteration 2 — reuse mandatory packed DY in dHidden

- Hypothesis: backward already gathers router-weighted DY for DW2. Reusing that identical BF16 buffer in dHidden avoids fetching the same DY from peers again. Forward dispatch and FC2/DX return remain fused; no extra standalone dispatch is introduced. Backward DY packing remains explicit for DW2 and is reused, so this is a dataflow optimization, not a claim that every backward transfer occurs inside GEMM.
- Correctness: peer cache-group unit passed (126.451s, not the earlier mistakenly skipped NativeTests invocation). Full independent FP32 reference at Qwen H8192/I2048/E16/K10/T19 compared every Y/DX/DW1/DW2/dScore element; masked/ragged and duplicate-skew patterns passed, worst relative RMS0.00494.
- Performance: same SN1 four hosts, wlf2.11, EP32, T1024/C512, cache16, TT;49.3500092ms, samples [49.5296509,49.3422821,49.3500092,49.3655273,49.3049500]. All32 parents and four gates passed.
- Gain:7.33% less time than cache16 without reuse53.2545166ms;23.45% less than original same-node cache0 64.4701233ms (1.3064x). This denominator is the previous bilateral implementation, not DeepEP.
- Decision: retain as an explicit option, defaultFalse until final EP32/64 and model gates. Next: profile the new bottleneck and compare a real installed DeepEP training baseline.

### Installed DeepEP training baseline — matched EP32 scale-up

- Implemented DeepEPMoEPlan using the existing wlf_sgl_2.2 DeepEP2.2.0 Buffer dispatch/combine, BF16 native NoPad grouped GEMMs, full forward recomputation in backward, FP32 DW1/DW2, and routing-score gradients. GPU count/prefix/permutation metadata and fixed worst-case receive capacity support dynamic graph replay. Local unpermute reduces top-k in registers, directly writes BF16, and avoids FP32 atomics or expanded [receive,topk,H] output. No installed DeepEP files modified.
- Correctness: independent primitive test passed; two-rank independent CPU reference plus128 graph replays across balanced, duplicate-skewed, masked and ragged routes passed, including empty rank0 and a valid route with zero score. Clean teardown and idle-node gate. Native2.10-built extension SHA256 ee614737d90c6cf207ac462c90f12fdecbd2068eccdd3a42a40c194d47f17c63 ran in wlf Torch2.11.
- Matched case qwen38-wlf-ep32-deepep-scaleup-baseline, same SN1 four hosts, Qwen H8192/I2048/E512/K10, T1024/C512, tiles128x256x64/S2,5x20 graph steps. DeepEP sailshmem forced verified-domain fabric scale-up (RDMA ranks1),8comm CUs:51.5132202ms; bilateral cache16+DYreuse49.3827667ms. All32 rank parents and all gates passed.
- Observed gain4.14% less time,1.04314x against this DeepEP implementation. This is a measured initial baseline, not a claim to have beaten a fully tuned DeepEP or all industry implementations. Tune communication CUs and local layout/GEMM choices before final comparison. Peer versus DeepEP sampled relative RMS: Y0.00269,DX0.00149,DW1/DW2/dScore below2.9e-7.
- Iteration2 Asight profile clean on all ranks: FC1 remains37.4% (~18.57ms/step), weight gradients20.4%(~10.13ms), FC2return8.7%, DXreturn8.3%, transpose5.2%. Instrumented timing excluded from performance conclusions.
- Next kernel direction: feed the contiguous CTA input cache through AIU with its matching swizzled A operand; arbitrary remote-row gather itself stays explicit inside GEMM.

### DeepEP communication-CU controls

- Same SN1 four hosts and Qwen EP32T1024/C512, forced fabric scale-up, same tiles and5x20 graph repetitions. All three phases and ownership gates passed.
- Requested comm CUs4:58.1821777ms;16:49.1189117ms;24:48.0826019ms. Initial8-CU baseline was51.5132202ms.
- Use at least the24-CU48.0826ms implementation as the current DeepEP denominator. It is2.63% faster than the prior bilateral49.3828ms result; the previous4.1% gain versus8CUs does not establish a gain over tuned DeepEP. More baseline tuning remains possible.
- Next: AIU contiguous-cache candidate is authored and built, but still awaiting correctness and performance validation. Keep it separate from this unchanged-DeepEP control sweep.

### DeepEP high-CU controls

Same Qwen EP32 configuration and SN1 hosts, unchanged DeepEP training dataflow. Both phases and all ownership gates passed.

- comm CUs32: 47.703472900 ms; samples [47.7762451171875, 47.694482421875, 47.70347290039062, 47.67678527832031, 47.74150085449219].
- comm CUs38: 46.940802002 ms; samples [47.030416870117186, 46.940802001953124, 46.92217102050781, 46.9336669921875, 46.999734497070314].

The best measured DeepEP communication setting replaces the initial8-CU and24-CU denominators for final comparisons. AIU-cache candidate remains under separate correctness validation.

### Iteration 3 — AIU loads from each CTA's contiguous input cache

- Direction: retain the in-GEMM arbitrary peer gather, then use the native PPU AIU to feed the now-contiguous cached A rows into a matching swizzled TSM/MMA operand. Separate dispatch_cache_aiu option, defaultFalse. Ordinary gather paths retain their original copy atom.
- Correctness: cache-group unit passed100.868s including partial-N groups and backward; full Qwen H8192/I2048/E16/K10/T19 independent FP32 reference passed every element for masked/ragged and duplicate-skew patterns. Worst relative RMS0.00494.
- Matched EP32 SN1 paired run, all32 parents and all gates passed: tuned DeepEP38CUs46.9861084ms; bilateral AIU-cache48.9703064ms, samples [48.9510254,48.9621429,48.9878479,49.0075073,48.9703064].
- Compared to iteration2 without AIU49.3827667ms, only0.84% less time; below3% significance threshold. Still4.22% slower than the freshly measured tuned DeepEP baseline. Do not call this a material optimization win. Leave AIU optional and continue a different direction.
- Source also exposes the peer cache/reuse/tile options in the existing complete-model public-text trainer and checkpoints so the chosen recipe can be tested end-to-end. Full model runs still pending for the final winner.
- Next: address FC1 input-cache/GEMM scheduling and occupancy rather than assuming a bulk copy atom is sufficient. Broaden to T4096/8192 and keep the tuned DeepEP comparison.

### Larger-token controls for iteration3

- Same actual Qwen H8192/I2048/E512/K10, EP32 SN1 four hosts, C512 and5x20 graph steps. Paired tuned DeepEP38CUs versus cache16+AIU+packed-DY bilateral, staged_return=False.
- T4096/rank: DeepEP176.3947021ms, bilateral184.1678833ms (bilateral4.41% slower).
- T8192/rank: DeepEP348.2614502ms, bilateral364.1226807ms (bilateral4.56% slower).
- Increasing token count alone did not establish a fusion gain over the tuned dispatch/combine baseline. Keep these negative results in the final report.
- Whole-phase controller receipts at logging time:
```json
{"case": "distributed_matrix_phase", "name": "qwen38-wlf-ep32-t4096-iter3-control", "returncode": 0}
{"case": "distributed_matrix_phase", "name": "qwen38-wlf-ep32-t8192-iter3-control", "returncode": 0}
```
The future staged-return source is present but disabled in these controls. Its independent small/Qwen references passed; its distributed graph and performance gates remain separate.

### Iteration 4 — TSM redistribution for contiguous BF16x8 return stores

- Direction: reuse drained GEMM mainloop shared memory for BF16 accumulator pairs, then redistribute into contiguous128-bit peer stores. Dispatch and return remain inside the GEMMs; no separate output/scatter buffer or return kernel. Explicit staged_return option, defaultFalse.
- Correctness: BN64/128/256 unit passed130.959s; complete Qwen expert-dimension reference passed both masked/ragged and duplicate-skew distributions, worstRMS0.00494. EP2 independent CPU reference and128dynamic-route graph replays passed, with normal teardown and clean gate.
- Paired Qwen EP32T1024 SN1: tuned DeepEP38CUs46.9889832ms; staged-return bilateral49.2693939ms, samples [49.2206207,49.2526611,49.2714874,49.2693939,49.2770874]. All32parents and allfour gates passed.
- Result:0.61% slower than iteration3unstaged48.9703064ms; do not enable by default. Two consecutive directions (AIU, staged return) have not materially improved the best fusion time.
- Next: inspect shared staging layout/transaction cost and FC1 scheduling. An aligned shared-row padding control can test whether the transpose layout, rather than remote store width, negates the return benefit. No claim of physical floor.

### Complete public-text LM checks in wlf runtime (partial held-out audit)

- Three matched seeds, EP2,57.8M4-layerH512/I1024/E8/K2,2000steps, public pinned TinyShakespeare. Iteration3 peer cache16/AIU/packed-DY compared with nativeBF16; small LM is not Qwen2.4T convergence validation.
- Seeds4827/6113 completed and fully evaluated111539held-out successors: Native1.5816298120/1.5929814162;Peer1.5815587536/1.5918998886. No quality-win claim.
- Original seed4891 BOTH paths failed identical unfrozen-dense preflight router gradient .042700646>unchanged.04, overallgradRMS.00333; branch choices differ by[0,0,2,1]tokens across layers. Preserve failures.
- Explicit fixed-route preflight records those unfrozen differences, then compares continuous gradients on actualselected expert IDs. Both paths passed: selected differences[0,0,0,0], overallRMS.00226,maxparameter.00449, unchangedtolerance. This option only affects the independent oracle, and real2000-steptrainingremainsdynamic. Both retried seed4891trainingjobs nowcompleted2000steps and clean gates; fullheldoutevaluationstillpending.

### Complete three-seed public-text LM held-out evaluation

- native: mean CE 1.5857625161, sample SD 0.0062736101.
- peer: mean CE 1.5857200174, sample SD 0.0054580995.
- Seeds4827/4891/6113 each trained2000steps, EP2, same parameter initialization and public text. Every checkpoint evaluated on all111539held-out successors. Seed4891 nativeCE1.5826763200, peerCE1.5837014101. All training retries/evaluation gates normal.
- The means differ by about0.00004nats, far below seed variation; no quality improvement claim. Frozen-route preflight did not freeze real training. Keep the original seed4891unfrozen oracle failures and exact branch discrepancy alongside the successful branch-matched checks.
- Trainer/evaluator now retain and reconstruct actual plan options. Standalone JSON evidence: outputs/qwen38/public-text-lm-wlf.json in the local delivery workspace. This validates iteration3small-model behavior; it does not substitute for final kernel correctness and performance gates.
