# Qwen3.8 bilateral MoE optimization iterations

## Post-iteration-10 quality and capacity validation

The strongest BF16 producer-input/consumer-AIU/activation-backward recipe and a
fresh Native BF16 control each trained the complete 57.8M-parameter four-layer
MoE language model for 2,000 steps on public TinyShakespeare, with unfrozen
routing, using seeds 4827, 4891 and 6113. All six training runs and six full
held-out evaluations completed with clean exclusive-node audits. Full validation
mean cross-entropy: peer 1.590015169244, Native 1.590561401184 (-0.0343421%);
sample standard deviations 0.0034741320 and 0.0062784261. This is a small-model
convergence check, not evidence of a quality improvement or Qwen2.4T convergence.
Evidence: outputs/qwen38/public-text-lm-best-paired.json in the campaign workspace.

The near-int32-boundary EP32 C768 all-to-expert-zero capacity test passed all
elements of Y, DX, DW1, DW2 and routing-score gradients for both DeepEP and peer
on all 32 ranks. It explicitly checks nonzero derivatives at valid zero scores
and exactly zero gradients for unused experts. Maximum relative RMS 0.00349705;
all four host audits clean. Evidence: outputs/qwen38/ep32-c768-skew-capacity.json.
The trainer/evaluator now persist and reconstruct activation-backward fusion.

The latest user request raises the optimization target to MoK-like 2–3x and
prioritizes measured communication/computation fractions and actual overlap.
Iteration10's ~12% time reduction remains an intermediate result. Current
producer/consumer launches have a full kernel boundary; they are not equivalent
to MoK's persistent communication/compute role pipeline. Collect DeepEP,
unfused-peer and fused-peer traces before the next architectural candidate.

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

### Iteration 5 — aligned padding in shared return staging

- Direction: add16BF16elements to the TSM row pitch (BN+16), preserving16-byte vector alignment and changing adjacent fragment-row bank offsets. Remote return remains inside GEMM. Record peer_staged_return_row_padding=16.
- Correctness: BN64/128/256 unit passed115.516s; both full Qwen expert-dimension references passed, worstRMS0.00494, clean gates.
- Paired Qwen EP32T1024 SN1, all32parents/gatespassed: DeepEP38CUs46.9747131ms; padded-staged peer49.2741119ms, samples [49.2392700,49.2922302,49.2727478,49.2808899,49.2741119].
- Result: no meaningful change from unpadded49.2693939ms; still slower than the original unstaged iteration3. Do not recommend staged return. Padding did not explain the loss in this workload.
- Stall audit: iterations3/4/5 produced no >=3% improvement over the best fusion result. Re-profile/review and primary-source search are required before iteration6; there is no physical-floor evidence and viable scheduling/geometry directions remain.
- Actual occupancy inspection (not a performance comparison): BN128/BK64/WM32/WN64/S2 supports3residentblocks/CU, but the cached launcher caps its grid at2*39=78. BN128/BK128/WM64/WN64/S2 supports only1block/CU. This motivates a tested private-cache concurrency option and role-specific FC1 tiles; it does not by itself establish a speedup.

### Stall reassessment before iteration6

- Reviewed iteration2Asight kernel totals and min/median/max distributions: FC1 remains~37.4% (~18.57ms/step), larger than return kernels combined. Iterations3/4/5 did not materially move total time. No timer, bandwidth, or compute-floor evidence supports stopping.
- Actual occupancy queries: FC1 BN128/BK64/WM32/WN64/S2 supports3blocks/CU; the private-cache launcher currently caps at2. Larger grid requires correspondingly larger CTA-private scratch; no cross-CTA synchronization is needed.
- Read primary T-Head memory-bound GEMM study (blog33,2026-09-28) via official API after web page rendering failed: https://developer.t-head.cn/blogDetail?bizType=BLOG&id=33 . It supports checking wave tails and geometry against39CUs, while warning that occupancy alone is not a performance predictor. Its workload and quoted throughput are not our Qwen FC1 workload and are not used as measured bounds here.
- Next hypothesis: expose1/2/4cached CTAs/CU, retain occupancy clamping, and test FC1-specific smaller tiles while leaving the tuned DeepEP denominator and other GEMM roles unchanged. Validate >78active tasks, not only tiny cases that never touch the additional scratch rows.

### Iteration 6 — cache concurrency matched to actual occupancy

- Direction: configurable1/2/4cached CTAs/CU, matched private scratch capacity and occupancy clamping. No inter-CTA dependency or barrier. Add peer-role-specific tile overrides so FC1 geometry changes leave other roles and the tuned DeepEP control unchanged.
- Correctness: cache-group unit on clean .63 passed90.603s; full independent Qwen H8192/I2048/E16/K10/T512 reference on clean .215 passed every element, masked/ragged and duplicate-skew patterns, worstRMS.00409. Balanced case activates96cache tasks, exercising storage beyond the original78-CTA cap. Original .220 attempt was blocked before GPU work by other tasks; original .219 unit had a foreign-monitor record and was rerun cleanly.
- Relocated measurements from SN1 to SN2 because only1-2SN1hosts were idle. Paired5x20graph trials, actual Qwen512routedexperts, EP32T1024/C512, DeepEP38CUs, unstaged bilateral. Allthree phases and allrank/gate receipts passed.
```json
[
  {
    "run": "qwen38-wlf-ep32-iter6-fc1-n128-ctas2-sn2-attempt0",
    "backend": "deepep",
    "ms": 47.06810607910156,
    "samples": [
      47.17891845703125,
      47.05702209472656,
      47.03438110351563,
      47.071044921875,
      47.06810607910156
    ],
    "ctas": null,
    "tiles": {
      "fc1": [
        256,
        64,
        64,
        64,
        2
      ],
      "fc2": [
        256,
        64,
        64,
        64,
        2
      ],
      "dhidden": [
        256,
        64,
        64,
        64,
        2
      ],
      "dx": [
        256,
        64,
        64,
        64,
        2
      ]
    },
    "errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter6-fc1-n128-ctas2-sn2-attempt0",
    "backend": "peer",
    "ms": 61.69306640625,
    "samples": [
      61.670172119140624,
      61.69306640625,
      61.67611083984375,
      61.7105712890625,
      61.7082275390625
    ],
    "ctas": 2,
    "tiles": {
      "fc1": [
        128,
        64,
        32,
        64,
        2
      ],
      "fc2": [
        256,
        64,
        64,
        64,
        2
      ],
      "dhidden": [
        256,
        64,
        64,
        64,
        2
      ],
      "dx": [
        256,
        64,
        64,
        64,
        2
      ]
    },
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.771342337837268e-07,
      "dw2": 2.786914308217092e-07,
      "dscores": 1.4615960708397324e-07
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter6-fc1-n128-ctas4-sn2-attempt0",
    "backend": "deepep",
    "ms": 47.02039489746094,
    "samples": [
      47.17173767089844,
      47.02039489746094,
      47.00090637207031,
      46.975991821289064,
      47.039422607421876
    ],
    "ctas": null,
    "tiles": {
      "fc1": [
        256,
        64,
        64,
        64,
        2
      ],
      "fc2": [
        256,
        64,
        64,
        64,
        2
      ],
      "dhidden": [
        256,
        64,
        64,
        64,
        2
      ],
      "dx": [
        256,
        64,
        64,
        64,
        2
      ]
    },
    "errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter6-fc1-n128-ctas4-sn2-attempt0",
    "backend": "peer",
    "ms": 49.840554809570314,
    "samples": [
      49.740902709960935,
      49.820370483398435,
      49.85562438964844,
      49.840554809570314,
      49.86471862792969
    ],
    "ctas": 4,
    "tiles": {
      "fc1": [
        128,
        64,
        32,
        64,
        2
      ],
      "fc2": [
        256,
        64,
        64,
        64,
        2
      ],
      "dhidden": [
        256,
        64,
        64,
        64,
        2
      ],
      "dx": [
        256,
        64,
        64,
        64,
        2
      ]
    },
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.642166521127365e-07,
      "dw2": 2.7665305424307007e-07,
      "dscores": 1.4615960708397324e-07
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter6-n256-ctas2-sn2-attempt0",
    "backend": "deepep",
    "ms": 47.02198791503906,
    "samples": [
      47.065286254882814,
      47.02198791503906,
      47.0070068359375,
      47.03162536621094,
      47.00060119628906
    ],
    "ctas": null,
    "tiles": {
      "fc1": [
        256,
        64,
        64,
        64,
        2
      ],
      "fc2": [
        256,
        64,
        64,
        64,
        2
      ],
      "dhidden": [
        256,
        64,
        64,
        64,
        2
      ],
      "dx": [
        256,
        64,
        64,
        64,
        2
      ]
    },
    "errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter6-n256-ctas2-sn2-attempt0",
    "backend": "peer",
    "ms": 50.12757568359375,
    "samples": [
      50.11595153808594,
      50.12693481445312,
      50.12757568359375,
      50.15898132324219,
      50.133447265625
    ],
    "ctas": 2,
    "tiles": {
      "fc1": [
        256,
        64,
        64,
        64,
        2
      ],
      "fc2": [
        256,
        64,
        64,
        64,
        2
      ],
      "dhidden": [
        256,
        64,
        64,
        64,
        2
      ],
      "dx": [
        256,
        64,
        64,
        64,
        2
      ]
    },
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.631226720950508e-07,
      "dw2": 2.564627834544808e-07,
      "dscores": 1.4615960708397324e-07
    }
  }
]
```
- Best peer 49.840554810ms, paired DeepEP 47.020394897ms; ratio DeepEP/peer 0.943416, time reduction -5.9977%. Versus same-domain N256control, time reduction 0.5726%.
- Cross-supernode DeepEP EP16 passed independent CPU reference and128dynamic-route graph replays, allparents/gatesnormal. CPU Gloo carries initialization metadata; DeepEP transports token/gradient payloads, with cross-host MNNVL disabled. OriginalEP64 NCCL-metadata attempt blocked in NativeMoEPlan recipe all_gather_object / pcclAllGather(socketaccept), before DeepEP initialization. Its own verified rank processes were terminated after timeout; allparents closed. No DeepEP64success claim yet.

## Iteration7 — producer GEMM slice followed by a local remainder

- Motivation: private N caching still limits FC1 parallelism. Compute the first1/2/4 N tiles while dispatching X inside GEMM, materialize packed X, then compute remaining N tiles locally. Kernel stream ordering publishes X; no cross-CTA residency barrier. Return writes remain fused. Both FC1 kernels and mandatory input writes are timed.
- Same Qwen H8192/I2048/E16/K10,T1024,C512; BF16,FP32 DW, full forward/backward replay,256/64/64/64/S2,5x20 graph samples. Real installed DeepEP Buffer38CU paired in every case, Torch2.11 wlf_sgl_2.2. The routed layer excludes the shared expert and optimizer.
- Producer unit covers1/2/4 tiles and the whole-FC1 case, masks, duplicates, ragged chunks and all gradients; passed. Qwen512-token full independent FP32 reference passed every Y/DX/DW1/DW2/dScore element, worst RMS0.004083. EP32 dynamic-route stress128replays passed all32ranks with clean ownership gates.
- All3 performance phases finished all32parents0 and every ownership gate clean. Candidate first case usedSN2.225/.226/.227/.222; others used.225/.226/.227/.223. Comparisons use the paired denominator.
- Native extension SHA2563b283ac13dc3a69e4b9edd6072a0f96dcb5eeab82299c6b14e7f67b57c8cfea2.

```json
[
  {
    "run": "qwen38-wlf-ep32-iter7-producer1-attempt0",
    "nodes": [
      "172.17.0.225",
      "172.17.0.226",
      "172.17.0.227",
      "172.17.0.222"
    ],
    "producer_tiles": 1,
    "deepep_ms": 47.0342529296875,
    "peer_ms": 44.82920532226562,
    "peer_samples": [
      44.82920532226562,
      44.83145446777344,
      44.84373779296875,
      44.8272216796875,
      44.82712097167969
    ],
    "time_reduction_percent": 4.6881739797552395,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.6730231184046715e-07,
      "dw2": 2.5834683015091287e-07,
      "dscores": 1.4615960708397324e-07
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter7-producer2-attempt0",
    "nodes": [
      "172.17.0.225",
      "172.17.0.226",
      "172.17.0.227",
      "172.17.0.223"
    ],
    "producer_tiles": 2,
    "deepep_ms": 47.000274658203125,
    "peer_ms": 51.00791931152344,
    "peer_samples": [
      50.952264404296876,
      51.00296325683594,
      51.02781372070312,
      51.00791931152344,
      51.01741333007813
    ],
    "time_reduction_percent": -8.52685368854722,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.6598377189657185e-07,
      "dw2": 2.5620744281695806e-07,
      "dscores": 1.4615960708397324e-07
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter7-producer4-attempt0",
    "nodes": [
      "172.17.0.225",
      "172.17.0.226",
      "172.17.0.227",
      "172.17.0.223"
    ],
    "producer_tiles": 4,
    "deepep_ms": 47.056539916992186,
    "peer_ms": 52.83516845703125,
    "peer_samples": [
      52.795050048828124,
      52.864324951171874,
      52.83516845703125,
      52.85762329101563,
      52.80686645507812
    ],
    "time_reduction_percent": -12.28018156505466,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.658341600181302e-07,
      "dw2": 2.601505570964946e-07,
      "dscores": 1.4615960708397324e-07
    }
  }
]
```
- Best paired reduction4.6882%, peer44.829205322ms vsDeepEP47.034252930ms, producer_tiles=1. This is a two-GEMM producer/consumer FC1 path, not the original single peer-read GEMM.
- Separate baseline integration: new every-element distributed local-expert FP32 oracle passed2ranks onQwen H8192/I2048,E8,K10 with masks/zero-score/empty-source/duplicates and changing-route graph replay. Initial oracle attempt failed a deliberate one-stream plan check; fixed the test to keep preparation/execution/capture on one stream, then reran clean. Cross-domain64 smallshape graph128 passed with8RDMA ranks; actualQwen64 oracle still pending.
- Isolated ElasticBuffer probe now passes forward,DX,dScore and4dynamic-route graph replays in nonexpanded mode after native header/thread-scope compatibility fixes. Expanded cached backward and expanded score combine are explicitly unsupported in this installed version. An initial nonexpanded run passed math but failed teardown JIT include order; isolated comm header order fixed and clean rerun passed. Installed package untouched; no Elastic timing claims yet.

## Iteration8 — contiguous local FC1 AIU loading

- Addedopt-inproducer_local_aiu. The consumer GEMM loads mandatory producer-packed X directly with the native AIU atom; only local N remainder kernels can use it. Padded rows were already explicitly written by the producer. Original gather path remains default.
- Asys EP16proxy (H8192/I2048/E16,K10,256totalexperts,not512) showed localconsumer19.2%,producer11.1%,wgrad23.2%; thisdiagnosticmotivated the change and is not a QwenEP32performance result.
- Six producer unit cases (1/2/4tiles,wholeFC1,AIU1/4) passed110.237s,clean63. QwenH8192/I2048/E16/K10,T512 independent FP32 every-element reference passed masks and duplicate-skew; adding existing fusedactivationbackward also passed. WorstRMS0.004083,precisionboundaries unchanged.
- NativeextensionSHA2566cf1dd828f0eca0491ea5749aca88bb8a791e69ec452d4919435b98aa08f095f.
- EP8single-host diagnostic matrix was attempted while EP32/domain4 and cross64/domain8 idle windows were unavailable. It exited75 before timing. No performance result or winner claim. Raw controller receipts:

```json
{
  "qwen38-wlf-ep8-proxy-iter8-control": "{\"case\": \"distributed_campaign_preflight\", \"name\": \"qwen38-wlf-ep8-proxy-iter8-control\", \"hosts\": [{\"node\": \"172.17.0.87\", \"returncode\": 75, \"output\": \"{\\\"case\\\": \\\"idle_node_gate\\\", \\\"host\\\": \\\"e01-cn-9yq4xwro401-a0004\\\", \\\"selected_gpus\\\": [0, 1, 2, 3, 4, 5, 6, 7], \\\"allowed\\\": false, \\\"processes\\\": [{\\\"gpu\\\": 0, \\\"pid\\\": 1539796, \\\"owned\\\": false}, {\\\"gpu\\\": 1, \\\"pid\\\": 1539797, \\\"owned\\\": false}, {\\\"gpu\\\": 2, \\\"pid\\\": 1539798, \\\"owned\\\": false}, {\\\"gpu\\\": 3, \\\"pid\\\": 1539799, \\\"owned\\\": false}, {\\\"gpu\\\": 4, \\\"pid\\\": 1539800, \\\"owned\\\": false}, {\\\"gpu\\\": 5, \\\"pid\\\": 1539801, \\\"owned\\\": false}, {\\\"gpu\\\": 6, \\\"pid\\\": 1539802, \\\"owned\\\": false}, {\\\"gpu\\\": 7, \\\"pid\\\": 1539803, \\\"owned\\\": false}]}\\n\", \"error\": \"\"}]}\n"
}
```
- Separate DeepEP cross64Qwen test reached Bufferdispatch but rejected38CUs with default12RDMAQPs (internode_config_check requires2*QPs>=CUs). Allrankparents exited; no numerical success claim. Added explicit/automaticQP selection max(12,ceil(CUs/2)),38CUuses19QPs; true cross64retry pending. This changes RDMA queue setup, not the expert arithmetic or timing work.
- Producer1three-seed public-text LM ran2000steps each with clean training/evaluation gates. Fullheldoutmeans:producer1.592712080623,priorNative1.585762516074 (+0.438%). Record exact difference; three seeds do not prove statistical equivalence or Qwen2.4T convergence. Fullrecords in outputs/qwen38/public-text-lm-producer-wlf.json.

### Iteration8 diagnostic retry — EP8 same-host controls

- Same physical node226, all8rankparents and every ownership gate clean in all3phases. H8192/I2048/E16/K10,T1024,C512,256/64/64/64/S2,5x20graphs; producer1+packedDYreuse.
- **128totalexperts on8ranks**. This is a local-load optimization signal, not Qwen512expertEP32/64 validation. No DeepEP speedup denominator is measured here. Finaltargetcomparison remains pending.
- Fullwork contract unchanged, no optimizer. Independent all-element Qwenexpert-dimension correctness had passed before this signal.

```json
[
  {
    "run": "qwen38-wlf-ep8-proxy-iter8-control-r1-attempt0",
    "aiu": false,
    "activation": false,
    "ms": 44.648681640625,
    "samples": [
      44.667868041992186,
      44.65000915527344,
      44.6106689453125,
      44.636297607421874,
      44.648681640625
    ],
    "reduction_vs_control_percent": 0.0
  },
  {
    "run": "qwen38-wlf-ep8-proxy-iter8-local-aiu-r1",
    "aiu": true,
    "activation": false,
    "ms": 44.44956665039062,
    "samples": [
      44.48501281738281,
      44.42349853515625,
      44.4324951171875,
      44.44956665039062,
      44.46959228515625
    ],
    "reduction_vs_control_percent": 0.44595939436027354
  },
  {
    "run": "qwen38-wlf-ep8-proxy-iter8-local-aiu-activation-r1",
    "aiu": true,
    "activation": true,
    "ms": 44.25270080566406,
    "samples": [
      44.20602111816406,
      44.25287780761719,
      44.26948852539063,
      44.19879150390625,
      44.25270080566406
    ],
    "reduction_vs_control_percent": 0.8868813600100611
  }
]
```
- Best signal44.252700806ms; relative reduction0.8869%. Keep the controls explicit; do not extrapolate this delta to cross-hostEP32.

### Iteration8 actualQwen EP32 validation

- All3cases use exactly the same SN2.225/.226/.227/.223, with paired actual installed DeepEP Buffer38CU. Every32rankparent0, everyownershipgateclean. Actual512routedexperts,H8192/I2048/E16/K10,T1024,C512,BF16,FP32DW,256/64/64/64/S2,5x20graphs, sharedexpert/optimizerexcluded.
- Only flags shown below differ; all producer1+packedDYreuse. NativeSDK2.10-built extension in wlfTorch2.11, hash6cf1dd... preserved. DeepEPauto19QP setup verified; within-domain RDMA rank count1.

```json
[
  {
    "run": "qwen38-wlf-ep32-iter8-control-attempt0",
    "aiu": false,
    "activation": false,
    "peer_ms": 44.829550170898436,
    "deepep_ms": 47.029782104492185,
    "samples": [
      44.79281921386719,
      44.845150756835935,
      44.819766235351565,
      44.829550170898436,
      44.84882507324219
    ],
    "reduction_vs_deepep_percent": 4.678380028861728,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.631126676533313e-07,
      "dw2": 2.7146441539116495e-07,
      "dscores": 1.4615960708397324e-07
    },
    "rdma_ranks": 1,
    "rdma_qps": 19,
    "reduction_vs_peer_control_percent": 0.0
  },
  {
    "run": "qwen38-wlf-ep32-iter8-local-aiu",
    "aiu": true,
    "activation": false,
    "peer_ms": 44.690188598632815,
    "deepep_ms": 47.073370361328124,
    "samples": [
      44.59389953613281,
      44.70243530273437,
      44.7089111328125,
      44.67789001464844,
      44.690188598632815
    ],
    "reduction_vs_deepep_percent": 5.062696264156063,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.5237204681616277e-07,
      "dw2": 2.659586471054354e-07,
      "dscores": 1.4615960708397324e-07
    },
    "rdma_ranks": 1,
    "rdma_qps": 19,
    "reduction_vs_peer_control_percent": 0.31086988768423973
  },
  {
    "run": "qwen38-wlf-ep32-iter8-local-aiu-activation",
    "aiu": true,
    "activation": true,
    "peer_ms": 44.45949096679688,
    "deepep_ms": 47.10954284667969,
    "samples": [
      44.414852905273435,
      44.493002319335936,
      44.45949096679688,
      44.4608642578125,
      44.457540893554686
    ],
    "reduction_vs_deepep_percent": 5.625297380846028,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.6259294827468693e-07,
      "dw2": 2.660789846231637e-07,
      "dscores": 1.4615960708397324e-07
    },
    "rdma_ranks": 1,
    "rdma_qps": 19,
    "reduction_vs_peer_control_percent": 0.8254805205290316
  }
]
```
- BestactualEP3244.459490967ms vs pairedDeepEP47.109542847ms (5.6253%less time); only0.8255% versus producercontrol. No>=3%material new win. Preserve options for final combinations and pursue producerinputstaging.
- Strongest combined option passed EP2 independent math reference and128dynamic-route graph replays, allparents/gateclean220. Both AIU configurations already passed every-element Qwen512-token reference.
- ActualQwen cross-domainEP64 **PASSED** after38CU/19QP correction: phaseqwen38-wlf-cross-ep64-large-reference-r3-attempt0, all64ranks*3patterns=192fullreference records and all8hostgatesclean. H8192,I2048,E8/rank=512global,K10,T19,C256, graphreplay with masks,emptyrank0,zero-score,duplicates,reverseexpertIDs. GlobalmaxRMSY.004171,DX.004491,DW1.003385,DW2.003414,dScore.006988. Token/gradientpayloadSailSHMEM8RDMAranks; CPU Gloo onlymetadata/reference sums. Artifactoutputs/qwen38/cross-ep64-reference.json. This is routed-layer correctness, not fullQwen2.4Ttraining or performance.

### Cross-domainEP64 communication-CU controls — first attempt failed

- ActualQwenH8192/I2048/E8,K10,T1024,C256,64ranks,CPUmetadataGloo andrealSailSHMEMpayload, MNNVLdisabled. First8CU phase exited1, so the remaining16/24/32/38CU cases were not launched. No performance result or comparison. Source is frozen during the attempt; currently includes uncommitted iteration9peer-only optional features, unused by this DeepEP baseline. _Csha256be4414a11d87e88fab554ff03e151d4a1b978ad4c0eb166f25d869c75f229c8b.
- Preserve all64rank error tails inwork/qwen38-cross-ep64-comm8-failure.json. Rank0 andrank63 excerpts before debugging:

```json
[
  {
    "rank": 0,
    "tail": [
      "device_props.name:ZW-M890P",
      "{\"case\": \"bounded_training_benchmark\", \"backend\": \"deepep\", \"compute\": \"bf16\", \"host\": \"e01-cn-2n550vu1t02-a0005\", \"device\": \"ZW-M890P\", \"torch\": \"2.11.0\", \"process_group_backend\": \"gloo\", \"ep_size\": 64, \"topology\": \"qwen38-wlf-cross-supernode-ep64-real-rdma\", \"graph\": true, \"instrumented\": false, \"tokens_per_rank\": 1024, \"hidden\": 8192, \"intermediate\": 2048, \"local_experts\": 8, \"topk\": 10, \"chunk_tokens\": 256, \"routing\": \"balanced\", \"gemm_configs\": {\"fc1\": [256, 64, 64, 64, 2], \"fc2\": [256, 64, 64, 64, 2], \"dhidden\": [256, 64, 64, 64, 2], \"dx\": [256, 64, 64, 64, 2]}, \"collective_m_schedule\": null, \"collective_active_pointwise\": null, \"collective_vectorized_routes\": null, \"collective_fuse_return\": null, \"collective_fuse_activation_backward\": null, \"collective_gemm_config\": null, \"fused_activation_quantization\": null, \"peer_fuse_dispatch\": null, \"peer_fuse_return\": null, \"peer_active_rows\": null, \"peer_fuse_local_grad\": null, \"peer_fuse_activation_backward\": null, \"peer_vectorized_copy\": null, \"peer_dispatch_cache_n\": null, \"peer_dispatch_cache_ctas\": null, \"peer_dispatch_producer\": null, \"peer_producer_local_aiu\": null, \"peer_producer_input_aiu\": null, \"peer_dispatch_producer_tiles\": null, \"peer_dispatch_cache_aiu\": null, \"peer_staged_return\": null, \"peer_staged_return_row_padding\": null, \"peer_reuse_packed_dy\": null, \"deepep_version\": \"2.2.0\", \"deepep_backend\": \"sailshmem\", \"deepep_rdma_ranks\": 8, \"deepep_comm_sms\": 8, \"deepep_rdma_qps\": 12, \"deepep_mnnvl_disabled\": \"1\", \"deepep_scaleup\": false, \"persistent_ctas\": null, \"persistent_resident_blocks_per_cu\": null, \"fast_weight_transpose\": true, \"warmup\": 5, \"iterations\": 20, \"repeats\": 5, \"median_max_rank_event_ms\": 125.3934326171875, \"median_max_rank_wall_ms\": 125.3963922499679, \"samples_max_rank_event_ms\": [126.74967041015626, 125.39716796875, 125.3934326171875, 125.3017822265625, 125.2912353515625], \"samples_max_rank_wall_ms\": [126.75183629617095, 125.40031920652837, 125.3963922499679, 125.30451285419986, 125.29492985922843], \"workspace_bytes\": 11214690408, \"activation_workspace_bytes\": 10409384040, \"max_rank_torch_allocated_bytes\": 16428980736, \"rank0_initial_torch_allocated_bytes\": 2449596928, \"sample_relative_rms_vs_first_backend\": {\"y\": 0.0, \"dx\": 0.0, \"dw1\": 0.0, \"dw2\": 0.0, \"dscores\": 0.0}, \"scope\": \"forward + full backward replay + FP32 wgrad zero/accumulation; no optimizer\", \"memory_scope\": \"logical workspace and PyTorch allocator; excludes driver/VMM reservations\"}"
    ]
  },
  {
    "rank": 63,
    "tail": [
      "device_props.name:ZW-M890P"
    ]
  }
]
```

- Audit correction for the preceding incomplete communication sweep: all64GPU rank parents actually returned0 and rank0 produced a125.393432617ms timing. The phase failed its ownership audit on213, which observed8PIDs2083691–2083698 distinct from the verified own rank PIDs2083363/65/67/71/72/73/74/75. Exclude the timing because of interference; this was not a mathematical/kernel failure. Other7hostgates were clean. Nextattempt excludes213and the concurrently validating220.

## Iteration9 — bulk producer packing followed by same-CTA AIU

- The first1Ntile producer now optionally packs peer X once in a coalesced uint4 pass, writes every padded row, executes a device fence and CTA barrier, then loads its own packed tile with AIU for real FC1 computation. No additional scratch, no cross-CTA wait, no standalone dispatch kernel. Later consumer remains stream ordered. Requires exactly one producer N tile. DefaultFalse preserves prior paths.
- Unit covers BN128/256, local gather/AIU consumers, duplicates,masks,zero-score,raggedchunks,allgradients and whole-FC1-with-no-consumer. Passed173.682s. Both QwenH8192/I2048/E16/K10,T512 all-element FP32 references passed, with/without activation-backward fusion. EP2 independentreference+128dynamicroutegraphs passed. All4validation stages and the220 ownershipgate clean.
- NativeextensionSHA256be4414a11d87e88fab554ff03e151d4a1b978ad4c0eb166f25d869c75f229c8b.
- Attempted EP8 diagnostic timing on220 exited75 at preflight before GPU launch. No iteration9timing or performance winner yet. Receipt:

```text
{"case": "distributed_campaign_preflight", "name": "qwen38-wlf-ep8-proxy-iter9-control", "hosts": [{"node": "172.17.0.220", "returncode": 0, "output": "{\"case\": \"idle_node_gate\", \"host\": \"e01-cn-2n550vu1t01-a0008\", \"selected_gpus\": [0, 1, 2, 3, 4, 5, 6, 7], \"allowed\": true, \"processes\": []}\n{\"case\": \"idle_node_postflight\", \"host\": \"e01-cn-2n550vu1t01-a0008\", \"allowed\": true, \"processes\": [], \"observed_foreign_during_command\": [], \"verified_owned_processes\": [], \"monitoring_errors\": [], \"monitor_interval_seconds\": 2, \"command_returncode\": 0}\n", "error": ""}]}
{"case": "distributed_master_port_unavailable", "node": "172.17.0.220", "port": 30621, "error": "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nOSError: [Errno 98] Address already in use\n"}

```

### Cross-domainEP64 communication-CU sweep — clean retry

- Actual512experts,H8192/I2048/E8,K10,T1024,C256,64GPUs acrossSN2fourhosts+SN0threehosts+SN1onehost. Same8hosts acrossall5cases:225/226/227/223/63/87/80/219. All64rankparents0andall8ownershipgatesclean in every case.213wasexcludedafterthepriorinterference.
- Real installedDeepEPBuffer2.2,SailSHMEM8RDMA-ranks,CPU/Gloometadata, MNNVLdisabled. This is a host-group/RDMA fallback control, not an optimally sized supernode hierarchy. NativeBF16NoPadGEMMs,FP32DW,fullbackwardreplay,256/64/64/64/S2,5x20graphs,nooptimizer/sharedexpert. _Csha256be4414a11d87e88fab554ff03e151d4a1b978ad4c0eb166f25d869c75f229c8b.

```json
[
  {
    "run": "qwen38-wlf-cross-ep64-comm8-r1-attempt0",
    "communication_cus": 8,
    "rdma_qps": 12,
    "rdma_ranks": 8,
    "mnnvl_disabled": "1",
    "ms": 125.01475830078125,
    "samples": [
      125.42451171875,
      124.8778564453125,
      125.02181396484374,
      124.7277099609375,
      125.01475830078125
    ]
  },
  {
    "run": "qwen38-wlf-cross-ep64-comm16-r1",
    "communication_cus": 16,
    "rdma_qps": 12,
    "rdma_ranks": 8,
    "mnnvl_disabled": "1",
    "ms": 107.7940185546875,
    "samples": [
      107.84200439453124,
      107.7910400390625,
      107.509521484375,
      107.84984130859375,
      107.7940185546875
    ]
  },
  {
    "run": "qwen38-wlf-cross-ep64-comm24-r1",
    "communication_cus": 24,
    "rdma_qps": 12,
    "rdma_ranks": 8,
    "mnnvl_disabled": "1",
    "ms": 95.46548461914062,
    "samples": [
      95.43392333984374,
      95.46548461914062,
      95.44957885742187,
      95.52686767578125,
      95.52781982421875
    ]
  },
  {
    "run": "qwen38-wlf-cross-ep64-comm32-r1",
    "communication_cus": 32,
    "rdma_qps": 16,
    "rdma_ranks": 8,
    "mnnvl_disabled": "1",
    "ms": 90.10323486328124,
    "samples": [
      90.12725830078125,
      90.27642822265625,
      90.10323486328124,
      90.07439575195312,
      90.044580078125
    ]
  },
  {
    "run": "qwen38-wlf-cross-ep64-comm38-r1",
    "communication_cus": 38,
    "rdma_qps": 19,
    "rdma_ranks": 8,
    "mnnvl_disabled": "1",
    "ms": 87.66292724609374,
    "samples": [
      87.60396728515624,
      87.6944091796875,
      87.66292724609374,
      87.63072509765625,
      87.76297607421876
    ]
  }
]
```
- Bestobserved38CU/19QP: 87.662927246ms. Use this instead of the initial8CUcontrol for subsequent same-topology large-T comparisons. DomainEP32fusion cannot be directly extended across supernodes via peer pointers. Cross64largeshapecorrectness has already passed.

### Iteration9 actualQwen EP32 controls

- SameSN2fourhosts225/226/227/223 acrossall3cases,all32rankparents0+allgatesclean. Actual512routedexpertsH8192/I2048/E16/K10,T1024,C512,256/64/64/64/S2,5x20graphs. Producer1+packedDYreuse, activationbackwardfused. FullFWD+BWDrematerialization+FP32DW,nooptimizer/sharedexpert.
- Paired installedDeepEPBuffer38CU,19QP,within-domainRDMA-ranks1,matchingGEMMrecipe. Source8317ff1a69,extensionbe4414...; no kernel/source changes during these phases.
- Both new input staging variants and whole-FC1 case had passed independent fullgrad references and128dynamicroutegraphs before timing.

```json
[
  {
    "run": "qwen38-wlf-ep32-iter9-control-attempt0",
    "producer_input_aiu": false,
    "producer_local_aiu": true,
    "activation": true,
    "peer_ms": 44.44829406738281,
    "deepep_ms": 47.06523132324219,
    "samples": [
      44.43448181152344,
      44.487799072265624,
      44.468630981445315,
      44.44829406738281,
      44.44420166015625
    ],
    "reduction_vs_deepep_percent": 5.560234555921662,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.727231844801281e-07,
      "dw2": 2.489265682470432e-07,
      "dscores": 1.4615960708397324e-07
    },
    "reduction_vs_peer_control_percent": 0.0
  },
  {
    "run": "qwen38-wlf-ep32-iter9-input-aiu",
    "producer_input_aiu": true,
    "producer_local_aiu": true,
    "activation": true,
    "peer_ms": 43.8803466796875,
    "deepep_ms": 47.10782470703125,
    "samples": [
      43.85205078125,
      43.894473266601565,
      43.895574951171874,
      43.8803466796875,
      43.86514587402344
    ],
    "reduction_vs_deepep_percent": 6.8512567655496515,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.6842042188945925e-07,
      "dw2": 2.735590953761857e-07,
      "dscores": 1.4615960708397324e-07
    },
    "reduction_vs_peer_control_percent": 1.2777709462467035
  },
  {
    "run": "qwen38-wlf-ep32-iter9-input-aiu-gather-consumer",
    "producer_input_aiu": true,
    "producer_local_aiu": false,
    "activation": true,
    "peer_ms": 44.086865234375,
    "deepep_ms": 47.04383239746094,
    "samples": [
      44.05244445800781,
      44.08622741699219,
      44.086865234375,
      44.09914855957031,
      44.09129638671875
    ],
    "reduction_vs_deepep_percent": 6.285557558540944,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 2.6045481149594707e-07,
      "dw2": 2.680600061921723e-07,
      "dscores": 1.4615960708397324e-07
    },
    "reduction_vs_peer_control_percent": 0.8131444425288681
  }
]
```
- Best43.880346680ms vspairedDeepEP47.107824707ms (6.8513%less time). Improvementoverpeercontrol1.2778%. Below3%materialthreshold; retain explicit controls and continue a distinct direction.

## Iteration10 — larger token counts and aligned chunk capacity

- All5cases use the same SN2fourhosts225/226/227/223,all32rankparents0andeveryownershipgateclean. Actual512routedexpertsH8192/I2048/E16/K10,producer1+inputAIU+localAIU+activationbackward+DYreuse,256/64/64/64/S2,5x20graphs. PairedDeepEPBuffer38CU19QPfabricscaleup32.
- Chunk768 remains inside the existing int32 guard: alignment128 and247808allocatedrows atEP32,rows*8192=2030043136elements<=INT32_MAX. Initial512/256power-of-twochoices were not absolute limits; HINTS corrected. No token/expert drops. FullFWD+BWDreplay+FP32DW, nooptimizer/sharedexpert.
- SingleGPUH8192/I2048/E16/K10,T768all-elementFP32masked/duplicate referencepassed, maxRMS0.004081. New independently summed-duplicate-weight skew oracle passed2ranksmall-shape checks for bothDeepEPandpeer; actualEP32near-limitoffset stress remains pending.

```json
[
  {
    "run": "qwen38-wlf-ep32-iter10-t4096-c512-attempt0",
    "tokens": 4096,
    "chunk": 512,
    "peer_ms": 164.041357421875,
    "deepep_ms": 176.67615966796876,
    "reduction_vs_paired_deepep_percent": 7.151390583675021,
    "peer_samples": [
      163.94080810546876,
      164.0297607421875,
      164.041357421875,
      164.063818359375,
      164.04788818359376
    ],
    "peer_peak_torch_gib": 18.566678047180176,
    "deepep_peak_torch_gib": 18.53543519973755,
    "errors": {
      "y": 0.0026847815606743097,
      "dx": 0.0014849959407001734,
      "dw1": 2.5937464442904457e-07,
      "dw2": 2.5325016395072453e-07,
      "dscores": 1.460545888676279e-07
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter10-t4096-c768",
    "tokens": 4096,
    "chunk": 768,
    "peer_ms": 144.710009765625,
    "deepep_ms": 164.65115966796876,
    "reduction_vs_paired_deepep_percent": 12.111150594114594,
    "peer_samples": [
      144.5995849609375,
      144.6011474609375,
      144.710009765625,
      144.78558349609375,
      144.71041259765624
    ],
    "peer_peak_torch_gib": 24.713059425354004,
    "deepep_peak_torch_gib": 24.681816577911377,
    "errors": {
      "y": 0.0026847815606743097,
      "dx": 0.0014849959407001734,
      "dw1": 3.5115400009999576e-07,
      "dw2": 3.4753790600916545e-07,
      "dscores": 1.460545888676279e-07
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter10-t8192-c512",
    "tokens": 8192,
    "chunk": 512,
    "peer_ms": 323.7109130859375,
    "deepep_ms": 349.591796875,
    "reduction_vs_paired_deepep_percent": 7.403172505880196,
    "peer_samples": [
      323.5693115234375,
      323.6948974609375,
      323.719775390625,
      323.735302734375,
      323.7109130859375
    ],
    "peer_peak_torch_gib": 18.69213581085205,
    "deepep_peak_torch_gib": 18.660892963409424,
    "errors": {
      "y": 0.0026137905661016703,
      "dx": 0.0014898645458742976,
      "dw1": 2.780894874376827e-07,
      "dw2": 2.75692570994579e-07,
      "dscores": 1.482959675058737e-07
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter10-t8192-c768",
    "tokens": 8192,
    "chunk": 768,
    "peer_ms": 273.1869873046875,
    "deepep_ms": 310.4890380859375,
    "reduction_vs_paired_deepep_percent": 12.013967066665177,
    "peer_samples": [
      273.1869873046875,
      273.0773681640625,
      273.117724609375,
      273.3026123046875,
      273.2547607421875
    ],
    "peer_peak_torch_gib": 24.83851718902588,
    "deepep_peak_torch_gib": 24.807274341583252,
    "errors": {
      "y": 0.0026137905661016703,
      "dx": 0.0014898645458742976,
      "dw1": 3.627561397934187e-07,
      "dw2": 3.646568700332864e-07,
      "dscores": 1.482959675058737e-07
    }
  },
  {
    "run": "qwen38-wlf-ep32-iter10-t1024-c768",
    "tokens": 1024,
    "chunk": 768,
    "peer_ms": 44.298519897460935,
    "deepep_ms": 50.536065673828126,
    "reduction_vs_paired_deepep_percent": 12.342760943492925,
    "peer_samples": [
      44.29283447265625,
      44.298519897460935,
      44.29408569335938,
      44.30423583984375,
      44.30420837402344
    ],
    "peer_peak_torch_gib": 24.618966102600098,
    "deepep_peak_torch_gib": 24.58772325515747,
    "errors": {
      "y": 0.002692387904971838,
      "dx": 0.0014900852693244815,
      "dw1": 3.2887786005630915e-07,
      "dw2": 3.228395542009821e-07,
      "dscores": 1.4615960708397324e-07
    }
  }
]
```

BestbaselineusesitsownbestmeasuredchunkforeachT;donotcompareonlyagainsttheweaker512setting: Include iteration9C512controls forT1024.

```json
[
  {
    "tokens": 1024,
    "best_peer_ms": 43.8803466796875,
    "peer_chunk": 512,
    "best_deepep_ms": 47.04383239746094,
    "deepep_chunk": 512,
    "reduction_vs_best_deepep_percent": 6.724549333153773
  },
  {
    "tokens": 4096,
    "best_peer_ms": 144.710009765625,
    "peer_chunk": 768,
    "best_deepep_ms": 164.65115966796876,
    "deepep_chunk": 768,
    "reduction_vs_best_deepep_percent": 12.111150594114594
  },
  {
    "tokens": 8192,
    "best_peer_ms": 273.1869873046875,
    "peer_chunk": 768,
    "best_deepep_ms": 310.4890380859375,
    "deepep_chunk": 768,
    "reduction_vs_best_deepep_percent": 12.013967066665177
  }
]
```
- T1024C768 is slower in absolute time for both implementations than the earlierC512controls. Its paired12.34% ratio is not a gain over the best DeepEP setting and must not be promoted as that. For4096/8192bothbestmeasuredsettingsareC768andthe~12%fusionreductionholdsagainstthebetterbaseline.


## Iteration11 — resident communication/compute tile roles (negative proxy)

Implemented opt-in dispatch_pipeline and return_pipeline. Communication CTAs
publish/acquire per-tile flags; compute CTAs use contiguous local AIU operands.
Return CTAs vector-copy each completed M/N output tile while other CTAs compute.
The existing PeerGemmParams ABI remains unchanged through a separate control
argument. Timeout/status and occupancy checks protect forward progress. Raw
per-CTA total/wait cycles are diagnostic, not device-wide overlap percentages.

Chip ZW-M890P/ppu0015, SDK2.2, nativeTorch2.10 extension build exercised in
wlf_sgl_2.2/Torch2.11, extension SHA256
ec982a2b7ffcde2c5ac47c0d1e24f4cb10176c70ec0fec9f9c88588f1a78e018.
H8192/I2048/E16 local/K10/T4096/C768; BF16/full-backward-replay/FP32DW/no
optimizer/shared expert, graph5x20 median-max-rank. EP8 on219 is only a tuning
proxy (128 total experts), not an actual Qwen512 EP32 performance result.

Independent full-element H8192/I2048/E16/T768 masked/duplicate-skew references
passed on80, maximum RMS0.00408018, clean gate. EP2 independent gradients and
128 dynamic-route graph replays passed on219, both ranks0 and clean gate.
Single-GPU20-test run had two CLI recipe conflicts in explicit legacy-mode
controls; fixed those controls to select compatible pipeline switches. Both
failed controls reran successfully (2 tests, clean220). Remaining applicable
tests passed; three unrelated tests were skipped. Initial unit invocation
rejected --reuse-packed-dy without --ep; rerun used its documented unit mode.
Initial large-reference attempt on215 was blocked by two foreign PIDs; moved80.

Measured completed control matrix (all8 parents0/clean audit for each):
[
  {
    "phase": "qwen38-wlf-ep8-iter11-serial-control-attempt0",
    "backend": "deepep",
    "ms": 134.80830078125
  },
  {
    "phase": "qwen38-wlf-ep8-iter11-serial-control-attempt0",
    "backend": "peer",
    "ms": 136.170263671875
  },
  {
    "phase": "qwen38-wlf-ep8-iter11-producer-control",
    "backend": "peer",
    "ms": 137.8072265625
  },
  {
    "phase": "qwen38-wlf-ep8-iter11-input4",
    "backend": "peer",
    "ms": 280.5714599609375
  },
  {
    "phase": "qwen38-wlf-ep8-iter11-return4",
    "backend": "peer",
    "ms": 242.2955078125
  },
  {
    "phase": "qwen38-wlf-ep8-iter11-both2",
    "backend": "peer",
    "ms": 702.7490234375
  }
]

Verdict: reject this candidate as a speed winner. Input4 compute-CTA wait/total
cycles were0.518–0.699 (median~0.682) in the last FC1. The launch used38 CTAs,
whereas the preceding GEMM's measured launch was78 (occupancy2). The initial
resident grid underfilled available GEMM concurrency; four copy CTAs also
exposed substantial dispatch stalls. Deferred unrun both4/8/12 variants after
completing the active both2 phase; no GPU jobs were killed. Next direction:
size the resident grid from actual occupancy and retune communication roles.

Evidence: outputs/qwen38/iter11-role-pipeline-proxy.json includes every metric
and host receipt. The EP32 uninstrumented serial-control rerun was contaminated
by foreign work on227 and223 despite all32 ranks finishing; exclude its timings.
The three clean EP32 profiler decompositions remain diagnostic evidence in
docs/qwen38_bilateral/overlap_analysis.md, not a speedup verdict.


## Iteration12 — fill resident pool; isolate tile overlap (negative overall)

The resident grid now uses (available CUs-1)*actual kernel occupancy:76 CTAs
for the measured256-thread/98KiB kernel, rather than38. Compile-only returns
the actual grid; optional statistics allocate outside graph capture afterward.
Added a same-code serialized control: payload copies, tile flags, global
completion counters, GEMMs, role counts and buffers are identical; only waits
switch between per-tile readiness and whole-stage completion. Counter overhead
is present in both modes. Defaults remain the previous implementation.

ZW-M890P/ppu0015, SDK2.2/nativeTorch2.10 extension in wlf_sgl_2.2/Torch2.11.
Extension SHA25676d59f1017d03141eba17af82cc6e1c60f57f6babdfd6e63472a169656a10a4d.
H8192/I2048/E16local/K10/T4096/C768, BF16/fullbackwardreplay/FP32DW zero and
accumulation, no optimizer/shared expert, graph5x20 median of max-rank times.
EP8 is a single-host local-shape proxy (128 total experts); EP32 has the actual
512 experts and uses225/226/227/223. Every phase completed all ranks and clean
whole-node audits. Initial pending EP32 allocation included222, which became
busy; no phase started there. Stopped only its local waiting scheduler, then
reused recovered227 after fresh gates, as explicitly permitted by the user.

Completed timings:
[
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-serial-control-attempt0",
    "backend": "deepep",
    "ms": 134.90191650390625
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-serial-control-attempt0",
    "backend": "peer",
    "ms": 136.29239501953126
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-producer-control",
    "backend": "peer",
    "ms": 138.029443359375
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-both8",
    "backend": "peer",
    "ms": 296.9100830078125
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-both16",
    "backend": "peer",
    "ms": 195.38494873046875
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-both24",
    "backend": "peer",
    "ms": 185.8459716796875
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-both32",
    "backend": "peer",
    "ms": 187.73592529296874
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-both8-serialized",
    "backend": "peer",
    "ms": 371.07939453125
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-both16-serialized",
    "backend": "peer",
    "ms": 273.27578125
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-input16",
    "backend": "peer",
    "ms": 170.4064453125
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter12-return16",
    "backend": "peer",
    "ms": 163.43189697265626
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-both8-attempt0",
    "backend": "deepep",
    "ms": 164.6185791015625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-both8-attempt0",
    "backend": "peer",
    "ms": 348.2681640625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-both16",
    "backend": "deepep",
    "ms": 164.6968505859375
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-both16",
    "backend": "peer",
    "ms": 230.349609375
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-both8-serialized",
    "backend": "deepep",
    "ms": 164.500244140625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-both8-serialized",
    "backend": "peer",
    "ms": 435.526025390625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-both16-serialized",
    "backend": "deepep",
    "ms": 164.65640869140626
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-both16-serialized",
    "backend": "peer",
    "ms": 310.6498291015625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-producer-control",
    "backend": "deepep",
    "ms": 164.53916015625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-producer-control",
    "backend": "peer",
    "ms": 144.77237548828126
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-unfused-control",
    "backend": "deepep",
    "ms": 164.5994140625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter12r1-unfused-control",
    "backend": "peer",
    "ms": 142.646337890625
  }
]

Isolated effective benefit of tile overlap over identical serialized roles:
[
  {
    "scope": "ep8",
    "communication_ctas": 8,
    "pipeline_ms": 296.9100830078125,
    "serial_ms": 371.07939453125,
    "time_reduction_percent": 19.98745083033475,
    "speedup_over_serial": 1.249803949977293
  },
  {
    "scope": "ep8",
    "communication_ctas": 16,
    "pipeline_ms": 195.38494873046875,
    "serial_ms": 273.27578125,
    "time_reduction_percent": 28.502647458639828,
    "speedup_over_serial": 1.3986531870834165
  },
  {
    "scope": "ep32",
    "communication_ctas": 8,
    "pipeline_ms": 348.2681640625,
    "serial_ms": 435.526025390625,
    "time_reduction_percent": 20.03505100524432,
    "speedup_over_serial": 1.25054791201778
  },
  {
    "scope": "ep32",
    "communication_ctas": 16,
    "pipeline_ms": 230.349609375,
    "serial_ms": 310.6498291015625,
    "time_reduction_percent": 25.849111186959473,
    "speedup_over_serial": 1.3486015016237207
  }
]

Verdict: overlap is measurable, but total time still loses to tuned DeepEP and
the preceding producer recipe. Do not promote this candidate as a performance
winner. New communication loops repeatedly resolve route IDs, peer pointers,
and divisions for every16-byte vector; next cache row addresses once per tile.
Per-CTA clock deltas diagnose waiting; they are not synchronized device-wide
timestamps or a direct union/intersection overlap percentage.

Correctness: both scheduling modes passed independent H8192/I2048/E16/T768
full-element masked/duplicate-skew Y/DX/DW1/DW2/dScore references, maxRMS0.004081,
and each passed EP2 plus128 dynamic-route graph replays. All host audits clean.
ActualEP32 worst-skew C768 passed32 full-element records and four clean audits,
including the near-int32 index boundary, nonzero zero-score derivatives, and
exactly zero unused-expert gradients. Artifact:
outputs/qwen38/ep32-resident-pipeline-skew-reference.json, maxRMS0.003497055.

Asight Compute diagnostic (separate from candidate timings) profiled one native
DW1 kernel at actual local dimensions: all16 group reductions512; grid39,
512threads,216registers/thread,262144B TSM,one residentCTA/CU,25% achieved warp
occupancy. Compute37.64%,DRAM36.54%,LLC55.95%; TSM load-group conflict ratio0.71.
The13-pass replay duration4.043ms is not an uninstrumented step measurement.
This does not establish a physical compute floor. Tune wgrad tiles and inspect
shared-memory group contention. Amperf was stopped/verified before collection
and restored, exclusive node80 audit clean. Report:
ROOT/build/campaign/qwen38-dw1-profile.acurep; local textwork/qwen38-dw1-acu-details.txt.

Full timing/audit evidence: outputs/qwen38/iter12-resident-overlap-controls.json.


## Iteration13 — cache communication row addresses (negative overall)

Precompute128 source/destination byte addresses in shared memory once per
M input tile or M/N return tile. Payload BF16x8 loops now avoid repeated route
loads, dependent peer-table loads and divisions for every16bytes. All payloads,
BF16 rounding, gradient arithmetic, barriers, timeouts and flags are preserved.
Static shared memory increases by1024bytes; actual occupancy remains queried.
Native extension76d59f1017d03141eba17af82cc6e1c60f57f6babdfd6e63472a169656a10a4d
is unchanged; JIT source changes select fresh kernels.

Added explicit benchmark --scope step/forward/backward, defaultstep unchanged.
All five outputs first receive the existing complete-reference comparison;
after timed graph replay the selected scope's outputs are compared again.
Backward retains full replay and FP32 gradient zero/accumulation; it is not
MoK's saved-context backward. No optimizer/shared expert is timed.

H8192/I2048/E16local/K10/T4096/C768,BF16,BN256/BK64/WM64/WN64/S2,graph5x20,
ZW-M890P/ppu0015,SDK2.2/nativeTorch2.10 extension in wlf_sgl_2.2/Torch2.11.
EP8 is the local-shape proxy with128totalexperts; actualEP32 has512experts on
225/226/227/223. All14phases completed all ranks and clean whole-node audits.

Completed controls:
[
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-both4-attempt0",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 380.2087646484375
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-both8",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 236.5939697265625
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-both12",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 193.885791015625
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-both16",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 175.43841552734375
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-both8-serialized",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 291.29326171875
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-both8-forward",
    "backend": "deepep",
    "timing_scope": "forward",
    "ms": 32.95517272949219
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-both8-forward",
    "backend": "peer",
    "timing_scope": "forward",
    "ms": 77.57359619140625
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-both8-backward",
    "backend": "deepep",
    "timing_scope": "backward",
    "ms": 101.86609497070313
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-both8-backward",
    "backend": "peer",
    "timing_scope": "backward",
    "ms": 158.7691162109375
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-producer-control",
    "backend": "deepep",
    "timing_scope": "step",
    "ms": 134.85760498046875
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter13-producer-control",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 137.8761962890625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-both8-attempt0",
    "backend": "deepep",
    "timing_scope": "step",
    "ms": 164.603125
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-both8-attempt0",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 275.1183837890625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-both16",
    "backend": "deepep",
    "timing_scope": "step",
    "ms": 164.64869384765626
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-both16",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 198.4080322265625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-both8-serialized",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 330.8304443359375
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-both8-forward",
    "backend": "deepep",
    "timing_scope": "forward",
    "ms": 46.38267822265625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-both8-forward",
    "backend": "peer",
    "timing_scope": "forward",
    "ms": 95.48218994140625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-both8-backward",
    "backend": "deepep",
    "timing_scope": "backward",
    "ms": 118.12615966796875
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-both8-backward",
    "backend": "peer",
    "timing_scope": "backward",
    "ms": 179.7521728515625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-producer-control",
    "backend": "deepep",
    "timing_scope": "step",
    "ms": 164.5527587890625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter13-producer-control",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 144.67276611328126
  }
]

Verdict: cached addresses substantially improve the slow role-pipeline
implementation, but it still loses to tuned DeepEP and the preceding producer
recipe. Keep it opt-in. The FWD/BWD scope checks pass, so the regression is
not hidden by combined-step reporting. Existing fastest validated producer
recipe and stronger explicit-copy peer control remain available.

Correctness: both scheduling modes passed independent H8192/I2048/E16/T768
masked/duplicate-skew all-element Y/DX/DW1/DW2/dScore checks, maxRMS0.004081;
EP2 each passed128dynamic-route graph replays, both ranks0 and clean gates.
Unique full timing/audit/EP2 records are archived at
outputs/qwen38/iter13-cached-address-controls.json. Earlier EP2 helper used
fixed rank-log paths; its initial iteration12 parent/mode/gate receipts remain
valid, and the helper now derives unique directories from each job tag.

Stall reassessment: iterations11–13 did not improve the global best. Re-read
measured kernel traces and DW1 Asight Compute evidence (37.64%compute,71%TSM
load-group conflict ratio, one resident512-thread CTA using256KiB). Re-check
primary PPU and MoK guidance before the next kernel candidate. The current
role design reserves full GEMM registers/TSM for communication CTAs and writes
intermediate output before copying it; pursue work-conserving input owners
and direct cached return addresses to remove those costs. Test wgrad layouts
and a ring over received routed rows to reduce repeated FP32DW traffic.


## Iteration14 — inline input owners and direct cached return (negative overall)

Each M tile's N0 task bulk-packs X and publishes a local ready flag, then computes
real FC1 columns. Other N tasks acquire that tile and use local AIU. All CTAs
compute; there are no reserved copy CTAs. The occupancy-verified resident grid
is all CUs*occupancy,78 for the measured tile. Global dispatch serialization is
rejected for owners because that would introduce a dependency cycle.
Direct-return GEMMs may precompute128 destination addresses before the mainloop
and retain the original BF16x2 peer-store epilogue, avoiding staged output copies.
All options defaultFalse. Extension SHA256
a944ea3f417bef5c35ecdcbb84a3cf9f5b401061a3b6a090a86f7de8355f1f64.

Source review during correctness validation found that reusing the shared return
address table needed a CTA barrier BEFORE the next tile writes it. Fixed before
any performance measurement. The initial broad BF16 oracle alone had passed;
new same-input cacheFalse/True graph checks require Y,DX,dScore BITWISE equality.
Balanced/ragged/duplicate/allmasked modes each passed with mutated inputs and
three replays. FP32DW order differences<=2.126e-6relativeRMS (<1e-5limit).
Postfix full H8192/I2048/E16/T768 independent FP32 masked+duplicate references
passed (maxRMS0.004081), EP2+128dynamicgraphs passed all ranks, and actualEP32
C768 near-int32 worstskew passed32records and four clean audits, maxRMS0.003497055.
All correctness audits clean. New meaningful regression test:
tests/test_moe_return_cache_equivalence.py.

Benchmark: BF16,H8192/I2048/E16local/K10/T4096/C768,BN256/BK64/WM64/WN64/S2,
full backward replay with FP32DW zero/accumulation,no optimizer/shared expert,
graph5x20 median-max-rank. wlf_sgl_2.2/Torch2.11 on ZW-M890P/ppu0015/SDK2.2;
native extension buildTorch2.10. All12phases completed all ranks and clean audits.
EP8 is128-expert local-shape proxy; actualEP32 is512experts on225/226/227/223.
[
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter14-producer-control-attempt0",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 137.97523193359376
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter14-inline-input",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 264.64931640625
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter14-cached-return",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 137.85975341796876
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter14-bilateral",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 264.6408935546875
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter14-bilateral-forward",
    "backend": "deepep",
    "timing_scope": "forward",
    "ms": 32.97703552246094
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter14-bilateral-forward",
    "backend": "peer",
    "timing_scope": "forward",
    "ms": 97.3546142578125
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter14-bilateral-backward",
    "backend": "deepep",
    "timing_scope": "backward",
    "ms": 101.91519775390626
  },
  {
    "scope": "ep8",
    "phase": "qwen38-wlf-ep8-iter14-bilateral-backward",
    "backend": "peer",
    "timing_scope": "backward",
    "ms": 165.8747802734375
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-producer-control-attempt0",
    "backend": "deepep",
    "timing_scope": "step",
    "ms": 164.61651611328125
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-producer-control-attempt0",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 144.81435546875
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-inline-input",
    "backend": "deepep",
    "timing_scope": "step",
    "ms": 164.61014404296876
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-inline-input",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 318.49453125
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-cached-return",
    "backend": "deepep",
    "timing_scope": "step",
    "ms": 164.6460693359375
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-cached-return",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 145.03409423828126
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-bilateral",
    "backend": "deepep",
    "timing_scope": "step",
    "ms": 164.610107421875
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-bilateral",
    "backend": "peer",
    "timing_scope": "step",
    "ms": 318.85009765625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-bilateral-forward",
    "backend": "deepep",
    "timing_scope": "forward",
    "ms": 46.35319213867187
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-bilateral-forward",
    "backend": "peer",
    "timing_scope": "forward",
    "ms": 123.55506591796875
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-bilateral-backward",
    "backend": "deepep",
    "timing_scope": "backward",
    "ms": 118.09283447265625
  },
  {
    "scope": "ep32",
    "phase": "qwen38-wlf-ep32-iter14-bilateral-backward",
    "backend": "peer",
    "timing_scope": "backward",
    "ms": 195.0868408203125
  }
]

Verdict: cached direct return alone is essentially neutral; inline input with
row-major work ordering regresses badly. Keep opt-in and reject as winner.
The first78 work items with16Ntiles start only five input owners; the remaining
CTAs wait instead of exposing more independent input copies. Next order every
M-tile owner before the consumer jobs, matching the existing producer's broad
dispatch concurrency while allowing consumers to start before its last owner
finishes. Inspect the conservative256-thread cap and actual compute occupancy
with512/1024-thread geometry, retuning DeepEP under the same compute settings.

After iterations11–13 stalled, rechecked primary MoK paper and official PPU
articles34/36 (full source archived in work/reference/t-head-blog-34.json and36).
The PPU article's TP inference case is not an EP training speedup claim. Asight
DW1 evidence and measured waiting, not kernel count, drive the next direction.
Full timing/audit/equality/EP2 records: outputs/qwen38/iter14-inline-owner-controls.json.


## Iteration15: owner-prefix schedule and wider CTA controls

Hypothesis: expose input-owner M tiles first, avoiding the previous row-major
owner convoy; test 256/512/1024-thread GEMMs with identical compute tiles in
DeepEP and peer. The public peer API now permits up to1024 threads while still
querying actual occupancy and rejecting unavailable launch resources. Defaults
remain the previous256-thread configuration. Prefix scheduling is opt-in via
inline dispatch; existing producer/return behavior is preserved.

Chip/toolchain: ZW-M890P/ppu0015, wlf_sgl_2.2 Torch2.11, native-built extension
SHA256 0569f84c49ee8952b244de10aa4cff3917209e6e682330662a0353a5ca152a88.
Qwen H8192/I2048/K10,T4096,C768, BN256/BK64/S2;256 threads WM64/WN64,
512 threads WM32/WN64,1024 threads WM32/WN32. EP32 is the actual512-expert
shape on225/226/227/223. EP8 on80 retains E16/rank but has128 total experts:
it is only a proxy, and differs from earlier proxy host219. Do not compare
absolute proxy timings between hosts. Five repeats of20 graph iterations,
5warmups, max-rank event medians; routed experts only, no optimizer/shared
expert, full backward rematerialization and FP32 DW zero/accumulation.

Correctness: every-element independent FP32 large-shape masked and duplicate
route checks passed for prefix256/512/1024 and native NoPad512/1024. Actual
EP32 C768 all-to-expert-zero skew at1024 threads also passed all32 ranks and
all four host idle audits, including preserved nonzero zero-score derivatives
and exactly zero unused-expert gradients. Both wide configurations therefore
apply to the baseline too; no asymmetric compute-speed attribution.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8|prefix-1024|135.876831|168.121777|
|ep8|prefix-1024-backward|103.458862|124.576953|
|ep8|prefix-1024-forward|32.344177|43.325641|
|ep8|prefix-256|138.035303|151.978906|
|ep8|prefix-512|140.914600|175.888879|
|ep8|prefix-512-backward|106.760583|129.374182|
|ep8|prefix-512-forward|34.121500|46.483472|
|ep8|producer-1024|135.825342|165.892712|
|ep8|producer-256-attempt0|138.050867|147.157605|
|ep8|producer-512|140.899304|173.460730|
|ep32|prefix-1024|162.435608|167.459888|
|ep32|prefix-1024-backward|116.777393|123.156689|
|ep32|prefix-1024-forward|45.673547|44.355209|
|ep32|prefix-256|164.548926|150.268311|
|ep32|prefix-512|167.609229|173.026282|
|ep32|prefix-512-backward|120.132788|126.610266|
|ep32|prefix-512-forward|47.449921|46.527969|
|ep32|producer-1024|162.437817|164.798328|
|ep32|producer-256-attempt0|164.684619|144.865076|
|ep32|producer-512|167.670325|170.124573|

All20 benchmark phases completed every parent rank with clean host idle audits.
The owner-prefix schedule removes the severe iteration14 convoy but does not
beat the best producer recipe overall. Wider CTAs do not establish a fused
winner. Retain the previously validated recipe as best, with the explicit-copy
peer control still mandatory when attributing overlap. Evidence:
outputs/qwen38/iter15-prefix-wide-controls.json and
ep32-prefix-wide-skew-reference.json in the campaign workspace.

The unrelated unchanged cross-domain EP64 DeepEP large-shape control also
completed64 physical ranks on eight hosts for all four cases: T1024/C384
85.403381ms, T4096/C384319.503271ms, T8192/C384634.097607ms, T4096/C256
344.069653ms. This is eight logical host RDMA groups with MNNVL disabled,
not a tuned supernode hierarchy or a cross-domain fused kernel. Evidence:
outputs/qwen38/cross-ep64-large-token-chunk.json.

Reassessment: separate-kernel communication is only16.63% of the measured
DeepEP trace. Its local top-k reduction costs21.02ms/step, and weight gradients
33.24ms. MoK's pinned benchmark utils use eager CUDA events (500warmups,
100timed iterations), and time backward after a saved-context forward outside
the measured range. Its DeepEP+Torch forward requests do_cpu_sync=True and
constructs a GPU count tensor from a host count list. Our graph-resident GPU
metadata/full-rematerialization contract is different. These differences must
be reported; they do not justify weakening the current baseline. Next pursue
whole-FFN depth-first tile fusion and local compute/layout work rather than
more per-GEMM communication-CTA partition sweeps.


## Iteration16: depth-first whole-FFN tiles in forward and backward

Hypothesis: replace the full FC1/activation/FC2 kernel boundaries with one
persistent M-tile loop. Each CTA gathers one M128 tile, computes its complete
FC1 N sweep, applies BF16 SwiGLU, completes FC2 and writes directly to source
peers, while other CTAs process their own tiles. Backward similarly joins
local dHidden, activation gradient and direct-return DX; DW2 precedes this
pair, and DW1 follows it. Existing global rank phase barriers remain. This
is a paired-GEMM FFN pipeline, not a single kernel for the entire training
step or optimizer. No cross-CTA readiness dependency is introduced.

Changes: new opt-in PeerMoEPlan.ffn_fusion, separate PeerFfnParams/API/JIT
entry points, SelectedMTileScheduler and a compile-time selected-tile mode
on the existing GEMM. Existing launch ABIs/defaults remain intact. Fused
input packing uses the same release fence/CTA barrier as the previously
validated producer; same-launch gate/up reads are volatile and phase
boundaries fence stores before AIU consumption. All four compute roles use
identical BN256/BK64/WM64/WN64/S2, also used by the real DeepEP control.

ZW-M890P/ppu0015, wlf_sgl_2.2 Torch2.11, extension SHA256
c33ffe92a544462caf82b3b00c4228ac9e12abd7f82fb4fb2068293ae39c1fb9.
Actual EP32=512 routed experts, H8192/I2048/top10/T4096/C768, same-supernode
225/226/227/223. EP8 on80 is a128-total-expert proxy only. Five repeats of20
graph iterations with5warmups; max-rank event median, full forward and
backward rematerialization plus FP32 DW zero/accumulation, no optimizer or
shared expert. Forward/backward scopes are separately labeled.

Independent every-element FP32 reference passed masked and duplicate-skew
Qwen shapes at E16/T80. The graph mutation equivalence test compared old
producer+AIU against the new FFN for balanced/ragged/skewed/fully masked
routes: Y/DX/dScore were bitwise identical, maximum DW relative RMS4.33e-6
(threshold1e-5). Actual EP32 C768 all-to-expert-zero skew passed all32 ranks
and four clean host audits; maximum relative RMS0.003497055. No unused expert
gradients or valid zero-score derivative regressions.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8|ffn-backward|104.923914|128.291321|
|ep8|ffn-forward|33.030093|46.532629|
|ep8|ffn-step|137.967114|174.988550|
|ep8|producer-control|138.042004|147.589978|
|ep8|unfused-control|137.928650|143.401831|
|ep32|ffn-backward|118.122412|127.212085|
|ep32|ffn-forward|46.337613|48.096307|
|ep32|ffn-step|164.618262|175.279150|
|ep32|producer-control-attempt0|164.605286|144.857605|
|ep32|unfused-control|164.673962|142.193433|

All10 benchmark phases completed every rank and idle-node audit cleanly.
The single-CTA whole-FFN schedule is slower than the old producer and
explicit-copy controls. Retain it as opt-in experimental functionality;
it is not a performance winner and does not meet the2–3x target. Evidence:
outputs/qwen38/iter16-ffn-pair-controls.json and
ep32-ffn-pair-skew-reference.json in the campaign workspace.

Reassessment: profile actual FFN launch resources and local memory/compute
stalls before adding CTA teams or more FFN variants. The prior DW1 profile
also identified limited compute utilization and TSM group contention; local
compute and redundant intermediate transfers remain candidates.

Independent timing diagnostic (not a candidate benchmark): four exclusive
39-resident-CTA probes on87 found the named PPU globaltimer has a stable
~539600-tick inter-CU-group offset at a common atomic barrier. After subtracting
each CU's initial-barrier timestamp, a second barrier had only312–456ticks
spread. Raw globaltimer values must not be treated as synchronized nanoseconds.
Any internal overlap timeline requires per-device/CU offset and rate calibration;
CTA cycle sums alone do not establish device-wide hidden communication.


## Iteration17: expert-side router gradients and local backward replay

Hypothesis: remove the unweighted backward Y return and weighted DY roundtrip.
Publish raw source DY and router scores before routing; keep replayed FC2 Y at
its expert. A vectorized warp-per-row kernel reads raw peer DY/scores, writes
mandatory BF16 packed DY and computes the FP32 score dot product, returning
one scalar per unique route. Valid zero scores retain their derivatives.
Backward no longer needs the Y-return and DY-publication barriers. Existing
bilateral input/return fusion remains in the best producer/direct-DX path.

New opt-in expert_local_grad and appended PeerOperation13; original ABI/defaults
are preserved. Peer constructors now agree on protocol flags before symmetric
allocation. The new recipe persists through trainer checkpoints and standalone
evaluation. Whole-FFN fusion currently remains a separate incompatible option.

ZW-M890P/ppu0015,wlf_sgl_2.2,Torch2.11. Native extension SHA256
5431e80e76a327201feb3474e32eb14a7a0c574e66530920e80313c3ee86a068.
Same BN256/BK64/WM64/WN64/S2 in both real DeepEP38/19QP and peer, five repeats
of20 graph iterations after5warmups. Actual EP32 is512 experts/H8192/I2048/K10,
C768,T4096 unless labeled T1024/T8192; hosts225/226/227/223. EP8 on80 is only
a128-expert local-shape proxy. All scopes retain full backward rematerialization
and FP32 DW zero/accumulation; no optimizer/shared expert.

Independent full-element Qwen FP32 masked and duplicate-skew oracles passed.
Changing-input graph equivalence passed balanced/ragged/duplicate-skew/masked:
Y/DX bitwise equal to the old producer, dScore relative RMS~1.5e-7, maximum
DW relative RMS6.90e-6. Actual EP32 near-int32 C768 worst skew passed all32
ranks/four clean host audits (maximum relative RMS0.003497055), preserving
valid zero-score derivatives and exact zeros for unused experts.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8|expert-backward|104.987964|106.127209|
|ep8|expert-forward|33.081882|37.117767|
|ep8|expert-step|137.950024|143.658936|
|ep8|expert-unfused|138.004700|137.136853|
|ep8|producer-control|138.112573|147.060620|
|ep32|expert-backward|118.250049|103.174658|
|ep32|expert-forward|46.333865|36.788474|
|ep32|expert-step|164.535986|140.009692|
|ep32|expert-t1024|50.557040|43.074203|
|ep32|expert-t8192|310.484473|264.111206|
|ep32|expert-unfused|164.558533|135.234729|
|ep32|producer-control-attempt0|164.723267|144.818054|

All12 benchmark phases completed every rank and idle-node audit cleanly.
The new expert-side-gradient recipe improves the old bilateral producer by
about3.3% at EP32/T4096, reaching roughly1.175x versus the current DeepEP
control. It remains far short of2–3x, and the explicit-copy control must still
be used to isolate the benefit of fusion from other algorithm changes.
Evidence: outputs/qwen38/iter17-expert-gradient-controls.json and
outputs/qwen38/ep32-expert-grad-skew-reference.json in the campaign workspace.

Three full57.8M/four-layer public-text MoE models trained2000 steps with seeds
4827/4891/6113, unfrozen routing. Seed4891's fixed-route option applies only to
the independent gradient preflight, not training. All three training jobs and
full111539-target held-out evaluations exited0 behind clean host audits.
The library was copied and hash-checked before launching, so subsequent
source edits cannot alter this quality run. Full cross-entropy mean
1.590049869233, sample SD0.004830758970.
Native paired control is recorded in public-text-lm-best-paired.json; these
three seeds establish a small-model validation, not statistical noninferiority
or Qwen2.4T convergence. Evidence: public-text-lm-expert-grad17.json.

Next: separate full source-token routing metadata from a small local routed-row
activation ring, avoiding repeated source-chunk dispatch and FP32 DW passes.
Give DeepEP the same full-source/local-row-window contract and keep dropless
worst-skew graph correctness. Profile the current recipe before changing its
kernels; hidden communication must not be inferred from mixed kernel names.


## Iteration18: source routing separated from a bounded local routed-row ring

Hypothesis: avoid repeated source-chunk communication/layout and FP32 DW I/O.
Keep full-source int32 route metadata, window only received expert rows into a
bounded activation workspace, and process every worst-skew window without
per-window rank barriers. Both peer and real DeepEP receive the same explicit
source chunk/window contract. No token is dropped and no int32 guard removed.

Changes: Native packed_rows capacity, global metadata bound, GPU route-window
operation14, local peer replay/gradient windows, and DeepEP window packing plus
warp/token-cached FP32 reduction operation25. DeepEP rounds to BF16 once before
combine, preserving the transport's existing numerical boundary. Inactive K
groups skip before output I/O (scheduler source inspected). Unused reference
transport worst-skew matrices are omitted while constructing these workspaces.
Previous launch ABIs/defaults remain intact.

ZW-M890P/ppu0015,wlf_sgl_2.2 Torch2.11. Extension SHA256
4463833d91052907076822f4d1dbc313516eb819e177c3846d2b0e73a4fdd652.
Both paths use BN256/BK64/WM64/WN64/S2; DeepEP38communication CUs/19QPs.
Actual EP32 is512 experts,H8192,I2048,top10,T4096. New C4096 source chunks
use S65536/32768/131072 local ring rows; old expert-gradient control usesC768.
T8192 uses two C4096 chunks; T1024 usesC1024/S16384. EP8 on80 is128 total
experts and only a local-shape proxy. Five repeats of20 graph iterations after
5warmups,max-rank event median; full forward/backward rematerialization and
FP32 DW zero/accumulation, no optimizer/shared expert. Actual host selections
and clean launchers are included in each phase artifact.

Correctness: independent full-element Qwen FP32 peer masked/skewed oracles
passed at S128; changing-route/input graph equivalence at S1024 gave exact
Y/DX/dScore and maximum DW relative RMS8.11e-6. Independent two-rank DeepEP
S128/48-window graph tests passed all3 route patterns. Both actual EP32
C768/S65536 and C4096/S65536 skew tests passed64 full-gradient records and
all host audits each. The C4096 skew processes1,310,720 rows using21 reserved
windows (last inactive); physical element offset remains <=536,870,911 and
metadata capacity1,312,768 fits int32. All valid zero-score derivatives and
exactly-zero unused-expert gradients were checked.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8|expert-c768-control|137.911145|143.323206|
|ep8|window-backward|79.385608|79.321344|
|ep8|window-forward|30.601791|32.930200|
|ep8|window-s65536|109.780432|112.208081|
|ep8|window-unfused|109.977405|112.168335|
|ep32|expert-c768-control-attempt0|164.588745|140.078113|
|ep32|window-backward|90.761066|79.782550|
|ep32|window-forward|38.869794|32.497171|
|ep32|window-s131072|128.562781|111.359058|
|ep32|window-s32768|131.091833|114.856482|
|ep32|window-s65536|129.608374|112.326050|
|ep32|window-t1024|42.149646|38.886850|
|ep32|window-t8192|255.016333|220.454199|
|ep32|window-unfused|129.528064|113.330713|

All14 benchmark phases completed every rank and idle-node audit cleanly.
The old expert-gradient path decreases from about140ms to111–112ms for
T4096. Same-contract DeepEP is about129ms, giving roughly1.15x, still far
below2–3x. Old DeepEP164.6/new peer111.4 includes the source/window scheduling
change and must not be described as pure fusion/overlap gain. S131072 is the
fastest observed timing; S65536 has explicit full-capacity skew proof and a
smaller workspace. The small difference versus the same-window explicit-copy
control does not establish a large independent overlap benefit.

Artifacts: outputs/qwen38/iter18-routed-row-window-controls.json,
ep32-route-window-skew-c768-reference.json and
 ep32-route-window-skew-c4096-reference.json in the campaign workspace.

Quality receipts are recorded in public-text-lm-window18-paired.json. Six
2000-step public-text MoE training jobs and six111539-target full evaluations
passed cleanly, with per-seed frozen libraries. Initial parameter hashes match
for all three seed pairs. The collector decodes individual JSON objects from
shared rank stdout, including records concatenated on the same output line.

Quality status: clean paired validation; all per-seed initial parameter hashes match.
Observed means: Native1.589084658165, peer1.588399054513.


Next: profile large-source compute and weight reuse, then tune matched BM256
compute/metadata alignment and direct-return vector stores. Keep scopes and
baseline settings explicit; do not count mixed kernels as pure compute.


## Iteration19: BM256 compute tiles and matching routing alignment

Hypothesis: larger M tiles reduce repeated weight reads. Add opt-in block_m128|256
to native, DeepEP and peer paths. Metadata/ring alignment follows the tile, active
pointwise operations use the actual block height, and incompatible resident/FFN
paths reject BM256. GPU parameter layouts are unchanged. Both comparison paths
use BN256/BK64/WM64/WN64/S2 and the same BM; no asymmetric baseline tuning.
Native extension SHA256: c47526a2f9d2e9fd55bcb0f4dafdf063282c7538c2827bd7b55466e30499e813.
Chip ZW-M890P/ppu0015, runtime wlf_sgl_2.2 Torch2.11; native build Torch2.10.

Correctness: full H8192/I2048/E16/K10 FP32 masked/skewed comparisons passed;
BM128/BM256 changing-route graph equivalence has exact Y/DX/dScore and DW
relative RMS about4e-7. Two-rank DeepEP full FP32 graphs passed all patterns.
Actual EP32 C4096/S65536 all-to-expert0 validates every Y/DX/DW/dScore element
on both backends;64 records and four host audits pass. No routed rows omitted.

Uninstrumented graph benchmarks, five repeats of20 iterations after5 warmups,
max-rank CUDA events; complete FWD and rematerialized BWD with FP32 DW zero/
accumulation, no optimizer. EP32 uses512 total routed experts, EP8 proxy128.

|Group|Phase|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8|bm128-step|109.822156|112.045581|
|ep8|bm256-backward|72.869116|73.836353|
|ep8|bm256-forward|27.513718|29.567972|
|ep8|bm256-step|100.379224|103.553369|
|ep32|bm128-step-attempt0|129.651379|112.240784|
|ep32|bm256-backward|84.300879|75.017804|
|ep32|bm256-forward|35.694443|29.861374|
|ep32|bm256-step|120.135889|104.716187|

All eight phases passed all rank exits and exclusive-node audits. BM256 reduces
actual EP32 peer step from112.241ms to104.716ms, while equally tuned DeepEP
falls from129.651ms to120.136ms. This is compute/weight-reuse improvement; the
net DeepEP/peer ratio is about1.147x, still below the2–3x target. EP8 peer stays
slower than DeepEP. Do not claim a new overlap ratio from this tile change.

Artifacts: outputs/qwen38/iter19-bm256-controls.json and
outputs/qwen38/ep32-bm256-c4096-reference.json. Iter18's matched DeepEP profile
first attempt rejected invalid CLI flags before timing; retry waits for a
fourth idle SN1 host. No result is inferred from that failed diagnostic.
Next: improve peer-return store throughput and profile the matched large-chunk
DeepEP/explicit controls; assess larger compute tiles and Wgrad TSM contention.


## Iteration20: coalesce direct peer-return stores

Combine adjacent tc02 lane-owned BF16 pairs into eight-byte stores, using
register shuffles with unchanged BF16 conversion. All lanes execute shuffles
before row-validity predicates; no TSM staging. Opt-in return_vector_bytes8
preserves four-byte control. Native extension SHA256
0330c680a91e1c358d4c013517564e11ba6cc7208aaa516959a0cd77a81fd1c9.

An independent native HGGC CuTe-coordinate probe verified BM128 and BM256
fragment ownership: no mapping errors and exactly one writer per output.
Torch inline compile probes failed due mixed CUDA/native headers; native
hgcc -arch=ppu_15 succeeds. Full Qwen FP32 masked/skewed gradients and dynamic
graph four/eight-byte equivalence pass; Y/DX/dScore exact, DW RMS~4e-7.
Actual EP32 C4096/S65536 all-to-expert0 validates every output/gradient on
all32 peer ranks, clean host audits.

Same BF16 routed layer, graph warmup5, repeats5,20 iterations, max-rank events.
EP32 has512 experts, EP8 proxy128; H8192/I2048/K10/T4096/C4096/S65536/BM256.
Both paths keep full backward replay and FP32 DW zero/accumulation, no optimizer.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8|vector4-step|100.286487|103.375500|
|ep8|vector8-step|100.369183|103.472974|
|ep32|vector4-step-attempt0|119.985620|104.552075|
|ep32|vector8-backward|84.266241|74.914783|
|ep32|vector8-forward|35.746500|29.840372|
|ep32|vector8-step|120.072571|104.798071|

All six benchmark phases completed rank exits and exclusive-node audits.
This change measures return-store throughput, not a new cross-stage overlap
mechanism. It does not reach the2–3x objective. Artifacts:
outputs/qwen38/iter20-vector8-controls.json and
outputs/qwen38/ep32-vector8-c4096-reference.json.

Matched iteration19 traces now isolate the larger issue: FC1's producer
sublaunch plus remainder costs~17.0ms per call, versus~15.3ms for explicit
input staging plus local GEMM. This input-side penalty cancels much of the
return-side fusion savings. The user explicitly prioritizes communication/
compute overlap; next implement a GPU task DAG spanning dispatch, FC1,
activation, FC2 and return, with a same-kernel forced-serial control and
measured internal communication/computation activity. Preserve the best
verified producer path as a comparison until the DAG proves faster.


## Iteration21: GPU task DAG with measured internal activity

Implement a persistent GPU queue spanning input dispatch, FC1 tiles, paired
gate/up activation, ready FC2 tiles and return copies. Separate communication
CTAs and dynamically shared FC1/FC2 compute tasks use release/acquire readiness
with a fully resident occupancy-checked grid. Apply the same DAG to backward
forward-replay without returning Y. The remaining backward WG/DG kernels stay
separate. A forced-serial switch adds stage dependencies without changing
precision, task math or CTA resources. Default production recipe stays the
iteration19 producer path; DAG is opt-in.

Optional diagnostic monitor: one resident CTA samples active task counters
using only its own clock64, avoiding unsynchronized cross-CU timestamps.
Counters cover task spans including memory waits, not exact physical wire or
Tensor Cell utilization. Production compilation removes monitor atomics.
Global timeout status persists across route windows. Graph slots preserve
per-window forward/replay samples.

Native extension SHA2561edc46ba0f37dc4d47fba71dcb664a6433d07951c94bdca153dd7a940935a058.
ZW-M890P/ppu0015, wlf_sgl_2.2 Torch2.11 runtime/nativeTorch2.10 build.
Full Qwen FP32 masked/skewed Y/DX/DW/dScore comparisons passed. Both
producer-versus-DAG and parallel-versus-serial graph tests have exact Y/DX/
dScore across four changing-route patterns, DW RMS~4e-7. Actual EP32 C4096/
S65536 all-to-one-expert verifies all outputs/gradients on32 ranks; all four
node audits clean. No dropped routes or altered rounding contract.

Benchmarks: BF16 H8192/I2048/K10/T4096/C4096/S65536,512 experts at EP32 and128
at EP8 proxy. Warmup5,20 graph iterations,5 repeats, max-rank event medians.
Full backward replay and FP32 DW zero/accumulation retained, no optimizer.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8|dag-bm128-parallel|109.896338|289.764844|
|ep8|dag-bm256-bn128-parallel|109.615576|297.560449|
|ep8|dag-bm256-parallel|100.331409|159.766895|
|ep8|dag-bm256-serial|100.251447|251.619897|
|ep8|producer-bm256|100.459967|103.486548|
|ep32|dag-bm128-parallel-r1|129.444434|279.169141|
|ep32|dag-bm256-bn128-parallel-r1|129.644556|286.202808|
|ep32|dag-bm256-parallel-forward-r1|35.712970|54.512189|
|ep32|dag-bm256-parallel-r1|120.030811|154.280042|
|ep32|dag-bm256-serial-forward-r1|35.700586|120.554846|
|ep32|dag-bm256-serial-r1|119.881824|244.623071|
|ep32|producer-bm256-forward-r1|35.667123|29.822406|
|ep32|producer-bm256-r1-attempt0|119.989209|104.573645|

All accepted phases passed all rank and node audits. One initial EP32
producer-control phase is excluded in full because foreign work appeared on
223 during/postflight. The complete retry uses225/226/227/222. The launcher
now propagates gate return75 and prefers replacement hosts on retry.

The DAG demonstrably overlaps work but is slower than the best production
path: EP32 step154.280ms versus244.623ms forced-serial and104.574ms producer.
The1.59x serialized-control ratio is not a gain versus DeepEP. Monitor
recapture on32 real PPUs has no dropped samples: rank0 FWD54,595,714cycles
with both task classes active96.0% of the window, versus130,819,434cycles
and0% in forced serial. Replay has53,104,804 versus86,518,808cycles. These
are instrumented per-rank task activity diagnostics, not max-rank latency
measurements or proof of instruction/wire overlap. The first smaller trace
capacity truncated timeline samples; its online integrals remained complete,
and full-capacity r1 captures supersede those timelines.

Single-device Asight Compute, local peer pointers:38CTAs x512threads,
224registers/thread,2176B static +131072B dynamic TSM,1CTA/CU/25% theoretical
warp occupancy; CU throughput37.59%, DRAM24.29%, LLC45.73%, TSM load group
conflict ratio41%, no stack allocation. This is resource evidence, not an
ICN bandwidth measurement or physical floor.

Artifacts: outputs/qwen38/iter21-task-dag-controls.json (accepted/excluded
phases),ep32-dag-c4096-reference.json,ep32-dag-activity-audits.json,
ep32-dag-activity-{parallel,serial}-r1.json.gz and
ep32-dag-forward-activity.png/svg. Resource report qwen38-dag-profile.acurep
and work/qwen38-dag-acu-details.txt in the campaign.

Next: test a shared pool of ready communication and compute tasks, allowing
available CTAs to borrow copy work instead of reserving a small fixed copy
pool. Preserve the strong tuned DeepEP/producer controls. The2–3x target
remains unmet; this measured overlap is not yet an acceptable production win.


## Iteration22: shared pool of ready communication and compute tasks

Replace fixed communication roles with one resident task pool. Any worker CTA
can copy an input tile, execute FC1, execute ready FC2 or return a completed
tile. Input completion publishes FC1 jobs; paired activation publishes FC2;
FC2 completion publishes return jobs. Only runnable compute jobs are queued,
so no worker blocks the input producer it requires. The same stage-serial
control uses this full resource pool. Keep the production default unchanged.

Native SHA256d241e318e1549dbce1c51b2d01a61e526baa551df3a407d9cfa4a2315480e68b.
ZW-M890P/ppu0015, wlf_sgl_2.2 Torch2.11 runtime/nativeTorch2.10 build.
Full H8192/I2048/E16/K10 FP32 masked/skewed oracles passed. Pool-versus-producer
and pool-serial-versus-parallel changing-route graphs have exact Y/DX/dScore
and DW RMS~4e-7. Actual EP32 C4096/S65536 all-to-expert0 validates every output
and gradient on32 ranks, clean host gates. The state grows to18 words and
queues remain bounded by physical ring capacity; GPU launch ABI is unchanged.

BF16 routed experts, EP32 E512 and EP8 proxyE128; T4096/C4096/S65536/BM256,
full backward replay, FP32 DW zero/accumulation, no optimizer. Warmup5,
20 graph iterations,5 repeats, max-rank event medians.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8|pool-bm256-bn128-parallel|109.566174|128.788586|
|ep8|pool-bm256-parallel|100.341650|109.910815|
|ep8|pool-bm256-serial|100.206250|114.881799|
|ep8|producer-bm256|100.234906|103.491003|
|ep32|pool-bm256-bn128-parallel|129.673401|130.552979|
|ep32|pool-bm256-parallel|119.954126|110.406958|
|ep32|pool-bm256-parallel-forward|35.644846|33.875565|
|ep32|pool-bm256-serial|120.023303|115.621350|
|ep32|pool-bm256-serial-forward|35.698233|37.801089|
|ep32|producer-bm256-attempt0|119.990332|104.579956|
|ep32|producer-bm256-forward|35.654544|29.814816|

The shared pool recovers much of iteration21's regression, but remains an
intermediate experimental path. Keep the equally tuned DeepEP and best
producer comparisons; a serial-control ratio is not a DeepEP speedup.
Internal activity diagnostics are captured separately with all-rank output
files, monitorCTA single-clock timestamps and full timeline capacity.

A native publication litmus checked3000epochs x16384U32words across two
resident CTAs, three repeats each: per-thread fence and block-joined leader
fence both give zero errors/timeouts. Their~38–40ms times are similar. The
no-fence diagnostic also happens to pass, which is not proof that ordering
can be omitted. No production fence was removed. Official HGGC release/
acquire and fence documentation was fetched from chapter1041's public read
endpoint; NVIDIA descriptions are not used to establish PPU semantics.

Artifacts: outputs/qwen38/iter22-shared-task-pool-controls.json,
ep32-pool22-c4096-reference.json; raw distributed activity traces remain
under build/campaign/qwen38-ep32-dag-activity-*-pool22*.json.gz.

Iterations20–22 have not established a3% improvement over the best producer
recipe. Before iteration23, reassess with the resource profile, these timing
controls, official PPU references and the user's inference MegaMoE source.
Prioritize a new dataflow/compute fusion direction; do not keep claiming that
activity overlap alone achieves the2–3x objective.


## Iteration23: persistent G8 parameter layout and FC1/SwiGLU epilogue

Add optional G8 gate/up parameter storage and register FC1/SwiGLU fusion.
Convert checkpoints once; optimizer parameters and DW1 remain in G8 order.
The public inverse conversion restores canonical gate/up ordering. Both
DeepEP and peer receive the same layout. DeepEP retains its separate
activation; split-layout and G8 separate-activation controls isolate the
layout effect from epilogue fusion. Defaults remain split and unfused.

The epilogue explicitly rounds gate and up accumulators to BF16 BEFORE
SiLU/multiplication, preserving the training precision boundary. Forward
discards temporary gate/up output; backward replay preserves it for the
activation gradient. No gradient or backward replay is omitted. This is
intra-kernel compute fusion; no new inter-kernel overlap is claimed.

Native SHA25646b44dec8e29b37316fd895b3e31c87aa71306227290d1b8f47a6ebac840dac1.
ZW-M890P/ppu0015, wlf_sgl_2.2, Torch2.11 runtime/native Torch2.10 build.
Full H8192/I2048/E16/K10 FP32 masked/skewed oracles passed. G8 fused versus
separate activation changing-route graphs have exact Y/DX/dScore and DW
RMS about4e-7. Two-rank DeepEP G8 full-gradient oracles passed. Actual EP32
C4096/S65536 all-to-expert0 passes output/all-gradient FP32 checks and
unused-expert exact zeros on32 ranks, with clean four-host audits.

Full BF16 routed-layer FWD + rematerialized BWD + FP32 DW zero/accumulation,
no optimizer. H8192/I2048/K10, T4096 unless named t8192, C4096/S65536,
BM256/BN256/BK64/WM64/WN64/S2. Warmup5,20 graph iterations,5 repeats;
median of max-rank event times. Each phase rechecks whole-host exclusivity.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8 proxy E128|g8-control|101.450116|103.620276|
|ep8 proxy E128|g8-fused|101.300336|102.805981|
|ep8 proxy E128|split-control|100.410327|103.596619|
|ep32 E512|g8-control|121.271008|104.810876|
|ep32 E512|g8-fused|121.330542|103.884631|
|ep32 E512|g8-fused-backward|85.042645|74.517993|
|ep32 E512|g8-fused-forward|36.125153|29.279465|
|ep32 E512|g8-fused-t8192|237.952075|202.558191|
|ep32 E512|split-control-attempt0|120.333325|104.789685|

This controlled group does not establish a2–3x gain against DeepEP. Any
gain must use matched layout/scope and complete-step timings, not the
forced-serial DAG denominator. Public-text paired full-LM validation is
running separately with three2000-step seeds, frozen libraries, persistent
G8 optimizer state and full111539-target held-out evaluation. Its outcome
will be recorded only after all training/evaluation audits complete.

Artifacts: outputs/qwen38/iter23-g8-fc1-{proxy,controls}.json;
full graph/oracle logs are in the campaign workspace. The G8 design is
adapted from the user-provided PPU inference MegaMoE fragment pairing,
with the additional BF16 intermediate-rounding and replay-storage contract.


## Iteration24: G8 FC1/SwiGLU inside the shared ready-task pool

Integrate the BF16-rounded G8 FC1 activation epilogue into the resident shared
pool. FC1 directly writes each compact hidden slice. The last FC1 N tile for
an M tile publishes FC2 jobs, removing paired gate/up waits and the separate
activation task. Backward replay preserves G8 gate/up; forward discards it.
Only opt-in shared-pool G8 uses this path. Fixed-role/canonical DAG defaults
remain unchanged. The GPU argument structure is unchanged; host API adds an
optional boolean. Activity metadata says FC1 now includes SwiGLU.

Native SHA2568f4ed938182e86fd9e841880ca4a557c522c5a623b7d0371fb48a6c5abf3d6b0,
ZW-M890P/ppu0015, wlf_sgl_2.2 Torch2.11 runtime/Torch2.10 native build.
Full H8192/I2048/E16/K10 masked/skewed FP32 checks pass. G8 shared-pool versus
producer and G8 shared-pool serial versus parallel changing-route graphs
have exact Y/DX/dScore, DW RMS below5e-7. Actual EP32 C4096/S65536 all-to-
expert0 validates every output/gradient and unused-expert zeros on32 ranks,
with clean four-host audits. One initial oracle CLI mistakenly passed an
unsupported --chunk-tokens option and exited before GPU work; corrected
invocation passed on219.

H8192/I2048/K10, EP32 E512 or explicitly labeled EP8 proxy E128,
T4096/C4096/S65536/BM256/BN256/BK64/WM64/WN64/S2. Complete FWD + replayed
BWD + FP32 DW zero/accumulation, no optimizer. Warmup5,20 graph iterations,
5 repeats; max-rank event medians. Serial controls change stage dependencies
on the same pool, but their ratio is not a DeepEP speedup.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8 proxy E128|pool-g8|101.354065|106.644177|
|ep8 proxy E128|pool-g8-serial|101.421783|112.517712|
|ep8 proxy E128|producer-g8|101.441827|102.599609|
|ep32 E512|pool-g8-attempt0|121.082349|107.088501|
|ep32 E512|pool-g8-forward|36.069968|32.057455|
|ep32 E512|pool-g8-serial|121.232251|114.100830|
|ep32 E512|pool-g8-serial-forward|36.106345|36.670197|
|ep32 E512|pool-split|120.231641|110.570361|
|ep32 E512|producer-g8|121.291089|103.751599|

The matched producer and equally tuned DeepEP controls remain necessary:
sharing a launch or observing task overlap does not by itself make the
complete step faster. This group has not established a2–3x gain. Next test
bounded input prefetch, since the current pool puts new input below all
ready compute work and may restrict sustained lookahead. Do not claim that
source-level scheduling observation is a measured physical bandwidth cause.

Iteration23 paired full57.8M public-text LM validation is complete:
three2000-step seeds per backend, dynamic training routing, frozen libraries,
matched canonical initial hashes, full111539-target held-out evaluations,
all training/evaluation audits clean. Native G8 loss mean1.5901788364
(SD.0042897528), producer G8 fusion1.5869032975(SD.0011653200), relative
mean difference-.205986%. This verifies small-model training behavior;
it is not Qwen2.4T convergence or statistical noninferiority.

Artifacts: outputs/qwen38/iter24-g8-pool-{proxy,controls}.json,
ep32-g8-pool-c4096-reference.json, public-text-lm-g8-iter23-paired.json.
Internal G8 task traces are separate diagnostics, not production timings.
