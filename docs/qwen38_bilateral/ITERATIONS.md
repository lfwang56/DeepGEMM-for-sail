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


## Iteration25: bounded input prefetch in the shared task pool

Prioritize new input tasks ahead of ready compute when copy credits are
available. Outstanding copies equal NextInput minus DoneInput. A leader
claims NextInput with CAS only below the chosen bound, avoiding races that
would oversubscribe the credit limit. Workers return to any ready job after
copying; no fixed CTA identity or compute capacity is reserved at the tail.
Return jobs keep highest priority. Prefetch is opt-in; the old pool is the
control. Test1/2/4/8/16 concurrent input copies with matched G8 math/tiles.

Native SHA256edcb0d7ae9860cc0201b6df1a4415f4f8d8f54f998293d56eae1ee935f28fd41.
ZW-M890P/ppu0015,wlf_sgl_2.2 Torch2.11 runtime/Torch2.10 native build.
Full masked/skewed FP32 checks pass with one input credit. Dynamic graphs
with four credits match producer and stage-serial controls: exact Y/DX/
dScore and DW RMS below5e-7 for all four changing-route patterns. Actual
EP32 C4096/S65536 all-to-expert0 checks every output/gradient and unused-
expert zeros on32 ranks, all host audits clean. No GPU launch structure
changed; host API adds one default-false boolean. Metadata reports the
actual input credit limit, separate from fixed communication CTA roles.

H8192/I2048/K10, EP32 E512 or labeled EP8 proxy E128,
T4096/C4096/S65536/BM256/BN256/BK64/WM64/WN64/S2. Complete FWD + replayed
BWD + FP32 DW zero/accumulation, no optimizer. Warmup5,20 graph iterations,
5 repeats, median of max-rank event timings. The forward-only phase is
labeled separately. Strong DeepEP and best producer controls remain.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|ep8 proxy E128|pool-control|101.458325|106.608716|
|ep8 proxy E128|prefetch1|101.526337|466.842773|
|ep8 proxy E128|prefetch16|101.544507|107.265454|
|ep8 proxy E128|prefetch2|101.447858|259.960791|
|ep8 proxy E128|prefetch4|101.339325|156.795532|
|ep8 proxy E128|prefetch8|101.421191|108.097925|
|ep8 proxy E128|producer-control|101.343323|102.742944|
|ep32 E512|pool-control|121.251111|107.276135|
|ep32 E512|prefetch1-attempt0|121.179053|451.054297|
|ep32 E512|prefetch16|121.002917|108.264978|
|ep32 E512|prefetch2|121.197693|252.464551|
|ep32 E512|prefetch4|121.344873|153.565808|
|ep32 E512|prefetch4-forward|36.059283|54.044434|
|ep32 E512|prefetch8|121.277966|108.566333|
|ep32 E512|producer-control|121.274475|103.796704|

The credit experiment must improve complete-step latency to be adopted;
activity overlap alone is insufficient. Frozen iteration24 diagnostics on
all32 ranks show forward communication/compute task-window intersection
86.15–90.00% of the monitored interval, while stage-serial controls give0%.
Backward forward-replay intersection is35.80–45.99%; this is not dHidden/
DX/weight-gradient overlap. The monitor uses a single clock per device;
these windows include memory waits and are neither wire nor MMA utilization.

Artifacts: outputs/qwen38/iter25-prefetch-{proxy,controls}.json,
ep32-pool-prefetch-c4096-reference.json; iteration24 all-rank summaries in
ep32-pool24-{parallel,serial}-all-rank-summary.json. The pending original
SN1 diagnostic has only three idle hosts; a separate frozen24 rerun on SN2
completed cleanly. No measurements from different library versions are
combined as if they were one candidate.

If iterations23–25 fail to improve the best complete step by3%, perform
the required stall reassessment before editing iteration26. The next
unimplemented direction is a backward task pipeline joining DY/score
fetch, dHidden+activation backward, DX and gradient return; DW1/DW2 still
must be computed in full. Record its actual scope and controls explicitly.


## Iteration26: backward task pipeline and optional direct GEMM return

Join raw DY/score fetch and score-gradient publication, dHidden with BF16-
rounded G8 activation backward, DX and gradient return in a shared ready-
task pool. It can accompany the producer forward independently of forward
DAG execution. Input consumes replayed Y before DX may reuse its workspace.
The existing activation-backward epilogue preserves forward hidden values,
so full FP32 DW2 and DW1 can run after the backward pool without retaining
another activation copy. No gradient, replay, valid route or precision
boundary is omitted. Per-window weight-gradient accumulation order remains.

Optional direct return writes FC2/DX results to source ranks in the GEMM
rather than local output plus a copy task. Replay still keeps local Y.
FC2/DX task spans then include peer traffic; their entire duration must not
be classified as pure compute or counted as independent communication
simultaneously. The GPU argument structure stays unchanged; optional host
flags and backward score pointer tables select the new path. All-rank plan
agreement includes the DAG protocol flags. Defaults remain unchanged.

Native SHA2565d86fffae382ef733d3486844886aca9bc458253ed6c87357bdaef21ece485a8.
ZW-M890P/ppu0015, wlf_sgl_2.2 Torch2.11 runtime/Torch2.10 native build.
Full H8192/I2048/E16/K10 FP32 masked/skewed checks pass. Five dynamic-graph
comparisons cover backward-only, both directions, direct/copy return and
serial/parallel controls; all four patterns have exact Y/DX/dScore and
DW RMS below5e-7. X, DY and routing weights change between patterns.
Actual EP32 E512 independent FP32 checks cover balanced/masked, duplicate
and reversed routing, including an empty input rank:96 records. A separate
C4096/S65536 all-to-expert0 oracle passes32 records. Clean whole-host audits
accompany every accepted phase. Node226 became busy between the two oracle
phases; the next fresh gate selected225/227/223/222 instead.

Stall reassessment preceded this edit: reviewed iterations23–25, all-rank
iteration24 activity and prior resource counters, then fetched current
primary T-Head HGGC guide1044 and blog36 about M890 TSM group conflicts.
They motivate examining resource balance, not a universal overlap target.
No physical performance floor was established. Documentation guidance and
the installed2.2.0 toolchain remain distinct.
Sources: https://developer.t-head.cn/api/developer/document/chapter/content?chapterId=1044
and https://developer.t-head.cn/api/developer/content/detail?bizType=BLOG&id=36 .

H8192/I2048/K10, T4096/C4096/S65536/BM256/BN256/BK64/WM64/WN64/S2.
EP32 has512 total experts; EP8 is explicitly an E128 proxy. Full FWD plus
replayed BWD and FP32 DW zero/accumulation, no optimizer; separately named
forward/backward scopes retain their stated work. Warmup5,20 graph iterations,
5 repeats, median of max-rank event times. DeepEP receives the same G8
parameter layout and GEMM tiles. Low input-copy credits remain disabled.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|EP8 proxy E128|backward-copy|101.535986|107.112939|
|EP8 proxy E128|backward-direct|101.518622|108.212610|
|EP8 proxy E128|both-copy|101.415167|111.031152|
|EP8 proxy E128|both-direct|101.350555|112.153772|
|EP8 proxy E128|forward-copy|101.597034|106.541467|
|EP8 proxy E128|forward-direct|101.363745|106.199634|
|EP8 proxy E128|producer-control|101.386554|102.558630|
|EP32 E512|backward-copy|121.159167|104.256702|
|EP32 E512|backward-direct|121.092053|105.855237|
|EP32 E512|backward-direct-backward|85.084546|77.317920|
|EP32 E512|both-copy|121.119971|107.635242|
|EP32 E512|both-direct|121.071582|109.075110|
|EP32 E512|both-direct-forward|36.115594|30.973901|
|EP32 E512|forward-copy|121.119983|104.207532|
|EP32 E512|forward-direct|120.956445|104.239185|
|EP32 E512|producer-control-attempt0|121.162427|100.879974|

Choose the complete-step winner against both producer and tuned DeepEP;
functional fusion and task overlap alone are insufficient. Nine complete
57.8M public-text LM trainings also finished2000 steps (Native, backward-
only pool and forward/backward pool, three seeds each). Full111539-target
held-out evaluation evidence is collected separately, without claiming
Qwen2.4T model convergence or statistical noninferiority.

Artifacts: outputs/qwen38/iter26-backward-dag-{proxy,controls}.json,
ep32-backward-dag-{balanced,c4096}-reference.json. Source-level scope and
mixed-kernel timing caveats must accompany future overlap figures.


## Iteration27: independent vector reads in input staging

Increase memory-level parallelism inside producer input copies: load2/4/8
independent uint4 vectors per thread into compile-time-indexed register
arrays, then store them. Keep the original single-vector loop as default1.
BF16 bits, row addresses, invalid-row zeros and the final per-thread fence
plus block join are unchanged. No cache mode, opaque alias, inline asm or
GPU communication instruction is substituted. The forward shared-pool copy
uses the same helper; backward DY/dScore reduction keeps its original order.
Only producer-copy kernels receive the nondefault template; local remainder
and gradient GEMMs keep1. Host APIs add optional integers, GPU structs stay.

Native SHA256f89f6c4caa2344cb4a0e38427b31b74b5165dd5d2bd898384a03608e664db560.
ZW-M890P/ppu0015,wlf_sgl_2.2 Torch2.11 runtime/Torch2.10 native build.
Full H8192/I2048/E16/K10 FP32 masked/skewed checks pass with ILP8.
Producer ILP1 versus2/4/8 and shared-pool four-credit ILP1 versus8 changing-
route graphs give exact Y/DX/dScore and DW RMS below5e-7, including changed
X,DY,scores and masked/duplicate routes. Actual EP32 E512 balanced/masked/
duplicate/reversed FP32 checks pass96 records. Full C4096/S65536 all-to-
expert0 with shared-pool four-credit ILP8 passes32 records. All accepted
host audits are clean. Cold JIT publication races used successfully compiled
fallback modules; compilation time is excluded from device measurements.

Asight Compute on one PPU, local peer pointers, producer FC1 first slice:
ILP1 uses224 registers/thread, ILP8 uses232; both have zero stack bytes and
25% theoretical occupancy. Instrumented CU utilization22.94% versus32.72%
is diagnostic, not an ICN bandwidth measurement or full-step speedup.
Both profile host gates are clean and amperf was restored. No spill-based
performance gain is assumed; the measured full step is decisive.

H8192/I2048/K10,T4096/C4096/S65536/BM256/BN256/BK64/WM64/WN64/S2.
EP32 E512 or explicitly labeled EP8 E128 proxy. Complete BF16 FWD + replayed
BWD + FP32 DW zero/accumulation, no optimizer. Warmup5,20 graph iterations,
5 repeats, median of max-rank event times. Equally tuned DeepEP uses the
same G8 parameter storage and GEMM tiles; input-copy ILP affects peer only.

|Group|Configuration/scope|DeepEP ms|Peer ms|
|---|---|---:|---:|
|EP8 proxy E128|dag-credit16-ilp8|101.453912|100.979785|
|EP8 proxy E128|dag-credit4-ilp1|101.364880|156.758118|
|EP8 proxy E128|dag-credit4-ilp8|101.537628|100.358197|
|EP8 proxy E128|dag-credit8-ilp8|101.451965|100.785461|
|EP8 proxy E128|dag-unbounded-ilp8|101.416364|100.705396|
|EP8 proxy E128|producer-ilp1|101.492419|102.635400|
|EP8 proxy E128|producer-ilp2|101.356232|96.733484|
|EP8 proxy E128|producer-ilp4|101.411432|95.514984|
|EP8 proxy E128|producer-ilp8|101.410077|95.648566|
|EP32 E512|dag-credit16-ilp8|121.191248|102.133545|
|EP32 E512|dag-credit4-ilp1|121.271191|153.495300|
|EP32 E512|dag-credit4-ilp8|121.129224|101.191620|
|EP32 E512|dag-credit8-ilp8|121.037585|101.588196|
|EP32 E512|dag-unbounded-ilp8|121.225317|101.781970|
|EP32 E512|producer-ilp1-attempt0|121.210168|103.923975|
|EP32 E512|producer-ilp2|121.259875|99.519922|
|EP32 E512|producer-ilp4|121.242456|98.428741|
|EP32 E512|producer-ilp8|121.093665|98.358118|

The matched EP32 producer control falls from about103.9ms to98.4ms (~5.3%).
The best observed ILP4/8 times are close; choose the production recipe after
matched repeats and resource checks. The ratio to this tuned G8 DeepEP
baseline is about1.23x, not2–3x. Prior iteration26 producer100.88ms used a
different four-host mapping, so that absolute difference is not attributed
to the new code. Preserve matched controls for every claimed improvement.

Artifacts: outputs/qwen38/iter27-input-ilp-{proxy,controls}.json,
ep32-input-ilp8-{balanced,c4096}-reference.json; resource details in campaign
work/qwen38-{producer-fc1-26,producer-ilp8-27}-acu-details.txt. Updated
full-model quality for the chosen recipe is a separate follow-up.


## Iteration 28 — weight-gradient geometry under an equally tuned DeepEP control

**Direction.** Full-token Acu27 measured DW1/DW2 TSM load conflicts 0.70/0.69,
216 registers per thread, 512 threads, zero stack and 25% theoretical occupancy.
These are local operator counters, not ICN bandwidth. Test existing native
runtime-LUT tile/warp/stage choices without changing arithmetic, buffers,
BF16 boundaries, FP32 accumulation or the peer communication algorithm.
The previous best producer input ILP8 recipe is fixed in all variants.
The same wgrad LUT applies to both installed DeepEP and peer plans.

**Validation.** Eight configurations pass every-element independent CPU FP32
checks for full Qwen widths, empty/skewed dynamic group lengths, nonzero FP32
accumulators and two graph replays; 32 oracle records, frozen iteration27 library.
No candidate spills to stack. Runtime attributes range104–240 registers per
thread. Canonical Torch2.11 host extension also passes actual32-rank independent
FP32 routed-layer validation before timing. See native-host-build-compatibility.json
for the separate native Torch2.10/2.11 build and GPU checks.

**Measured scope.** Same-domain EP32, H8192/I2048/E512/top10, T4096/rank,
C4096, BM256, G8, ring65536. Complete forward plus rematerialized backward,
including FP32 gradient zero/accumulation; no optimizer/shared expert.
Median maximum-rank device event milliseconds. Every row uses its own matched
DeepEP Buffer dispatch/combine control,38 communication CUs.

| Wgrad geometry | Peer full step ms | DeepEP full step ms | DeepEP / peer |
| --- | ---: | ---: | ---: |
| 128x128-w32x32-s2 | 118.296069 | 141.073083 | 1.1925x |
| 128x128-w32x64-s2 | 114.864392 | 137.493066 | 1.1970x |
| 128x128-w32x64-s4 | 114.354053 | 137.135596 | 1.1992x |
| 128x256-w32x64-s2 | 114.686279 | 137.457166 | 1.1985x |
| 256x128-w64x32-s2 | 116.284082 | 139.201062 | 1.1971x |
| 256x256-w32x128-s4 | 98.494623 | 121.163403 | 1.2302x |
| 256x256-w64x64-s2 | 102.946973 | 125.385681 | 1.2180x |
| auto-attempt0 | 98.464209 | 121.192639 | 1.2308x |

Accepted clean phases: 8/8 collected; eight phases were planned.
Best observed geometry: `qwen38-wlf-ep32-iter28-wgrad-auto-attempt0`, peer98.464209ms, paired DeepEP121.192639ms (1.2308x). This is a full-step comparison; it does not establish physical network utilization or a2–3x gain.

Artifacts: `outputs/qwen38/iter28-wgrad-geometry-controls.json`, `iter28-wgrad-geometry-reference.json`, `ep32-native211-reference.json`; LUT sources under `docs/qwen38_bilateral/wgrad_luts/`.


## Iteration 29 — overlap independent weight gradients with backward GEMMs and return

**Direction.** The fastest producer schedule still serialized every local GEMM.
Use one auxiliary native Torch stream: DW2 reads saved forward Hidden while
the main stream computes dHidden/activation gradients; DW1 then waits for
those gradients and runs alongside DX/return. An explicit join precedes
every metadata/activation-window reuse. Peer joins before rank barriers.
Both FP32 wgrads retain per-window accumulation order. Default `none` retains
the original schedule; `dw1` is a single-switch control and `both` overlaps
both gradients. No GPU math, precision boundaries or instructions change.

**Strong baseline.** Installed DeepEP gets the same overlap choices. Its `both`
mode writes dHidden into an additional bounded BF16 I-wide ring, preserving
forward Hidden for DW2 without a copy. Final comparisons must use the best
DeepEP mode, not a slower serial denominator. Base-plan all-rank agreement
includes the mode. Side-stream tensor lifetime is recorded with the allocator.

**Validation.** Producer serial/parallel graphs pass four changed-input,
changed-score and changed-routing patterns each for `dw1` and `both`: exact
Y/DX/dScore, DW relative RMS below5e-7. Full H8192/I2048/E16/top10 FP32 oracle
passes masked and duplicate/skewed patterns. DeepEP EP2 graph protocol oracle
passes all six records. The initial DeepEP allocation preflight caught a
missing dtype argument; it was fixed before any timing. Domain EP32 balanced
peer/DeepEP and full C4096 all-to-expert0 oracles are recorded separately;
inspect their clean audits and completion before accepting domain timings.

**Measured scope.** Qwen H8192/I2048/E512/top10, T4096/rank, C4096,
ring65536, BM256, G8, producer input ILP8. EP16 uses32 experts/rank;
EP32 uses16. Each is inside one supernode. Complete forward and rematerialized
backward with FP32 gradient zero/accumulation; no optimizer/shared expert.
Median maximum-rank event milliseconds; native Torch2.11 extension2ec4d537.

| Domain EP | Wgrad mode | Peer full step ms | DeepEP full step ms | Paired ratio |
| ---: | --- | ---: | ---: | ---: |
| 16 | qwen38-wlf-ep16-iter29-wgrad-dw1 | incomplete/excluded | incomplete/excluded | — |
| 16 | none | 106.704114 | 120.539600 | 1.1297x |

EP16: best peer `none` 106.704114ms; strongest measured DeepEP `none` 120.539600ms; best/strongest ratio1.1297x. Peer change versus its matched serial control106.704114ms: 0.000% reduction.

**Failed candidate; do not promote.** The actual EP32 DeepEP independent
oracle failed DW1 with relative RMS0.87–0.93 on several ranks. The two-rank
single-window check passed, but it did not exercise the missing dependency:
the new join was accidentally inserted in the forward window loop rather
than the backward window loop. Backward could overwrite packed X and lengths
while auxiliary DW1 still read them. Peer has the intended per-window join
and passed its independent oracle, but every nonserial paired comparison
in this iteration is excluded. No EP32 candidate timing was launched after
this failure. Preserve raw partial EP16 timings only as invalid diagnostics;
fix and repeat the multi-window checks in the next iteration.

The intended stream schedule enables concurrency, not a guarantee of faster
execution or measured network/MMA overlap. Both kernels still compete for CU,
register, TSM and memory resources. Use separate Asys spans and complete-step
timings to evaluate the outcome; retain the faster serial recipe if necessary.

Artifacts: `outputs/qwen38/iter29-wgrad-stream-ep{16,32}-controls.json`,
`ep32-wgrad-stream-reference.json`; local preflight logs `work/qwen38-wgrad29-*`.


## Iteration 30 — correct the window-reuse dependency and measure wgrad concurrency

**Direction.** The fastest producer schedule still serialized every local GEMM.
Use one auxiliary native Torch stream: DW2 reads saved forward Hidden while
the main stream computes dHidden/activation gradients; DW1 then waits for
those gradients and runs alongside DX/return. An explicit join precedes
every metadata/activation-window reuse. Peer joins before rank barriers.
Both FP32 wgrads retain per-window accumulation order. Default `none` retains
the original schedule; `dw1` is a single-switch control and `both` overlaps
both gradients. No GPU math, precision boundaries or instructions change.

**Strong baseline.** Installed DeepEP gets the same overlap choices. Its `both`
mode writes dHidden into an additional bounded BF16 I-wide ring, preserving
forward Hidden for DW2 without a copy. Final comparisons must use the best
DeepEP mode, not a slower serial denominator. Base-plan all-rank agreement
includes the mode. Side-stream tensor lifetime is recorded with the allocator.

**Validation.** Producer serial/parallel graphs pass four changed-input,
changed-score and changed-routing patterns each for `dw1` and `both`: exact
Y/DX/dScore, DW relative RMS below5e-7. Full H8192/I2048/E16/top10 FP32 oracle
passes masked and duplicate/skewed patterns. DeepEP EP2 graph protocol oracle
passes all six records. The earlier prototype allocation bug was fixed before timing; the added
36-window protocol check covers actual buffer reuse. Domain EP32 balanced
peer/DeepEP and full C4096 all-to-expert0 oracles are recorded separately;
inspect their clean audits and completion before accepting domain timings.

**Measured scope.** Qwen H8192/I2048/E512/top10, T4096/rank, C4096,
ring65536, BM256, G8, producer input ILP8. EP16 uses32 experts/rank;
EP32 uses16. Each is inside one supernode. Complete forward and rematerialized
backward with FP32 gradient zero/accumulation; no optimizer/shared expert.
Median maximum-rank event milliseconds; native Torch2.11 extension2ec4d537.
The source and precision remain fixed within the group.

| Domain EP | Wgrad mode | Peer full step ms | DeepEP full step ms | Paired ratio |
| ---: | --- | ---: | ---: | ---: |
| 16 | both | 109.910461 | 121.740552 | 1.1076x |
| 16 | dw1 | 107.813721 | 121.387207 | 1.1259x |
| 16 | none | 106.360901 | 120.368408 | 1.1317x |
| 32 | both | 100.497021 | 122.694983 | 1.2209x |
| 32 | dw1 | 98.918170 | 122.361157 | 1.2370x |
| 32 | none | 97.575024 | 121.022290 | 1.2403x |

EP16: best peer `none` 106.360901ms; strongest measured DeepEP `none` 120.368408ms; best/strongest ratio1.1317x. Peer change versus its matched serial control106.360901ms: 0.000% reduction.

EP32: best peer `none` 97.575024ms; strongest measured DeepEP `none` 121.022290ms; best/strongest ratio1.2403x. Peer change versus its matched serial control97.575024ms: 0.000% reduction.

**Corrected dependency.** Iteration29 exposed an omitted backward-window
join. The join now lives inside DeepEP `_set_window`, the exact mutation
boundary shared by forward and backward. A36-window EP2 dynamic graph oracle
and the corrected actual32-rank balanced/masked/reversed oracle passed before
this group. Full C4096 all-to-expert0 validation gates both timing matrices.
The peer implementation is unchanged from its passing iteration29 oracle.
The rank launcher now stops only its owned sibling ranks when one fails;
the benchmark also agrees on numerical failure across ranks outside timed
ranges, preventing survivors from entering a later unmatched collective.
A CPU process regression verifies prompt termination of only launcher children.

The intended stream schedule enables concurrency, not a guarantee of faster
execution or measured network/MMA overlap. Both kernels still compete for CU,
register, TSM and memory resources. Use separate Asys spans and complete-step
timings to evaluate the outcome; retain the faster serial recipe if necessary.

Artifacts: `outputs/qwen38/iter30-wgrad-stream-ep{16,32}-controls.json`,
`ep32-wgrad-stream-reference.json`; local preflight logs `work/qwen38-wgrad30-*`.


## Iteration 31 — native wgrad operand issue order and warp ownership

**Stall review.** Iteration28 smaller tiles reduced registers but slowed the
complete step;29 exposed a dependency bug;30 fixed it but legal coarse
concurrency still slowed execution. Fresh actual EP32 Asys30 confirms13.287ms
of wgrad/mixed-GEMM overlap plus6.137ms with other local GEMMs, while DW1 grew
10.832→15.505ms, DW2 5.869→8.043ms and mixed kernels27.246→37.322ms. These
are task intervals, not physical network overlap. Full-token Acu27 separately
measured TSM load conflict0.70/0.69,216 registers and zero stack for DW1/DW2.
No physical floor has been established.

Fresh primary-source review: official T-Head blogs36 and41, including their
public REST content, and the installed ACTLIZE v1.0.0
`ppu_mma_aiu_multistage.hpp` AIU_BEFORE_TSMLD branches. Blog36 distinguishes
TSM groups from banks and warns that changing tile geometry alone is
insufficient; blog41 separates global AIU supply from TSM operand supply.
Archives: `work/reference/t-head-blog{36,41}-iter31-live.json` and matching
clean plaintext. No NVIDIA instruction, bank map or fixed PU placement is
assumed.

**Direction.** Keep identical WG tiles, BF16 operands, FP32 accumulation and
K reduction order. Test identity/N-major/two bijective XOR warp ownership
maps, applied consistently to both operands and output coordinates; physical
AIU copy roles stay unchanged. Independently move next-fragment TSM loads
before/after the existing kblock0 AIU refill, using the two issue orders from
native ACTLIZE. Defaults identity+AIU-before-TSM preserve the old schedule.
JIT code includes both constants; runtime guards check ranges and XOR grids.
Environment controls `DG_BF16_WGRAD_WARP_MAP` and
`DG_BF16_WGRAD_AIU_BEFORE_TSM` apply equally to DeepEP and peer WG kernels.
Native Torch2.11 extension SHAadb6688d6ceed51fb7c787338cdc0cd9a195c8674af7b5d8fef6852a578e5c07.

**Validation.** All8 combinations pass full-width independent CPU FP32 WG
checks with empty/skewed lengths, nonzero accumulators and mutable-length
repeated graphs:32 oracle records. Map3/AIU-after also passes full Qwen MoE
FP32 masked/skewed checks and actual32-rank balanced/masked/reversed output
and gradient checks; map2/AIU-after passes the independent32-rank DeepEP
oracle. Full C4096 all-to-expert0 checks both transports before timing.
See reference artifacts and clean host audits for exact completion evidence.

**Measured scope.** Domain EP32, Qwen H8192/I2048/E512/top10, T4096/rank,
C4096, BM256, G8, producer input ILP8, ring65536, serial WG schedule. Full
forward and rematerialized backward with FP32 gradient zero/accumulation;
no optimizer/shared expert. Installed DeepEP Buffer uses38 communication CUs.
Every candidate has the same WG issue settings in its matched DeepEP control.

| WG issue configuration | Peer full step ms | DeepEP full step ms | DeepEP / peer |
| --- | ---: | ---: | ---: |
| map0-aiu0 | 99.195062 | 121.848157 | 1.2284x |
| qwen38-wlf-ep32-iter31-wgrad-map0-aiu1-attempt0 | incomplete/excluded | incomplete/excluded | — |
| map0-aiu1-attempt1 | 98.714551 | 121.165234 | 1.2274x |
| map1-aiu0 | 99.568506 | 122.260388 | 1.2279x |
| map1-aiu1 | 98.601544 | 121.296619 | 1.2302x |
| map2-aiu0 | 99.848297 | 122.534131 | 1.2272x |
| map2-aiu1 | 98.496906 | 121.168225 | 1.2302x |
| map3-aiu0 | 99.505035 | 122.018225 | 1.2263x |
| map3-aiu1 | 98.531250 | 121.252368 | 1.2306x |

Accepted clean phases: 8/9 collected; eight phases were planned.
Best observed geometry: `qwen38-wlf-ep32-iter31-wgrad-map2-aiu1`, peer98.496906ms, paired DeepEP121.168225ms (1.2302x). Matched auto peer control98.714551ms; peer reduction0.220%. Strongest DeepEP across this group121.165234ms; best-peer/strongest-DeepEP ratio1.2301x. This is a full-step comparison; it does not establish physical network utilization or a2–3x gain.

Local full-token DW1 Acu (not EP/wire timing): default map0/AIU-before216regs, stack0,25% occupancy,TSM load conflict0.69; map1/AIU-before also216/0/25%/0.69; map2 or3/AIU-after216/0/25%/0.63. Lower conflicts did not establish lower complete-step latency; inspect uninstrumented results above.

Artifacts: `outputs/qwen38/iter31-wgrad-issue-controls.json`, `iter31-wgrad-issue-reference.json`, `ep32-wgrad-issue31-reference.json`; Acu logs `work/qwen38-wgrad31-map*-aiu*-dw1-acu*`.


## Iteration 32 — low-resource raw-DY communication during rematerialization

**Direction.** The observed19.4ms wgrad/GEMM concurrency in iteration30
lengthened both compute kernels. Instead overlap a small communication
worker with local replay. After the producer's X dispatch completes, fork
raw DY into a bounded BF16 staging ring during FC1's remaining columns and
FC2 replay. Join at its first consumer. Extra staging is one H-wide ring,
independent of max_tokens/layer count. Keep raw, unweighted DY so valid zero
routing scores retain their correct score derivative.

New peer operation15 uses128 threads, warp-per-row vector16B copies and
eight independent loads before stores; padding is zeroed. Existing operation13
selects either staged/local or peer DY at runtime, then executes the same
dot reduction, BF16 weighted-DY stores and FP32 score return. GPU Params ABI
and old eight-tensor callers remain compatible; staging is an optional ninth
tensor. Per-window joins protect all reuse. Default behavior remains off.

**Strong baseline.** Installed DeepEP gets the same `overlap_expert_dy` choice:
cached DY dispatch starts on an auxiliary stream while the main stream replays
the experts, with a join immediately before `_expert_backward_rows` consumes
DY. Its returned DY records its main-stream consumer lifetime. Communication
CU counts8/16/24/38 are tested for overlap; the final ratio uses the fastest
observed DeepEP setting, including the original serial38-CU control.

**Validation.** Peer serial/prefetch graphs pass four changed-X/DY/score/route
patterns with exact Y/DX/dScore and DW RMS below5e-7. Full Qwen FP32 masked and
duplicate/skewed oracle passes. DeepEP's36-window EP2 graph oracle passes.
Actual EP32 independent FP32 peer and DeepEP balanced/masked/reversed checks,
plus full C4096 all-to-expert0 for both transports, gate every timing phase.
Native Torch2.11 extension SHA3a9c589a13be79cbeb736adfc45c028113277859f96d31c90261e55b8df80273.
Every benchmark record includes its extension path and SHA.

**Scope.** Domain EP32, H8192/I2048/E512/top10, T4096/rank, C4096,
ring65536, BM256, G8, producer input ILP8, serial WG. Full forward and
rematerialized backward, including FP32 weight-gradient zero/accumulation;
no optimizer/shared expert. Median maximum-rank event milliseconds.

| Configuration | Backend | DY overlap | DeepEP CUs | Complete step ms |
| --- | --- | --- | ---: | ---: |
| deepep-prefetch16 | deepep | True | 16 | 123.821484 |
| deepep-prefetch24 | deepep | True | 24 | 120.275134 |
| deepep-prefetch8 | deepep | True | 8 | 136.804797 |
| prefetch38 | deepep | True | 38 | 119.972473 |
| prefetch38 | peer | True | None | 99.848474 |
| serial38-attempt0 | deepep | False | 38 | 121.241980 |
| serial38-attempt0 | peer | False | None | 97.358685 |

Best peer: overlap=False, 97.358685ms. Strongest DeepEP: overlap=True, CUs=38, 119.972473ms. Ratio1.2323x. Peer reduction versus matched serial control97.358685ms: 0.000%. Task concurrency must still be verified with Asys; this ratio does not measure physical wire utilization.

Artifacts: `outputs/qwen38/iter32-dy-prefetch-controls.json` and `ep32-dy-prefetch-reference.json`.


## Iteration 33 — low-register tile assistants with inline GEMM fallback

**Direction.** Remove the whole-input boundary between producer FC1 N=0
and the remaining FC1 columns. Extend inline tile owners to the same BM256,
G8 activation epilogue and bounded routed-window recipe as the fastest control.
An independent128-thread copy kernel races the GEMM's N=0 owners to claim
individual M tiles, copies raw BF16 X in ILP8 vectors and publishes each tile.
All GEMM column consumers acquire that tile's ready flag. A GEMM owner that
wins still performs its own input copy; it cannot rely solely on a future
assistant CTA. Existing fully resident scheduling is retained.

**Protocol.** Flag0=unclaimed,2=busy,1=ready. CAS claims each tile exactly
once, existing device release/acquire publishes the data, and all loser CTAs
join before reusing shared claim state. Auxiliary completion is joined after
FC1 and before subsequent window reuse. No cross-rank protocol, GPU Params
layout, arithmetic, FP32 accumulation order or BF16 intermediate rounding changes.
Zero assisting CTAs exercises the inline-owner path without auxiliary work;
8/16/32/39 CTAs test the same native copy code at different concurrency.
Default producer remains unchanged. No NVIDIA occupancy assumptions are used.

**Validation.** Full-width independent FP32 masked/skewed reference, exact
mutable graph comparison against producer G8 ILP8 (changed X/DY/scores/routes),
actual EP32 balanced/masked/duplicate/reversed and C4096 full all-to-expert0
references gate timing. Every accepted phase requires all four host gates and
all32 ranks. Strong installed DeepEP serial and early-DY settings are retained.
Native Torch2.11 extension SHAe875fd910fc304c90ad7bbe1eb9a09caeec2ef914ce73f42ffe0efd07f08ef53.

**Scope.** Domain EP32 H8192/I2048/E512/top10, T4096/rank,C4096,ring65536,
BM256,G8,full forward plus rematerialized backward and FP32 DW zero/accumulation.
Median maximum-rank event milliseconds; optimizer/shared expert excluded.

| Configuration | Backend | Assistant CTAs | Full step ms |
| --- | --- | ---: | ---: |
| deepep-early-dy | deepep | None | 120.065222 |
| inline-assist0 | peer | 0 | 100.085876 |
| inline-assist16 | peer | 16 | 103.384839 |
| inline-assist32 | peer | 32 | 105.023706 |
| inline-assist39 | peer | 39 | 104.216602 |
| inline-assist8 | peer | 8 | 104.233716 |
| qwen38-wlf-ep32-iter33-input-producer-control-attempt0 | excluded/incomplete | — | — |
| producer-control-attempt2 | deepep | None | 121.310986 |
| producer-control-attempt2 | peer | 0 | 97.400525 |

Best peerqwen38-wlf-ep32-iter33-input-producer-control-attempt2: 97.400525ms. Strongest DeepEPqwen38-wlf-ep32-iter33-input-deepep-early-dy: 120.065222ms. Ratio1.2327x. Reduction versus matched producer97.400525ms: 0.000%. Kernel-span concurrency alone is not wire utilization or a latency win.

Artifacts: `outputs/qwen38/iter33-input-assist-controls.json`, `ep32-input-assist-reference.json`.


## Iteration 34 — specialized addressing and separate input-consumer footprint

**Review.** Three successive negative iterations31–33 triggered a new stall
review (`stall_review_before34.md`), including fresh primary PPU SDK/Asys/Acu
sources and new frozen33 resources. No physical floor has been established.
The local-pointer33 trace shows real copy/compute overlap, but the inline
GEMM carries240registers/thread and the copy kernel64. Its auxiliary vector
positions divide by runtime64-bit width. Source cost is a hypothesis, not yet
an attributed hardware counter result.

**Changes.** Specialize input assistant H/BM in JIT, leaving raw BF16 bytes,
padding, tile CAS and publication identical. Environment
`DG_INPUT_ASSIST_STATIC_GEOMETRY=0` exactly retains the dynamic addressing
geometry,1 enables constants. Compare both with the same inline assistant8.

New `dispatch_external` compiles a GEMM consumer without either inline input
copy or resident communication-CTA code. Its readiness scheduler acquires each
tile before local AIU loads; an independent128-thread assistant supplies data.
A compile-time ExternalInput template flag is necessary: runtime zero
communication CTAs alone would retain both branches in the same register
allocation. Native occupancy bounds the consumer grid to(numCUs-1)*occupancy,
leaving room for progress of a lower-resource copy CTA. Retain timeout/status,
per-window event joins, all-rank protocol agreement and graph-safe lifetimes.
No NVIDIA-specific resource rule, precision change or payload omission.

**Validation.** Independent full-width FP32, exact mutable graph comparisons
for dynamic/static inline and external consumer against the producer, and
actual EP32 balanced/masked/reversed/duplicate plus full C4096 all-to-expert0
references gate timing. All accepted rows require four clean whole-host gates
and32 clean ranks. Default producer and WG/DY scheduling remain unchanged.

**Scope.** Domain EP32 H8192/I2048/E512/top10,T4096/rank,C4096,ring65536,
BM256,G8,full FWD+rematerialized BWD and FP32 DW zero/accumulation.
Strong installed DeepEP serial/early-DY38 controls are retained. Times are
uninstrumented median maximum-rank device-event milliseconds.

| Configuration | Backend | External | Static geometry | Assist CTAs | Full step ms |
| --- | --- | --- | --- | ---: | ---: |
| deepep-early-dy | deepep | None | 1 | None | 119.939490 |
| external-static16 | peer | True | 1 | 16 | 228.583472 |
| external-static32 | peer | True | 1 | 32 | 231.126782 |
| external-static39 | peer | True | 1 | 39 | 239.948633 |
| external-static8 | peer | True | 1 | 8 | 226.628149 |
| inline-dynamic8 | peer | False | 0 | 8 | 104.075488 |
| inline-static8 | peer | False | 1 | 8 | 102.269965 |
| producer-control-attempt0 | deepep | None | 1 | None | 121.204871 |
| producer-control-attempt0 | peer | False | 1 | 0 | 97.389264 |

Best peerqwen38-wlf-ep32-iter34-input-producer-control-attempt0: 97.389264ms. Strongest DeepEPqwen38-wlf-ep32-iter34-input-deepep-early-dy: 119.939490ms. Ratio1.2315x. Reduction versus matched producer97.389264ms: 0.000%. Kernel-span concurrency alone is not wire utilization or a latency win.

Native extension SHA256: 5548ec2a3d31327d6f7d043cb21afb9aa19fb19d3f9ec5ce210f8df3eca711ca

Artifacts: `outputs/qwen38/iter34-input-assist-controls.json`, `ep32-input-assist-reference.json`.


## Iteration 35 — warp-row input copies and a matched serialization control

**Direction.** Iteration34 had117–134ms of actual copy/mixed-GEMM intersection
while complete latency regressed226–240ms. Overlap was real but expensive.
Native resource records also show the static linear mover uses80registers,
compared with56for the earlier warp-row DY mover. Source-level inference:
reuse a row address per warp and limit live vectors to eight, without changing
payload, padding, CAS0/2/1 ownership or release/acquire publication.

New `DG_INPUT_ASSIST_WARP_ROWS=1` selects warp-row copies;0 retains the linear
kernel. Both use the same compile-time geometry and ILP8. Explicitly disable
outer row/column-loop unrolling to bound vector lifetimes; the eight independent
loads and stores remain unrolled. No speculative ISA or precision change.

**Counterfactual.** `dispatch_external_serial=True` joins the existing
auxiliary dispatch before launching the identical external GEMM consumer.
False joins only after GEMM. Data, flags, mover/consumer kernels and launch
geometries are identical. This measures the net effect of simultaneous work;
it is not a substitute for the fastest installed DeepEP denominator. All-rank
recipe agreement includes the serialization mode and joins protect ring reuse.

**Validation.** Exact mutable graph producer comparisons for warp-row inline,
external parallel and external serial, full-width independent FP32, actual
EP32 balanced/masked/duplicate/reversed and C4096 all-to-expert0 checks gate
candidate timing. Default producer remains unchanged. All accepted phases
require four clean host gates and32 clean ranks.

**Scope.** Domain EP32 H8192/I2048/E512/top10,T4096/rank,C4096,ring65536,
BM256,G8,full FWD+rematerialized BWD and FP32 DW zero/accumulation.
Strong DeepEP serial/early-DY38 controls remain. Uninstrumented median
maximum-rank device-event milliseconds.

| Configuration | Backend | External | Warp rows / serial | Assist CTAs | Full step ms |
| --- | --- | --- | --- | ---: | ---: |
| deepep-early-dy | deepep | None | 1 / None | None | 120.096179 |
| linear-inline8 | peer | False | 0 / False | 8 | 102.278467 |
| linear-serial8 | peer | True | 0 / True | 8 | 124.936951 |
| producer-control-attempt0 | deepep | None | 1 / None | None | 121.213025 |
| producer-control-attempt0 | peer | False | 1 / False | 0 | 97.532886 |
| warp-inline8 | peer | False | 1 / False | 8 | 100.911017 |
| warp-parallel32 | peer | True | 1 / False | 32 | 181.066919 |
| warp-parallel8 | peer | True | 1 / False | 8 | 162.378015 |
| warp-serial32 | peer | True | 1 / True | 32 | 99.994977 |
| warp-serial8 | peer | True | 1 / True | 8 | 113.376074 |

Best peerqwen38-wlf-ep32-iter35-input-producer-control-attempt0: 97.532886ms. Strongest DeepEPqwen38-wlf-ep32-iter35-input-deepep-early-dy: 120.096179ms. Ratio1.2313x. Reduction versus matched producer97.532886ms: 0.000%. Kernel-span concurrency alone is not wire utilization or a latency win.

Native extension SHA256: 91d45f9e6ba9465682d8f2e9fd96a657cd1d151c6d5baefed7e397a196beaed9

Artifacts: `outputs/qwen38/iter35-input-assist-controls.json`, `ep32-input-assist-reference.json`.


## Iteration 36 — native PPU yield hints at input-ready waits

**Evidence.** Iteration35 compares identical copy and consumer kernels:
warp-row32 parallel181.066919ms versus serialized99.994977ms. Default producer
97.532886ms remains faster. The purpose is to test poll scheduling cost while
preserving the same release/acquire handoff, copy payload and BF16 math.

The installed SDK2.2 header `hgrt/hggc_extend_device_functions.h` defines
`__ppu_yield(){return __ppukernel_yield_void();}`. Use this native PPU symbol;
no NVIDIA nanosleep instruction or preemption semantics is assumed.

**Change.** `DG_INPUT_READY_YIELD_EVERY`0(default) leaves polling unchanged.
Power-of-two intervals1/8/32/128 issue the hint after that many unsuccessful
ready polls. Apply it to the scheduler input wait and inline owner's loser
wait. Keep `tix::ld_acq`, target/status checks, timeout, stores, fences and
publication exactly intact. Zero is compiled out, and default producer kernels
have no input-ready pipeline. Kernel names and benchmark metadata record the
interval; public API/GPU Params remain compatible.

**Validation.** Full-width independent FP32, exact mutable graphs for external,
inline and serialized consumers at1/8/32/128, plus all-rank EP32 balanced and
C4096 all-to-expert0 checks gate timing. No precision/gradient tolerance changes.
All accepted phases require four clean host gates and32 clean ranks.

**Scope.** Domain EP32 H8192/I2048/E512/top10,T4096/rank,C4096,ring65536,
BM256,G8,full FWD+rematerialized BWD with FP32 DW zero/accumulation. Strong
installed DeepEP serial/early-DY38 controls remain the denominator.
Uninstrumented median maximum-rank device-event milliseconds.

| Configuration | Backend | External | Yield interval / serial | Assist CTAs | Full step ms |
| --- | --- | --- | --- | ---: | ---: |
| deepep-early-dy | deepep | None | 0 / None | None | 119.925647 |
| inline8-y1 | peer | False | 1 / False | 8 | 101.056158 |
| parallel32-y0 | peer | True | 0 / False | 32 | 181.770715 |
| parallel32-y1 | peer | True | 1 / False | 32 | 181.579590 |
| parallel32-y128 | peer | True | 128 / False | 32 | 182.188379 |
| parallel32-y32 | peer | True | 32 / False | 32 | 182.055884 |
| parallel32-y8 | peer | True | 8 / False | 32 | 181.964392 |
| parallel8-y1 | peer | True | 1 / False | 8 | 162.896313 |
| producer-control-attempt1 | deepep | None | 0 / None | None | 121.358972 |
| producer-control-attempt1 | peer | False | 0 / False | 0 | 97.596191 |
| serial32-y1 | peer | True | 1 / True | 32 | 99.884131 |

Best peerqwen38-wlf-ep32-iter36-yield-producer-control-attempt1: 97.596191ms. Strongest DeepEPqwen38-wlf-ep32-iter36-yield-deepep-early-dy: 119.925647ms. Ratio1.2288x. Reduction versus matched producer97.596191ms: 0.000%. Kernel-span concurrency alone is not wire utilization or a latency win.

Native extension SHA256: 51c05037ef0150ef0e6e9d865eee54cf6d139627b4694480e9b722d26d72bc51

Artifacts: `outputs/qwen38/iter36-yield-assist-controls.json`, `ep32-input-assist-reference.json`.


## Iteration 37 — full-width peer GEMM geometry

**Evidence and change.** Iterations34–36 show that extra concurrent input copying lengthens both copy and ready-wait GEMM spans. The required stall review is archived in `stall-review-before37.md` with the live primary PPU guide. Test stage depth, BK, warp shapes and BN using the existing native API, without precision changes. Four fixed GPU queues gate every configuration on independent full-width FP32 and mutable graph checks; unsupported resource configurations are rejected rather than timed. BK32 is outside the API contract.

**Validation.** H8192/I2048/E16/top10, FP32 comparisons of all output and gradient elements, four graph input patterns, then actual EP32 distributed every-element references before timing. BF16, G8, producer ILP8, BM256, ring65536, T4096/C4096, full forward + rematerialized backward including FP32 DW zero/accumulation. DeepEP uses matching geometry; an additional early-DY control is retained. All phases require four clean whole-host audits. No kernel implementation or default geometry changes.

| Configuration | Backend | GEMM configurations | Complete layer ms |
| --- | --- | --- | ---: |
| qwen38-wlf-ep32-iter37-gemm-bn128 | deepep | {'fc1': [128, 64, 64, 64, 2], 'fc2': [128, 64, 64, 64, 2], 'dhidden': [128, 64, 64, 64, 2], 'dx': [128, 64, 64, 64, 2]} | 130.798035 |
| qwen38-wlf-ep32-iter37-gemm-bn128 | peer | {'fc1': [128, 64, 64, 64, 2], 'fc2': [128, 64, 64, 64, 2], 'dhidden': [128, 64, 64, 64, 2], 'dx': [128, 64, 64, 64, 2]} | 103.317969 |
| qwen38-wlf-ep32-iter37-gemm-bn128-bk128 | deepep | {'fc1': [128, 128, 64, 64, 2], 'fc2': [128, 128, 64, 64, 2], 'dhidden': [128, 128, 64, 64, 2], 'dx': [128, 128, 64, 64, 2]} | 137.183154 |
| qwen38-wlf-ep32-iter37-gemm-bn128-bk128 | peer | {'fc1': [128, 128, 64, 64, 2], 'fc2': [128, 128, 64, 64, 2], 'dhidden': [128, 128, 64, 64, 2], 'dx': [128, 128, 64, 64, 2]} | 117.061731 |
| qwen38-wlf-ep32-iter37-gemm-control-attempt0 | deepep | {'fc1': [256, 64, 64, 64, 2], 'fc2': [256, 64, 64, 64, 2], 'dhidden': [256, 64, 64, 64, 2], 'dx': [256, 64, 64, 64, 2]} | 121.185852 |
| qwen38-wlf-ep32-iter37-gemm-control-attempt0 | peer | {'fc1': [256, 64, 64, 64, 2], 'fc2': [256, 64, 64, 64, 2], 'dhidden': [256, 64, 64, 64, 2], 'dx': [256, 64, 64, 64, 2]} | 97.500012 |
| qwen38-wlf-ep32-iter37-gemm-deepep-early-control | deepep | {'fc1': [256, 64, 64, 64, 2], 'fc2': [256, 64, 64, 64, 2], 'dhidden': [256, 64, 64, 64, 2], 'dx': [256, 64, 64, 64, 2]} | 120.214099 |
| qwen38-wlf-ep32-iter37-gemm-stage3 | deepep | {'fc1': [256, 64, 64, 64, 3], 'fc2': [256, 64, 64, 64, 3], 'dhidden': [256, 64, 64, 64, 3], 'dx': [256, 64, 64, 64, 3]} | 110.157800 |
| qwen38-wlf-ep32-iter37-gemm-stage3 | peer | {'fc1': [256, 64, 64, 64, 3], 'fc2': [256, 64, 64, 64, 3], 'dhidden': [256, 64, 64, 64, 3], 'dx': [256, 64, 64, 64, 3]} | 84.258929 |
| qwen38-wlf-ep32-iter37-gemm-wm32 | deepep | {'fc1': [256, 64, 32, 64, 2], 'fc2': [256, 64, 32, 64, 2], 'dhidden': [256, 64, 32, 64, 2], 'dx': [256, 64, 32, 64, 2]} | 121.791284 |
| qwen38-wlf-ep32-iter37-gemm-wm32 | peer | {'fc1': [256, 64, 32, 64, 2], 'fc2': [256, 64, 32, 64, 2], 'dhidden': [256, 64, 32, 64, 2], 'dx': [256, 64, 32, 64, 2]} | 94.207941 |
| qwen38-wlf-ep32-iter37-gemm-wn32 | deepep | {'fc1': [256, 64, 64, 32, 2], 'fc2': [256, 64, 64, 32, 2], 'dhidden': [256, 64, 64, 32, 2], 'dx': [256, 64, 64, 32, 2]} | 118.128162 |
| qwen38-wlf-ep32-iter37-gemm-wn32 | peer | {'fc1': [256, 64, 64, 32, 2], 'fc2': [256, 64, 64, 32, 2], 'dhidden': [256, 64, 64, 32, 2], 'dx': [256, 64, 64, 32, 2]} | 93.508429 |

Fastest peer: ('peer', 84.25892944335938, 'qwen38-wlf-ep32-iter37-gemm-stage3').

Fastest deepep: ('deepep', 110.15780029296874, 'qwen38-wlf-ep32-iter37-gemm-stage3').

Preflight configurations: [('control', [256, 64, 64, 64, 2], True), ('wn32', [256, 64, 64, 32, 2], True), ('stage3', [256, 64, 64, 64, 3], True), ('bn128', [128, 64, 64, 64, 2], True), ('bk128', [256, 128, 64, 64, 2], False), ('bn128-bk128', [128, 128, 64, 64, 2], True), ('wm32', [256, 64, 32, 64, 2], True), ('stage4', [256, 64, 64, 64, 4], False)]


## Iteration 38 — two-domain bilateral composition and ElasticBuffer baseline

**Change.** Two contiguous physical domains of32 ranks each. Matching two-rank PCCL groups all-gather X/IDs/scores across domains. Each domain runs the existing bilateral PeerMoEPlan on its own expert subspace; pair reduce-scatter returns source outputs. Backward regathers inputs, all-gathers raw DY, rematerializes the complete inner operator, then reduce-scatters DX/dScore while retaining each expert owner's FP32 DW. GPU payload only; Gloo is metadata/oracle control. Fixed maximum input shapes preserve graph capture; global-ID validation precedes domain masking. Fresh outputs prevent later plan calls from overwriting autograd storage. Group lifetime is caller-owned.

**Strong baseline.** Installed DeepEP ElasticBuffer nonexpanded hybrid dispatch/combine shares the exact DeepEPMoEPlan GEMM/rematerialization/FP32 DW implementation. Native PPU header/thread-scope compatibility uses an isolated package copy; installed package is untouched. Elastic hybrid rejects deterministic routing, so record nondeterministic mode explicitly. Its top-k expert IDs must be distinct; compare balanced/masked/reversed and uniquely skewed routes. Peer additionally retains duplicate-route tests. Legacy flat-RDMA Buffer controls are separate and disable MNNVL; the fastest accepted baseline, including Elastic, is the denominator.

**Preflight evidence.** Two physical hosts/16 ranks proxy checks passed48 every-element FP32 records and mutable graphs. Qwen H8192/I2048/E512/top10 two-domain EP64 passed192 records, max relative RMS0.006988, all eight whole-host gates clean. Additional stage3 hybrid/Elastic oracles gate performance. The initial Elastic probes rejected deterministic mode and then a missing PCCL include path; the native compatibility environment resolves both and passes forward/DX/dScore plus4 changed-route graph replays at scaleup8/scaleout2. Preserve failed attempts.

**Precision and model validation.** Domain BF16 partial rounding differs from flat dispatch/combine. In a full57.8M LM, the unfrozen BF16-vs-FP32 preflight changes selected expert sets in later layers (0/0/6/5) and fails the original4% per-parameter limit; do not relax it. On identical selected IDs the largest parameter-gradient difference is0.6851%, global L2~0.23%. Training remains dynamically routed. The training harness uses the global GPU group for auxiliary load gradients, separate from Gloo plan metadata, and serializes no process-group handles. Checkpoints explicitly retain training backend and consolidated single-device evaluation backend. Public-text seed runs and heldout evaluation are reported separately; performance is not convergence proof.

**Timing scope.** Full BF16 routed layer plus rematerialized backward and FP32 DW zero/accumulation; stage3, G8, ring65536, T4096, E512/top10. Pure matched C2048 controls first; separately tuned Hybrid C4096 and Elastic C3840 are labeled. The legacy/Elastic global receive matrix forbids C4096 at EP64 due to int32 indexing. Pair AG/RS composition changes cross-domain traffic and is not a measurement of domain-only fusion.

| Configuration | Backend | Source chunk | Complete routed-layer ms |
| --- | --- | ---: | ---: |
| qwen38-wlf-cross-ep64-iter38-hybrid-c2048-attempt0 | excluded/incomplete | — | — |


## Iteration 39 — accept routed hybrid/elastic benchmark configurations

**Change.** Iteration38 completed independent EP64 hybrid and Elastic validation, but its first timing command was rejected by a legacy CLI guard before device initialization. Extend that guard to hybrid/elastic transports, validate positive domain size, and add --validate-configuration to check all nine recipes without device initialization. Correct activation-workspace metadata to exclude the inner weight-transpose buffers, matching the base plan convention. Kernels and numerical contracts are unchanged. Two contiguous physical domains of32 ranks each. Matching two-rank PCCL groups all-gather X/IDs/scores across domains. Each domain runs the existing bilateral PeerMoEPlan on its own expert subspace; pair reduce-scatter returns source outputs. Backward regathers inputs, all-gathers raw DY, rematerializes the complete inner operator, then reduce-scatters DX/dScore while retaining each expert owner's FP32 DW. GPU payload only; Gloo is metadata/oracle control. Fixed maximum input shapes preserve graph capture; global-ID validation precedes domain masking. Fresh outputs prevent later plan calls from overwriting autograd storage. Group lifetime is caller-owned.

**Strong baseline.** Installed DeepEP ElasticBuffer nonexpanded hybrid dispatch/combine shares the exact DeepEPMoEPlan GEMM/rematerialization/FP32 DW implementation. Native PPU header/thread-scope compatibility uses an isolated package copy; installed package is untouched. Elastic hybrid rejects deterministic routing, so record nondeterministic mode explicitly. Its top-k expert IDs must be distinct; compare balanced/masked/reversed and uniquely skewed routes. Peer additionally retains duplicate-route tests. Legacy flat-RDMA Buffer controls are separate and disable MNNVL; the fastest accepted baseline, including Elastic, is the denominator.

**Preflight evidence.** Two physical hosts/16 ranks proxy checks passed48 every-element FP32 records and mutable graphs. Qwen H8192/I2048/E512/top10 two-domain EP64 passed192 records, max relative RMS0.006988, all eight whole-host gates clean. Additional stage3 hybrid/Elastic oracles gate performance. The initial Elastic probes rejected deterministic mode and then a missing PCCL include path; the native compatibility environment resolves both and passes forward/DX/dScore plus4 changed-route graph replays at scaleup8/scaleout2. Preserve failed attempts. Elastic no-CPU-sync receive buffers initialize only a GPU-counted prefix; the adapter masks padded expert IDs to -1 and weights to0 using the documented device prefix sum. Its initial full-shape padding failure is preserved. Corrected Elastic EP16 and stage3 EP64 references passed before these timings.

**Precision and model validation.** Domain BF16 partial rounding differs from flat dispatch/combine. In a full57.8M LM, the unfrozen BF16-vs-FP32 preflight changes selected expert sets in later layers (0/0/6/5) and fails the original4% per-parameter limit; do not relax it. On identical selected IDs the largest parameter-gradient difference is0.6851%, global L2~0.23%. Training remains dynamically routed. The training harness uses the global GPU group for auxiliary load gradients, separate from Gloo plan metadata, and serializes no process-group handles. Checkpoints explicitly retain training backend and consolidated single-device evaluation backend. Public-text seed runs and heldout evaluation are reported separately; performance is not convergence proof.

**Timing scope.** Full BF16 routed layer plus rematerialized backward and FP32 DW zero/accumulation; stage3, G8, ring65536, T4096, E512/top10. Pure matched C2048 controls first; separately tuned Hybrid C4096 and Elastic C3840 are labeled. The legacy/Elastic global receive matrix forbids C4096 at EP64 due to int32 indexing. Pair AG/RS composition changes cross-domain traffic and is not a measurement of domain-only fusion.

| Configuration | Backend | Source chunk | Complete routed-layer ms |
| --- | --- | ---: | ---: |
| qwen38-wlf-cross-ep64-iter39-elastic16-c2048 | elastic | 2048 | 203.010681 |
| qwen38-wlf-cross-ep64-iter39-elastic38-c2048 | elastic | 2048 | 160.894861 |
| qwen38-wlf-cross-ep64-iter39-elastic38-c3840 | elastic | 3840 | 171.938867 |
| qwen38-wlf-cross-ep64-iter39-elastic38-early-c2048 | elastic | 2048 | 158.100024 |
| qwen38-wlf-cross-ep64-iter39-elastic8-c2048 | elastic | 2048 | 267.476953 |
| qwen38-wlf-cross-ep64-iter39-hybrid-c2048-attempt0 | excluded/incomplete | — | — |
| qwen38-wlf-cross-ep64-iter39-hybrid-c2048-attempt1 | hybrid | 2048 | 110.592810 |
| qwen38-wlf-cross-ep64-iter39-hybrid-c4096 | hybrid | 4096 | 103.742053 |
| qwen38-wlf-cross-ep64-iter39-legacy38-c2048 | deepep | 2048 | 256.243555 |
| qwen38-wlf-cross-ep64-iter39-legacy38-early-c2048 | deepep | 2048 | 255.487964 |

Fastest accepted hybrid: ('hybrid', 103.74205322265625, 'qwen38-wlf-cross-ep64-iter39-hybrid-c4096', 4096).

Fastest accepted elastic: ('elastic', 158.1000244140625, 'qwen38-wlf-cross-ep64-iter39-elastic38-early-c2048', 2048).

Fastest accepted deepep: ('deepep', 255.4879638671875, 'qwen38-wlf-cross-ep64-iter39-legacy38-early-c2048', 2048).


## Iteration 40 — persistent compute-grid budget and paced ready polling

**Evidence.** Identical serial versus concurrent copy/consumer kernels in35 cost99.99 versus181.07ms. Native yield hints36 did not improve that concurrency. The faster stage3 recipe37 still shows zero cross-kernel intersection: local44.16ms (including WG16.92), mixed23.92ms in a85.54ms instrumented rank0 union. No physical resource floor was established.

**Change.** DG_EXTERNAL_INPUT_COMPUTE_CUS0 retains the old external grid of(numCUs-1)*measured occupancy; positive budgets cap its persistent CTA grid. This does not pin CTAs to CUs or guarantee hardware partitioning. DG_INPUT_READY_PAUSE_CYCLES0..16384 adds a local clock64 pause after unsuccessful acquire/status/timeout checks in the scheduler and inline-owner loser wait. Defaults0 compile away. The existing DAG hardware sampler already uses the same native clock64 pacing pattern. All tix acquire/release operations, fences, ready/status semantics, BF16 arithmetic and timeout remain intact. Kernel names and metrics expose pause and grid settings. Public GPU Params and native API arguments remain compatible.

**Validation.** Stage3 producer/consumer exact mutable graph comparisons, independent full-width FP32, actual32 all-element balanced/masked/skewed/reversed references, and fullC4096 skew gate the matrix. The installed ElasticBuffer domain-EP32 path is additionally validated and timed at8/16/38 communication CUs; the strongest accepted Elastic or legacy dispatch/combine is the denominator. Stage3, G8, producerILP8, H8192/I2048/E512/top10,T4096/C4096,ring65536,BM256,full rematerialized backward and FP32 DW zero/accumulation.

| Configuration | Backend | Compute budget | Copy CTAs | Pause cycles | Complete layer ms |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen38-wlf-ep32-iter40-grid-c16-a16-p0 | peer | 16 | 16 | 0 | 104.976965 |
| qwen38-wlf-ep32-iter40-grid-c24-a16-p0 | peer | 24 | 16 | 0 | 99.635968 |
| qwen38-wlf-ep32-iter40-grid-c24-a16-p0-serial | peer | 24 | 16 | 0 | 104.205542 |
| qwen38-wlf-ep32-iter40-grid-c24-a16-p1024 | peer | 24 | 16 | 1024 | 99.427966 |
| qwen38-wlf-ep32-iter40-grid-c31-a8-p0 | peer | 31 | 8 | 0 | 88.781000 |
| qwen38-wlf-ep32-iter40-grid-c31-a8-p256 | peer | 31 | 8 | 256 | 88.757782 |
| qwen38-wlf-ep32-iter40-grid-c35-a4-p0 | peer | 35 | 4 | 0 | 114.677051 |
| qwen38-wlf-ep32-iter40-grid-c38-a32-p0 | peer | 38 | 32 | 0 | 198.385901 |
| qwen38-wlf-ep32-iter40-grid-c38-a32-p2048 | peer | 38 | 32 | 2048 | 198.041785 |
| qwen38-wlf-ep32-iter40-grid-c38-a32-p512 | peer | 38 | 32 | 512 | 198.823389 |
| qwen38-wlf-ep32-iter40-grid-c8-a32-p0 | peer | 8 | 32 | 0 | 152.647607 |
| qwen38-wlf-ep32-iter40-grid-deepep-early | deepep | 0 | None | 0 | 109.021887 |
| qwen38-wlf-ep32-iter40-grid-elastic16 | elastic | 0 | None | 0 | 114.402283 |
| qwen38-wlf-ep32-iter40-grid-elastic38 | elastic | 0 | None | 0 | 107.803589 |
| qwen38-wlf-ep32-iter40-grid-elastic8 | elastic | 0 | None | 0 | 115.138745 |
| qwen38-wlf-ep32-iter40-grid-inline-a0-p0 | peer | 0 | 0 | 0 | 88.842365 |
| qwen38-wlf-ep32-iter40-grid-inline-a8-p512 | peer | 0 | 8 | 512 | 90.222229 |
| qwen38-wlf-ep32-iter40-grid-producer-control-attempt0 | deepep | 0 | None | 0 | 110.083630 |
| qwen38-wlf-ep32-iter40-grid-producer-control-attempt0 | peer | 0 | 0 | 0 | 85.411011 |

Fastest accepted peer: ('peer', 85.4110107421875, 'qwen38-wlf-ep32-iter40-grid-producer-control-attempt0').

Fastest accepted elastic: ('elastic', 107.8035888671875, 'qwen38-wlf-ep32-iter40-grid-elastic38').

Fastest accepted deepep: ('deepep', 109.02188720703126, 'qwen38-wlf-ep32-iter40-grid-deepep-early').

**Local-pointer resource screening.** GlobalE16 on one PPU, not ICN or EP32. Whole-host audit clean=True. These values do not enter distributed speedups. See iter40-local-grid-screen.json.

Only a lower complete-step latency counts as a win; resource budgets, busy-loop hints and visible concurrency are hypotheses rather than wire/MMA utilization measurements.


## Iteration 41 — additional full-width WGrad tiles and fixed-thread warp layouts

The stage3 profile leaves44.16ms of local kernel spans, including16.92ms WGrad.
Iteration37 improved GEMM staging but did not test these four additional WGrad
geometries or the constant512-thread WM32/WN128 and WM128/WN32 mappings.
WGrad options:256x256x64 stages3;256x256x128 stages2;
256x128x128 stages2;128x256x128 stages2. Earlier28 used different BK64 tiles.
The directory wgrad_luts40 retains its preliminary preflight name; these are41
timing trials. Every WGrad geometry passed independent full-width CPU FP32,
nonzero beta=1 accumulation and changing device K lengths under repeated graph
replay. Four M-group warp options passed independent full-width FP32 and exact
mutable graph comparisons. Actual32-rank FP32 checks cover both Peer and legacy
DeepEP layouts before timing; failed or incomplete phases remain excluded.

The fastest accepted installed ElasticBuffer or legacy Buffer recipe from40
sets the communication baseline. Each compute geometry applies equally to both
backends, with H8192/I2048/E512/top10,T4096/C4096,ring65536,BM256,BF16 and
complete rematerialized backward/FP32 DW zero+accumulation. Source/extension are
frozen from40; LUT hashes are recorded separately. This changes compute tiles,
not the communication protocol or instruction-overlap claims.

| Phase | Backend | Complete layer ms |
| --- | --- | ---: |
| qwen38-wlf-ep32-iter41-geometry-control-elastic | elastic | 107.921216 |
| qwen38-wlf-ep32-iter41-geometry-control-peer-attempt0 | peer | 84.225812 |
| qwen38-wlf-ep32-iter41-geometry-wg-bk128-stage2-elastic | elastic | 108.239172 |
| qwen38-wlf-ep32-iter41-geometry-wg-bk128-stage2-peer | peer | 84.681580 |
| qwen38-wlf-ep32-iter41-geometry-wg-m128-bk128-stage2-elastic | elastic | 120.573389 |
| qwen38-wlf-ep32-iter41-geometry-wg-m128-bk128-stage2-peer | peer | 97.017572 |
| qwen38-wlf-ep32-iter41-geometry-wg-n128-bk128-stage2-elastic | elastic | 120.128357 |
| qwen38-wlf-ep32-iter41-geometry-wg-n128-bk128-stage2-peer | peer | 96.554108 |
| qwen38-wlf-ep32-iter41-geometry-wg-stage3-elastic | elastic | 106.848181 |
| qwen38-wlf-ep32-iter41-geometry-wg-stage3-peer | peer | 83.194653 |
| qwen38-wlf-ep32-iter41-geometry-wm128-wn32-s2-elastic | elastic | 117.677051 |
| qwen38-wlf-ep32-iter41-geometry-wm128-wn32-s2-peer | peer | 96.751685 |
| qwen38-wlf-ep32-iter41-geometry-wm128-wn32-s3-elastic | elastic | 108.188721 |
| qwen38-wlf-ep32-iter41-geometry-wm128-wn32-s3-peer | peer | 84.046893 |
| qwen38-wlf-ep32-iter41-geometry-wm32-wn128-s2-elastic | elastic | 117.835010 |
| qwen38-wlf-ep32-iter41-geometry-wm32-wn128-s2-peer | peer | 96.867694 |
| qwen38-wlf-ep32-iter41-geometry-wm32-wn128-s3-elastic | elastic | 107.571790 |
| qwen38-wlf-ep32-iter41-geometry-wm32-wn128-s3-peer | peer | 84.915588 |

Fastest accepted peer: {"name": "qwen38-wlf-ep32-iter41-geometry-wg-stage3-peer", "backend": "peer", "ms": 83.1946533203125, "gemm_configs": {"fc1": [256, 64, 64, 64, 3], "fc2": [256, 64, 64, 64, 3], "dhidden": [256, 64, 64, 64, 3], "dx": [256, 64, 64, 64, 3]}, "wgrad_lut_sha256": "0651bc1981d18a9291b4f27b53794cafec98f9b15f34280565824810dd438fad"}

Fastest accepted elastic: {"name": "qwen38-wlf-ep32-iter41-geometry-wg-stage3-elastic", "backend": "elastic", "ms": 106.84818115234376, "gemm_configs": {"fc1": [256, 64, 64, 64, 3], "fc2": [256, 64, 64, 64, 3], "dhidden": [256, 64, 64, 64, 3], "dx": [256, 64, 64, 64, 3]}, "wgrad_lut_sha256": "0651bc1981d18a9291b4f27b53794cafec98f9b15f34280565824810dd438fad"}

See iter41-geometry-controls.json, iter41-geometry-reference.json and iter41-baseline-selection.json.


## Iteration 42 — coupled WGrad and input-gradient persistent CTA budgets

Earlier30 exposed real DW/dX kernel intersections but regressed total latency.
The standard persistent grids can occupy all available slots independently.
Add DG_BF16_WGRAD_COMPUTE_CUS0(default) or1..actualCUs, capping its grid by
requested CUs times measured per-kernel occupancy. PeerMoEPlan additionally
accepts dgrad_compute_cus0(default) or a positive budget for dHidden/DX only.
The internal native peer API appends a default-zero compute_cus argument;
new Python requires the matching extension. Both schedulers stride by gridDim.x,
so reduced grids still cover every tile. GPU Params, MMA arithmetic, accumulation,
barriers, stream joins and buffer reuse protocols are unchanged. These budgets
do not pin CTAs, guarantee CU partitions, or prove joint kernel residency.

Six exact mutable graph cases passed. Full H8192/I2048/E16/K10 FP32 passed both
WG4/DGrad35 and WG31/DGrad8 extremes. The first two large preflight commands
omitted ring capacity and were rejected by constructor validation; corrected
commands passed and initial logs remain. Actual32-rank checks passed three Peer
budgets, Elastic WG8/both and fullC4096 skew. The selected41 compute geometry
has an additional all-rank gate before timing. One skew launcher used an unsupported --dispatch-producer CLI flag;
the corrected recipe passed without changing tolerances. No failed launch timing
is included. All accepted performance phases require four clean whole-host audits.

Compare each backend's best41 geometry and preserve BF16,H8192/I2048/E512/top10,
T4096/C4096,ring65536,BM256,complete rematerialized backward,DX/DW/dScore and
FP32 DW zero/accumulation. Controls include identical-budget serial DW, old
uncapped overlap and installed DeepEP with early DY plus bounded DW overlap.
Four additional small WGrad tiles passed full-width independent FP32 and mutable
K-length graph checks: M64/N128/S2, M128/N64/S2, M64/N64/S2 and M64/N128/S3,
all BK64 and128threads. Measured SMEM49,152/49,152/32,768/73,728bytes;
registers144/160/104/144; per-kernel occupancy5/5/8/3 and no stack. Their
directory wgrad_luts43 retains the initial exploratory label; all their timings
belong to42. Each small tile has actual-domain Peer and Elastic preflights and
serial/overlap controls. Joint residency remains a hypothesis until measured.

| Phase | Backend | WG budget | dGrad budget | WG stream | Complete layer ms |
| --- | --- | ---: | ---: | --- | ---: |
| qwen38-wlf-ep32-iter42-wgrad-baseline-control | elastic | 0 | None | none | 106.872559 |
| qwen38-wlf-ep32-iter42-wgrad-baseline-early-dy | elastic | 0 | None | none | 105.703552 |
| qwen38-wlf-ep32-iter42-wgrad-baseline-w16-both-early | elastic | 16 | None | both | 117.181787 |
| qwen38-wlf-ep32-iter42-wgrad-peer-control-attempt0 | peer | 0 | 0 | none | 83.173517 |
| qwen38-wlf-ep32-iter42-wgrad-peer-uncapped-both | peer | 0 | 0 | both | 87.769409 |
| qwen38-wlf-ep32-iter42-wgrad-peer-w16-d0-both | peer | 16 | 0 | both | 98.685828 |
| qwen38-wlf-ep32-iter42-wgrad-peer-w16-d23-both | peer | 16 | 23 | both | 89.682416 |
| qwen38-wlf-ep32-iter42-wgrad-peer-w16-d23-dw1 | peer | 16 | 23 | dw1 | 100.819659 |
| qwen38-wlf-ep32-iter42-wgrad-peer-w16-d23-serial | peer | 16 | 23 | none | 116.295813 |
| qwen38-wlf-ep32-iter42-wgrad-peer-w31-d8-both | peer | 31 | 8 | both | 126.615649 |
| qwen38-wlf-ep32-iter42-wgrad-peer-w4-d35-both | peer | 4 | 35 | both | 210.766260 |
| qwen38-wlf-ep32-iter42-wgrad-small-m128-n64-s2-baseline-w8-both | elastic | 8 | None | both | 255.854541 |
| qwen38-wlf-ep32-iter42-wgrad-small-m128-n64-s2-peer-w0-none | peer | 0 | 0 | none | 109.673157 |
| qwen38-wlf-ep32-iter42-wgrad-small-m128-n64-s2-peer-w4-both | peer | 4 | 0 | both | 327.417407 |
| qwen38-wlf-ep32-iter42-wgrad-small-m128-n64-s2-peer-w8-both | peer | 8 | 0 | both | 226.479419 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n128-s2-baseline-w8-both | elastic | 8 | None | both | 239.558789 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n128-s2-peer-w0-none | peer | 0 | 0 | none | 111.773206 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n128-s2-peer-w4-both | peer | 4 | 0 | both | 308.135278 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n128-s2-peer-w8-both | peer | 8 | 0 | both | 207.107715 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n128-s3-baseline-w8-both | elastic | 8 | None | both | 203.389624 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n128-s3-peer-w0-none | peer | 0 | 0 | none | 111.730566 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n128-s3-peer-w8-both | peer | 8 | 0 | both | 184.548010 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n64-s2-baseline-w8-both | elastic | 8 | None | both | 264.717603 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n64-s2-peer-w0-none | peer | 0 | 0 | none | 121.736230 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n64-s2-peer-w4-both | peer | 4 | 0 | both | 334.959180 |
| qwen38-wlf-ep32-iter42-wgrad-small-m64-n64-s2-peer-w8-both | peer | 8 | 0 | both | 251.015356 |

Fastest accepted peer: {"name": "qwen38-wlf-ep32-iter42-wgrad-peer-control-attempt0", "backend": "peer", "ms": 83.17351684570312}

Fastest accepted elastic: {"name": "qwen38-wlf-ep32-iter42-wgrad-baseline-early-dy", "backend": "elastic", "ms": 105.70355224609375}

Only a lower complete-layer latency is an optimization win. See iter42-wgrad-controls.json, iter42-wgrad-reference.json and iter42-compute-selection.json.


## Iteration 43 — fuse raw-DY and score-dot work into replay FC2

Reassessment after40–42 is in stall_review_before43.md. More kernel overlap had
not beaten the83.17ms producer: even positive paired-budget overlap89.68 versus
116.30ms stayed slower. Choose a different dataflow: during FC2 backward replay,
load raw peer DY and source FP32 scores, round expert Y to the SAME BF16 boundary,
produce the SAME BF16 weighted DY, and save per-N-tile FP32 score-dot partials.
New pointwise peer opcode17 reduces these partials and returns FP32 dScore.
This removes the expert-Y write/read and standalone H-wide DY weighting/dot pass.
The existing packed-Y buffer remains allocated; this is a traffic change, not
an advertised activation-allocation reduction. BM256 uses4KB aligned static
shared scratch. Shared float atomicAdd was validated on the actual PPU.

PeerGemmParams ABI stays unchanged: an optional separate PeerReplayGradients
launch argument accompanies only the uniquely named _replay_grad kernels. Old
false-mode source/API calls remain compatible; opcode17 avoids the old13 cache.
Replay gradients require routed bilateral expert-local gradients, packed DY,
fused activation backward and an explicit FC2 tile; DAG, DY prefetch and staged
return combinations are rejected. Kernel names expose the mixed peer-DY path.

Two graph cases(serial/both-WG) passed changing inputs, masks, ragged valid counts,
duplicate/skew and zero-score cases. Y/DX exact and original DW RMS<1e-5 remain.
The new FP32 score reduction is reassociated: BEFORE testing, its graph contract
was RMS<1e-5 plus elementwise rtol2e-5/atol2e-6. Other ablations retain their
previous exact score contract. Independent every-element fullH8192/I2048/E16/K10
FP32 checks passed BM128 andBM256 with unchanged .04 thresholds. Actual32-rank
balanced/BM128/overlap and fullC4096 skew checks gate timing. No failed or incomplete
host receipt contributes a latency. All phases retain complete remat, allDX/DW/
dScore and FP32 DW zero/accumulation, BF16,H8192/I2048/E512/top10,T4096/C4096,
ring65536,G8 and identical MGroup/WGrad stage3 tuning opportunity for DeepEP.

| Phase | Backend | Replay-gradient fusion | Complete layer ms |
| --- | --- | --- | ---: |
| qwen38-wlf-ep32-iter43-replay-control-attempt0 | peer | False | 83.099426 |
| qwen38-wlf-ep32-iter43-replay-elastic-early | elastic | None | 105.717981 |
| qwen38-wlf-ep32-iter43-replay-fused | peer | True | 90.186145 |
| qwen38-wlf-ep32-iter43-replay-fused-bm128 | peer | True | 118.545337 |
| qwen38-wlf-ep32-iter43-replay-fused-overlap | peer | True | 94.672382 |
| qwen38-wlf-ep32-iter43-replay-fused-w16-d23 | peer | True | 96.613776 |
| qwen38-wlf-ep32-iter43-replay-legacy-early | deepep | None | 107.871680 |

Fastest accepted peer: {"name": "qwen38-wlf-ep32-iter43-replay-control-attempt0", "backend": "peer", "ms": 83.09942626953125}

Fastest accepted deepep: {"name": "qwen38-wlf-ep32-iter43-replay-legacy-early", "backend": "deepep", "ms": 107.8716796875}

Fastest accepted elastic: {"name": "qwen38-wlf-ep32-iter43-replay-elastic-early", "backend": "elastic", "ms": 105.71798095703124}

Mixed kernel duration includes computation and peer traffic; it is not pure wire time. See iter43-replay-controls.json and iter43-replay-reference.json.


## Iteration 44 — reduce replay score atomics inside each warp

43's fused replay passed every numerical check but cost90.19ms versus83.10ms
unfused control. Test DG_REPLAY_GRAD_WARP_REDUCTION0(default)/1: combine the
tc02 C-coordinate lane contributions belonging to one row with uniform XOR
shuffles, then one shared float atomic per row's lane group. All lanes keep
participating for ragged M tails. C-coordinate groups being closed under XOR
is a tested layout property, not a general hardware assumption. Kernel names
append _warp_reduce and the explicit environment setting is recorded. The
default0 source retains43's kernel call/ABI; no base GPU Params changes.
Native debug output now exposes registers and stack bytes: both BM128/BM256
new kernels use232registers/thread, no local stack, occupancy1; dynamic SMEM
147456/196608bytes respectively, in addition to the epilogue static scratch.

Strict serial/both-WG graph checks and independent fullH8192/I2048/E16/K10
FP32 BM128/BM256 checks passed with unchanged43 thresholds. Actual32-rank and
fullC4096 skew gate timing. The initial CPU build on215 observed foreign work
and returned75; no timing was accepted there and no foreign process was stopped.
Build/preflight repeated cleanly on219. Retain complete routed-layer FWD+remat
BWD,DX/DW/dScore andFP32 DW zero, H8192/I2048/E512/top10,T4096/C4096,ring65536,
MGroup/WGrad stage3,BF16. Only whole-host-clean four-node phases count.

| Phase | Backend | Warp reduction | Complete layer ms |
| --- | --- | --- | ---: |
| qwen38-wlf-ep32-iter44-warp-control-attempt0 | peer | 0 | 83.176367 |
| qwen38-wlf-ep32-iter44-warp-elastic-early | elastic | 0 | 105.676208 |
| qwen38-wlf-ep32-iter44-warp-fused-old | peer | 0 | 90.149237 |
| qwen38-wlf-ep32-iter44-warp-fused-warp | peer | 1 | 89.398389 |
| qwen38-wlf-ep32-iter44-warp-fused-warp-overlap | peer | 1 | 93.930542 |

Fastest accepted peer: {"name": "qwen38-wlf-ep32-iter44-warp-control-attempt0", "backend": "peer", "ms": 83.1763671875}

Fastest accepted elastic: {"name": "qwen38-wlf-ep32-iter44-warp-elastic-early", "backend": "elastic", "ms": 105.67620849609375}


## Iteration45 — per-CTA clocks to distinguish residency from kernel spans

The resumed24h campaign first corrected the prior DW causal attribution. Matched
immutable-wheel profiles, same4hosts, sameH8192/I2048/E512/top10,T4096/C4096,
G8,BM256,MGroup/WGradstage3 gave:

| Configuration | DWtotalms | GPUkernelunionms | WGgrid |
| --- | ---: | ---: | ---: |
| Fullserial |16.184150|83.539595|39|
| WG16/DGrad23serial |38.071336|117.683016|16|
| WG16/DGrad23parallel |38.217898|89.997003|16|
| Fullparallel |31.772179|87.855552|39|

Thus the old16.6->37.8 comparison mostly mixed CTA capacity with concurrency.
Matchedcappedparalleladds0.4%DWspan, whilefullgridparallelnearlydoublesDWspan.
This diagnosticadds an optionalInt64CUDAprobe to the dynamic BF16 WGrad API,
with a distinct kernel name/signature; defaultNone leaves the old call and
KGroupedGemmArgs GPUABI unchanged. It records native SDK2.2 __ppu_read_clock64
perCTAentry/exit, finalCTAbarrier, rejectsincorrectdtype/device/contiguity/alias
orcapacity. PerCTAelapsed includesstalls/preemptionafterentry and is notpureMMA
time. CrossCUclockvaluesareNOTcompared. A manualdiagnosticplan subclass passes
a separate caller-ownedbufferperWGcall and retainslifetimesonauxiliarystream.
FullQwenoutputs/gradientsarecheckedbitwiseto uninstrumentedplansbeforeprofiling.
Thisisdiagnostic-only, NOTanoptimizationwin. No change to numerical thresholds,
modelwork, strong baselineorproductionrecipe. Wholehostauditsgovernacceptance.

Closure records:
[]


### Iteration45 diagnostic follow-up

The firstdistributedlaunchwasrejectedbeforeGPUworkbecause219becameforeign;
switchedto220216217218. Retryr2failedbeforetheprofilerAPI: the newdiagnostic
harnessincorrectlyrequiredbitwiseFP32DWacrossseparatelyrebuiltatomicroute
permutations. Rank1usedTWOuninstrumentedPeerplansandfailedthesameassertion
(91.5% ofDW1elementsdiffered,maxabs7.08e-8). Thus this was an invalidcontrol,
not evidenceofprobe arithmeticerrors. The dedicatedsame-packed-layoutprobe
APItestkeepsbitwiseequality and is extendedtoBOTHfullQwenwgradtemplates.
The full-layerdiagnosticuses the PRE-EXISTING1e-5RMSgraph/ablationcontract;
no existing numericaltest, thresholdorindependentFP32oracle was changed.
Allfailedlogsarepreserved; no timingswere accepted fromr2.


### Iteration45 completed validation

SinglePPUidentical-packed-inputchecksPASSbitwiseforbothactualQwenDWtemplates,
plusdynamiclength/inputgraphandfiveinvalidprobe/aliascases. Retriedactual32
on220216217218afterthe218singlePPUpreflightfinished(r3gatecorrectlydeniedthe
occupiedhostbeforelaunch). Allfourr4profilephasespassedall32ranksand4wholehost
audits. Y/DX/dScorebitwisechecksremain; FP32DWusesexisting1e-5RMSbecauseatomic
routepackingchangesKsumorderbetweenseparateplans. Numericalthresholdsinall
existingtestsareunchanged. Runtime4b5e0c6aaf9203585925e73ea2e4fd9bb5a8814418a7f3ef58b3a8fe548261aa.

CTAclockaggregates(toalignwithAsyslastgraphnext;notactiveMMAcycles):
[
  {
    "phase": "qwen38-r2-cta45-profile-capped-parallel-r4",
    "dw1_max_cta_cycles": 32284770,
    "dw2_max_cta_cycles": 16220116,
    "cta_counts": [
      16
    ],
    "layer_rms": [
      0.0,
      0.0,
      3.543921138771111e-07,
      3.5421322763795615e-07,
      0.0
    ],
    "clean": true
  },
  {
    "phase": "qwen38-r2-cta45-profile-capped-serial-r4",
    "dw1_max_cta_cycles": 32606572,
    "dw2_max_cta_cycles": 16528126,
    "cta_counts": [
      16
    ],
    "layer_rms": [
      0.0,
      0.0,
      3.6304953709986876e-07,
      3.62731725545018e-07,
      0.0
    ],
    "clean": true
  },
  {
    "phase": "qwen38-r2-cta45-profile-full-parallel-r4",
    "dw1_max_cta_cycles": 14101880,
    "dw2_max_cta_cycles": 7004562,
    "cta_counts": [
      39
    ],
    "layer_rms": [
      0.0,
      0.0,
      3.4575123208924197e-07,
      3.4556069294922054e-07,
      0.0
    ],
    "clean": true
  },
  {
    "phase": "qwen38-r2-cta45-profile-full-serial-r4",
    "dw1_max_cta_cycles": 13716536,
    "dw2_max_cta_cycles": 6893144,
    "cta_counts": [
      39
    ],
    "layer_rms": [
      0.0,
      0.0,
      3.468078659807361e-07,
      3.4659564107641927e-07,
      0.0
    ],
    "clean": true
  }
]


## Iteration46 — optional dHidden score fusion removes replay FC2

Compute dScore from original BF16 hidden and unweighted FP32 dHidden accumulators.
Stage raw DY into the former packed-output buffer and BF16 weighted DY into the
existing packed-DY buffer for DW2, retaining full gradients. The dHidden
epilogue scales raw accumulators by each route score, rounds to BF16, applies
activation backward, and reduces score partials. No division, preserving
valid-zero, tiny and negative-score derivatives. Retained hidden is not
overwritten, supporting concurrent DW2. This deliberately reassociates BF16
scaling from DY to dHidden and must pass independent unchanged FP32 limits
and new training-quality validation before any production adoption.

The flag defaults false, uses a separate optional GPU sidecar and kernel
name, and preserves PeerGemmParams / PeerReplayGradients ABIs. Reject unsupported
path combinations. Existing full-reference tolerances and benchmark work are
unchanged; new dynamic graph test changes inputs/weights/routes and exercises
backward-only replay. Whole-host audit gates every performance result.

Timing records (partial and rejected phases retained in provenance):
[
  {
    "phase": "qwen38-r2-dhidden46-control",
    "clean": true,
    "backend": "peer",
    "ms": 83.21393432617188
  },
  {
    "phase": "qwen38-r2-dhidden46-elastic-early",
    "clean": true,
    "backend": "elastic",
    "ms": 105.7523193359375
  },
  {
    "phase": "qwen38-r2-dhidden46-fused",
    "clean": true,
    "backend": "peer",
    "ms": 79.90184326171875
  },
  {
    "phase": "qwen38-r2-dhidden46-fused-overlap",
    "clean": true,
    "backend": "peer",
    "ms": 84.60529174804688
  }
]

Preflights: native4df7f1fbd8f4 built clean218. Initial new harness bound the plan to a different stream (correctly rejected); fixed the harness to warm and capture on one stream. Initial full-shape preflight caught an incorrect score-partial width using H instead of I; corrected before timing. Unchanged production FP32 limits retained. Dedicated serial/both dynamic checks each passed all six patterns, including valid zero/tiny/negative scores, updated weights, and backward-only replay. FullH8192/I2048/E16/K10 single-device every-element references passed balanced+duplicate patterns, RMS0.0020–0.0038. Logs retained under work/qwen38-r2-dhidden46-*.log.


## Iteration47 — short output-tile WGrad launches with full CTA grids

The matched CTA clocks show far less in-CTA elapsed growth than kernel spans
under full-grid concurrency. Test short complete-K output-tile intervals,
letting dGrad/communication dispatch between WGrad launches without cutting
its persistent grid. DG_BF16_WGRAD_TILE_CHUNK0(default) retains the prior
launch; positive1..2^20 partitions only device-length WGrad output tiles.
A distinct _tile_range kernel gets a separate uint2 argument, preserving
KGroupedGemmArgs. Every tile has the original complete K accumulation order
and beta=1; no work, precision, gradient or route is omitted.
Reject negative/oversized chunks and multi-launch CTA-probe ambiguity before
output mutation. Static CPU-length APIs retain their original behavior.
The first CPU build on215 encountered foreign GPU work and was excluded by
its whole-node gate; switch to218 for clean build and PPU preflights. No
foreign task was stopped. Frozen iteration46 LM quality runs are isolated.

Complete H8192/I2048/E512/K10,T4096/C4096,ring65536,G8,BM256,MGroup/WGrad3.
Strong Elastic38 early-DY denominator and all-rank audits remain unchanged.
Raw dHidden score fusion46 is kept constant in all Peer cases. Serial chunks
control launch overhead versus concurrency. Timings including failed/partial
phases are retained in provenance; only clean complete steps count.

Timing records:
[]

Candidate blocked before distributed timing: new full-Qwen identical-input chunk reference failed bitwise comparison (3.2% elements, maxabs0.007013624); small shapes had passed. No speedup accepted. Preserve work/qwen38-r2-wgchunk47-preflight.log. Investigate implementation and test stream ordering after closing this attempt.


## Iteration48 — short output-tile WGrad launches with full CTA grids

The matched CTA clocks show far less in-CTA elapsed growth than kernel spans
under full-grid concurrency. Test short complete-K output-tile intervals,
letting dGrad/communication dispatch between WGrad launches without cutting
its persistent grid. DG_BF16_WGRAD_TILE_CHUNK0(default) retains the prior
launch; positive1..2^20 partitions only device-length WGrad output tiles.
A distinct _tile_range kernel gets a separate uint2 argument, preserving
KGroupedGemmArgs. Every tile has the original complete K accumulation order
and beta=1; no work, precision, gradient or route is omitted.
Reject negative/oversized chunks and multi-launch CTA-probe ambiguity before
output mutation. Static CPU-length APIs retain their original behavior.
The first CPU build on215 encountered foreign GPU work and was excluded by
its whole-node gate; switch to218 for clean build and PPU preflights. No
foreign task was stopped. Frozen iteration46 LM quality runs are isolated.

Complete H8192/I2048/E512/K10,T4096/C4096,ring65536,G8,BM256,MGroup/WGrad3.
Strong Elastic38 early-DY denominator and all-rank audits remain unchanged.
Raw dHidden score fusion46 is kept constant in all Peer cases. Serial chunks
control launch overhead versus concurrency. Timings including failed/partial
phases are retained in provenance; only clean complete steps count.

Timing records:
[
  {
    "phase": "qwen38-r2-wgchunk48-both-1024",
    "clean": true,
    "backend": "peer",
    "ms": 83.254248046875
  },
  {
    "phase": "qwen38-r2-wgchunk48-both-2048",
    "clean": true,
    "backend": "peer",
    "ms": 83.1755859375
  },
  {
    "phase": "qwen38-r2-wgchunk48-both-4096",
    "clean": true,
    "backend": "peer",
    "ms": 83.87572021484375
  },
  {
    "phase": "qwen38-r2-wgchunk48-both-512",
    "clean": true,
    "backend": "peer",
    "ms": 84.01217041015624
  },
  {
    "phase": "qwen38-r2-wgchunk48-control",
    "clean": true,
    "backend": "peer",
    "ms": 79.9606201171875
  },
  {
    "phase": "qwen38-r2-wgchunk48-elastic-early",
    "clean": true,
    "backend": "elastic",
    "ms": 105.69415283203125
  },
  {
    "phase": "qwen38-r2-wgchunk48-serial-2048",
    "clean": true,
    "backend": "peer",
    "ms": 79.92532348632812
  }
]

47 failed the newly authored reference test before any timing. The test wrote a/b/device lengths on the caller stream and immediately read them on another stream without a dependency. Add wait_stream before each reference read; preserve bitwise equality, all shapes, dynamic inputs and invalid-argument gates. Kernel implementation and native7f30de6b are unchanged, so no JIT name change is required for this harness correction.


## Iteration49 — direct original expert weights for dHidden/DX

Use the native PPU AIU Trans=true B operand already validated by the existing
KGrouped GEMM: conceptual [N,K], physical original [K,N] weights. A compile-time
WeightKN flag and distinct _weight_kn name leave PeerGemmParams ABI unchanged.
The appended default-false native/Python options retain previous behavior.
Peer dHidden/DX use original W2/W1, freeing unused transpose buffers and
avoiding the per-step transpose. Reject unsupported pipelines/DAG/FFN mixes
and include the flag in the all-rank recipe agreement. Test math and layout
against identical-input GEMMs plus unchanged full FP32/graph/distributed/skew
gates before timing. New kernels use unique names because the JIT cache key
does not hash included header contents.

Keep score fusion46, H8192/I2048/E512/K10,T4096/C4096,ring65536,G8,BM256,
MGroup/WGrad3, full FWD+BWD with FP32 DW zero/accumulation and strongest
Elastic38 early-DY baseline. No reduced compute or missing gradients.
48 full-grid chunks gave best concurrent83.176ms but serial79.925ms, so
retain tile_chunk0 as the default/control.46 three-seed2000-step public LM
full validation mean1.589567693 versusNative27 1.586408262(+0.199156%), all
canonical initial hashes and clean audits matched; not Qwen convergence.

Timings, with failed/partial phases retained in provenance:
[
  {
    "phase": "qwen38-r2-weight49-control",
    "clean": true,
    "backend": "peer",
    "ms": 79.96337280273437
  },
  {
    "phase": "qwen38-r2-weight49-direct",
    "clean": true,
    "backend": "peer",
    "ms": 77.261962890625
  },
  {
    "phase": "qwen38-r2-weight49-direct-overlap",
    "clean": true,
    "backend": "peer",
    "ms": 82.04461669921875
  },
  {
    "phase": "qwen38-r2-weight49-elastic-early",
    "clean": true,
    "backend": "elastic",
    "ms": 105.7501708984375
  }
]

First actual32 balanced/masked/skewed/reversed references passed. The following C4096-skew phase was interrupted by a whole-host gate75 on217; cancel only exact-tag owned partial ranks and switch to213/215/216/218. Preserve the rejected phase and rerun both references there before timings.


## Iteration50 — WGrad B shared-memory group offset, blocked before timing

Clean isolated stage3 WGrad1 counters (39x512,216registers,196608SMEM,
ZW-M890P/SDK2.2) show TSM load-group conflict ratio0.70 versus0.12 in the
local FC1 kernel. Hypothesis: shift B's shared-memory base by1024/2048/3072
bytes and enlarge the launch allocation by the same amount. PPU15's documented
linear mapping has four groups interleaved at1024bytes; active hashing may
differ, so only counters and timing can establish a benefit. Default0 retains
old generated code and positive offsets have unique kernel names. Preserve
KGroupedGemmArgs GPU ABI and all complete-K beta=1 arithmetic. Reject invalid
offsets and allocations beyond256KiB. No CU affinity or cross-CU TSM claim.

Native09d4fb2a built clean221, but the new identical-input bitwise preflight
failed HGCC JIT compilation on the unpadded control before any timing. No
performance or numerical win is accepted. Retain the failed source and log,
then repair the compile-time template error in the next attempt. The unchanged
best49 remains frozen, tested at77.261963ms versus105.750171ms Elastic.

Compiler diagnostics:

```text
/root/lfwang/Projects/t-head/worktrees/DeepGEMM-for-sail-mok-bf16-training-20261120/build/training-validation/jit-cache/tmp/767736-3dd691f6-0c26473c-2eb9a4f7/kernel.cu:22:17: error: expected unqualified-id
/root/lfwang/Projects/t-head/worktrees/DeepGEMM-for-sail-mok-bf16-training-20261120/build/training-validation/jit-cache/tmp/767736-3dd691f6-0c26473c-2eb9a4f7/kernel.cu:22:17: error: expected ')'
hgcc error: Exited with error code 1
RuntimeError: Assertion error (/root/lfwang/Projects/t-head/worktrees/DeepGEMM-for-sail-mok-bf16-training-20261120/csrc/apis/../jit_kernels/impls/../../jit/compiler.hpp:360): false and "HGCC compilation failed"
```


## Iteration51 — repair optional padding code generation and screen local WGrad

50 added a template suffix placeholder after the kernel signature, but put
its format argument before kernel_name. Move only that argument to the proper
position; unpadded generated code is exactly the pre50 control again. Native
28c2caa977bf built clean221. Positive padding has distinct kernel names and
unchanged complete-K FP32 accumulation order. Identical-input beta=1, dynamic
K lengths/graphs and all outputs must remain bitwise equal.

The local signal uses both actual Qwen WGrad shapes,E16,K2560 per group,
40960 active rows/65536 physical capacity, explicit stage3 LUT, pads0/1024/
2048/3072. Five warmups,20 graph iterations x5repeats, CUDA events on the
calling stream, mandatory FP32 output zero included identically. Every
output is checked bitwise before and after timing. This is a single-device
compute screening signal, not a distributed layer speedup or concurrency
measurement. Full-layer checks and strongest matched baseline remain required.

Whole-host-clean signal: True.

[
  {
    "case": "local_wgrad_bpad_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "bitwise_equal": true,
    "median_event_ms": 10.725743103027344,
    "samples": [
      10.739514923095703,
      10.694802856445312,
      10.726995086669922,
      10.689097595214843,
      10.725743103027344
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "28c2caa977bf5d9974e627876d47c12c50e748a4c9553dd50ccd348e7706c7cf",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_bpad_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 1024,
    "bitwise_equal": true,
    "median_event_ms": 10.843592834472656,
    "samples": [
      10.838162994384765,
      10.806593322753907,
      10.902048492431641,
      10.843592834472656,
      10.920320129394531
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "28c2caa977bf5d9974e627876d47c12c50e748a4c9553dd50ccd348e7706c7cf",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_bpad_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "bitwise_equal": true,
    "median_event_ms": 10.534927368164062,
    "samples": [
      10.501466369628906,
      10.534927368164062,
      10.575544738769532,
      10.566062927246094,
      10.53423843383789
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "28c2caa977bf5d9974e627876d47c12c50e748a4c9553dd50ccd348e7706c7cf",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_bpad_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 3072,
    "bitwise_equal": true,
    "median_event_ms": 10.6969482421875,
    "samples": [
      10.67537612915039,
      10.715386199951173,
      10.67709732055664,
      10.6969482421875,
      10.73496322631836
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "28c2caa977bf5d9974e627876d47c12c50e748a4c9553dd50ccd348e7706c7cf",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_bpad_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "bitwise_equal": true,
    "median_event_ms": 5.541792678833008,
    "samples": [
      5.572383880615234,
      5.541792678833008,
      5.541344451904297,
      5.53904914855957,
      5.556304931640625
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "28c2caa977bf5d9974e627876d47c12c50e748a4c9553dd50ccd348e7706c7cf",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_bpad_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 1024,
    "bitwise_equal": true,
    "median_event_ms": 5.6313018798828125,
    "samples": [
      5.662535095214844,
      5.614418411254883,
      5.6313018798828125,
      5.607024002075195,
      5.642911148071289
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "28c2caa977bf5d9974e627876d47c12c50e748a4c9553dd50ccd348e7706c7cf",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_bpad_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "bitwise_equal": true,
    "median_event_ms": 5.511712646484375,
    "samples": [
      5.514414978027344,
      5.552880477905274,
      5.486806488037109,
      5.511712646484375,
      5.506344604492187
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "28c2caa977bf5d9974e627876d47c12c50e748a4c9553dd50ccd348e7706c7cf",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_bpad_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 3072,
    "bitwise_equal": true,
    "median_event_ms": 5.555102920532226,
    "samples": [
      5.535223388671875,
      5.547711944580078,
      5.555102920532226,
      5.589346313476563,
      5.562659835815429
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "28c2caa977bf5d9974e627876d47c12c50e748a4c9553dd50ccd348e7706c7cf",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  }
]


## Iteration52 — stagger independent A/B TSM loads across physical warps

Accepted stage3 Asight Compute DW1 counters show elevated TSM load group
conflicts. Padding B alone (51) reduced local DW1+DW2+zero by only1.36%.
Add optional DG_BF16_WGRAD_TSM_LOAD_ORDER=1/2 to reverse A/B TSM request
order on alternating warps/warp pairs, retaining MMA ownership, every K
accumulation, barriers, beta=1 and output store order. A compiler memory
clobber separates independent operand reads. Default0 retains original
request order and generated shader string; positive orders use unique JIT
kernel names. GPU KGroupedGemmArgs ABI is unchanged.

Reference gates use identical packed inputs, nonzero initial FP32 gradients,
both full Qwen WGrad templates, changed input/K lengths, all-empty and graphs.
Local factorial pads0/2048 x orders0/1/2 uses five warmups,20 graph iterations
x5repeats and includes mandatory FP32 output zero identically. It is a local
screening signal, not a distributed or complete-layer speedup. Any candidate
needs full EP checks and a strongest matched compute baseline before adoption.

Whole-host-clean local signal: True.

[
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 0,
    "bitwise_equal": true,
    "median_event_ms": 10.515222930908203,
    "samples": [
      10.379389190673828,
      10.515222930908203,
      10.47003173828125,
      10.584101104736328,
      10.523377990722656
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 1,
    "bitwise_equal": true,
    "median_event_ms": 12.184712219238282,
    "samples": [
      12.18862533569336,
      12.196791839599609,
      12.184712219238282,
      12.177423858642578,
      12.170243835449218
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 2,
    "bitwise_equal": true,
    "median_event_ms": 12.123152160644532,
    "samples": [
      12.120340728759766,
      12.093617248535157,
      12.123152160644532,
      12.127861785888673,
      12.134794616699219
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 0,
    "bitwise_equal": true,
    "median_event_ms": 10.502095794677734,
    "samples": [
      10.612409210205078,
      10.502095794677734,
      10.430024719238281,
      10.560294342041015,
      10.491192626953126
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 1,
    "bitwise_equal": true,
    "median_event_ms": 11.700691223144531,
    "samples": [
      11.700691223144531,
      11.643817138671874,
      11.638277435302735,
      11.710547637939452,
      11.732985687255859
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 2,
    "bitwise_equal": true,
    "median_event_ms": 12.08740234375,
    "samples": [
      12.083716583251952,
      12.031949615478515,
      12.1749267578125,
      12.09560775756836,
      12.08740234375
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 0,
    "bitwise_equal": true,
    "median_event_ms": 5.446236419677734,
    "samples": [
      5.446236419677734,
      5.489290618896485,
      5.431325912475586,
      5.42365608215332,
      5.497438049316406
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 1,
    "bitwise_equal": true,
    "median_event_ms": 6.191566467285156,
    "samples": [
      6.201131057739258,
      6.1910655975341795,
      6.191717147827148,
      6.1867115020751955,
      6.191566467285156
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 2,
    "bitwise_equal": true,
    "median_event_ms": 6.151581954956055,
    "samples": [
      6.117538070678711,
      6.151581954956055,
      6.1365234375,
      6.159386444091797,
      6.169504928588867
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 0,
    "bitwise_equal": true,
    "median_event_ms": 5.5250701904296875,
    "samples": [
      5.509661483764648,
      5.5250701904296875,
      5.558026504516602,
      5.491828918457031,
      5.55901985168457
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 1,
    "bitwise_equal": true,
    "median_event_ms": 5.996044921875,
    "samples": [
      5.990584182739258,
      5.996044921875,
      5.995930099487305,
      6.012250518798828,
      5.998077011108398
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 2,
    "bitwise_equal": true,
    "median_event_ms": 6.168782043457031,
    "samples": [
      6.182635498046875,
      6.194488143920898,
      6.143127059936523,
      6.167647933959961,
      6.168782043457031
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "27864c71ead2d53a816796370c09e1b5f459b7de2f2ac8dd5539f894d05323c9",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  }
]


## Iteration53 — remove memory-clobber inline asm from staggered TSM loads

Stall reassessment after50/51/52: reviewed28 tile geometry,31 warp mapping,
42 resource-sharing controls, current51/52 local signal and accepted Acu
counters; reread PPU ISA verified facts and primary ACTLIZE/MoK sources.
No physical floor established. Neither padding nor source-level load
reordering has established a complete-layer win.

52 incorrectly described an empty memory-clobber asm as compiler-only.
SDK2.2 verified-facts documents a decoder-specific counterexample emitting
a hardware fence. Preserve52's results but do not attribute all regression
to operand order alone. CPU disassembly of the exact52 cache is collected
separately.53 removes only these two newly introduced asm statements.
Original pipeline barriers/fences remain unchanged; positive variants use
new _tsmorder_v2 kernel names to prevent stale-header cache reuse.
Default0 source request order and generated shader string are unchanged.

The unchanged reference gates require identical packed inputs, full-K
beta=1, both full Qwen templates, mutable input/K lengths and graphs.
The same local pad/order factorial,20x5 event samples and FP32 zero scope
are retained. No standalone result is a complete-layer speedup.

Whole-host-clean local signal: True.

[
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 0,
    "bitwise_equal": true,
    "median_event_ms": 10.510628509521485,
    "samples": [
      10.613648223876954,
      10.462553405761719,
      10.50616683959961,
      10.510628509521485,
      10.625038146972656
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 1,
    "bitwise_equal": true,
    "median_event_ms": 11.119340515136718,
    "samples": [
      11.0816650390625,
      11.139041900634766,
      11.074727630615234,
      11.119340515136718,
      11.143112182617188
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 2,
    "bitwise_equal": true,
    "median_event_ms": 11.191362762451172,
    "samples": [
      11.0983154296875,
      11.191362762451172,
      11.256623840332031,
      11.176551055908202,
      11.214768218994141
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 0,
    "bitwise_equal": true,
    "median_event_ms": 10.548089599609375,
    "samples": [
      10.548089599609375,
      10.58721466064453,
      10.510488128662109,
      10.467182159423828,
      10.548863983154297
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 1,
    "bitwise_equal": true,
    "median_event_ms": 11.136955261230469,
    "samples": [
      11.136955261230469,
      11.143759155273438,
      11.12403564453125,
      11.090891265869141,
      11.186321258544922
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw1",
    "groups": 16,
    "m": 4096,
    "n": 8192,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 2,
    "bitwise_equal": true,
    "median_event_ms": 11.215116119384765,
    "samples": [
      11.111787414550781,
      11.215116119384765,
      11.19534912109375,
      11.224751281738282,
      11.242208862304688
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 0,
    "bitwise_equal": true,
    "median_event_ms": 5.465765380859375,
    "samples": [
      5.48167839050293,
      5.515176391601562,
      5.427734375,
      5.447474670410156,
      5.465765380859375
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 1,
    "bitwise_equal": true,
    "median_event_ms": 5.717491531372071,
    "samples": [
      5.675785064697266,
      5.704736328125,
      5.722481536865234,
      5.74766731262207,
      5.717491531372071
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 0,
    "tsm_load_order": 2,
    "bitwise_equal": true,
    "median_event_ms": 5.7720893859863285,
    "samples": [
      5.793983078002929,
      5.7531780242919925,
      5.7720893859863285,
      5.73590202331543,
      5.779933166503906
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 0,
    "bitwise_equal": true,
    "median_event_ms": 5.540801620483398,
    "samples": [
      5.528287506103515,
      5.591686630249024,
      5.550641632080078,
      5.505049133300782,
      5.540801620483398
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 1,
    "bitwise_equal": true,
    "median_event_ms": 5.772444152832032,
    "samples": [
      5.791408538818359,
      5.759337997436523,
      5.788284683227539,
      5.772444152832032,
      5.769624328613281
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  },
  {
    "case": "local_wgrad_tsm_order_nofence_signal",
    "role": "dw2",
    "groups": 16,
    "m": 8192,
    "n": 2048,
    "capacity": 65536,
    "active_rows": 40960,
    "pad_bytes": 2048,
    "tsm_load_order": 2,
    "bitwise_equal": true,
    "median_event_ms": 5.813864517211914,
    "samples": [
      5.829595947265625,
      5.8563495635986325,
      5.811613845825195,
      5.807963943481445,
      5.813864517211914
    ],
    "warmup": 5,
    "iterations": 20,
    "repeats": 5,
    "extension_sha256": "6e39083eabd01c1b180ea4e4343e31ec713e15d20b4de0821615a69894a9de13",
    "scope": "single-device grouped WGrad plus mandatory FP32 output zero; no communication or complete-layer speedup"
  }
]


## Iteration54 — register-resident BF16 Tensor Cell throughput diagnostic

No production path changes. After small TSM scheduling changes failed to
improve local WG, measure independent register-resident BF16/FP32 MMA using
the actual production256x256/64x64 and two alternative warp geometries.
Reserve196608B shared as stage3 WG; no TSM or global operand supply in the
MMA mainloop. Fresh runtime scalar input pairs1/1,2/3,3/1 must produce the
exact independently calculated result at every output element. Poison all
outputs first and verify again after timing. Only exact representable FP32
integer sums are used, to validate all MMA iterations without tolerances.

HGGC-O3 ppu15 executable, five warmups,20 launches x5 event repeats on its
calling stream. Includes launch prologue and final stores. Record actual
register/local resources. This is a diagnostic of measured arithmetic
throughput, not a formal physical ceiling or achievable full-layer floor.
Production GEMMs still include AIU, TSM, nonlinearities and networking.
Original full-layer reference/quality/baseline requirements remain.

Clean diagnostic: True.

[
  {
    "case": "register_bf16_compute_diagnostic",
    "chip": "ZW-M890P",
    "active_cus": 39,
    "block_m": 256,
    "block_n": 256,
    "warp_m": 64,
    "warp_n": 64,
    "threads": 512,
    "registers": 144,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "steps": 4096,
    "every_output_exact": true,
    "fresh_seed_pairs": 3,
    "median_event_ms": 3.255326509,
    "observed_tflops": 411.6422093,
    "samples_ms": [
      3.255448818,
      3.255326509,
      3.255305052,
      3.255278111,
      3.255420685
    ],
    "scope": "Register-resident BF16/FP32 MMA plus launch prologue and result store; no AIU mainloop or networking; measured throughput is not an upper bound"
  },
  {
    "case": "register_bf16_compute_diagnostic",
    "chip": "ZW-M890P",
    "active_cus": 39,
    "block_m": 128,
    "block_n": 256,
    "warp_m": 64,
    "warp_n": 64,
    "threads": 256,
    "registers": 144,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "steps": 4096,
    "every_output_exact": true,
    "fresh_seed_pairs": 3,
    "median_event_ms": 1.661400437,
    "observed_tflops": 403.283208016,
    "samples_ms": [
      1.661400437,
      1.661334038,
      1.661414504,
      1.661476493,
      1.661341071
    ],
    "scope": "Register-resident BF16/FP32 MMA plus launch prologue and result store; no AIU mainloop or networking; measured throughput is not an upper bound"
  },
  {
    "case": "register_bf16_compute_diagnostic",
    "chip": "ZW-M890P",
    "active_cus": 39,
    "block_m": 256,
    "block_n": 256,
    "warp_m": 32,
    "warp_n": 128,
    "threads": 512,
    "registers": 144,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "steps": 4096,
    "every_output_exact": true,
    "fresh_seed_pairs": 3,
    "median_event_ms": 3.255348921,
    "observed_tflops": 411.639375362,
    "samples_ms": [
      3.255348921,
      3.25535059,
      3.25532341,
      3.255336285,
      3.255414486
    ],
    "scope": "Register-resident BF16/FP32 MMA plus launch prologue and result store; no AIU mainloop or networking; measured throughput is not an upper bound"
  }
]


## Iteration55 — finite in-CTA helper/consumer publication diagnostic

No production path changes yet. Validate the source-backed SDK2.2 builtin
__ppu_barrier_sync(id,count,fc31) before trying persistent FC1 helper warps.
A544/576-thread CTA splits into512 consumers and32/64 helpers. Helpers
write the next phase's disjoint global payload, each thread executes its
global release fence, then the entire team meets at named barrier1.
Consumers check every payload word and independently reuse TSM behind
512-participant barrier0 while helpers prepare the next phase. No kernel
depends on progress of another CTA/kernel. Default FC31 is preserved;
no undocumented flag interpretation or empty inline-asm clobber is used.

Three fresh tags,1/2/7/31 phases,1/39 CTAs and32/64 helpers yield48 cases.
All global words are checked by both GPU and independent CPU arithmetic,
with allocation canaries and zero status required.196608B dynamic shared
reservation mimics stage3, but low diagnostic register resources do not
prove production GEMM occupancy, performance, or multi-GPU correctness.
This finite proof is a gate to actual integrated kernel tests, not a win.

Clean reference: True.

[
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 1,
    "tag": 551197,
    "payload_words_checked": 4096,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 1,
    "tag": 551197,
    "payload_words_checked": 4096,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 1,
    "tag": 551197,
    "payload_words_checked": 159744,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 1,
    "tag": 551197,
    "payload_words_checked": 159744,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 2,
    "tag": 551197,
    "payload_words_checked": 8192,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 2,
    "tag": 551197,
    "payload_words_checked": 8192,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 2,
    "tag": 551197,
    "payload_words_checked": 319488,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 2,
    "tag": 551197,
    "payload_words_checked": 319488,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 7,
    "tag": 551197,
    "payload_words_checked": 28672,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 7,
    "tag": 551197,
    "payload_words_checked": 28672,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 7,
    "tag": 551197,
    "payload_words_checked": 1118208,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 7,
    "tag": 551197,
    "payload_words_checked": 1118208,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 31,
    "tag": 551197,
    "payload_words_checked": 126976,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 31,
    "tag": 551197,
    "payload_words_checked": 126976,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 31,
    "tag": 551197,
    "payload_words_checked": 4952064,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 31,
    "tag": 551197,
    "payload_words_checked": 4952064,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 1,
    "tag": 551229,
    "payload_words_checked": 4096,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 1,
    "tag": 551229,
    "payload_words_checked": 4096,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 1,
    "tag": 551229,
    "payload_words_checked": 159744,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 1,
    "tag": 551229,
    "payload_words_checked": 159744,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 2,
    "tag": 551229,
    "payload_words_checked": 8192,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 2,
    "tag": 551229,
    "payload_words_checked": 8192,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 2,
    "tag": 551229,
    "payload_words_checked": 319488,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 2,
    "tag": 551229,
    "payload_words_checked": 319488,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 7,
    "tag": 551229,
    "payload_words_checked": 28672,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 7,
    "tag": 551229,
    "payload_words_checked": 28672,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 7,
    "tag": 551229,
    "payload_words_checked": 1118208,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 7,
    "tag": 551229,
    "payload_words_checked": 1118208,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 31,
    "tag": 551229,
    "payload_words_checked": 126976,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 31,
    "tag": 551229,
    "payload_words_checked": 126976,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 31,
    "tag": 551229,
    "payload_words_checked": 4952064,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 31,
    "tag": 551229,
    "payload_words_checked": 4952064,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 1,
    "tag": 551263,
    "payload_words_checked": 4096,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 1,
    "tag": 551263,
    "payload_words_checked": 4096,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 1,
    "tag": 551263,
    "payload_words_checked": 159744,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 1,
    "tag": 551263,
    "payload_words_checked": 159744,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 2,
    "tag": 551263,
    "payload_words_checked": 8192,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 2,
    "tag": 551263,
    "payload_words_checked": 8192,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 2,
    "tag": 551263,
    "payload_words_checked": 319488,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 2,
    "tag": 551263,
    "payload_words_checked": 319488,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 7,
    "tag": 551263,
    "payload_words_checked": 28672,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 7,
    "tag": 551263,
    "payload_words_checked": 28672,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 7,
    "tag": 551263,
    "payload_words_checked": 1118208,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 7,
    "tag": 551263,
    "payload_words_checked": 1118208,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 31,
    "tag": 551263,
    "payload_words_checked": 126976,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 1,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 31,
    "tag": 551263,
    "payload_words_checked": 126976,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 32,
    "total_threads": 544,
    "phases": 31,
    "tag": 551263,
    "payload_words_checked": 4952064,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  },
  {
    "case": "ppu_subgroup_pipeline_reference",
    "blocks": 39,
    "compute_threads": 512,
    "helper_threads": 64,
    "total_threads": 576,
    "phases": 31,
    "tag": 551263,
    "payload_words_checked": 4952064,
    "registers": 32,
    "local_bytes": 0,
    "dynamic_shared_bytes": 196608,
    "gpu_and_cpu_reference_exact": true,
    "canaries_exact": true,
    "passed": true
  }
]


## Iteration56 — in-CTA FC1 helpers prefetch the next M tile

55 validated consumer-only and whole-team named barriers on this SDK.
Add an optional native fc1_helper_threads32/64 entry with unique kernel
name.32/64 helper threads gather next M tile's peer X into the existing
packed-X ring while256/512 Tensor Cell threads compute all N columns of
the previous M tile, including G8 SwiGLU. Handoff uses team barrier1;
consumers use builtin barrier0 with their own count and helper metadata
uses barrier2. Every helper retains its global release fence. No spin
wait depends on another CTA or kernel's admission. Shared/Params ABI is
unchanged; default0 remains the old block synchronization path.

Raw tests compare ALL packed-X/gate-up/hidden buffers bitwise with the
existing producer-first-N plus local-remaining-N implementation. Test both
BM128/256,32/64 helpers, keep/discard, changed source/weights/routes,
partial experts, worst skew, all-empty and graph replay. Small cases also
check an independent FP32 matrix/activation oracle without relaxing any
existing gate. Actual production register/local resources still require
inspection; the low-register55 test did not establish GEMM occupancy.

Local canonical Qwen signal usesE16,H8192,I2048,40960 active rows/65536
capacity,5warm20x5 event samples. Four source owners are emulated by local
arrays: this is a local FC1 screening signal, NOT an EP/network/full-layer
win. An actual32-rank full-reference/skew/full-step verdict is required.

Whole-host-clean local signal: False.

[]

Failure tail:
{"case": "idle_node_gate", "host": "e01-cn-2n550vu1t02-a0001", "selected_gpus": [0, 1, 2, 3, 4, 5, 6, 7], "allowed": true, "processes": []}
device_props.name:ZW-M890P
[PeerTraining_BF16:]
kernel:bf16_peer_rtrue_wfalse_n256_k256_bn256_bk64_wm64_wn64_s3_nslice0_1_producer_aiu_g8_swiglu_keep_copy_ilp8, vreg:232, local_size_bytes:0
groups:3, n:256, k:256, read_peer:1, write_peer:0, capacity_rows:1024
tile:[128,256,64], warp:[64,64], stages:3, threads:256, dynamic_smem_bytes:147456, blocks_per_cu:1, grid:39
[PeerTraining_BF16:]
kernel:bf16_peer_rtrue_wfalse_n256_k256_bn256_bk64_wm64_wn64_s3_input_helpers32_g8_swiglu_keep_copy_ilp8, vreg:216, local_size_bytes:0
groups:3, n:256, k:256, read_peer:1, write_peer:0, capacity_rows:1024
tile:[128,256,64], warp:[64,64], stages:3, threads:288, dynamic_smem_bytes:147456, blocks_per_cu:1, grid:39
Traceback (most recent call last):
  File "/root/lfwang/Projects/t-head/worktrees/DeepGEMM-for-sail-mok-bf16-training-20261120/tests/test_moe_input_helper_gemm.py", line 71, in <module>
    with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):both()
  File "/usr/local/lib/python3.12/site-packages/torch/cuda/graphs.py", line 257, in __enter__
    self.cuda_graph.capture_begin(
  File "/usr/local/lib/python3.12/site-packages/torch/cuda/graphs.py", line 115, in capture_begin
    super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)
RuntimeError: CUDA graphs must be captured on a non-default stream. (However, after capture, it's ok to replay them on the default stream.)
{"case": "idle_node_postflight", "host": "e01-cn-2n550vu1t02-a0001", "allowed": true, "processes": [], "observed_foreign_during_command": [], "verified_owned_processes": [{"pid": 1952984, "start_ticks": "63924260"}], "monitoring_errors": [], "monitor_interval_seconds": 2, "command_returncode": 1}


## Iteration57 — run FC1 helper graph gates on a supported nondefault stream

56 passed its first eager ALL-buffer bitwise comparison, with BM128 helper32
reporting216registers/zero local bytes versus232/zero for the old producer.
It then stopped before timing because the new harness explicitly captured
on CUDA's default stream. Preserve that failed attempt.57 changes ONLY
the harness: enter one explicit nondefault stream before input allocations,
metadata writes, eager references, graph capture/replays and event timing.
The stream first waits on the previous calling stream; no cross-stream
input/reference race is introduced. Production kernel and nativea59b8d8e
are unchanged from56. All original numerical/shape gates are retained.

55 validated consumer-only and whole-team named barriers on this SDK.
Add an optional native fc1_helper_threads32/64 entry with unique kernel
name.32/64 helper threads gather next M tile's peer X into the existing
packed-X ring while256/512 Tensor Cell threads compute all N columns of
the previous M tile, including G8 SwiGLU. Handoff uses team barrier1;
consumers use builtin barrier0 with their own count and helper metadata
uses barrier2. Every helper retains its global release fence. No spin
wait depends on another CTA or kernel's admission. Shared/Params ABI is
unchanged; default0 remains the old block synchronization path.

Raw tests compare ALL packed-X/gate-up/hidden buffers bitwise with the
existing producer-first-N plus local-remaining-N implementation. Test both
BM128/256,32/64 helpers, keep/discard, changed source/weights/routes,
partial experts, worst skew, all-empty and graph replay. Small cases also
check an independent FP32 matrix/activation oracle without relaxing any
existing gate. Actual production register/local resources still require
inspection; the low-register55 test did not establish GEMM occupancy.

Local canonical Qwen signal usesE16,H8192,I2048,40960 active rows/65536
capacity,5warm20x5 event samples. Four source owners are emulated by local
arrays: this is a local FC1 screening signal, NOT an EP/network/full-layer
win. An actual32-rank full-reference/skew/full-step verdict is required.

Whole-host-clean local signal: True.

[
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 128,
    "helpers": 0,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 14.539401245117187,
    "samples": [
      14.535475158691407,
      14.545388793945312,
      14.540341186523438,
      14.537950134277343,
      14.539401245117187
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 128,
    "helpers": 32,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 15.376962280273437,
    "samples": [
      15.376962280273437,
      15.502821350097657,
      15.387742614746093,
      15.37667236328125,
      15.322479248046875
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 128,
    "helpers": 64,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 13.961695861816406,
    "samples": [
      13.976528930664063,
      13.940188598632812,
      13.961695861816406,
      13.934919738769532,
      13.961842346191407
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 128,
    "helpers": 0,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 14.3793701171875,
    "samples": [
      14.384782409667968,
      14.380381774902343,
      14.375318908691407,
      14.37064208984375,
      14.3793701171875
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 128,
    "helpers": 32,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 16.056321716308595,
    "samples": [
      16.083554077148438,
      16.078201293945312,
      16.040647888183592,
      16.056321716308595,
      16.053140258789064
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 128,
    "helpers": 64,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 14.62542724609375,
    "samples": [
      14.600859069824219,
      14.653945922851562,
      14.589454650878906,
      14.62542724609375,
      14.630978393554688
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "helpers": 0,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 9.081780242919923,
    "samples": [
      9.086385345458984,
      9.079440307617187,
      9.082825469970704,
      9.081780242919923,
      9.077075958251953
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "helpers": 32,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 13.206838989257813,
    "samples": [
      13.236795043945312,
      13.208392333984374,
      13.206838989257813,
      13.193093872070312,
      13.152359008789062
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "helpers": 64,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 12.367601776123047,
    "samples": [
      12.366859436035156,
      12.367601776123047,
      12.367463684082031,
      12.385978698730469,
      12.385240173339843
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "helpers": 0,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 9.06277084350586,
    "samples": [
      9.062126159667969,
      9.064111328125,
      9.06451187133789,
      9.061769104003906,
      9.06277084350586
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "helpers": 32,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 13.064839172363282,
    "samples": [
      13.063455200195312,
      13.064839172363282,
      13.067764282226562,
      13.069068908691406,
      13.063284301757813
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_fc1_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "helpers": 64,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 12.415025329589843,
    "samples": [
      12.417955780029297,
      12.420437622070313,
      12.413756561279296,
      12.411897277832031,
      12.415025329589843
    ],
    "extension_sha256": "a59b8d8e314dddd1eab108a7ceb551bdc06c2b5cd9dc9825e9d09c74b773a71b",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  }
]


## Iteration58 — wider N tiles and smaller M compute tiles preserve route metadata

57 passed16 all-buffer bitwise references and12 local measurements but
BM256 helpers64 regressed9.08->12.37ms locally. Fresh paired Acu gates are
clean: ordinary remaining15Ntiles(3840cols) ComputeCU79.79%,7.788ms,
DRAM708.8GB/s,TSMconflict.12; helper16tiles(4096cols)54.00%,12.288ms,
DRAM1089.3GB/s,conflict.16,168regs/zero local. Inferred transferred DRAM
bytes rise5.52->13.38GB despite only6.7%more output columns AND additional
input staging. This is evidence of a locality/supply problem, not proof
that all regression is one specific cache mechanism.

Test BM128/BN512/S3 so each CTA reuses A over a wider N tile, with
WM64/WN64 or WM32/WN128 and32/64 helper threads. SMEM245760B plus helper
address scratch fits the device limit. Preserve the original256-row route
metadata and bounded rings: SelectedMTileScheduler optionally maps each
metadata block to two128-row compute blocks with correct group row bases,
valid tails and source positions. Default metadata geometry remains BM.
No Params ABI, accumulation order, synchronization primitive or precision
boundary changes. Host rejects unsupported subdivisions before GPU work.

Compare with the previous two-kernel BM256/BN256 producer/local control
on identical fresh data. Small FP32, fullQwen all-buffer BITWISE, mutable
routing/weight graphs, ragged experts, empty and full-skew are mandatory.
Local5warm20x5 remains a screening signal without real peer networking
or full routed-layer work. No EP/full-step winner is inferred.

Whole-host-clean local signal: True.

[
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 256,
    "block_n": 256,
    "warp": "64x64",
    "helpers": 0,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 9.074748992919922,
    "samples": [
      9.074748992919922,
      9.071237945556641,
      9.073707580566406,
      9.075790405273438,
      9.075820159912109
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 128,
    "block_n": 512,
    "warp": "64x64",
    "helpers": 32,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 11.256562805175781,
    "samples": [
      11.236678314208984,
      11.233284759521485,
      11.256562805175781,
      11.287483215332031,
      11.361042022705078
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 128,
    "block_n": 512,
    "warp": "64x64",
    "helpers": 64,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 10.863043212890625,
    "samples": [
      10.864515686035157,
      10.863374328613281,
      10.861647796630859,
      10.861298370361329,
      10.863043212890625
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 256,
    "block_n": 256,
    "warp": "64x64",
    "helpers": 0,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 9.040080261230468,
    "samples": [
      9.040080261230468,
      9.040476989746093,
      9.037613677978516,
      9.042330169677735,
      9.037637329101562
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 128,
    "block_n": 512,
    "warp": "64x64",
    "helpers": 32,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 11.194194030761718,
    "samples": [
      11.194194030761718,
      11.195843505859376,
      11.157728576660157,
      11.180686950683594,
      11.222759246826172
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 128,
    "block_n": 512,
    "warp": "64x64",
    "helpers": 64,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 10.820321655273437,
    "samples": [
      10.821986389160156,
      10.823141479492188,
      10.82029037475586,
      10.819991302490234,
      10.820321655273437
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 256,
    "block_n": 256,
    "warp": "32x128",
    "helpers": 0,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 9.109687805175781,
    "samples": [
      9.114808654785156,
      9.109197998046875,
      9.1100341796875,
      9.108560180664062,
      9.109687805175781
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 128,
    "block_n": 512,
    "warp": "32x128",
    "helpers": 32,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 28.375631713867186,
    "samples": [
      28.375631713867186,
      28.36962890625,
      28.373916625976562,
      28.385015869140624,
      28.383245849609374
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 128,
    "block_n": 512,
    "warp": "32x128",
    "helpers": 64,
    "keep_gate_up": true,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 27.437863159179688,
    "samples": [
      27.437863159179688,
      27.418011474609376,
      27.43993225097656,
      27.423483276367186,
      27.44508056640625
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 256,
    "block_n": 256,
    "warp": "32x128",
    "helpers": 0,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 9.069805908203126,
    "samples": [
      9.07144546508789,
      9.069805908203126,
      9.059544372558594,
      9.064885711669922,
      9.076593017578125
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 128,
    "block_n": 512,
    "warp": "32x128",
    "helpers": 32,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 12.594880676269531,
    "samples": [
      12.594880676269531,
      12.643661499023438,
      12.439649963378907,
      12.538642883300781,
      12.653858184814453
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  },
  {
    "case": "local_input_helper_wide_signal",
    "hidden": 8192,
    "intermediate": 2048,
    "experts": 16,
    "block_m": 256,
    "compute_block_m": 128,
    "block_n": 512,
    "warp": "32x128",
    "helpers": 64,
    "keep_gate_up": false,
    "capacity": 65536,
    "active_rows": 40960,
    "median_event_ms": 12.695458221435548,
    "samples": [
      12.727931213378906,
      12.565179443359375,
      12.637451934814454,
      12.728486633300781,
      12.695458221435548
    ],
    "extension_sha256": "d9460c6ee5eb75ad7a298274a6fa70ce6e7cca4c0aefa860d2f8077e4adcc260",
    "scope": "Single-device FC1 input staging+GEMM+SwiGLU; source owners emulated by local arrays; no EP/full-step speedup"
  }
]


## Iteration59 — integrate bounded in-CTA input helpers into bilateral EP32

Expose the already validated58 native input helper through PeerMoEPlan as an
opt-in recipe. Require the G8 producer, input/local AIU, bilateral bounded
ring and expert-local gradient recipe, reject unsupported pipelines, and
include helper geometry in the cross-rank recipe agreement. Default0 keeps
the existing best49 path. Override only FC1 geometry to BM128/BN512/S3
with64 helpers; metadata remains256 rows, other roles keep prior configs.
No precision, work scope, Params ABI, gradient or capacity reduction.

58 local all-buffer bitwise, small FP32 and mutable graph gates passed.
Locally the wider helper is still slower than original two-kernel FC1
(10.863 versus9.075ms), so real peer networking and full-step work decide
acceptance. Run actual EP32 balanced/masked/duplicates/reversed graphs and
full C4096 all-to-one expert skew, then same-host complete routed FWD+BWD
against helper0 best49 and tuned Elastic38 early-DY. H8192/I2048/E512/K10.
Failed or partial phase provenance is retained. No local-signal speedup
is treated as a distributed win.

[
  {
    "phase": "qwen38-r2-input59-control-r1",
    "clean": true,
    "backend": "peer",
    "ms": 76.94671630859375,
    "passed": null
  },
  {
    "phase": "qwen38-r2-input59-elastic-early-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 105.28189697265626,
    "passed": null
  },
  {
    "phase": "qwen38-r2-input59-reference-balanced-r1-attempt0",
    "clean": true,
    "backend": "peer",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-input59-reference-balanced-r1-attempt0",
    "clean": true,
    "backend": "peer",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-input59-reference-balanced-r1-attempt0",
    "clean": true,
    "backend": "peer",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-input59-reference-c4096-skew-r1",
    "clean": true,
    "backend": "peer",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-input59-wide64-r1",
    "clean": true,
    "backend": "peer",
    "ms": 76.88698120117188,
    "passed": null
  }
]


## Iteration60 — FP8 expert compute with DeepEP BF16 dispatch/combine

Add FP8 E4M3 grouped expert GEMMs to the existing DeepEP adapter, retaining
BF16 transport, row/128 activation scales,128x128 weight scales, BF16
intermediate and weight-gradient inputs and FP32 DW. This is not MXFP8
communication or the still-BF16-only Peer fusion path. Allow only the
DeepEP FP8 bounded workspace/DY prefetch recipe, retain Peer precision
guards and the FP8 BM128/split-layout contract. Quantize current weights
every forward and backward, inside timed/captured work, so in-place
updates cannot reuse stale initialization quantization. Forward/backward
activation epilogues hand prequantized data to the next GEMM.

Actual eight-rank gates compare all five complete outputs/gradients
against an independent CPU quantized STE oracle and the pre-existing
Native FP8 implementation. Graph replays mutate inputs, weights and routes;
cover balanced/ragged/zero-weight, skewed unique routes, empty and reversed
assignments with small bounded rings, both serial and early DY dispatch.
Do not infer an FP8 fusion speedup from this baseline implementation.

[
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0009797634556889534,
        "dw2": 0.0009211087599396706,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.000979763688519597,
        "dw2": 0.000921109109185636,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 2.95291613383597e-08,
      "dw2": 3.006816129413892e-08,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0010193610796704888,
        "dw2": 0.001108555938117206,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0010193618945777416,
        "dw2": 0.001108555938117206,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 1.1034011748733974e-07,
      "dw2": 1.1073542083295251e-07,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0009675679029896855,
        "dw2": 0.0010721925646066666,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0009675677865743637,
        "dw2": 0.0010721924481913447,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 3.207346566114211e-08,
      "dw2": 3.3279029310051556e-08,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.005749503616243601,
        "dx": 0.0035084006376564503,
        "dw1": 0.0008764806552790105,
        "dw2": 0.0008849724545143545,
        "dscores": 0.0037320167757570744
      },
      {
        "y": 0.005974738392978907,
        "dx": 0.003775127464905381,
        "dw1": 0.0008764804806560278,
        "dw2": 0.0008849722798913717,
        "dscores": 0.0037320272531360388
      }
    ],
    "pair_errors": {
      "y": 0.002666722983121872,
      "dx": 0.0021963652689009905,
      "dw1": 3.35638823401041e-08,
      "dw2": 3.5307351708979695e-08,
      "dscores": 7.563368598084708e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.006175451911985874,
        "dx": 0.004165369551628828,
        "dw1": 0.0009858265984803438,
        "dw2": 0.0009318398660980165,
        "dscores": 0.005884362384676933
      },
      {
        "y": 0.006306886672973633,
        "dx": 0.00441815285012126,
        "dw1": 0.0009858262492343783,
        "dw2": 0.0009318368975073099,
        "dscores": 0.00588435772806406
      }
    ],
    "pair_errors": {
      "y": 0.003043977078050375,
      "dx": 0.002435138914734125,
      "dw1": 1.1499844276841031e-07,
      "dw2": 1.1288880585880179e-07,
      "dscores": 5.820074733264846e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.005158334504812956,
        "dx": 0.004827829077839851,
        "dw1": 0.0011402064701542258,
        "dw2": 0.0011466400464996696,
        "dscores": 0.0035877262707799673
      },
      {
        "y": 0.005530246999114752,
        "dx": 0.004958080593496561,
        "dw1": 0.0011402067029848695,
        "dw2": 0.001146640395745635,
        "dscores": 0.003587728599086404
      }
    ],
    "pair_errors": {
      "y": 0.0029574413783848286,
      "dx": 0.002161348471418023,
      "dw1": 3.930743730506947e-08,
      "dw2": 3.8469757157599815e-08,
      "dscores": 6.60805952179544e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.005954194813966751,
        "dx": 0.004026231821626425,
        "dw1": 0.0010632252087816596,
        "dw2": 0.0009076944552361965,
        "dscores": 0.003801827784627676
      },
      {
        "y": 0.0062162079848349094,
        "dx": 0.004381174221634865,
        "dw1": 0.0010632254416123033,
        "dw2": 0.0009076945716515183,
        "dscores": 0.003801847342401743
      }
    ],
    "pair_errors": {
      "y": 0.0026962996926158667,
      "dx": 0.0024321351666003466,
      "dw1": 3.102983470171239e-08,
      "dw2": 3.1092476149297e-08,
      "dscores": 7.288414849426772e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.0037538371980190277,
        "dx": 0.004831607919186354,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.002434397581964731
      },
      {
        "y": 0.004216491244733334,
        "dx": 0.004983427003026009,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.002434408525004983
      }
    ],
    "pair_errors": {
      "y": 0.0028754351660609245,
      "dx": 0.002251222962513566,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 9.113195176269073e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.004053820855915546,
        "dx": 0.004405945539474487,
        "dw1": 0.0009274499025195837,
        "dw2": 0.001250903238542378,
        "dscores": 0.006841504015028477
      },
      {
        "y": 0.004463759250938892,
        "dx": 0.00460955360904336,
        "dw1": 0.0009274507756344974,
        "dw2": 0.0012509031221270561,
        "dscores": 0.006841497030109167
      }
    ],
    "pair_errors": {
      "y": 0.002858137944713235,
      "dx": 0.0021063294261693954,
      "dw1": 3.367806655774075e-08,
      "dw2": 3.38893997309242e-08,
      "dscores": 7.139234980968467e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.009609762579202652,
        "dx": 0.006750408560037613,
        "dw1": 0.0007893602014519274,
        "dw2": 0.001160333282314241,
        "dscores": 0.005733679514378309
      },
      {
        "y": 0.009827636182308197,
        "dx": 0.0069536990486085415,
        "dw1": 0.0007893601432442665,
        "dw2": 0.0011603337479755282,
        "dscores": 0.005733699072152376
      }
    ],
    "pair_errors": {
      "y": 0.0029052437748759985,
      "dx": 0.002529871417209506,
      "dw1": 3.0874264922431394e-08,
      "dw2": 3.196395681470676e-08,
      "dscores": 7.676641899934111e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.003784189000725746,
        "dx": 0.0036107900086790323,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.005372003652155399
      },
      {
        "y": 0.004212968982756138,
        "dx": 0.003919914364814758,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.005371996201574802
      }
    ],
    "pair_errors": {
      "y": 0.0027675738092511892,
      "dx": 0.00244304072111845,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 8.674193452407053e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.0048638577573001385,
        "dx": 0.004649926908314228,
        "dw1": 0.0008222328615374863,
        "dw2": 0.0014591347426176071,
        "dscores": 0.003594357054680586
      },
      {
        "y": 0.005147759336978197,
        "dx": 0.0048376936465501785,
        "dw1": 0.0008222322212532163,
        "dw2": 0.001459134859032929,
        "dscores": 0.003594369860365987
      }
    ],
    "pair_errors": {
      "y": 0.0026250656228512526,
      "dx": 0.0021549873054027557,
      "dw1": 3.923759805957161e-08,
      "dw2": 3.8225905996114307e-08,
      "dscores": 8.308526133760097e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.0024684143718332052,
        "dx": 0.004153745248913765,
        "dw1": 0.0014325251104310155,
        "dw2": 0.001141942455433309,
        "dscores": 0.003194952616468072
      },
      {
        "y": 0.0030542383901774883,
        "dx": 0.0043297428637743,
        "dw1": 0.0014325265074148774,
        "dw2": 0.0011419432703405619,
        "dscores": 0.0031949521508067846
      }
    ],
    "pair_errors": {
      "y": 0.0029683883767575026,
      "dx": 0.002253004815429449,
      "dw1": 3.708748508302051e-08,
      "dw2": 3.538066550845542e-08,
      "dscores": 7.875654262079479e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.006130004767328501,
        "dx": 0.003506612963974476,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0051892478950321674
      },
      {
        "y": 0.006387365981936455,
        "dx": 0.0038243767339736223,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.005189246963709593
      }
    ],
    "pair_errors": {
      "y": 0.002992002060636878,
      "dx": 0.0022191850002855062,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 7.548849367822186e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.015895957127213478,
        "dx": 0.004867431242018938,
        "dw1": 0.001439899206161499,
        "dw2": 0.0011206489289179444,
        "dscores": 0.003322869073599577
      },
      {
        "y": 0.016021139919757843,
        "dx": 0.005038414150476456,
        "dw1": 0.0014398988569155335,
        "dw2": 0.0011206490453332663,
        "dscores": 0.003322865581139922
      }
    ],
    "pair_errors": {
      "y": 0.002896495396271348,
      "dx": 0.002235159743577242,
      "dw1": 2.7744414765606962e-08,
      "dw2": 2.8398877915947196e-08,
      "dscores": 5.7554853327701494e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.007033649366348982,
        "dx": 0.005095875356346369,
        "dw1": 0.0010150743182748556,
        "dw2": 0.0010275128297507763,
        "dscores": 0.0030790595337748528
      },
      {
        "y": 0.007247815374284983,
        "dx": 0.005194628611207008,
        "dw1": 0.0010150742018595338,
        "dw2": 0.0010275129461660981,
        "dscores": 0.0030790455639362335
      }
    ],
    "pair_errors": {
      "y": 0.0027121263556182384,
      "dx": 0.001946851727552712,
      "dw1": 3.586537289379521e-08,
      "dw2": 3.663369696482732e-08,
      "dscores": 8.8006650855732e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.005969054996967316,
        "dx": 0.005827776622027159,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0046165152452886105
      },
      {
        "y": 0.00617145886644721,
        "dx": 0.006034857593476772,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.004616504535079002
      }
    ],
    "pair_errors": {
      "y": 0.002855906495824456,
      "dx": 0.002538139931857586,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 7.746922392470879e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.0039886911399662495,
        "dx": 0.00679578073322773,
        "dw1": 0.001060010981746018,
        "dw2": 0.0012960480526089668,
        "dscores": 0.005062390584498644
      },
      {
        "y": 0.004255945328623056,
        "dx": 0.00695143872871995,
        "dw1": 0.0010600112145766616,
        "dw2": 0.0012960477033630013,
        "dscores": 0.005062407348304987
      }
    ],
    "pair_errors": {
      "y": 0.0026494725607335567,
      "dx": 0.0020926492288708687,
      "dw1": 3.358111655415996e-08,
      "dw2": 3.4426651751573445e-08,
      "dscores": 7.440431204486231e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.0025499328039586544,
        "dx": 0.004391548223793507,
        "dw1": 0.0010562811512500048,
        "dw2": 0.0021391522604972124,
        "dscores": 0.0035956664942204952
      },
      {
        "y": 0.003086566459387541,
        "dx": 0.004504705313593149,
        "dw1": 0.001056281733326614,
        "dw2": 0.002139149233698845,
        "dscores": 0.003595672082155943
      }
    ],
    "pair_errors": {
      "y": 0.0029516217764467,
      "dx": 0.0021585258655250072,
      "dw1": 3.929251946033219e-08,
      "dw2": 4.024045097139606e-08,
      "dscores": 7.052420869513298e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.006652104202657938,
        "dx": 0.004580227192491293,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.002081364393234253
      },
      {
        "y": 0.006877524312585592,
        "dx": 0.004740349482744932,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0020813634619116783
      }
    ],
    "pair_errors": {
      "y": 0.002801918424665928,
      "dx": 0.002261708490550518,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 5.936544056339699e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.005441670771688223,
        "dx": 0.0067390454933047295,
        "dw1": 0.0009156723390333354,
        "dw2": 0.0011216806014999747,
        "dscores": 0.0040494888089597225
      },
      {
        "y": 0.0056888144463300705,
        "dx": 0.006831966806203127,
        "dw1": 0.0009156725136563182,
        "dw2": 0.0011216782731935382,
        "dscores": 0.004049481824040413
      }
    ],
    "pair_errors": {
      "y": 0.0028637133073061705,
      "dx": 0.002056804718449712,
      "dw1": 3.795823033669876e-08,
      "dw2": 3.6731577779391955e-08,
      "dscores": 7.336902996257777e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.006567209959030151,
        "dx": 0.004163537174463272,
        "dw1": 0.0009234387543983757,
        "dw2": 0.0008430402376689017,
        "dscores": 0.006260508671402931
      },
      {
        "y": 0.006847612094134092,
        "dx": 0.004313563462346792,
        "dw1": 0.0009234387543983757,
        "dw2": 0.0008430403540842235,
        "dscores": 0.006260516121983528
      }
    ],
    "pair_errors": {
      "y": 0.0028938527684658766,
      "dx": 0.002210951177403331,
      "dw1": 3.695343409049201e-08,
      "dw2": 3.737110887414019e-08,
      "dscores": 9.018554436579507e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.005932551808655262,
        "dx": 0.003840808058157563,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.004815223626792431
      },
      {
        "y": 0.0061243027448654175,
        "dx": 0.004071954637765884,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.004815234337002039
      }
    ],
    "pair_errors": {
      "y": 0.002710625994950533,
      "dx": 0.0023031525779515505,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 7.592100814690639e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-early-dy-r1",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.007777142804116011,
        "dx": 0.004247371573001146,
        "dw1": 0.0008610884542576969,
        "dw2": 0.0014045534189790487,
        "dscores": 0.00372659252025187
      },
      {
        "y": 0.00792275182902813,
        "dx": 0.0044506993144750595,
        "dw1": 0.000861087697558105,
        "dw2": 0.0014045517891645432,
        "dscores": 0.003726585302501917
      }
    ],
    "pair_errors": {
      "y": 0.002774151973426342,
      "dx": 0.0023274344857782125,
      "dw1": 3.324130304349637e-08,
      "dw2": 3.5564525546760706e-08,
      "dscores": 6.728679835532603e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0009797638049349189,
        "dw2": 0.0009211089345626533,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0009797639213502407,
        "dw2": 0.000921109109185636,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 3.023570727123115e-08,
      "dw2": 3.186150365763751e-08,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.001019362360239029,
        "dw2": 0.0011085578007623553,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0010193604975938797,
        "dw2": 0.001108555938117206,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 1.1199536942285704e-07,
      "dw2": 1.108632901036799e-07,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0009675673791207373,
        "dw2": 0.0010721926810219884,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0009675677865743637,
        "dw2": 0.0010721927974373102,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 3.580183971507722e-08,
      "dw2": 3.565961037566012e-08,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.005749503616243601,
        "dx": 0.0035084006376564503,
        "dw1": 0.0008764807716943324,
        "dw2": 0.0008849721634760499,
        "dscores": 0.0037320167757570744
      },
      {
        "y": 0.005974738392978907,
        "dx": 0.003775127464905381,
        "dw1": 0.0008764804224483669,
        "dw2": 0.0008849722798913717,
        "dscores": 0.0037320272531360388
      }
    ],
    "pair_errors": {
      "y": 0.002666722983121872,
      "dx": 0.0021963652689009905,
      "dw1": 2.645946217683104e-08,
      "dw2": 2.8338362767499348e-08,
      "dscores": 7.563368598084708e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.006175451911985874,
        "dx": 0.004165369551628828,
        "dw1": 0.000985829159617424,
        "dw2": 0.0009318387019447982,
        "dscores": 0.005884362384676933
      },
      {
        "y": 0.006306886672973633,
        "dx": 0.00441815285012126,
        "dw1": 0.0009858261328190565,
        "dw2": 0.00093183817807585,
        "dscores": 0.00588435772806406
      }
    ],
    "pair_errors": {
      "y": 0.003043977078050375,
      "dx": 0.002435138914734125,
      "dw1": 1.1532334553976398e-07,
      "dw2": 1.1473689198737702e-07,
      "dscores": 5.820074733264846e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.005158334504812956,
        "dx": 0.004827829077839851,
        "dw1": 0.0011402067029848695,
        "dw2": 0.0011466400464996696,
        "dscores": 0.0035877262707799673
      },
      {
        "y": 0.005530246999114752,
        "dx": 0.004958080593496561,
        "dw1": 0.0011402067029848695,
        "dw2": 0.0011466394644230604,
        "dscores": 0.003587728599086404
      }
    ],
    "pair_errors": {
      "y": 0.0029574413783848286,
      "dx": 0.002161348471418023,
      "dw1": 4.007369014402684e-08,
      "dw2": 4.180948209864255e-08,
      "dscores": 6.60805952179544e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.005954194813966751,
        "dx": 0.004026231821626425,
        "dw1": 0.0010632246267050505,
        "dw2": 0.000907694164197892,
        "dscores": 0.003801827784627676
      },
      {
        "y": 0.0062162079848349094,
        "dx": 0.004381174221634865,
        "dw1": 0.0010632250923663378,
        "dw2": 0.0009076945716515183,
        "dscores": 0.003801847342401743
      }
    ],
    "pair_errors": {
      "y": 0.0026962996926158667,
      "dx": 0.0024321351666003466,
      "dw1": 3.7428023347274575e-08,
      "dw2": 3.649754276580097e-08,
      "dscores": 7.288414849426772e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.0037538371980190277,
        "dx": 0.004831607919186354,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.002434397581964731
      },
      {
        "y": 0.004216491244733334,
        "dx": 0.004983427003026009,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.002434408525004983
      }
    ],
    "pair_errors": {
      "y": 0.0028754351660609245,
      "dx": 0.002251222962513566,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 9.113195176269073e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.004053820855915546,
        "dx": 0.004405945539474487,
        "dw1": 0.000927449727896601,
        "dw2": 0.0012509021908044815,
        "dscores": 0.006841504015028477
      },
      {
        "y": 0.004463759250938892,
        "dx": 0.00460955360904336,
        "dw1": 0.0009274499607272446,
        "dw2": 0.0012509034713730216,
        "dscores": 0.006841497030109167
      }
    ],
    "pair_errors": {
      "y": 0.002858137944713235,
      "dx": 0.0021063294261693954,
      "dw1": 3.4986825880878314e-08,
      "dw2": 3.5494952754788756e-08,
      "dscores": 7.139234980968467e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.009609762579202652,
        "dx": 0.006750408560037613,
        "dw1": 0.0007893603760749102,
        "dw2": 0.001160333980806172,
        "dscores": 0.005733679514378309
      },
      {
        "y": 0.009827636182308197,
        "dx": 0.0069536990486085415,
        "dw1": 0.0007893598522059619,
        "dw2": 0.0011603340972214937,
        "dscores": 0.005733699072152376
      }
    ],
    "pair_errors": {
      "y": 0.0029052437748759985,
      "dx": 0.002529871417209506,
      "dw1": 3.617047283910324e-08,
      "dw2": 3.6135748615606644e-08,
      "dscores": 7.676641899934111e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.003784189000725746,
        "dx": 0.0036107900086790323,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.005372003652155399
      },
      {
        "y": 0.004212968982756138,
        "dx": 0.003919914364814758,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.005371996201574802
      }
    ],
    "pair_errors": {
      "y": 0.0027675738092511892,
      "dx": 0.00244304072111845,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 8.674193452407053e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.0048638577573001385,
        "dx": 0.004649926908314228,
        "dw1": 0.0008222327451221645,
        "dw2": 0.001459134859032929,
        "dscores": 0.003594357054680586
      },
      {
        "y": 0.005147759336978197,
        "dx": 0.0048376936465501785,
        "dw1": 0.0008222322212532163,
        "dw2": 0.0014591329963877797,
        "dscores": 0.003594369860365987
      }
    ],
    "pair_errors": {
      "y": 0.0026250656228512526,
      "dx": 0.0021549873054027557,
      "dw1": 3.668089476605019e-08,
      "dw2": 3.738829690291823e-08,
      "dscores": 8.308526133760097e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.0024684143718332052,
        "dx": 0.004153745248913765,
        "dw1": 0.0014325273223221302,
        "dw2": 0.0011419435031712055,
        "dscores": 0.003194952616468072
      },
      {
        "y": 0.0030542383901774883,
        "dx": 0.0043297428637743,
        "dw1": 0.001432526740245521,
        "dw2": 0.0011419435031712055,
        "dscores": 0.0031949521508067846
      }
    ],
    "pair_errors": {
      "y": 0.0029683883767575026,
      "dx": 0.002253004815429449,
      "dw1": 3.516359114996703e-08,
      "dw2": 3.320906571957494e-08,
      "dscores": 7.875654262079479e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.006130004767328501,
        "dx": 0.003506612963974476,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0051892478950321674
      },
      {
        "y": 0.006387365981936455,
        "dx": 0.0038243767339736223,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.005189246963709593
      }
    ],
    "pair_errors": {
      "y": 0.002992002060636878,
      "dx": 0.0022191850002855062,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 7.548849367822186e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.015895957127213478,
        "dx": 0.004867431242018938,
        "dw1": 0.0014398981584236026,
        "dw2": 0.0011206488125026226,
        "dscores": 0.003322869073599577
      },
      {
        "y": 0.016021139919757843,
        "dx": 0.005038414150476456,
        "dw1": 0.001439897809177637,
        "dw2": 0.0011206484632566571,
        "dscores": 0.003322865581139922
      }
    ],
    "pair_errors": {
      "y": 0.002896495396271348,
      "dx": 0.002235159743577242,
      "dw1": 2.4218826055744103e-08,
      "dw2": 2.7092825760632877e-08,
      "dscores": 5.7554853327701494e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.007033649366348982,
        "dx": 0.005095875356346369,
        "dw1": 0.0010150743182748556,
        "dw2": 0.0010275131789967418,
        "dscores": 0.0030790595337748528
      },
      {
        "y": 0.007247815374284983,
        "dx": 0.005194628611207008,
        "dw1": 0.001015074667520821,
        "dw2": 0.0010275127133354545,
        "dscores": 0.0030790455639362335
      }
    ],
    "pair_errors": {
      "y": 0.0027121263556182384,
      "dx": 0.001946851727552712,
      "dw1": 3.818638205643765e-08,
      "dw2": 3.9189721690036095e-08,
      "dscores": 8.8006650855732e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.005969054996967316,
        "dx": 0.005827776622027159,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0046165152452886105
      },
      {
        "y": 0.00617145886644721,
        "dx": 0.006034857593476772,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.004616504535079002
      }
    ],
    "pair_errors": {
      "y": 0.002855906495824456,
      "dx": 0.002538139931857586,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 7.746922392470879e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.0039886911399662495,
        "dx": 0.00679578073322773,
        "dw1": 0.0010600099340081215,
        "dw2": 0.001296047237701714,
        "dscores": 0.005062390584498644
      },
      {
        "y": 0.004255945328623056,
        "dx": 0.00695143872871995,
        "dw1": 0.0010600107489153743,
        "dw2": 0.0012960480526089668,
        "dscores": 0.005062407348304987
      }
    ],
    "pair_errors": {
      "y": 0.0026494725607335567,
      "dx": 0.0020926492288708687,
      "dw1": 3.6455212182318064e-08,
      "dw2": 3.749319077428481e-08,
      "dscores": 7.440431204486231e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.0025499328039586544,
        "dx": 0.004391548223793507,
        "dw1": 0.0010562811512500048,
        "dw2": 0.0021391522604972124,
        "dscores": 0.0035956664942204952
      },
      {
        "y": 0.003086566459387541,
        "dx": 0.004504705313593149,
        "dw1": 0.0010562819661572576,
        "dw2": 0.0021391515620052814,
        "dscores": 0.003595672082155943
      }
    ],
    "pair_errors": {
      "y": 0.0029516217764467,
      "dx": 0.0021585258655250072,
      "dw1": 3.9375780858108556e-08,
      "dw2": 3.889908484211446e-08,
      "dscores": 7.052420869513298e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.006652104202657938,
        "dx": 0.004580227192491293,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.002081364393234253
      },
      {
        "y": 0.006877524312585592,
        "dx": 0.004740349482744932,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0020813634619116783
      }
    ],
    "pair_errors": {
      "y": 0.002801918424665928,
      "dx": 0.002261708490550518,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 5.936544056339699e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.005441670771688223,
        "dx": 0.0067390454933047295,
        "dw1": 0.0009156722226180136,
        "dw2": 0.0011216799030080438,
        "dscores": 0.0040494888089597225
      },
      {
        "y": 0.0056888144463300705,
        "dx": 0.006831966806203127,
        "dw1": 0.0009156723972409964,
        "dw2": 0.001121680485084653,
        "dscores": 0.004049481824040413
      }
    ],
    "pair_errors": {
      "y": 0.0028637133073061705,
      "dx": 0.002056804718449712,
      "dw1": 3.5285104615923046e-08,
      "dw2": 3.8510393096657936e-08,
      "dscores": 7.336902996257777e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "balanced",
    "oracle_errors": [
      {
        "y": 0.006567209959030151,
        "dx": 0.004163537174463272,
        "dw1": 0.0009234389872290194,
        "dw2": 0.0008430405287072062,
        "dscores": 0.006260508671402931
      },
      {
        "y": 0.006847612094134092,
        "dx": 0.004313563462346792,
        "dw1": 0.0009234391618520021,
        "dw2": 0.0008430402376689017,
        "dscores": 0.006260516121983528
      }
    ],
    "pair_errors": {
      "y": 0.0028938527684658766,
      "dx": 0.002210951177403331,
      "dw1": 3.735705078611318e-08,
      "dw2": 3.8580711958502434e-08,
      "dscores": 9.018554436579507e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "skewed_unique",
    "oracle_errors": [
      {
        "y": 0.005932551808655262,
        "dx": 0.003840808058157563,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.004815223626792431
      },
      {
        "y": 0.0061243027448654175,
        "dx": 0.004071954637765884,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.004815234337002039
      }
    ],
    "pair_errors": {
      "y": 0.002710625994950533,
      "dx": 0.0023031525779515505,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 7.592100814690639e-08
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "empty",
    "oracle_errors": [
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      },
      {
        "y": 0.0,
        "dx": 0.0,
        "dw1": 0.0,
        "dw2": 0.0,
        "dscores": 0.0
      }
    ],
    "pair_errors": {
      "y": 0.0,
      "dx": 0.0,
      "dw1": 0.0,
      "dw2": 0.0,
      "dscores": 0.0
    }
  },
  {
    "phase": "qwen38-r2-fp8compute60-reference-serial-r1-attempt0",
    "clean": false,
    "backend": null,
    "ms": null,
    "passed": true,
    "pattern": "reversed_changed_weights",
    "oracle_errors": [
      {
        "y": 0.007777142804116011,
        "dx": 0.004247371573001146,
        "dw1": 0.0008610881050117314,
        "dw2": 0.001404553884640336,
        "dscores": 0.00372659252025187
      },
      {
        "y": 0.00792275182902813,
        "dx": 0.0044506993144750595,
        "dw1": 0.0008610879303887486,
        "dw2": 0.0014045506250113249,
        "dscores": 0.003726585302501917
      }
    ],
    "pair_errors": {
      "y": 0.002774151973426342,
      "dx": 0.0023274344857782125,
      "dw1": 3.263541614728638e-08,
      "dw2": 3.359162903393553e-08,
      "dscores": 6.728679835532603e-08
    }
  }
]


## Iteration61 — Qwen FP8 DeepEP complete-step configuration verdict

60 passed64 every-element eight-rank mutable-graph records (serial and
early-DY), independent CPU E4M3 STE oracle maximum RMS.0160212, Native
transport pair maximum.00304398, all idle audits clean. This validates
FP8 compute/BF16 transport, not FP8 peer fusion or model convergence.

Extend the existing large Elastic FP32 reference with an explicit compute
flag and documented FP8.13/BF16.04 accuracy limits, preserving BF16's
previous default geometry. First exercise actual EP32 H8192/I2048/E512/K10
full outputs/gradients and mutable route graphs, with serial/early-DY.
Then same-host T4096/C4096/ring65536 complete FWD+rematerialized BWD,
DX,FP32 DW zero/accumulation and dScores. Measure tuned BF16 control,
FP8 default and BN128/256/512 BK128 S3 tiles. Quantization/transposes
stay inside timing; communication stays BF16. No mixed-precision ratio
is reported as a same-precision fusion speedup.

[
  {
    "phase": "qwen38-r2-fp8compute61-bf16-control-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 105.3615966796875,
    "passed": null
  },
  {
    "phase": "qwen38-r2-fp8compute61-bn128s3-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 618.85478515625,
    "passed": null
  },
  {
    "phase": "qwen38-r2-fp8compute61-bn256s3-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 1101.04658203125,
    "passed": null
  },
  {
    "phase": "qwen38-r2-fp8compute61-bn512s3-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 741.149951171875,
    "passed": null
  },
  {
    "phase": "qwen38-r2-fp8compute61-default-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 519.723828125,
    "passed": null
  },
  {
    "phase": "qwen38-r2-fp8compute61-reference-early-r1",
    "clean": true,
    "backend": "elastic",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8compute61-reference-early-r1",
    "clean": true,
    "backend": "elastic",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8compute61-reference-early-r1",
    "clean": true,
    "backend": "elastic",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8compute61-reference-serial-r1-attempt0",
    "clean": true,
    "backend": "elastic",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8compute61-reference-serial-r1-attempt0",
    "clean": true,
    "backend": "elastic",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8compute61-reference-serial-r1-attempt0",
    "clean": true,
    "backend": "elastic",
    "ms": null,
    "passed": true
  }
]


## Iteration62 — persistent active FP8 quantization and direct scale layout

61 exact rank0 three-graph profile spans519.875602ms: row quantization
243.934170, activation+quantization156.166241, weight quantization26.907956,
FP8 forward/input GEMM24.671306, BF16 DW15.429916, scale layout10.584852.
The first400.100411ms is about77% of the graph. Physical ring65536 covers
21 windows at EP32/T4096, but most windows are empty for balanced routing.
Unprofiled best FP8 default519.723828ms versus BF16105.361597; no FP8
speedup claimed. Source-backed scale helper copies a full SF matrix unless
its strides are(1,m), even when the GEMM has zero active rows.

Append three operations26/27/28: a bounded156-CTA grid reads GPU active
rows, validates capacity, and loops over only active warp groups. Reuse the
original9/14/15 arithmetic with compile-time MN scale indexing; old default
paths retain their scale arithmetic and existing Params ABI. Distinct positive
row/MN kernel names and both precompiled variants protect warmup. Optional
DeepEP flags allow separate active-only and active+MN-layout ablations.
Direct SF storage is physically[segments,capacity], viewed as[capacity,segments]
with strides(1,capacity), eliminating native layout copies. Default flags false.
Initialize cold-preparation row counts and dispatch activation warmup through
the plan method so the opted-in noncontiguous SF view is never passed to
the legacy contiguous quantizer API. No quantization recipe or precision
boundary changes; no token or gradient reduction.

Require all-buffer bitwise comparison with old quantizers, inactive canaries,
mutable row/input graph replay, counts0/1/17/tail/full and invalid negative/
over-capacity status with no writes. Test small and full Qwen ring shapes.
Isolated quantizer events are screening signals, not EP/full-layer speedups.

Whole-host-clean local signal: True.

[
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 8192,
    "old_operation": 9,
    "mn_major": false,
    "rows": 0,
    "variant": "full_capacity_control",
    "median_event_ms": 3.001832962036133,
    "samples": [
      3.001832962036133,
      3.0021400451660156,
      3.00177059173584,
      3.002101516723633,
      3.001813507080078
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 8192,
    "old_operation": 9,
    "mn_major": false,
    "rows": 0,
    "variant": "active_rows",
    "median_event_ms": 0.004498500004410743,
    "samples": [
      0.004620499908924103,
      0.004509500041604042,
      0.0044824998825788494,
      0.004475999996066094,
      0.004498500004410743
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 8192,
    "old_operation": 9,
    "mn_major": false,
    "rows": 40960,
    "variant": "full_capacity_control",
    "median_event_ms": 3.001811408996582,
    "samples": [
      3.0014930725097657,
      3.001811408996582,
      3.001470947265625,
      3.001910591125488,
      3.0019079208374024
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 8192,
    "old_operation": 9,
    "mn_major": false,
    "rows": 40960,
    "variant": "active_rows",
    "median_event_ms": 2.2939594268798826,
    "samples": [
      2.2939594268798826,
      2.293613052368164,
      2.292753982543945,
      2.2940305709838866,
      2.2940635681152344
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 8192,
    "old_operation": 9,
    "mn_major": true,
    "rows": 0,
    "variant": "full_capacity_control",
    "median_event_ms": 3.0021120071411134,
    "samples": [
      3.00179443359375,
      3.0021120071411134,
      3.001832962036133,
      3.0021520614624024,
      3.002161979675293
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 8192,
    "old_operation": 9,
    "mn_major": true,
    "rows": 0,
    "variant": "active_rows",
    "median_event_ms": 0.004453499987721443,
    "samples": [
      0.004453499987721443,
      0.004368999972939491,
      0.004475999996066094,
      0.004462499916553497,
      0.004351500049233436
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 8192,
    "old_operation": 9,
    "mn_major": true,
    "rows": 40960,
    "variant": "full_capacity_control",
    "median_event_ms": 3.0013425827026365,
    "samples": [
      3.001077079772949,
      3.0015470504760744,
      3.0010515213012696,
      3.001417541503906,
      3.0013425827026365
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 8192,
    "old_operation": 9,
    "mn_major": true,
    "rows": 40960,
    "variant": "active_rows",
    "median_event_ms": 2.6347335815429687,
    "samples": [
      2.6343870162963867,
      2.6360040664672852,
      2.633822441101074,
      2.6352054595947267,
      2.6347335815429687
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 14,
    "mn_major": false,
    "rows": 0,
    "variant": "full_capacity_control",
    "median_event_ms": 1.6221115112304687,
    "samples": [
      1.621728515625,
      1.6223539352416991,
      1.621983528137207,
      1.622126007080078,
      1.6221115112304687
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 14,
    "mn_major": false,
    "rows": 0,
    "variant": "active_rows",
    "median_event_ms": 0.004450000077486038,
    "samples": [
      0.004526999965310097,
      0.004450000077486038,
      0.004484000056982041,
      0.0043969999998807905,
      0.004411499947309494
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 14,
    "mn_major": false,
    "rows": 40960,
    "variant": "full_capacity_control",
    "median_event_ms": 1.622074508666992,
    "samples": [
      1.6217395782470703,
      1.622074508666992,
      1.6220384597778321,
      1.6220855712890625,
      1.6221139907836915
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 14,
    "mn_major": false,
    "rows": 40960,
    "variant": "active_rows",
    "median_event_ms": 1.33503999710083,
    "samples": [
      1.334689998626709,
      1.33503999710083,
      1.3349069595336913,
      1.3352785110473633,
      1.33518648147583
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 14,
    "mn_major": true,
    "rows": 0,
    "variant": "full_capacity_control",
    "median_event_ms": 1.622307586669922,
    "samples": [
      1.6218965530395508,
      1.6223464965820313,
      1.622307586669922,
      1.6223054885864259,
      1.6223129272460937
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 14,
    "mn_major": true,
    "rows": 0,
    "variant": "active_rows",
    "median_event_ms": 0.004473999887704849,
    "samples": [
      0.004473999887704849,
      0.00451049990952015,
      0.0044755000621080395,
      0.004407500103116036,
      0.004469000175595284
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 14,
    "mn_major": true,
    "rows": 40960,
    "variant": "full_capacity_control",
    "median_event_ms": 1.622085952758789,
    "samples": [
      1.6220365524291993,
      1.622085952758789,
      1.6221920013427735,
      1.6220844268798829,
      1.622185516357422
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 14,
    "mn_major": true,
    "rows": 40960,
    "variant": "active_rows",
    "median_event_ms": 1.2823499679565429,
    "samples": [
      1.2823499679565429,
      1.2820605278015136,
      1.2819464683532715,
      1.2830109596252441,
      1.2826279640197753
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 15,
    "mn_major": false,
    "rows": 0,
    "variant": "full_capacity_control",
    "median_event_ms": 2.485993576049805,
    "samples": [
      2.485760498046875,
      2.4857610702514648,
      2.485993576049805,
      2.4860464096069337,
      2.4861644744873046
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 15,
    "mn_major": false,
    "rows": 0,
    "variant": "active_rows",
    "median_event_ms": 0.004493999853730201,
    "samples": [
      0.004500000178813935,
      0.004491500183939933,
      0.004619999974966049,
      0.004446500167250633,
      0.004493999853730201
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 15,
    "mn_major": false,
    "rows": 40960,
    "variant": "full_capacity_control",
    "median_event_ms": 2.485722541809082,
    "samples": [
      2.485626983642578,
      2.4856164932250975,
      2.4858095169067385,
      2.485894966125488,
      2.485722541809082
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 15,
    "mn_major": false,
    "rows": 40960,
    "variant": "active_rows",
    "median_event_ms": 2.2038810729980467,
    "samples": [
      2.2037450790405275,
      2.2046749114990236,
      2.2045360565185548,
      2.2036890029907226,
      2.2038810729980467
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 15,
    "mn_major": true,
    "rows": 0,
    "variant": "full_capacity_control",
    "median_event_ms": 2.485866928100586,
    "samples": [
      2.485866928100586,
      2.486200523376465,
      2.4856950759887697,
      2.4858169555664062,
      2.4862285614013673
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 15,
    "mn_major": true,
    "rows": 0,
    "variant": "active_rows",
    "median_event_ms": 0.004456000030040741,
    "samples": [
      0.0044824998825788494,
      0.004452500119805336,
      0.004619999974966049,
      0.004441500082612038,
      0.004456000030040741
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 15,
    "mn_major": true,
    "rows": 40960,
    "variant": "full_capacity_control",
    "median_event_ms": 2.4859460830688476,
    "samples": [
      2.48584098815918,
      2.485960578918457,
      2.485726547241211,
      2.4859460830688476,
      2.486136054992676
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  },
  {
    "case": "active_fp8_quantization_signal",
    "capacity": 65536,
    "width": 2048,
    "old_operation": 15,
    "mn_major": true,
    "rows": 40960,
    "variant": "active_rows",
    "median_event_ms": 2.120042419433594,
    "samples": [
      2.1198034286499023,
      2.1196830749511717,
      2.121517562866211,
      2.120042419433594,
      2.1201820373535156
    ],
    "scope": "isolated quantizer; no EP/full-step speedup"
  }
]


## Iteration63 — active FP8 quantization integration and EP32 verdict

62 compiled clean with nativee412c5cabbd6f904af6d6f7d7bd5c951cfee9cbf88b92265f84ec6bb487bb070;
whole-host63 passed114 all-buffer bitwise/canary/mutable graph/invalid-count
references and24 isolated signals. Empty full-ring H8192 row quantization
3.002ms becomes.00445ms; active40960 row-major2.294ms, MN2.635ms.
These are isolated kernels, not routed-step gains.

Keep62 arithmetic unchanged. Extend full-skew reference with an explicit
compute flag and documented FP8.13/BF16.04 FP32 tolerances; permit FP8
only in DeepEP. Include the two formerly uncommitted cross64 reference
wiring changes now independently validated by frozen49 actual64 three
reference phases: hybrid192records,Elastic192records,and fullC4096skew64,
all eight-host audits clean. No production math changes in this iteration.

First actual8 independent quantized STE/native pair and mutable weight/route
graph checks for active-row and active+MN, with early-DY. Then actual32
Qwen H8192/I2048/E512/K10 full FP32 route graphs and dropless full C4096
all-to-expert0 reference, before same-host full routed FWD+rematerialized
BWD+FP32 DW/DX/dScore: unchanged FP8 control, active-only, active+MN,
and strong BF16 control. Quantization and transposes remain inside timing.
Any speedup here is FP8 baseline optimization, not an FP8 fusion speedup.

[
  {
    "phase": "qwen38-r2-fp8quant63-active-mn-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 121.88546142578124,
    "passed": null
  },
  {
    "phase": "qwen38-r2-fp8quant63-active-row-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 131.63558349609374,
    "passed": null
  },
  {
    "phase": "qwen38-r2-fp8quant63-bf16-control-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 105.27081298828125,
    "passed": null
  },
  {
    "phase": "qwen38-r2-fp8quant63-control-r1",
    "clean": true,
    "backend": "elastic",
    "ms": 519.701953125,
    "passed": null
  },
  {
    "phase": "qwen38-r2-fp8quant63-ep8-mn-r1",
    "clean": true,
    "backend": null,
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-ep8-mn-r1",
    "clean": true,
    "backend": null,
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-ep8-mn-r1",
    "clean": true,
    "backend": null,
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-ep8-mn-r1",
    "clean": true,
    "backend": null,
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-ep8-row-r1",
    "clean": true,
    "backend": null,
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-ep8-row-r1",
    "clean": true,
    "backend": null,
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-ep8-row-r1",
    "clean": true,
    "backend": null,
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-ep8-row-r1",
    "clean": true,
    "backend": null,
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-reference-c4096-skew-r1",
    "clean": true,
    "backend": "deepep",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-reference-qwen-r1-attempt0",
    "clean": true,
    "backend": "elastic",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-reference-qwen-r1-attempt0",
    "clean": true,
    "backend": "elastic",
    "ms": null,
    "passed": true
  },
  {
    "phase": "qwen38-r2-fp8quant63-reference-qwen-r1-attempt0",
    "clean": true,
    "backend": "elastic",
    "ms": null,
    "passed": true
  }
]
