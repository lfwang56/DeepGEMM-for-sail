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
