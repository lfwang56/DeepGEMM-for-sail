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
