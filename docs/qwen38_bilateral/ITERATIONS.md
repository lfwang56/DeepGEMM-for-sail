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

