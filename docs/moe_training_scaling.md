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
The overall 3.65x comparison to the current fixed-capacity collective includes
these scheduling improvements, and must not be described as isolated GEMM
communication fusion. A stronger collective pointwise baseline is planned.

Correctness: 15-method peer suite passed (two unrelated methods skipped),
including graph/input updates, masks, skew, empty inputs and four new control
combinations. Independent FP32 Torch full-element H7168/I2048/E8/T128 checks
passed for balanced/masked and duplicate/skewed routing; maximum relative RMS
0.004075. Logs: `scale-active-pointwise-unit.log`,
`scale-active-pointwise-full-reference.log`,
`scale-peer-ablation-large-ep32-c1024-active/`.
