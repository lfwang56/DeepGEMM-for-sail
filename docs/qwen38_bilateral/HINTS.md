# Qwen3.8 bilateral MoE optimization requirements

User steering on 2026-11-21 prioritizes the bilateral fusion path
(`fuse_dispatch=True`, `fuse_return=True`) and the actual Qwen3.8-2.4T model
dimensions. The hybrid path is a comparison, not the optimization objective.

Pinned official configuration:
https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/912e40d903bfc891be8fe3c51faa0a09d182ce95/config.json

- H8192, routed expert intermediate2048, 512 routed experts, routed top10.
- Shared expert intermediate2048, one shared expert. Report its inclusion or
  exclusion explicitly; routed-layer timing is not complete-model timing.
- EP32 uses16 local routed experts; EP64 uses8. Keep512 total routed experts.
- Sweep1024/4096/8192 tokens per rank independently of model context capacity.
- Existing int32 element offsets constrain EP32 chunks to512 and EP64 to256
  for the worst-skew allocation. Do not silently drop tokens or expert routes.

The primary deployment comparison should use PPU dispatch/combine with native
grouped GEMM where available, following the type of baseline used by MoK.
The existing all-gather/reduce-scatter implementation is an auxiliary reference;
speedup against it is not a speedup against DeepEP or an industry-best baseline.
Distinguish complete implementations from same-kernel single-switch controls.

Use only currently idle authorized nodes, eight hosts per supernode. Switch
nodes after interruption; previously stopped nodes may be reused after fresh
idle checks, as explicitly authorized. Never kill foreign jobs. Wait for every
rank of a distributed phase before starting another phase on those hosts.

Keep full forward/backward including FP32 weight gradients and routing-score
gradients. Preserve precision boundaries, graph replay and independent reference
checks. Do not change benchmark work, timing streams, or correctness checks to
create gains. Record each candidate and failed attempt. No commit attribution,
model metadata or contribution tracking, per the user's AGENTS.md preference.

Reuse the established codex worktree and native distributed benchmark. Adapt the
AKO loop to the PPU toolchain: Asight Systems first, then selected Asight Compute
counters when useful. NVIDIA ncu is not a PPU profiler. The reference and input
generator live in existing tests; kernel edits remain in their package paths.

The user explicitly selected **wlf_sgl_2.2** in their latest container clarification.
Use this existing container for primary DeepEP/peer comparisons; record its
Torch2.11 runtime separately from the nativeTorch2.10 extension build. The
installed DeepEP2.2.0 Buffer path is functional and must be tuned as a real
baseline; the initial8-CU setting is superseded by the measured38-CU setting.
