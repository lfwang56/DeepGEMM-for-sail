# MoE training on M890

`deep_gemm.training.native_moe` provides first-order autograd for routed SwiGLU
experts with BF16 inputs, BF16 or FP32 master weights, FP32 routing scores and
FP32 weight-gradient accumulation. It uses native PPU Tensor Cell GEMMs and a
reusable activation ring. Validation uses native PyTorch 2.10 and SAIL SDK 2.2.0
on M890 (PPU0015). CUDA-compatible API names do not imply NVIDIA instruction
support.

## Backends and topology

| Plan | Expert GEMMs | Communication | Core execution |
| --- | --- | --- | --- |
| `NativeMoEPlan` | BF16 or blockwise FP8 | PCCL all-gather/reduce-scatter, including cross-supernode RDMA | Several kernels per chunk |
| `DeepEPMoEPlan` | BF16 | Installed DeepEP Buffer dispatch/combine | Several kernels per chunk, with optional routed-row windows |
| `PeerMoEPlan` | BF16 | Direct peer reads in dispatch GEMMs and peer writes in return GEMMs | Separate metadata, GEMM and activation kernels |
| `PersistentMoEPlan` | BF16 | Direct peer reads and writes | One forward core kernel and one backward core kernel over all chunks |
| `CollectiveFusedMoEPlan` | BF16 | PCCL all-gather/reduce-scatter, including RDMA | Local GEMM epilogues write directly into the collective return ring |

Peer and persistent plans require an EP group inside one direct-access domain,
which can span several hosts in a supernode. Containers need the PPU devices
and the driver's IMEX channel for cross-host mappings. The plan honors the
runtime direct-access flag when present. Native PPU Torch 2.10 omits this flag;
its CUDA symmetric-memory backend requires successful driver import/mapping of
every peer buffer. Unknown backends cannot be treated as direct peer memory.

Native collective EP can span supernodes over PCCL/RDMA. It exchanges fixed
capacities, including zero-filled routes; it is not a sparse hierarchical RDMA
dispatcher. Select groups and chunk sizes for the actual topology. The peer
metadata's 64-rank construction limit is not a 64-rank benchmark claim.

`CollectiveFusedMoEPlan` retains that collective transport and fuses the local
return-ring scatter into GEMM. It needs no symmetric-memory mappings and does
not perform RDMA inside the GEMM. `fuse_return=False` retains a matched local
copy control. It accepts the same per-role `gemm_configs` as the peer plan.

Persistent execution is opt-in. Its core includes GPU routing, FC1, SwiGLU,
FC2, full backward replay and both wgrads. Weight casts/transposes, shared
experts, the router and optimizer remain outside that core. Benchmark all
applicable backends; fewer launches do not guarantee lower latency.

Peer plans expose `fuse_dispatch` and `fuse_return` switches, both `True` by
default. Disabling dispatch first packs peer input with a separate copy kernel;
disabling return scatters packed GEMM results with a separate peer-store kernel.
Both separate-copy controls visit current padded expert rows selected on the
GPU, rather than zero-filling the entire worst-skew ring capacity.
The routes, peer transport, GEMM tiles, activations and FP32 wgrads stay the same.
These switches support controlled fusion measurements and topology-specific
staging choices. Persistent plans always fuse both directions.

Use these names when reporting results:

| Path | Dispatch | Return | Topology |
| --- | --- | --- | --- |
| Collective baseline | PCCL all-gather, then local packing | Local scatter, then PCCL reduce-scatter | Within or across supernodes |
| Peer copy control (`False, False`) | Separate peer-copy kernel | Separate peer-store kernel | One direct-access domain |
| Peer hybrid (`False, True`) | Separate peer-copy kernel | GEMM epilogue stores to the source rank | One direct-access domain |
| Peer bilateral fusion (`True, True`) | GEMM reads from peer memory | GEMM epilogue stores to the source rank | One direct-access domain |
| Collective local fusion | PCCL all-gather, then local packing | GEMM writes the local return ring, then PCCL reduce-scatter | Within or across supernodes |

Here “hybrid” describes staging dispatch while fusing return. It does not mean
a hierarchical combination of within-domain direct access and cross-domain
RDMA. Such a sparse hierarchical transport is not implemented. Direct fusion
still has synchronization and separate routing/activation kernels. Neither peer
fusion switch means the whole MoE executes in a single kernel.

The current deployment performance baseline is tuned installed DeepEP
dispatch/combine through `DeepEPMoEPlan`, with native grouped GEMMs. The
PCCL all-gather/reduce-scatter `NativeMoEPlan` remains a transport reference.
A controlled fusion baseline uses the same peer GEMMs with a fusion switch
disabled. Independent Torch references verify numerical results; they are
not performance denominators. State the baseline and GEMM configuration
with every speedup: changing transport, scheduling and tiles measures a
whole implementation comparison rather than isolated overlap.

Peer plans default to GPU active-row pointwise work, fused local score/weighted
gradients and BF16x8 copies. `active_rows=False`, `fuse_local_grad=False` and
`vectorized_copy=False` retain benchmark controls. Both peer and collective-fused
plans accept `fuse_activation_backward=True` to put SwiGLU backward in the
dHidden epilogue; its default is `False`, since the measured EP16 gain was small.
See `docs/moe_training_scaling.md` for versioned baseline and controlled results.

Early smaller-shape M890 EP8/EP16/EP32 measurements favored staging input and
fusing the return. They do not establish the best path at Qwen3.8 dimensions.
The Qwen3.8 campaign instead prioritizes bilateral fusion with a producer
input stage; see `qwen38_bilateral/ITERATIONS.md` for matched DeepEP controls,
current shapes and experimental task pipelines. A measured GEMM configuration is
`gemm_configs={role: (256,64,64,64,2) for role in
("fc1","fc2","dhidden","dx")}`: `(block_N,block_K,warp_M,warp_N,stages)`;
block M was128 in those early measurements; the Qwen3.8 BF16 recipe uses256.
Sweep chunk size for the workload's memory budget. These
measurements do not establish a universal best tile or a sparse-RDMA speedup.

`gate_up_layout="g8"` selects persistent `[gate8,up8,...]` expert-weight
storage for BF16 native/DeepEP/direct-peer training. Convert canonical W1
once with `to_g8_weight`; convert checkpoint weights or DW1 back with
`from_g8_weight`. Optimizer parameters and gradients use the same G8 order,
so no layout conversion is hidden outside each timed training step.
`fuse_fc1_activation=True` optionally puts SwiGLU into a peer FC1 epilogue.
It rounds gate and up to BF16 before activation, and preserves those values
during backward replay. G8 supports the opt-in shared task pool as well;
it is not supported by the older persistent/whole-FFN paths.

A local GEMM kernel reads local tensors; that name does not imply that the
network is idle. Independent tile communication can overlap it. The current
best producer path retains some serial boundaries. The experimental shared
task pool pipelines ready dispatch/FC1/activation/FC2/return tasks; it uses
the same pipeline for forward replay in backward. Optional `dag_backward=True`
adds DY/score fetch, score-gradient publication, dHidden/activation backward,
DX and gradient return to a shared pool. It requires G8 routed-row windows
and can accompany producer forward independently. Full FP32 weight gradients
remain separate. `dag_direct_return=True` optionally embeds the return in
FC2/DX; those task spans include traffic and are not pure compute. Report full-step times
alongside activity overlap, since communication and computation contend for
CU, register and memory resources.

`input_copy_ilp=2`, `4` or `8` issues several independent BF16 vector reads
before storing a producer's packed input. It applies to producer input
staging and shared-pool forward input tasks; its default is1. It preserves
raw BF16 values, padding and publication fences. The backward DY/dScore dot
product retains its original floating-point order. Tune this with the actual
topology and payload: restricting input workers without enough independent
reads reduced throughput in the controlled Qwen3.8 measurements.

## Calling contract

```python
import torch
from deep_gemm.training import NativeMoEPlan, native_moe

device = torch.device("cuda", torch.cuda.current_device())
plan = NativeMoEPlan(
    hidden=1024, intermediate=512, local_experts=4, topk=2,
    max_tokens=4096, chunk_tokens=256, compute="bf16",
    device=device, group=None,  # or the initialized EP process group
)
x = torch.randn(1024, 1024, device=device).bfloat16().requires_grad_()
w1 = torch.randn(4, 1024, 1024, device=device, requires_grad=True)
w2 = torch.randn(4, 1024, 512, device=device, requires_grad=True)
ids = torch.randint(4, (1024, 2), device=device)
scores = torch.rand(1024, 2, device=device, requires_grad=True)
y = native_moe(x, ids, scores, w1, w2, plan=plan)
y.float().square().mean().backward()
```

- `x`: contiguous BF16 `[T,H]`; positive `H` divisible by 256.
- `w1`: contiguous `[E_local,2I,H]`; `w2`: contiguous `[E_local,H,I]`.
  Positive `I` divisible by 128. Weights are BF16 or FP32 masters.
- `ids`: contiguous int32/int64 `[T,K]`, with global expert IDs in
  `[0,EP_size*E_local)`. `-1` masks a slot; duplicates are allowed.
- `scores`: contiguous FP32 `[T,K]`. The caller supplies normalization.
  Scores apply once after FC2; zero scores retain their score derivative.
- Optional `valid_tokens`: one-element int32 device tensor selecting the leading
  valid input rows. Padded outputs/gradients are zero. Its value can change
  between graph replays within the captured input shape.
- All tensors and the plan use one device; inputs and scores are finite.
  Ranks can have different `T`, including zero, up to `max_tokens`.

Reused Tensor Cell paths use 32-bit element offsets. Each weight and ring matrix
must contain at most `2**31-1` elements; the constructor rejects larger capacities
before allocation. The collective nopad scheduler supports at most 1024 local
experts. Input/output token storage is separate from the fixed ring.

EP ranks construct matching dimensions, capacities, chunks, compute and transport,
then issue operations in matching order. Keep inputs, IDs, scores, weights and
valid-token tensors unchanged until their backward; autograd enforces saved
tensor version checks. Several pending forwards and `retain_graph=True` are
supported with matching rank order. Higher-order gradients and
`torch.compile(fullgraph=True)` are unsupported.

Shared experts, their gates, router logits, top-k normalization and balancing
loss remain ordinary differentiable PyTorch code; see `tests/train_moe_lm.py`.

## Workspace and recomputation

The activation ring holds one chunk. For chunk size `C`, EP size `P` and top-k
`K`, route capacity is `P*C*K` plus per-expert alignment padding. Counts,
offsets, permutations and GEMM/wgrad lengths are built on the device. Increasing
`max_tokens` or the number of sequential layers sharing a plan does not enlarge
this ring; actual inputs, outputs and saved input tensors still scale with shape.

Backward replays FC1, SwiGLU **and FC2** per chunk. It saves no full-batch
`[T,K,H]` expert-output tensor for score gradients. BF16 operands for the two
native FP32 wgrads are packed with zero padding to multiples of 128.

Collective plans default to `m_schedule="nopad"`: grouped GEMMs use GPU expert
lengths and skip unused trailing capacity. `"contiguous"` retains the original
capacity-wide schedule as a benchmark control. BF16 NoPad collective plans also
default to `active_pointwise=True` and `vectorized_routes=True`: packing,
activations and route transforms visit current padded expert rows and use
BF16x8 copies. The controls retain scalar/allocated-capacity work. FP8
quantization/pointwise kernels still cover their allocated capacity. Use the
optimized nopad baseline when attributing gains to communication fusion.

All plans default to `fast_weight_transpose=True`. A combined grouped transpose
prepares backward weights with BF16 bit preservation, including signed zero and
NaN payloads. It falls back to Torch copies for unaligned source pointers.
`fast_weight_transpose=False` retains the original Torch transpose-copy control;
both sides of a performance comparison should use the same setting.

Logical `workspace_bytes`/`activation_workspace_bytes` exclude caller tensors,
optimizer state, driver/VMM reservations and spills. Compare allocator peaks in
separate processes; sequential executable graphs can retain older allocations.

A plan serializes submissions on one execution stream. Share it across sequential
layers, or use independent native/peer plans for independent streams. Persistent
grids require every CTA to be resident: serialize persistent expert work on the
device. The default grid leaves capacity for framework communication. Advanced
`num_ctas`/`maximize_ctas=True` options are checked against both kernels' actual
occupancy; maximizing residency requires separate framework-overlap validation.

## Precision and gradients

BF16 GEMMs accumulate in FP32 and write BF16 intermediates. SwiGLU and its
derivative evaluate in FP32 before BF16 rounding. Combination and routing-score
derivatives accumulate in FP32. FP32 master weights receive FP32 wgrad.

`compute="fp8"` uses E4M3FN, per-row/per-128-channel activation scales and
per-128-by-128 weight scales. The FP32 scale is `max(amax,1e-4)/448`. This is
blockwise FP8, not Blackwell MXFP8; communication stays BF16. SwiGLU producers
fuse activation/backward and FP8 quantization, retaining the same BF16 values
needed by wgrad. Transposed dgrad weights have separately prepared scales.
Wgrad remains BF16-to-FP32. Quantization rounding is not differentiated: this is
an approximate training gradient.

`fuse_activation_quant=False` selects the equivalent separate BF16 activation
and row-quantization kernels for controlled measurement. The default is `True`.

`weight_grad_accumulators=(main_grad_w1,main_grad_w2)` optionally writes beta=1
into caller-owned contiguous FP32 buffers matching weight shapes. Zero them at
the optimizer boundary. Buffers must be disjoint from each other, weights,
other inputs and plan workspace. Disjoint slices of one flat gradient bucket
are supported. Expert `.grad` is absent in this mode; the caller
integrates optimizer/distributed hooks. Standard returned gradients are the path
tested with DDP/FSDP2.

Returned gradients follow the parameter dtype seen by autograd. For FSDP2,
`param_dtype=torch.bfloat16` rounds them at that framework boundary, even with
FP32 reduction. Keep FSDP parameters in FP32 when preserving the unrounded FP32
expert gradient is required; `native_moe` casts internally for BF16 compute.

Atomic packing can change reduction order; training is not bitwise deterministic.
Checkpoint checks require exact restored model/AdamW/batch-generator state and
compare subsequent drift with an independent uninterrupted repeat.

## Preparation and graph replay

Construct the plan outside capture. Warm forward/backward on its execution
stream first. `prepare(w1,w2)` compiles local kernels before payload collectives;
ranks agree on readiness before entering peer waits. Shared-cache publication
conflicts retain a valid temporary module and canonical in-process alias.

Use `check_errors=False` in the hot path, `reset_errors()` at the iteration
boundary, and `check()` after replay outside capture. Sticky flags are:
1 invalid ID, 2 capacity violation, 4 invalid valid-token count, 8 synchronization
timeout. Discard the plan after a timeout; clearing flags cannot restore phases.

Delete executable graphs and collect their Python references before destroying
the process group. Release captured outputs, gradients and plan buffers too.
Multi-host test harnesses synchronize each device, rendezvous through the CPU
store before destroying communicators, and retain the store through a final
shutdown handshake. This avoids starting communicator destruction while a
different host is still draining device work.
Full-model examples capture forward, loss,
backward, clipping and optimizer. The distributed example also reduces replicated
gradients and forms a global clipping norm. Its gathered final checkpoint is an
evaluation model; rank-local checkpoints include local AdamW state.

## Validation and measurement

Build the C++ extension in the native PPU environment; keep `hgobjdump` on `PATH`.

```bash
PYTHONPATH=. python tests/test_moe_training_kernels.py
PYTHONPATH=. python tests/test_native_moe_training.py
PYTHONPATH=. python tests/test_native_moe_training.py --fp8
PYTHONPATH=. python tests/test_native_moe_training.py --peer
PYTHONPATH=. python tests/test_native_moe_training.py --persistent
PYTHONPATH=. torchrun --standalone --nproc-per-node=8 \
  tests/test_native_moe_training.py --peer --ep --graph --stress
PYTHONPATH=. torchrun --standalone --nproc-per-node=4 \
  tests/test_moe_training_parallel.py --native-ring --mode fsdp2 --ep-size=2 --fp8
PYTHONPATH=. torchrun --standalone --nproc-per-node=8 \
  tests/benchmark_native_moe.py --backend all --maximize-ctas
```

`tests/launch_moe_local_ranks.py` supports multi-host runs with one log per rank.
Run one instance per host with a common master address/port and distinct node
ranks. It avoids an elastic agent's extra TCPStore exit barrier after test
shutdown. Distinguish same-host, same-supernode cross-host and cross-supernode
results. In the supplied cluster each supernode has eight hosts, each with eight
PPUs; these are different topology levels.

Benchmarks include forward, full backward replay and FP32 wgrad zero/accumulation;
optimizer and JIT are excluded. Output/gradient samples are compared with the
first backend. Sweep chunk size as well as GEMM tiles: small chunks bound memory
but repeat large FP32 wgrad updates. Report graph/eager timings separately.

Standard UT dump/replay covers reused eager grouped GEMMs, but cannot serialize
distributed peer-pointer plans or dynamic GPU-length wgrad into a host K-list.
Use dedicated training tests/external profiling for these paths. Disable
parameter dumping during capture.

For BF16 tile controls, `benchmark_native_moe.py --collective-gemm-config`
selects explicit existing nopad tiles. Pass the same five values to
`--gemm-config` for peer kernels; also compare both against the automatic nopad
baseline. `--separate-fp8-quantization` isolates FP8 activation/quantizer fusion.
`--peer-ablation` compares all four dispatch/return combinations;
`--fp8-ablation --backend collective --compute fp8` compares separate and fused
activation quantization on identical input tensors in one process.

```bash
PYTHONPATH=. python tests/train_moe_lm.py --backend native --compute fp8 \
  --graph --steps 2000 --checkpoint build/lm-fp8.pt
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  tests/train_moe_lm_distributed.py --compute bf16 --graph --reference-check \
  --steps 2000 --checkpoint build/lm-ep2.pt
PYTHONPATH=. python tests/evaluate_moe_lm.py --checkpoint build/lm-fp8.pt
PYTHONPATH=. python tests/train_moe_lm.py --backend peer --no-fuse-dispatch \
  --graph --steps 2000 --checkpoint build/lm-peer-hybrid.pt
```

The four-layer model uses pinned public TinyShakespeare, eight routed experts,
top-2 routing and a shared expert. Training monitors fixed validation samples;
full evaluation visits every held-out successor once with nonoverlapping causal
windows. BF16/FP8 can choose different experts in later layers. FP8's complete
gradient diagnostic uses an independent dequantized-input Torch STE reference
on the selected branches; training always recomputes dynamic routes. Dense Torch
computes all experts and is a convergence reference, not a sparse speed baseline.
Small-model convergence does not establish large-model pretraining convergence.

## Earlier reference API

`bf16_moe`, `bf16_moe_forward` and `bf16_moe_backward` remain correctness-oriented
APIs with native dense/grouped GEMMs or a float64 CPU Torch reference. This
legacy path reads route/layout data on the host, retains source expert outputs,
and rejects graph capture. Its `recompute=True` replays FC1/SwiGLU only. Its
timings/memory contract differ from ring plans. Tests are in
`tests/test_moe_training.py`; it is not a persistent megakernel.
