# Experimental PPU MoE training

`deep_gemm.training` provides a correctness-first routed MoE training backend
for M890. It uses DeepGEMM's native BF16/FP8 NT GEMMs, BF16 K-grouped weight-gradient
GEMM and either PCCL collectives or PPU symmetric peer memory. It does not require
NVIDIA multicast, TMEM or two-CTA cluster execution.

This initial backend is a sequence of kernels and communication operations. It
is not the inference MegaMoE kernel with an automatically generated backward,
and no training-throughput improvement is claimed.

## Autograd interface

```python
from deep_gemm.training import bf16_moe

# x: [tokens, H], BF16
# expert_ids: [tokens, topk], int32 or int64, global expert IDs
# scores: [tokens, topk], floating point
# w1: [local_experts, 2 * I, H], BF16, concatenated [gate | up]
# w2: [local_experts, H, I], BF16
y = bf16_moe(
    x, expert_ids, scores, w1, w2,
    group=expert_parallel_group,  # None for local experts without EP
    transport="symmetric",       # or "collective" for PCCL all-to-all
    recompute=True,
)
loss = loss_function(y)
loss.backward()
optimizer.step()
```

The caller owns all parameters, optimizer state and checkpoints. Shared experts
and optional shared-output gates remain external and can be added to `y` through
ordinary differentiable composition. Biases and other gated activations are not
implemented by this API.

Native inputs and weights must be contiguous BF16 device tensors; H must be a
multiple of 256 and I a multiple of 128. Rank r owns the consecutive expert IDs
`[r * local_experts, (r + 1) * local_experts)`. All ranks must agree on H, I,
local expert count, top-k, backend and transport. Per-rank token counts may differ,
including zero. Masked slots use expert ID -1. Valid zero routing scores retain
their score derivative; the implementation does not divide by routing scores.

The default autograd gradients have the corresponding parameter dtype. Only
first-order differentiation is supported. Higher-order differentiation is
explicitly disabled.

## Functional interface and FP32 accumulation

```python
from deep_gemm.training import bf16_moe_forward, bf16_moe_backward

y, context = bf16_moe_forward(
    x, expert_ids, scores, w1, w2,
    group=expert_parallel_group, transport="collective", recompute=True,
)
dx, dw1, dw2, dscores = bf16_moe_backward(dy, context)
# Native dw1 and dw2 are FP32. Alternatively, accumulate directly:
# dx, dw1, dw2, dscores = bf16_moe_backward(
#     dy, context, weight_grad_accumulators=(main_grad_w1, main_grad_w2))
```

The autograd wrapper also accepts `weight_grad_accumulators=(main_grad_w1,
main_grad_w2)`. Both must be contiguous, non-differentiable FP32 tensors matching
the weight shapes and devices. Every backward adds into them. The caller clears
them at the desired optimizer boundary and supplies the optimizer integration.
In this mode weight `.grad` is not populated. This option does not implicitly
implement Megatron-DDP or FSDP gradient hooks.

The functional backward accepts the same accumulators and returns aliases of
them. Native K-grouped GEMM uses these buffers as both C and D, without a separate
temporary weight-gradient allocation or subsequent add. Empty experts preserve
the existing accumulator contents.

The native wgrad primitive pads each expert's physical token count to a multiple
of 128 with zero rows. It consumes a host length list and matching device layout.
Empty experts produce zero new gradients. The BF16 data-gradient paths construct
contiguous transposed weights and use native NT kernels instead of unimplemented
NN aliases.

## Precision and gradient contract

FC1 produces BF16 gate/up values. SwiGLU is evaluated in FP32 and its result is
rounded to BF16 before FC2. FC2 produces BF16 expert outputs. Routing weights
are applied exactly once after FC2, and contributions are summed in FP32 before
the final BF16 output conversion.

Backward scales upstream expert gradients by the routing scores, computes BF16
data gradients, and accumulates weight gradients in FP32. Score gradients use a
dot product with the saved, unweighted expert output. These rounding boundaries
are explicit; bitwise equivalence with other MoE precision recipes is not claimed.

`backend="torch"` is an independent debugging/reference route. CPU float64 is
supported there for numerical gradcheck; it is not a native PPU performance path.

### Experimental E4M3 recipe

Select `backend="deep_gemm_fp8"` for FP8 forward and data-gradient GEMMs.
Parameters, communicated payloads, nonlinear activations and GEMM outputs remain
BF16; wgrad still uses BF16 operands and FP32 accumulation. Each GEMM dynamically
quantizes its LHS per row per 128 K elements and RHS per 128x128 block, using
E4M3FN values and FP32 scales (`max(amax, 1e-4) / 448`). Scales are not rounded to
UE8M0, and no delayed scaling history is maintained. Data-gradient weights are
transposed in BF16 first, then independently requantized in the required NT layout.
Quantized forward weight buffers cannot be reused as transposed dgrad weights.

Backward implements an approximate training gradient; it does not differentiate
through discrete FP8 rounding. Native tests check output/gradient error against
an FP64 MoE and check the scaling layout against an independent CPU quantization
model. These numerical checks do not establish full-model convergence.
Quantization currently uses separate tensor operations and is repeated at each
GEMM, so this recipe does not imply a throughput improvement.

### Grouped BF16 compute

With `backend="deep_gemm", grouped_gemm=True`, FC1, FC2 and both dgrad GEMMs
use native M-grouped contiguous GEMMs. A per-forward permutation and aligned
ragged expert layout are reused by forward, recompute, dgrad and K-grouped wgrad.
Packing uses a single indexed copy per operand. Empty experts occupy no rows;
other experts are zero-padded to the native M alignment, at least 128. This
avoids padding every expert to the largest expert's token count.

This option reuses the native grouped GEMM scheduler. Communication and activation
operations remain separate. It is opt-in and currently supports BF16 compute
only; combining it with the FP8 recipe is rejected.

## State, recomputation and communication

Every forward owns a separate context, including its route permutation and
returned expert outputs. Multiple forwards can remain pending before backward;
one forward does not overwrite another's context. Inputs, IDs, scores and weights
must not be changed before the matching backward, and version checks reject such
changes.

The autograd wrapper registers its tensor state with `save_for_backward`, so
normal backward releases saved activations even if the output remains alive.
Its transport cache is released after backward. Repeated backward with
`retain_graph=True` is supported when all EP ranks repeat the same sequence;
transport buffers are reacquired as needed. The functional API's context remains
caller-owned until the caller releases it.

With `recompute=True`, the context omits received expert inputs and gate/up/hidden
activations. Backward re-dispatches the original input and rebuilds only FC1 and
SwiGLU. It does not replay FC2 or combine. This first version still retains the
source-side expert outputs for score gradients; it does not implement MoK's
bounded activation ring or in-kernel ring replay.

`transport="collective"` uses the initialized EP group's all-to-all operations.
`transport="symmetric"` uses the same PyTorch PPU symmetric-memory allocation
and peer-pointer foundation as the existing PPU inference MegaMoE. Each sender
publishes into a separate source-rank slice at the destination. Explicit device
completion and rank barriers protect publication and buffer reuse. It requires
a peer-accessible ICN domain and is not a cross-supernode RDMA implementation.

All EP ranks, including empty ranks, must enter forward/backward collectives in
the same order. Executions sharing an EP group must be serialized. Route building
and wgrad layout preparation currently synchronize with the host. CUDA Graph
capture is rejected explicitly; `torch.compile(fullgraph=True)` and full-iteration
graph replay are not supported by this initial implementation.

## Validation

```bash
# CPU analytical/finite-difference checks (use the source package).
PYTHONPATH=. python tests/test_moe_training.py

# Native BF16 forward, backward, accumulation, mutation checks and recompute.
PYTHONPATH=. python tests/test_moe_training.py --native

# Grouped BF16 scheduler and experimental FP8 recipe.
PYTHONPATH=. python tests/test_moe_training.py --grouped
PYTHONPATH=. python tests/test_moe_training.py --fp8

# Two-rank EP, both transports, saved/recomputed contexts, skew and empty ranks.
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  tests/test_moe_training.py --native --ep

# Data-parallel and sharded-parameter optimizer integration.
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  tests/test_moe_training_parallel.py --mode ddp
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  tests/test_moe_training_parallel.py --mode fsdp2

# EP=2 and FSDP=2, with symmetric peer transport and grouped BF16 GEMMs.
PYTHONPATH=. torchrun --standalone --nproc-per-node=4 \
  tests/test_moe_training_parallel.py --mode fsdp2 --ep-size 2 \
  --transport symmetric --grouped

# Warm wall time includes host scheduling, communication, forward and backward.
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  tests/benchmark_moe_training.py --grouped --transport collective
```

Use a native SAIL PyTorch build compatible with this repository's native HGGC
headers. A CUDA-compatibility inference build can run tensors while exposing a
different C++ header ABI; mixing unconverted CUDA headers with HGGC headers is
not a supported build recipe.

Separate cold JIT compilation from a runtime hang. During initial investigation,
the generic BLAS BF16 autograd path remained inside HGRTC compilation past a
30-second timeout, then completed successfully when allowed to finish. Warm
execution completed normally. Native training tests should retain their JIT
cache and allow first-compilation time.

Further work includes GPU-resident route metadata, reusable bounded workspaces,
communication/GEMM fusion, training graph replay, fused quantization and full-model
convergence validation of the FP8 recipe. These are not implied by passing the
operator correctness checks.

## GPU schedules and bounded native replay

`NativeMoEPlan` and `native_moe` provide a second backend with native GPU route
count/prefix/packing, fused SwiGLU forward/backward and fused E4M3 quantizers.
Unlike the reference backend above, this path does not read route counts or K
lengths on the CPU. Its K-grouped wgrad API consumes a GPU length vector and a
static upper bound used only for tile selection.

```python
from deep_gemm.training import NativeMoEPlan, native_moe

plan = NativeMoEPlan(
    H, I, local_experts, topk, max_tokens,
    chunk_tokens=128, compute="bf16", group=ep_group, device=x.device,
)
y = native_moe(x, ids, scores, w1, w2, plan=plan)
y.backward(dy)
```

The native plan accepts BF16 inputs/weights and FP32 scores. IDs may be int32 or
int64. An optional int32 device scalar `valid_tokens` masks the padded tail of
the input without changing the captured shape. All EP ranks create matching
plans, but their actual input tensor token counts may differ up to `max_tokens`.

The activation ring holds one chunk and has a fixed capacity independent of
`max_tokens`. Each expert's physical rows are aligned separately; capacity covers
the worst-case route distribution plus per-expert alignment. Backward replays
FC1, SwiGLU **and FC2**, avoiding a full-batch saved `[T,topk,H]` tensor for score
gradients. Input/parameter snapshots still follow ordinary autograd lifetimes.
The same plan can be shared by sequential layers or pending forwards with matching
shapes because their expanded activations are reconstructed during backward.
One plan is restricted to one stream; separate streams require separate plans.

This path uses fixed-size all-gather/reduce-scatter communication per chunk. The
collectives work both within a peer domain and across domains through PCCL's
selected network transport; metadata and activation capacity never depend on a
host-side count exchange. It currently performs communication separately from
the GEMM kernels.

For graph capture, construct a separate plan and warm it on the stream that will
be captured. Capture forward, loss/backward and optimizer operations in the
normal PyTorch manner. Route metadata is rebuilt on every replay, including when
the routing distribution or `valid_tokens` changes. Call `plan.check()` outside
capture/replay to inspect device-side invalid-ID/capacity checks.

Executable graphs retain PCCL communicator resources. Before destroying a process
group, release its graph objects and collect them (`del graph; gc.collect()`).
Tests verify that numerical graph replay **and process-group shutdown** complete.

```bash
PYTHONPATH=. python tests/test_moe_training_kernels.py
PYTHONPATH=. python tests/test_native_moe_training.py
PYTHONPATH=. python tests/test_native_moe_training.py --fp8
PYTHONPATH=. torchrun --standalone --nproc-per-node=2 \
  tests/test_native_moe_training.py --ep --graph
```

On the supplied topology, each eight listed hosts form one supernode. Same-host,
same-supernode cross-host, and cross-supernode evidence must be recorded
separately. The same-supernode cross-host PCCL test used `P2P/IPC pointer
inter-node`; the cross-supernode test used NET/IB. Containers need RDMA devices
for the latter and `/dev/alixpu-caps-imex-channels/channel0` for the inter-host
fabric mapping used by the former. A missing IMEX device must not be diagnosed
as a kernel arithmetic failure.
