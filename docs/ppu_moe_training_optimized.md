# Selected PPU MoE training implementation

This snapshot packages the fastest validated BF16 and FP8 paths from the training campaign. It contains the measured source321 implementation, with portable EP32 and EP64 recipes and the exact DW LUT files.

## Performance and scope

Target: ZW-M890P / ppu0015, PPU SDK 2.2.0-663c22, PyTorch 2.11.0. Each host has eight devices; each physical supernode has eight hosts. EP32 uses four hosts and EP64 uses all eight hosts. Both require direct peer access inside one physical supernode.

| Shape | Tokens/device | Precision | Selected FWD+BWD | Tuned DeepEP dispatch/combine | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| MoK H4096/I1024 | 4096 | BF16 | 20.12 ms | 30.89 ms | 1.54x |
| MoK H4096/I1024 | 4096 | FP8 | 19.68 ms | 29.89 ms | 1.52x |
| Qwen H8192/I2048 | 16384 | BF16 | 205.89 ms | 326.73 ms | 1.59x |
| Qwen H8192/I2048 | 16384 | FP8 | 194.54 ms | 292.67 ms | 1.50x |

All rows use EP32, 512 experts, top10 and balanced routing. Fresh FP8 repeats on a second idle four-host cohort measured 19.6866 ms and 194.7472 ms. The max-rank GPU-event median includes current-weight preparation, forward, full backward, dX, dScores and fresh FP32 weight-gradient initialization/accumulation. Optimizer updates, router/shared experts and the rest of a complete model are outside this performance scope. These are model-dimension benchmarks.

**FP8 precision in this snapshot:** forward and input-gradient matrix operations use FP8; DW retains BF16 operands with FP32 accumulation and output. Direct peer payloads remain BF16. The later FP8-DW experiments have not improved complete-layer performance and are outside this selected snapshot. The measured gain remains below the 2–3x target.

## Implementation

The BF16 path fuses peer input/return with GEMM, retains the gradient producer and avoids clearing routes that the final combine masks. It uses G8 gate/up weights and fused activation/score handling.

The FP8 path pipelines expert-batch dispatch and quantization against native GEMM in both the forward and gradient directions. Source producers prefetch two independent peer vectors before consuming them. Return GEMMs write directly to peers. MoK selects forward/gradient expert batches 8/4; Qwen selects 2/1.

Fresh DW outputs use a first-nonempty beta0 write, including when initial windows or experts are empty. GPU seen flags preserve subsequent beta1 accumulation; only never-seen experts need final zeroing. Caller-provided accumulators keep ordinary beta1 behavior. Multiple retained chunks may use a bounded segmented DW schedule with FP32 accumulation. The independent FP32 gate limits the continuous-K bound to 10240 padded rows/expert. Concurrent DW/dGrad and the optional DW/source schedule are disabled in the selected recipes.

## Build and reproduce

Build the project and its pinned dependencies in the PPU environment as described in the main README. Rebuild the extension before using these training APIs. The preserved measured package had extension SHA256 `5a33f2fccd6a5fedf052f590b447463219dc42f43efb039a6bad19df33b51d30`; rebuilding may change the binary fingerprint.

The [recipe file](ppu_moe_training_recipes.json) contains all eight complete environments, benchmark arguments, original measurement identities and portable LUT checksums. It has no campaign-specific library, cache or host paths. The launcher verifies the selected LUT before importing Torch.

Inspect a configuration without initializing a device:

```bash
python tests/benchmark_moe_training_best.py --model mok --precision fp8 --dry-run
```

On each of four hosts in the same physical supernode, supply the common master address and the host's rank (0–3):

```bash
torchrun --nnodes=4 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
  tests/benchmark_moe_training_best.py --model mok --precision fp8
```

Use `--model qwen` for H8192/I2048 and `--precision bf16` for the BF16 path. Defaults preserve five warmups, twenty iterations and five repeats. `--profile` instruments the timed interval; compare performance with unprofiled runs. Use a new process when changing a LUT, because the runtime caches the first loaded table. Existing int32 element-capacity checks bound ring sizes.

## Validation

The code was validated with independent FP32 references, mutable graph replay, fresh and external FP32 accumulators, empty experts/initial windows, routed capacity/skew checks, current-weight changes and distributed source/interference audits. `tests/test_training_segmented_wgrad.py` covers the bounded DW path, including first writes over NaNs, guards and nonzero accumulators. Existing context-generation and vector-combine tests remain applicable.

The selected FP8 recipe also completed a 57.8M four-layer public-text MoE LM (H512/I1024/E8/top2), three matched EP2 seeds and 2000 optimizer updates. All 111539 held-out targets per checkpoint were evaluated with a common original-c6 Native FP8 evaluator. The mean validation cross entropy was 1.59061243 nats. This validates a complete small model; it does not establish statistical noninferiority or convergence of a 2.4T model.

## EP64 reproduction

Use eight hosts in the **same physical supernode**, with eight devices per host, and host ranks0–7. EP64 keeps512 total experts and uses8 local experts/device; EP32 uses16. The default launcher remains EP32, so explicitly pass `--ep 64`.

```bash
python tests/benchmark_moe_training_best.py --ep 64 --model qwen --precision fp8 --dry-run

torchrun --nnodes=8 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port=29500 \
  tests/benchmark_moe_training_best.py --ep 64 --model qwen --precision fp8
```

Select the other three EP64 cases with `--model mok|qwen --precision bf16|fp8`. Do not put hosts from different supernodes in this direct-peer configuration. The launcher checks WORLD_SIZE=64 before importing Torch for a real run; it cannot infer the physical network domain from a host address.

| EP64 recipe | Tokens/device | Chunk tokens | Ring rows | FP8 expert batches FWD/grad | Historical FWD+BWD | DeepEP | Speedup | Measured source |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| MoK BF16 | 4096 | 4096 | 520192 | disabled | 20.3970 ms | 35.2424 ms | 1.73x | 283 |
| MoK FP8 | 4096 | 4096 | 520192 | 4/2 | 19.8898 ms | 34.5029 ms | 1.73x | 283 |
| Qwen BF16 | 16384 | 16384 | 260096 | disabled | 213.8715 ms | 364.0971 ms | 1.70x | 291 |
| Qwen FP8 | 16384 | **3840** | 260096 | 4/2 | 204.4079 ms | 332.9872 ms | 1.63x | 291 |

EP64 uses source ILP1, while the selected EP32 FP8 recipes use ILP2. Qwen EP64 also retains the original allocator setting and exact Qwen DW LUT. These parameters are copied from the original283/291 frozen-package runs. The JSON stores each original phase, extension SHA256 and `current_snapshot_ep_revalidated=false`; dry-run output includes this provenance. Running this commit uses source321 with those parameters. **The EP64 table is historical evidence; source321 EP64 has not been retimed.** No new EP64 performance measurement is claimed by this configuration-only update.

This commit is based on the previously validated training baseline `c6b0c5ec1de495e5867b1a75d3ba36da881d1cd6`. Its optimized implementation is the measured `cd69ba8f65fb28480100728fce18f1644496346c` source snapshot; the packaging does not select the subsequent experimental FP8 DW paths.
