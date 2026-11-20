#pragma once

#include <cstdint>
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <cutlass/numeric_types.h>
#include <deep_gemm/common/moe_training.cuh>
#include <deep_gemm/epilogue/activation_quant_ops.cuh>

namespace deep_gemm::training {

using Bf16 = __ppu_bfloat16;

template <typename T>
__device__ __forceinline__ T* pointer(const Params& p, int index) {
    return reinterpret_cast<T*>(p.ptr[index]);
}

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, delta);
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
    return __shfl_sync(0xffffffffu, value, 0);
}

template <Operation Op>
__device__ __forceinline__ void run(const Params& p) {
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int warp = thread >> 5;
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + thread;

    if constexpr (Op == Operation::CountRoutes) {
        // ids, counts, route expert, within-expert row, status.
        // dims: route count, local experts, rank, number of ranks.
        if (index >= p.dim[0]) return;
        const int64_t id = pointer<int64_t>(p, 0)[index];
        const int experts = p.dim[1];
        int expert = -1, row = -1;
        if (id < -1 || id >= int64_t(experts) * p.dim[3]) {
            atomicOr(pointer<int32_t>(p, 4), 1);
        } else if (id >= 0 && id / experts == p.dim[2]) {
            expert = id % experts;
            row = atomicAdd(pointer<int32_t>(p, 1) + expert, 1);
        }
        pointer<int32_t>(p, 2)[index] = expert;
        pointer<int32_t>(p, 3)[index] = row;
    } else if constexpr (Op == Operation::PrefixRoutes) {
        // counts, offsets, aligned K lengths. dims: experts, alignment.
        if (index != 0) return;
        int offset = 0;
        pointer<int32_t>(p, 1)[0] = 0;
        for (int expert = 0; expert < p.dim[0]; ++expert) {
            const int count = pointer<int32_t>(p, 0)[expert];
            const int length = ((count + p.dim[1] - 1) / p.dim[1]) * p.dim[1];
            pointer<int32_t>(p, 2)[expert] = length;
            offset += length;
            pointer<int32_t>(p, 1)[expert + 1] = offset;
        }
    } else if constexpr (Op == Operation::InitializeLayout) {
        // offsets, per-row expert IDs, packed-row -> global route permutation.
        // dims: physical row capacity, experts.
        if (index >= p.dim[0]) return;
        const auto* offsets = pointer<int32_t>(p, 0);
        int expert = 0;
        if (index < offsets[p.dim[1]]) {
            int low = 0, high = p.dim[1];
            while (low < high) {
                const int middle = (low + high) / 2;
                if (offsets[middle + 1] <= index) low = middle + 1;
                else high = middle;
            }
            expert = low;
        }
        // Unused trailing rows are zero and use valid expert 0. This also
        // supports kernels that do not recognize a negative expert sentinel.
        pointer<int32_t>(p, 1)[index] = expert;
        pointer<int32_t>(p, 2)[index] = -1;
    } else if constexpr (Op == Operation::PackInput) {
        // gathered x, route experts, local rows, expert offsets, packed x,
        // permutation, status. dims: routes, H, top-k, physical capacity.
        const int64_t route = blockIdx.x;
        if (route >= p.dim[0]) return;
        const int expert = pointer<int32_t>(p, 1)[route];
        if (expert < 0) return;
        const int row = pointer<int32_t>(p, 3)[expert] + pointer<int32_t>(p, 2)[route];
        if (row >= p.dim[3]) {
            if (thread == 0) atomicOr(pointer<int32_t>(p, 6), 2);
            return;
        }
        if (thread == 0) pointer<int32_t>(p, 5)[row] = route;
        const auto* source = pointer<Bf16>(p, 0) + (route / p.dim[2]) * p.dim[1];
        auto* destination = pointer<Bf16>(p, 4) + int64_t(row) * p.dim[1];
        for (int64_t column = thread; column < p.dim[1]; column += blockDim.x)
            destination[column] = source[column];
    } else if constexpr (Op == Operation::UnpackRoutes) {
        // packed values, permutation, zero-initialized global route output.
        // dims: physical rows, H, global routes.
        if (index >= p.dim[0] * p.dim[1]) return;
        const int row = index / p.dim[1];
        const int route = pointer<int32_t>(p, 1)[row];
        if (route >= 0 && route < p.dim[2])
            pointer<Bf16>(p, 2)[int64_t(route) * p.dim[1] + index % p.dim[1]] = pointer<Bf16>(p, 0)[index];
    } else if constexpr (Op == Operation::PackRouteGrad) {
        // global route gradients, permutation, packed gradients.
        // dims: physical rows, H, global routes.
        if (index >= p.dim[0] * p.dim[1]) return;
        const int route = pointer<int32_t>(p, 1)[index / p.dim[1]];
        pointer<Bf16>(p, 2)[index] = route >= 0 && route < p.dim[2]
            ? pointer<Bf16>(p, 0)[int64_t(route) * p.dim[1] + index % p.dim[1]]
            : __float2bfloat16(0.f);
    } else if constexpr (Op == Operation::Combine) {
        // local route outputs, full-input scores, full-input y/dx.
        // dims: chunk tokens, H, top-k, input tokens, chunk start, apply scores.
        const int64_t token = index / p.dim[1], column = index % p.dim[1];
        const int64_t source_token = token + p.dim[4];
        if (token >= p.dim[0] || source_token >= p.dim[3]) return;
        float value = 0;
        for (int slot = 0; slot < p.dim[2]; ++slot) {
            const float score = p.dim[5] ? pointer<float>(p, 1)[source_token * p.dim[2] + slot] : 1.f;
            value += __bfloat162float(pointer<Bf16>(p, 0)[(token * p.dim[2] + slot) * p.dim[1] + column]) * score;
        }
        pointer<Bf16>(p, 2)[source_token * p.dim[1] + column] = __float2bfloat16(value);
    } else if constexpr (Op == Operation::Activation || Op == Operation::ActivationBackward) {
        // Forward: gate/up, hidden. Backward: gate/up, dHidden, dGateUp.
        // dims: physical rows, intermediate size.
        if (index >= p.dim[0] * p.dim[1]) return;
        const int64_t row = index / p.dim[1], column = index % p.dim[1];
        const int64_t offset = row * 2 * p.dim[1] + column;
        const float gate = __bfloat162float(pointer<Bf16>(p, 0)[offset]);
        const float up = __bfloat162float(pointer<Bf16>(p, 0)[offset + p.dim[1]]);
        const float sigmoid = 1.f / (1.f + expf(-gate));
        const float silu = gate * sigmoid;
        if constexpr (Op == Operation::Activation) {
            pointer<Bf16>(p, 1)[index] = __float2bfloat16(silu * up);
        } else {
            const float dh = __bfloat162float(pointer<Bf16>(p, 1)[index]);
            pointer<Bf16>(p, 2)[offset] = __float2bfloat16(dh * up * sigmoid * (1.f + gate * (1.f - sigmoid)));
            pointer<Bf16>(p, 2)[offset + p.dim[1]] = __float2bfloat16(dh * silu);
        }
    } else if constexpr (Op == Operation::QuantizeRows) {
        // BF16 matrix, E4M3 matrix, FP32 scales. dims: rows, K (128 aligned).
        const int64_t group = int64_t(blockIdx.x) * (blockDim.x / 32) + warp;
        const int64_t groups = p.dim[0] * (p.dim[1] / 128);
        if (group >= groups) return;  // Uniform within each participating warp.
        const int64_t base = group * 128;
        float values[4], maximum = 0.f;
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            values[element] = __bfloat162float(pointer<Bf16>(p, 0)[base + lane * 2 + (element / 2) * 64 + element % 2]);
            maximum = fmaxf(maximum, fabsf(values[element]));
        }
        maximum = warp_max(maximum);
        const float scale = fmaxf(maximum, 1e-4f) / 448.f;
        if (lane == 0) pointer<float>(p, 2)[group] = scale;
        auto* output = pointer<uint16_t>(p, 1);
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair)
            output[base / 2 + lane + pair * 32] = epilogue::quant::fp8_pair(
                values[pair * 2] / scale, values[pair * 2 + 1] / scale);
    } else if constexpr (Op == Operation::QuantizeWeights) {
        // BF16 weights[G,N,K], E4M3 weights, FP32 block scales[G,N/128,K/128].
        // dims: G,N,K; N/K are 128 aligned.
        const int n_blocks = p.dim[1] / 128, k_blocks = p.dim[2] / 128;
        const int group = blockIdx.x / (n_blocks * k_blocks);
        const int tile = blockIdx.x % (n_blocks * k_blocks);
        const int row_base = (tile / k_blocks) * 128, col_base = (tile % k_blocks) * 128;
        float maximum = 0.f;
        for (int element = thread; element < 128 * 128; element += blockDim.x) {
            const int64_t offset = (int64_t(group) * p.dim[1] + row_base + element / 128) * p.dim[2] + col_base + element % 128;
            maximum = fmaxf(maximum, fabsf(__bfloat162float(pointer<Bf16>(p, 0)[offset])));
        }
        maximum = warp_max(maximum);
        __shared__ float maxima[8];
        if (lane == 0) maxima[warp] = maximum;
        __syncthreads();
        if (warp == 0) {
            maximum = warp_max(lane < blockDim.x / 32 ? maxima[lane] : 0.f);
            if (lane == 0) maxima[0] = fmaxf(maximum, 1e-4f) / 448.f;
        }
        __syncthreads();
        const float scale = maxima[0];
        if (thread == 0) pointer<float>(p, 2)[blockIdx.x] = scale;
        for (int pair = thread; pair < 128 * 64; pair += blockDim.x) {
            const int64_t offset = (int64_t(group) * p.dim[1] + row_base + pair / 64) * p.dim[2] + col_base + (pair % 64) * 2;
            pointer<uint16_t>(p, 1)[offset / 2] = epilogue::quant::fp8_pair(
                __bfloat162float(pointer<Bf16>(p, 0)[offset]) / scale,
                __bfloat162float(pointer<Bf16>(p, 0)[offset + 1]) / scale);
        }
    } else if constexpr (Op == Operation::WeightedGradient) {
        // full-input dy, full-input scores, local per-route dy.
        // dims: chunk tokens,H,top-k,input tokens,chunk start.
        if (index >= p.dim[0] * p.dim[1] * p.dim[2]) return;
        const int64_t route = index / p.dim[1], column = index % p.dim[1];
        const int64_t token = route / p.dim[2] + p.dim[4];
        float value = 0;
        if (token < p.dim[3])
            value = __bfloat162float(pointer<Bf16>(p, 0)[token * p.dim[1] + column])
                    * pointer<float>(p, 1)[token * p.dim[2] + route % p.dim[2]];
        pointer<Bf16>(p, 2)[index] = __float2bfloat16(value);
    } else if constexpr (Op == Operation::ScoreGradient) {
        // local unweighted expert outputs, full-input dy, full-input dscores.
        // dims: chunk tokens,H,top-k,input tokens,chunk start.
        const int64_t route = blockIdx.x;
        const int64_t token = route / p.dim[2] + p.dim[4];
        if (route >= p.dim[0] * p.dim[2] || token >= p.dim[3]) return;
        float value = 0;
        for (int64_t column = thread; column < p.dim[1]; column += blockDim.x)
            value += __bfloat162float(pointer<Bf16>(p, 0)[route * p.dim[1] + column])
                     * __bfloat162float(pointer<Bf16>(p, 1)[token * p.dim[1] + column]);
        value = warp_sum(value);
        __shared__ float sums[8];
        if (lane == 0) sums[warp] = value;
        __syncthreads();
        if (warp == 0) {
            value = warp_sum(lane < blockDim.x / 32 ? sums[lane] : 0.f);
            if (lane == 0) pointer<float>(p, 2)[token * p.dim[2] + route % p.dim[2]] = value;
        }
    } else if constexpr (Op == Operation::ReadChunk) {
        // full x, full IDs, chunk x, chunk IDs, valid token count, status.
        // dims: input capacity, chunk tokens,H,top-k,chunk start.
        const int64_t valid = pointer<int32_t>(p, 4)[0];
        if (index == 0 && (valid < 0 || valid > p.dim[0])) atomicOr(pointer<int32_t>(p, 5), 4);
        const int64_t available = valid < 0 ? 0 : valid > p.dim[0] ? p.dim[0] : valid;
        if (index < p.dim[1] * p.dim[2]) {
            const int64_t token = index / p.dim[2] + p.dim[4];
            pointer<Bf16>(p, 2)[index] = token < available
                ? pointer<Bf16>(p, 0)[token * p.dim[2] + index % p.dim[2]] : __float2bfloat16(0.f);
        }
        if (index < p.dim[1] * p.dim[3]) {
            const int64_t token = index / p.dim[3] + p.dim[4];
            pointer<int64_t>(p, 3)[index] = token < available
                ? pointer<int64_t>(p, 1)[token * p.dim[3] + index % p.dim[3]] : -1;
        }
    }
}

}  // namespace deep_gemm::training
