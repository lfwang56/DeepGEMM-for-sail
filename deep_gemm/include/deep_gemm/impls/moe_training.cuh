#pragma once

#include <cstdint>
#include <type_traits>
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

template <Operation Op, bool MnMajorScales = false>
__device__ __forceinline__ void run(const Params& p, int64_t block_index = -1) {
    if (block_index < 0) block_index = blockIdx.x;
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int warp = thread >> 5;
    const int64_t index = block_index * blockDim.x + thread;

    if constexpr (Op == Operation::QuantizeRowsDeviceRows ||
                  Op == Operation::ActivationQuantizeDeviceRows ||
                  Op == Operation::ActivationBackwardQuantizeDeviceRows) {
        constexpr Operation Base = Op == Operation::QuantizeRowsDeviceRows ? Operation::QuantizeRows :
            Op == Operation::ActivationQuantizeDeviceRows ? Operation::ActivationQuantize :
            Operation::ActivationBackwardQuantize;
        constexpr int count_index = Op == Operation::QuantizeRowsDeviceRows ? 3 :
            Op == Operation::ActivationQuantizeDeviceRows ? 4 : 5;
        const int32_t rows = __ldg(pointer<int32_t>(p, count_index));
        if (rows < 0 || int64_t(rows) > p.dim[0]) {
            if (thread == 0 && blockIdx.x == 0)
                atomicOr(pointer<int32_t>(p, count_index + 1), 2);
            return;
        }
        Params active = p;
        active.dim[0] = rows;
        active.dim[2] = MnMajorScales ? p.dim[0] : 0;
        const int64_t blocks = (int64_t(rows) * (p.dim[1] / 128) + 7) / 8;
        // The reused arithmetic is warp-local. Its last partial warp returns
        // uniformly without touching inactive rows or requiring a CTA barrier.
        for (int64_t block = block_index; block < blocks; block += gridDim.x)
            run<Base, MnMajorScales>(active, block);
    } else if constexpr (Op == Operation::CountRoutes) {
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
        const int64_t route = block_index;
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
        const int64_t offset = row * 2 * p.dim[1]
            + (p.dim[2] == 8 ? column / 8 * 16 + column % 8 : column);
        const int64_t up_step = p.dim[2] == 8 ? 8 : p.dim[1];
        const float gate = __bfloat162float(pointer<Bf16>(p, 0)[offset]);
        const float up = __bfloat162float(pointer<Bf16>(p, 0)[offset + up_step]);
        const float sigmoid = 1.f / (1.f + expf(-gate));
        const float silu = gate * sigmoid;
        if constexpr (Op == Operation::Activation) {
            pointer<Bf16>(p, 1)[index] = __float2bfloat16(silu * up);
        } else {
            const float dh = __bfloat162float(pointer<Bf16>(p, 1)[index]);
            pointer<Bf16>(p, 2)[offset] = __float2bfloat16(dh * up * sigmoid * (1.f + gate * (1.f - sigmoid)));
            pointer<Bf16>(p, 2)[offset + up_step] = __float2bfloat16(dh * silu);
        }
    } else if constexpr (Op == Operation::ActivationQuantize || Op == Operation::ActivationBackwardQuantize) {
        // Preserve the BF16 activation/gradient rounding boundary for wgrad,
        // then immediately quantize those same values for the next FP8 GEMM.
        constexpr bool backward = Op == Operation::ActivationBackwardQuantize;
        const int64_t group = block_index * (blockDim.x / 32) + warp;
        const int64_t segments = p.dim[1] / 128;
        if (group >= p.dim[0] * segments) return;
        const int64_t row = group / segments, segment = group % segments;
        const int64_t input_base = row * 2 * p.dim[1] + segment * 128;
        const int64_t output_base = row * (backward ? 2 : 1) * p.dim[1] + segment * 128;
        float first[4], second[4], maximum_first = 0.f, maximum_second = 0.f;
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            const int column = lane * 2 + (element / 2) * 64 + element % 2;
            const float gate = __bfloat162float(pointer<Bf16>(p, 0)[input_base + column]);
            const float up = __bfloat162float(pointer<Bf16>(p, 0)[input_base + column + p.dim[1]]);
            const float sigmoid = 1.f / (1.f + expf(-gate));
            const float silu = gate * sigmoid;
            if constexpr (backward) {
                const float dh = __bfloat162float(pointer<Bf16>(p, 1)[row * p.dim[1] + segment * 128 + column]);
                const Bf16 dg = __float2bfloat16(dh * up * sigmoid * (1.f + gate * (1.f - sigmoid)));
                const Bf16 du = __float2bfloat16(dh * silu);
                pointer<Bf16>(p, 2)[output_base + column] = dg;
                pointer<Bf16>(p, 2)[output_base + column + p.dim[1]] = du;
                first[element] = __bfloat162float(dg);
                second[element] = __bfloat162float(du);
                maximum_second = fmaxf(maximum_second, fabsf(second[element]));
            } else {
                const Bf16 hidden = __float2bfloat16(silu * up);
                pointer<Bf16>(p, 1)[output_base + column] = hidden;
                first[element] = __bfloat162float(hidden);
            }
            maximum_first = fmaxf(maximum_first, fabsf(first[element]));
        }
        const float scale_first = fmaxf(warp_max(maximum_first), 1e-4f) / 448.f;
        auto* quantized = pointer<uint16_t>(p, backward ? 3 : 2);
        auto* scales = pointer<float>(p, backward ? 4 : 3);
        if (lane == 0) {
            if constexpr (MnMajorScales) scales[segment * p.dim[2] + row] = scale_first;
            else scales[output_base / 128] = scale_first;
        }
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair)
            quantized[output_base / 2 + lane + pair * 32] = epilogue::quant::fp8_pair(
                first[2 * pair] / scale_first, first[2 * pair + 1] / scale_first);
        if constexpr (backward) {
            const float scale_second = fmaxf(warp_max(maximum_second), 1e-4f) / 448.f;
            if (lane == 0) {
                if constexpr (MnMajorScales) scales[(segment + segments) * p.dim[2] + row] = scale_second;
                else scales[(output_base + p.dim[1]) / 128] = scale_second;
            }
            #pragma unroll
            for (int pair = 0; pair < 2; ++pair)
                quantized[(output_base + p.dim[1]) / 2 + lane + pair * 32] = epilogue::quant::fp8_pair(
                    second[2 * pair] / scale_second, second[2 * pair + 1] / scale_second);
        }
    } else if constexpr (Op == Operation::QuantizeRows) {
        // BF16 matrix, E4M3 matrix, FP32 scales. dims: rows, K (128 aligned).
        const int64_t group = block_index * (blockDim.x / 32) + warp;
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
        if (lane == 0) {
            if constexpr (MnMajorScales) {
                const int64_t segments = p.dim[1] / 128;
                pointer<float>(p, 2)[(group % segments) * p.dim[2] + group / segments] = scale;
            } else pointer<float>(p, 2)[group] = scale;
        }
        auto* output = pointer<uint16_t>(p, 1);
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair)
            output[base / 2 + lane + pair * 32] = epilogue::quant::fp8_pair(
                values[pair * 2] / scale, values[pair * 2 + 1] / scale);
    } else if constexpr (Op == Operation::QuantizeWeightsCachedFloat ||
                         Op == Operation::QuantizeWeightsCachedInteger) {
        // One 128x128 block, with every BF16 input loaded once and retained
        // as packed registers until the block scale is known.
        const int n_blocks = p.dim[1] / 128, k_blocks = p.dim[2] / 128;
        const int group = block_index / (n_blocks * k_blocks);
        const int tile = block_index % (n_blocks * k_blocks);
        const int row_base = (tile / k_blocks) * 128, col_base = (tile % k_blocks) * 128;
        uint4 cached[8];
        float maximums[4] = {0.f, 0.f, 0.f, 0.f};
        unsigned integer_maximum = 0;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int vector = thread + slot * 256;
            const int64_t offset = (int64_t(group) * p.dim[1] + row_base + vector / 16) * p.dim[2]
                                   + col_base + (vector % 16) * 8;
            cached[slot] = *reinterpret_cast<const uint4*>(pointer<Bf16>(p, 0) + offset);
            const unsigned words[4] = {cached[slot].x, cached[slot].y, cached[slot].z, cached[slot].w};
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const unsigned word = words[pair];
                if constexpr (Op == Operation::QuantizeWeightsCachedInteger) {
                    const unsigned lo = word & 0x7fffu, hi = (word >> 16) & 0x7fffu;
                    // Match fmaxf: ignore NaNs, retain infinity and unsigned zero.
                    if (lo <= 0x7f80u && lo > integer_maximum) integer_maximum = lo;
                    if (hi <= 0x7f80u && hi > integer_maximum) integer_maximum = hi;
                } else {
                    maximums[pair] = fmaxf(maximums[pair], fabsf(__uint_as_float(word << 16)));
                    maximums[pair] = fmaxf(maximums[pair], fabsf(__uint_as_float(word & 0xffff0000u)));
                }
            }
        }
        float maximum;
        if constexpr (Op == Operation::QuantizeWeightsCachedInteger)
            maximum = __uint_as_float(integer_maximum << 16);
        else maximum = fmaxf(fmaxf(maximums[0], maximums[1]), fmaxf(maximums[2], maximums[3]));
        maximum = warp_max(maximum);
        __shared__ float maxima[8];
        if (lane == 0) maxima[warp] = maximum;
        __syncthreads();
        if (warp == 0) {
            maximum = warp_max(lane < 8 ? maxima[lane] : 0.f);
            if (lane == 0) maxima[0] = fmaxf(maximum, 1e-4f) / 448.f;
        }
        __syncthreads();
        const float scale = maxima[0];
        if (thread == 0) pointer<float>(p, 2)[block_index] = scale;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const unsigned words[4] = {cached[slot].x, cached[slot].y, cached[slot].z, cached[slot].w};
            unsigned converted[4];
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair)
                converted[pair] = epilogue::quant::fp8_pair(
                    __uint_as_float(words[pair] << 16) / scale,
                    __uint_as_float(words[pair] & 0xffff0000u) / scale);
            const uint2 packed = make_uint2(converted[0] | (converted[1] << 16),
                                            converted[2] | (converted[3] << 16));
            const int vector = thread + slot * 256;
            const int64_t offset = (int64_t(group) * p.dim[1] + row_base + vector / 16) * p.dim[2]
                                   + col_base + (vector % 16) * 8;
            *reinterpret_cast<uint2*>(pointer<uint8_t>(p, 1) + offset) = packed;
        }
    } else if constexpr (Op == Operation::QuantizeWeights) {
        // BF16 weights[G,N,K], E4M3 weights, FP32 block scales[G,N/128,K/128].
        // dims: G,N,K; N/K are 128 aligned.
        const int n_blocks = p.dim[1] / 128, k_blocks = p.dim[2] / 128;
        const int group = block_index / (n_blocks * k_blocks);
        const int tile = block_index % (n_blocks * k_blocks);
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
        if (thread == 0) pointer<float>(p, 2)[block_index] = scale;
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
        const int64_t route = block_index;
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
    } else if constexpr (Op == Operation::ActivationDeviceRows) {
        const int64_t rows = pointer<int32_t>(p, 2)[0];
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 3), 2);
        } else for (int64_t position = index; position < rows * p.dim[1];
                    position += int64_t(gridDim.x) * blockDim.x) {
            const int64_t column = position % p.dim[1];
            const int64_t offset = position / p.dim[1] * 2 * p.dim[1]
                + (p.dim[3] == 8 ? column / 8 * 16 + column % 8 : column);
            const int64_t up_step = p.dim[3] == 8 ? 8 : p.dim[1];
            const float gate = __bfloat162float(pointer<Bf16>(p, 0)[offset]);
            const float up = __bfloat162float(pointer<Bf16>(p, 0)[offset + up_step]);
            const float sigmoid = 1.f / (1.f + expf(-gate)), silu = gate * sigmoid;
            if (!p.dim[2]) pointer<Bf16>(p, 1)[position] = __float2bfloat16(silu * up);
            else {
                const float dh = __bfloat162float(pointer<Bf16>(p, 1)[position]);
                pointer<Bf16>(p, 0)[offset] = __float2bfloat16(dh * up * sigmoid * (1.f + gate * (1.f - sigmoid)));
                pointer<Bf16>(p, 0)[offset + up_step] = __float2bfloat16(dh * silu);
            }
        }
    } else if constexpr (Op == Operation::TransformDeviceRows || Op == Operation::TransformDeviceRowsVector) {
        const int64_t rows = pointer<int32_t>(p, 3)[0];
        constexpr int elements = Op == Operation::TransformDeviceRowsVector ? 8 : 1;
        using Value = std::conditional_t<elements == 8, uint4, Bf16>;
        const int64_t width = p.dim[1] / elements;
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 4), 2);
        } else for (int64_t position = index; position < rows * width;
                    position += int64_t(gridDim.x) * blockDim.x) {
            const int route = pointer<int32_t>(p, 1)[position / width];
            const bool valid = route >= 0 && route < p.dim[2];
            const int64_t offset = int64_t(route) * width + position % width;
            if (p.dim[3]) pointer<Value>(p, 2)[position] = valid ? pointer<Value>(p, 0)[offset] : Value{};
            else if (valid) pointer<Value>(p, 2)[offset] = pointer<Value>(p, 0)[position];
        }
    } else if constexpr (Op == Operation::ZeroDeviceRows) {
        const int64_t rows = pointer<int32_t>(p, 1)[0];
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 2), 2);
        } else for (int64_t position = index; position < rows * p.dim[1];
                    position += int64_t(gridDim.x) * blockDim.x)
            pointer<Bf16>(p, 0)[position] = __float2bfloat16(0.f);
    } else if constexpr (Op == Operation::BuildPermutationOnly) {
        if (index >= p.dim[0]) return;
        const int expert = pointer<int32_t>(p, 0)[index];
        if (expert < 0) return;
        const int row = pointer<int32_t>(p, 2)[expert] + pointer<int32_t>(p, 1)[index];
        if (row >= p.dim[1]) atomicOr(pointer<int32_t>(p, 4), 2);
        else pointer<int32_t>(p, 3)[row] = index;
    } else if constexpr (Op == Operation::PackInputDeviceRows) {
        const int64_t rows = pointer<int32_t>(p, 3)[0], width = p.dim[1] / 8;
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 4), 2);
        } else for (int64_t position = index; position < rows * width;
                    position += int64_t(gridDim.x) * blockDim.x) {
            const int route = pointer<int32_t>(p, 1)[position / width];
            uint4 value{};
            if (route >= 0) {
                const int64_t source_row = route / p.dim[3];
                if (source_row < p.dim[2]) value = pointer<uint4>(p, 0)[source_row * width + position % width];
                else atomicOr(pointer<int32_t>(p, 4), 2);
            }
            pointer<uint4>(p, 2)[position] = value;
        }
    } else if constexpr (Op == Operation::DeepEPReduceWindowRows) {
        const int64_t active_rows = pointer<int32_t>(p, 6)[0];
        if (active_rows < 0 || active_rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 7), 2);
            return;
        }
        if (active_rows == 0) return;
        const int64_t width = p.dim[1] / 8;
        for (int64_t token = block_index * 8 + warp; token < p.dim[2];
             token += int64_t(gridDim.x) * 8) {
            // Cache route positions/weights across the warp once per token.
            // The receive accumulator starts at zero, so absent/padded tokens
            // and inactive windows require no H-wide writes.
            int packed_row = -1;
            float weight = 1.f;
            if (lane < p.dim[3]) {
                const int64_t route = token * p.dim[3] + lane;
                const int expert = pointer<int32_t>(p, 1)[route];
                if (expert >= p.dim[4]) atomicOr(pointer<int32_t>(p, 7), 2);
                else if (expert >= 0) {
                    const int64_t global_row = int64_t(pointer<int32_t>(p, 3)[expert])
                                               + pointer<int32_t>(p, 2)[route];
                    if (global_row < 0 || global_row >= p.dim[7]) atomicOr(pointer<int32_t>(p, 7), 2);
                    else if (global_row >= p.dim[5] && global_row < p.dim[5] + active_rows) {
                        packed_row = global_row - p.dim[5];
                        if (p.dim[6]) weight = pointer<float>(p, 4)[route];
                    }
                }
            }
            const uint32_t mask = __ballot_sync(0xffffffffu, packed_row >= 0);
            if (!mask) continue;
            for (int64_t column = lane; column < width; column += 32) {
                float sum[8]{};
                for (uint32_t pending = mask; pending; pending &= pending - 1) {
                    const int slot = __ffs(pending) - 1;
                    const int row = __shfl_sync(0xffffffffu, packed_row, slot);
                    const float score = __shfl_sync(0xffffffffu, weight, slot);
                    const uint4 raw = pointer<uint4>(p, 0)[int64_t(row) * width + column];
                    const auto* values = reinterpret_cast<const Bf16*>(&raw);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) sum[j] += __bfloat162float(values[j]) * score;
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    pointer<float>(p, 5)[token * p.dim[1] + column * 8 + j] += sum[j];
            }
        }
    } else if constexpr (Op == Operation::DeepEPReduceRows) {
        // Invert the route layout from count/prefix metadata. This reduces
        // top-k in registers and writes BF16 directly, without FP32 atomics,
        // a zeroed FP32 receive matrix, or a separate conversion pass.
        const int64_t width = p.dim[1] / 8;
        for (int64_t position = index; position < p.dim[2] * width;
             position += int64_t(gridDim.x) * blockDim.x) {
            const int64_t token = position / width, column = position % width;
            float sum[8]{};
            for (int slot = 0; slot < p.dim[3]; ++slot) {
                const int64_t route = token * p.dim[3] + slot;
                const int expert = pointer<int32_t>(p, 1)[route];
                if (expert < 0) continue;
                if (expert >= p.dim[4]) {
                    atomicOr(pointer<int32_t>(p, 6), 2);
                    continue;
                }
                const int row = pointer<int32_t>(p, 3)[expert] + pointer<int32_t>(p, 2)[route];
                if (row < 0 || row >= p.dim[0]) {
                    atomicOr(pointer<int32_t>(p, 6), 2);
                    continue;
                }
                const uint4 raw = pointer<uint4>(p, 0)[int64_t(row) * width + column];
                const auto* values = reinterpret_cast<const Bf16*>(&raw);
                const float weight = p.dim[5] ? pointer<float>(p, 4)[route] : 1.f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sum[j] += __bfloat162float(values[j]) * weight;
            }
            uint4 raw{};
            auto* values = reinterpret_cast<Bf16*>(&raw);
            #pragma unroll
            for (int j = 0; j < 8; ++j) values[j] = __float2bfloat16(sum[j]);
            pointer<uint4>(p, 5)[position] = raw;
        }
    } else if constexpr (Op == Operation::DeepEPBackwardRows) {
        // Gather DY, apply router weights, and reduce dScore without a
        // [receive_tokens, topk, hidden] expanded temporary.
        const int64_t rows = pointer<int32_t>(p, 6)[0];
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 7), 2);
            return;
        }
        for (int64_t row = block_index * 8 + warp; row < rows;
             row += int64_t(gridDim.x) * 8) {
            const int route = pointer<int32_t>(p, 2)[row];
            const bool valid = route >= 0 && route < p.dim[2] * p.dim[3];
            if (route >= p.dim[2] * p.dim[3] && lane == 0)
                atomicOr(pointer<int32_t>(p, 7), 2);
            const float weight = valid ? pointer<float>(p, 1)[route] : 0.f;
            float dot = 0.f;
            for (int64_t col = lane; col < p.dim[1]; col += 32) {
                const float dy = valid ? __bfloat162float(pointer<Bf16>(p, 0)[
                    int64_t(route / p.dim[3]) * p.dim[1] + col]) : 0.f;
                pointer<Bf16>(p, 4)[row * p.dim[1] + col] = __float2bfloat16(dy * weight);
                if (valid) dot += dy * __bfloat162float(pointer<Bf16>(p, 3)[row * p.dim[1] + col]);
            }
            dot = warp_sum(dot);
            if (valid && lane == 0) pointer<float>(p, 5)[route] = dot;
        }
    } else if constexpr (Op == Operation::TransposeWeights) {
        // Two independent expert-weight transposes share one launch. Transfer
        // BF16 bits in aligned vectors. XOR vector columns avoid a single
        // repeated TSM column for the transpose's strided lane accesses.
        constexpr int tile_size = 128;
        __shared__ __align__(16) uint4 tile[tile_size][tile_size / 8];
        const int64_t first_tiles = p.dim[0] * (2 * p.dim[2] / tile_size) * (p.dim[1] / tile_size);
        const bool first = block_index < first_tiles;
        const int64_t rows = first ? 2 * p.dim[2] : p.dim[1];
        const int64_t columns = first ? p.dim[1] : p.dim[2];
        const int64_t tile_index = first ? block_index : block_index - first_tiles;
        const int64_t column_tiles = columns / tile_size;
        const int64_t tiles_per_group = (rows / tile_size) * column_tiles;
        const int64_t group = tile_index / tiles_per_group;
        const int64_t local_tile = tile_index % tiles_per_group;
        const int64_t tile_row = local_tile / column_tiles, tile_column = local_tile % column_tiles;
        const auto* source = pointer<uint4>(p, first ? 0 : 1);
        auto* destination = pointer<uint4>(p, first ? 2 : 3);
        #pragma unroll
        for (int vector = thread; vector < tile_size * tile_size / 8; vector += 256) {
            const int row = vector / (tile_size / 8), column = vector % (tile_size / 8) * 8;
            const int64_t offset = ((group * rows + tile_row * tile_size + row) * columns
                                   + tile_column * tile_size + column) / 8;
            tile[row][(column / 8) ^ (row / 8)] = source[offset];
        }
        __syncthreads();
        #pragma unroll
        for (int vector = thread; vector < tile_size * tile_size / 8; vector += 256) {
            const int row = vector / (tile_size / 8), column = vector % (tile_size / 8) * 8;
            uint32_t words[4];
            const auto read_bits = [&](int source_row) {
                const uint4 value = tile[source_row][(row / 8) ^ (source_row / 8)];
                const int part = (row % 8) / 2;
                const uint32_t word = part == 0 ? value.x : part == 1 ? value.y : part == 2 ? value.z : value.w;
                return (word >> (16 * (row & 1))) & 0xffffu;
            };
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair)
                words[pair] = read_bits(column + 2 * pair) | (read_bits(column + 2 * pair + 1) << 16);
            const int64_t offset = ((group * columns + tile_column * tile_size + row) * rows
                                   + tile_row * tile_size + column) / 8;
            destination[offset] = make_uint4(words[0], words[1], words[2], words[3]);
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
