#pragma once

#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <cutlass/numeric_types.h>
#include <type_traits>
#include <deep_gemm/common/peer_moe_training.cuh>
#include <deep_gemm/tix/sys_atomics.cuh>

namespace deep_gemm::training {

template <typename T>
__device__ __forceinline__ T* peer_pointer(const Params& p, int index) {
    return reinterpret_cast<T*>(p.ptr[index]);
}

template <PeerOperation Op>
__device__ __forceinline__ void run_peer(const Params& p, int64_t block_index = -1) {
    if (block_index < 0) block_index = blockIdx.x;
    const int64_t index = block_index * blockDim.x + threadIdx.x;
    if constexpr (Op == PeerOperation::SetRouteWindow) {
        const auto* offsets = peer_pointer<int32_t>(p, 0);
        const int64_t total = offsets[p.dim[0]];
        if (total < 0 || total > p.dim[1]) {
            if (index == 0) atomicOr(peer_pointer<int32_t>(p, 6), 2);
            return;
        }
        if (index <= p.dim[0]) {
            const int64_t begin = min(p.dim[2], max(int64_t(0), int64_t(offsets[index]) - p.dim[3]));
            peer_pointer<int32_t>(p, 2)[index] = begin;
            if (index < p.dim[0]) {
                const int64_t end = min(p.dim[2], max(int64_t(0), int64_t(offsets[index + 1]) - p.dim[3]));
                peer_pointer<int32_t>(p, 3)[index] = end - begin;
            }
        }
        const int64_t active = min(p.dim[2], max(int64_t(0), total - p.dim[3]));
        for (int64_t row = index; row < active; row += int64_t(gridDim.x) * blockDim.x) {
            const int64_t global_row = p.dim[3] + row;
            const int route = peer_pointer<int32_t>(p, 1)[global_row];
            peer_pointer<int32_t>(p, 4)[row] = route;
            peer_pointer<int32_t>(p, 5)[row] = route < 0 ? p.dim[5] : route / p.dim[4];
            int low = 0, high = p.dim[0];
            while (low < high) {
                const int middle = (low + high) / 2;
                if (offsets[middle + 1] <= global_row) low = middle + 1;
                else high = middle;
            }
            peer_pointer<int32_t>(p, 7)[row] = low;
        }
    } else if constexpr (Op == PeerOperation::AssistInputTiles) {
        const int64_t capacity = p.dim[0], width = p.dim[1] / 8;
        const int64_t bm = p.dim[4], blocks = peer_pointer<int32_t>(p, 4)[0];
        if (blocks < 0 || blocks > capacity / bm) {
            if (index == 0) atomicOr(peer_pointer<int32_t>(p, 6), 2);
            return;
        }
        auto* ready = peer_pointer<uint32_t>(p, 5);
        __shared__ int claimed;
        __shared__ uint64_t addresses[256];
        for (int64_t tile = block_index; tile < blocks; tile += gridDim.x) {
            if (threadIdx.x == 0) claimed = atomicCAS(ready + tile, 0u, 2u) == 0u;
            __syncthreads();
            if (!claimed) { __syncthreads(); continue; }
            const uint4 record = reinterpret_cast<const uint4*>(p.ptr[3])[tile];
            const int64_t base_row = record.w + (tile - record.z) * bm;
            for (int64_t row = threadIdx.x; row < bm; row += blockDim.x) {
                const int source = peer_pointer<int32_t>(p, 1)[base_row + row];
                const bool valid = source >= 0 && source < p.dim[2] * p.dim[3];
                addresses[row] = valid ? peer_pointer<uint64_t>(p, 0)[source / p.dim[2]]
                    + uint64_t(source % p.dim[2]) * p.dim[1] * 2 : 0;
            }
            __syncthreads();
            for (int64_t base = threadIdx.x; base < bm * width; base += blockDim.x * 8) {
                uint4 values[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int64_t element = base + j * blockDim.x;
                    const uint64_t address = element < bm * width ? addresses[element / width] : 0;
                    values[j] = address ? reinterpret_cast<const uint4*>(address)[element % width] : uint4{};
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int64_t element = base + j * blockDim.x;
                    if (element < bm * width)
                        peer_pointer<uint4>(p, 2)[base_row * width + element] = values[j];
                }
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                tix::red_add_rel(ready + tile, uint32_t(-1)); // busy=2 -> ready=1
                tix::red_add_rel(ready + capacity / bm, 1u);
                atomicAdd(peer_pointer<int32_t>(p, 7), 1);
            }
            __syncthreads();
        }
    } else if constexpr (Op == PeerOperation::PrefetchRawDy) {
        // Raw (unweighted) DY must survive even when a valid score is zero.
        // One warp owns a row; independent vector loads precede their stores.
        const int64_t rows = int64_t(peer_pointer<int32_t>(p, 3)[0]) * p.dim[5];
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(peer_pointer<int32_t>(p, 4), 2);
            return;
        }
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int warps = blockDim.x / 32;
        const int64_t stride = p.dim[2] * p.dim[3], width = p.dim[1] / 8;
        for (int64_t row = block_index * warps + warp; row < rows;
             row += int64_t(gridDim.x) * warps) {
            const int route = peer_pointer<int32_t>(p, 1)[row];
            const bool valid = route >= 0 && route < p.dim[4] * stride;
            if (route >= p.dim[4] * stride && lane == 0)
                atomicOr(peer_pointer<int32_t>(p, 4), 2);
            const int owner = valid ? route / stride : 0;
            const int64_t local_route = valid ? route % stride : 0;
            const uint4* source = valid ? reinterpret_cast<const uint4*>(peer_pointer<uint64_t>(p, 0)[owner])
                + (local_route / p.dim[3]) * width : nullptr;
            for (int64_t base = lane; base < width; base += 32 * 8) {
                uint4 values[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int64_t column = base + j * 32;
                    values[j] = valid && column < width ? source[column] : uint4{};
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int64_t column = base + j * 32;
                    if (column < width) peer_pointer<uint4>(p, 2)[row * width + column] = values[j];
                }
            }
        }
    } else if constexpr (Op == PeerOperation::ExpertLocalGradients) {
        // Keep replayed Y at its expert. Pull raw source DY and scores once
        // here, produce the mandatory BF16 packed DY and return one FP32
        // derivative per unique source route. Valid zero scores still have a
        // score derivative; padded routes produce zero packed gradients.
        const int64_t rows = int64_t(peer_pointer<int32_t>(p, 6)[0]) * (p.dim[5] ? p.dim[5] : 128);
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(peer_pointer<int32_t>(p, 7), 2);
            return;
        }
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int64_t stride = p.dim[2] * p.dim[3], width = p.dim[1] / 8;
        for (int64_t row = block_index * 8 + warp; row < rows; row += int64_t(gridDim.x) * 8) {
            const int route = peer_pointer<int32_t>(p, 3)[row];
            const bool valid = route >= 0 && route < p.dim[4] * stride;
            if (route >= p.dim[4] * stride && lane == 0) atomicOr(peer_pointer<int32_t>(p, 7), 2);
            const int owner = valid ? route / stride : 0;
            const int64_t local_route = valid ? route % stride : 0;
            // Both modes share this exact dot/weighting kernel. Only the DY
            // address changes, preserving its FP32 accumulation order.
            const uint4* dy = valid ? (p.ptr[8] ? peer_pointer<uint4>(p, 8) + row * width
                : reinterpret_cast<const uint4*>(peer_pointer<uint64_t>(p, 0)[owner])
                    + (local_route / p.dim[3]) * width) : nullptr;
            const float score = valid ? reinterpret_cast<const float*>(peer_pointer<uint64_t>(p, 1)[owner])[local_route] : 0.f;
            float dot = 0.f;
            for (int64_t column = lane; column < width; column += 32) {
                const uint4 raw_dy = valid ? dy[column] : uint4{};
                const uint4 raw_y = valid ? peer_pointer<uint4>(p, 4)[row * width + column] : uint4{};
                const auto* d = reinterpret_cast<const __ppu_bfloat16*>(&raw_dy);
                const auto* y = reinterpret_cast<const __ppu_bfloat16*>(&raw_y);
                uint4 raw_weighted{};
                auto* weighted = reinterpret_cast<__ppu_bfloat16*>(&raw_weighted);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float value = __bfloat162float(d[j]);
                    weighted[j] = __float2bfloat16(value * score);
                    dot += value * __bfloat162float(y[j]);
                }
                peer_pointer<uint4>(p, 5)[row * width + column] = raw_weighted;
            }
            #pragma unroll
            for (int delta = 16; delta; delta >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, delta);
            if (valid && lane == 0)
                reinterpret_cast<float*>(peer_pointer<uint64_t>(p, 2)[owner])[local_route] = dot;
        }
    } else if constexpr (Op == PeerOperation::CountRoutes) {
        // ID pointer table, counts, route expert, row, error flag.
        // dims: R, E_local, rank, P, routes per source rank.
        if (index >= p.dim[0]) return;
        const int owner = index / p.dim[4];
        const auto* ids = reinterpret_cast<const int64_t*>(peer_pointer<uint64_t>(p, 0)[owner]);
        const int64_t id = ids[index % p.dim[4]];
        int expert = -1, row = -1;
        if (id < -1 || id >= p.dim[1] * p.dim[3]) {
            atomicOr(peer_pointer<int32_t>(p, 4), 1);
        } else if (id >= 0 && id / p.dim[1] == p.dim[2]) {
            expert = id % p.dim[1];
            row = atomicAdd(peer_pointer<int32_t>(p, 1) + expert, 1);
        }
        peer_pointer<int32_t>(p, 2)[index] = expert;
        peer_pointer<int32_t>(p, 3)[index] = row;
    } else if constexpr (Op == PeerOperation::InitializeLayout) {
        // Permutation and source-token indices. dims: capacity, source tokens.
        if (index >= p.dim[0]) return;
        peer_pointer<int32_t>(p, 0)[index] = -1;
        peer_pointer<int32_t>(p, 1)[index] = p.dim[1];
    } else if constexpr (Op == PeerOperation::BuildPermutation) {
        // Route experts, local rows, expert offsets, permutation, source rows,
        // status. dims: routes, capacity, topk.
        if (index >= p.dim[0]) return;
        const int expert = peer_pointer<int32_t>(p, 0)[index];
        if (expert < 0) return;
        const int row = peer_pointer<int32_t>(p, 2)[expert] + peer_pointer<int32_t>(p, 1)[index];
        if (row >= p.dim[1]) {
            atomicOr(peer_pointer<int32_t>(p, 5), 2);
            return;
        }
        peer_pointer<int32_t>(p, 3)[row] = index;
        peer_pointer<int32_t>(p, 4)[row] = index / p.dim[2];
    } else if constexpr (Op == PeerOperation::BuildWorkInfo) {
        // Offsets, lengths, uint4 work records, work count.
        // dims: experts, capacity, block M.
        const auto* offsets = peer_pointer<int32_t>(p, 0);
        const int total_blocks = offsets[p.dim[0]] / p.dim[2];
        if (index == 0) peer_pointer<int32_t>(p, 3)[0] = total_blocks;
        if (index >= total_blocks) return;
        const int row = index * p.dim[2];
        int low = 0, high = p.dim[0];
        while (low < high) {
            const int middle = (low + high) / 2;
            if (offsets[middle + 1] <= row) low = middle + 1;
            else high = middle;
        }
        const int expert = low;
        peer_pointer<uint4>(p, 2)[index] = make_uint4(
            expert, peer_pointer<int32_t>(p, 1)[expert], offsets[expert] / p.dim[2], offsets[expert]);
    } else if constexpr (Op == PeerOperation::Pack) {
        // Source pointer table, permutation, packed output, status.
        // dims: capacity, H, source rows per rank, divisor, P.
        if (index >= p.dim[0] * p.dim[1]) return;
        const int route = peer_pointer<int32_t>(p, 1)[index / p.dim[1]];
        float value = 0.f;
        if (route >= 0) {
            const int source_row = route / p.dim[3];
            const int owner = source_row / p.dim[2];
            if (owner < p.dim[4]) {
                const auto* source = reinterpret_cast<const __ppu_bfloat16*>(peer_pointer<uint64_t>(p, 0)[owner]);
                value = __bfloat162float(source[int64_t(source_row % p.dim[2]) * p.dim[1] + index % p.dim[1]]);
            } else {
                atomicOr(peer_pointer<int32_t>(p, 3), 2);
            }
        }
        peer_pointer<__ppu_bfloat16>(p, 2)[index] = __float2bfloat16(value);
    } else if constexpr (Op == PeerOperation::PackActive || Op == PeerOperation::PackActiveVector) {
        // Explicit dispatch control: stage only the current padded rows. Avoid
        // making the unfused control pay for the worst-skew ring capacity.
        // Peer pointers, permutation, output, status, work count.
        const int64_t rows = int64_t(peer_pointer<int32_t>(p, 4)[0]) * (p.dim[5] ? p.dim[5] : 128);
        constexpr int elements = Op == PeerOperation::PackActiveVector ? 8 : 1;
        const int64_t width = p.dim[1] / elements;
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(peer_pointer<int32_t>(p, 3), 2);
        } else for (int64_t position = index; position < rows * width;
                    position += int64_t(gridDim.x) * blockDim.x) {
            const int route = peer_pointer<int32_t>(p, 1)[position / width];
            using Value = std::conditional_t<elements == 8, uint4, __ppu_bfloat16>;
            Value value{};
            if (route >= 0) {
                const int source_row = route / p.dim[3], owner = source_row / p.dim[2];
                if (owner < p.dim[4]) {
                    const auto* source = reinterpret_cast<const __ppu_bfloat16*>(peer_pointer<uint64_t>(p, 0)[owner]);
                    const auto* address = source + int64_t(source_row % p.dim[2]) * p.dim[1] + position % width * elements;
                    value = *reinterpret_cast<const Value*>(address);
                } else atomicOr(peer_pointer<int32_t>(p, 3), 2);
            }
            reinterpret_cast<Value*>(p.ptr[2])[position] = value;
        }
    } else if constexpr (Op == PeerOperation::ActivationActive || Op == PeerOperation::ActivationBackwardActive) {
        // Both matrices keep their physical worst-skew capacity. Only rows
        // referenced by the GPU work list are consumed by the expert GEMMs.
        const int64_t rows = int64_t(peer_pointer<int32_t>(p, 2)[0]) * (p.dim[2] ? p.dim[2] : 128);
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(peer_pointer<int32_t>(p, 3), 2);
        } else for (int64_t position = index; position < rows * p.dim[1];
                    position += int64_t(gridDim.x) * blockDim.x) {
            const int64_t column = position % p.dim[1];
            const int64_t offset = (position / p.dim[1]) * 2 * p.dim[1]
                + (p.dim[3] == 8 ? column / 8 * 16 + column % 8 : column);
            const int64_t up_step = p.dim[3] == 8 ? 8 : p.dim[1];
            auto* gate_up = peer_pointer<__ppu_bfloat16>(p, 0);
            auto* hidden = peer_pointer<__ppu_bfloat16>(p, 1);
            const float gate = __bfloat162float(gate_up[offset]);
            const float up = __bfloat162float(gate_up[offset + up_step]);
            const float sigmoid = 1.f / (1.f + expf(-gate));
            const float silu = gate * sigmoid;
            if constexpr (Op == PeerOperation::ActivationActive) {
                hidden[position] = __float2bfloat16(silu * up);
            } else {
                const float dh = __bfloat162float(hidden[position]);
                gate_up[offset] = __float2bfloat16(dh * up * sigmoid * (1.f + gate * (1.f - sigmoid)));
                gate_up[offset + up_step] = __float2bfloat16(dh * silu);
            }
        }
    } else if constexpr (Op == PeerOperation::LocalGradients) {
        // One CTA owns one source route: reuse dy for the score dot product
        // and weighted route gradient, preserving the BF16 rounding boundary.
        // local values, dy, scores, dscores, local dy. dims C,H,K,T,start.
        const int64_t route = block_index;
        const int64_t token = route / p.dim[2] + p.dim[4];
        const bool valid = token < p.dim[3];
        const float score = valid ? peer_pointer<float>(p, 2)[token * p.dim[2] + route % p.dim[2]] : 0.f;
        float sum = 0.f;
        for (int64_t column = threadIdx.x; column < p.dim[1]; column += blockDim.x) {
            const float dy = valid ? __bfloat162float(peer_pointer<__ppu_bfloat16>(p, 1)[token * p.dim[1] + column]) : 0.f;
            peer_pointer<__ppu_bfloat16>(p, 4)[route * p.dim[1] + column] = __float2bfloat16(dy * score);
            if (valid) sum += __bfloat162float(peer_pointer<__ppu_bfloat16>(p, 0)[route * p.dim[1] + column]) * dy;
        }
        #pragma unroll
        for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, delta);
        __shared__ float sums[8];
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) sums[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = lane < blockDim.x / 32 ? sums[lane] : 0.f;
            #pragma unroll
            for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, delta);
            if (lane == 0 && valid) peer_pointer<float>(p, 3)[token * p.dim[2] + route % p.dim[2]] = sum;
        }
    } else if constexpr (Op == PeerOperation::Scatter || Op == PeerOperation::ScatterVector) {
        // Packed result, permutation, destination pointers, status, work count.
        // dims: capacity, H, destination rows per rank, P.
        const int64_t rows = int64_t(peer_pointer<int32_t>(p, 4)[0]) * (p.dim[4] ? p.dim[4] : 128);
        constexpr int elements = Op == PeerOperation::ScatterVector ? 8 : 1;
        using Value = std::conditional_t<elements == 8, uint4, __ppu_bfloat16>;
        const int64_t width = p.dim[1] / elements;
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(peer_pointer<int32_t>(p, 3), 2);
        } else for (int64_t position = index; position < rows * width;
                    position += int64_t(gridDim.x) * blockDim.x) {
            const int route = peer_pointer<int32_t>(p, 1)[position / width];
            if (route < 0) continue;
            const int owner = route / p.dim[2];
            if (owner >= p.dim[3]) {
                atomicOr(peer_pointer<int32_t>(p, 3), 2);
                continue;
            }
            auto* destination = reinterpret_cast<__ppu_bfloat16*>(peer_pointer<uint64_t>(p, 2)[owner]);
            auto* address = destination + int64_t(route % p.dim[2]) * p.dim[1] + position % width * elements;
            *reinterpret_cast<Value*>(address) = reinterpret_cast<const Value*>(p.ptr[0])[position];
        }
    } else if constexpr (Op == PeerOperation::Barrier) {
        // Peer signal pointer table, local signals, error flag.
        // dims: P, rank, timeout cycles, diagnostic tag. One resident CTA.
        // Signals use the same alternating phase/sign protocol as MegaMoE.
        auto* local = peer_pointer<int32_t>(p, 1);
        if (*peer_pointer<int32_t>(p, 2) & 8) return;
        const uint32_t status = uint32_t(local[0]) & 3u;
        const uint32_t phase = status & 1u, sign = status >> 1;
        if (threadIdx.x < p.dim[0]) {
            auto* remote = reinterpret_cast<int32_t*>(peer_pointer<uint64_t>(p, 0)[threadIdx.x]);
            tix::red_add_rel_sys(remote + 1 + phase, sign ? -1 : 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int target = sign ? 0 : p.dim[0];
            const uint64_t begin = clock64();
            while (tix::ld_acq_sys(local + 1 + phase) != target) {
                if (clock64() - begin > uint64_t(p.dim[2])) {
                    atomicOr(peer_pointer<int32_t>(p, 2), 8);
                    printf("PPU training peer barrier timeout: rank=%lld, tag=%lld, phase=%u, sign=%u\n",
                           (long long)p.dim[1], (long long)p.dim[3], phase, sign);
                    break;
                }
            }
            reinterpret_cast<uint32_t*>(local)[0] += 1u;
        }
        __syncthreads();
    }
}

}  // namespace deep_gemm::training
