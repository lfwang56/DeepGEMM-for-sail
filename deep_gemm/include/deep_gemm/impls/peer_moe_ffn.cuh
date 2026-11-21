#pragma once
#include <deep_gemm/common/peer_moe_ffn.cuh>
#include <deep_gemm/impls/peer_moe_gemm.cuh>

namespace deep_gemm::training {

template <uint32_t N, uint32_t K, uint32_t E, uint32_t BN, uint32_t BK,
          uint32_t WM, uint32_t WN, uint32_t Threads, uint32_t Stages,
          bool Return, bool ActivationBackward>
__device__ __forceinline__ void ffn_local_gemm(const PeerGemmParams& params, uint32_t tile) {
    training_bf16_peer_gemm_impl<cutlass::bfloat16_t, GemmType::MGroupedFused,
        N, K, E, 128, BN, BK, WM, WN, Threads, Stages, false, Return,
        false, ActivationBackward, 0, false, false, 0, 0, true, false,
        false, false, false, true>(params, {}, tile);
}

// Depth-first FFN tiles: one CTA gathers a complete input tile, computes both
// GEMMs and the intervening activation, then returns its own output directly.
// Different CTAs can gather/compute/return different tiles concurrently. There
// is no cross-CTA readiness dependency or grid barrier inside this kernel.
template <uint32_t H, uint32_t I, uint32_t E, uint32_t BN, uint32_t BK,
          uint32_t WM, uint32_t WN, uint32_t Threads, uint32_t Stages, bool Backward>
__device__ __forceinline__ void peer_ffn_impl(const PeerFfnParams params) {
    constexpr uint32_t BM = 128;
    const uint32_t tiles = __ldg(params.first.gemm.aligned_num_m_blocks);
    __shared__ uint64_t addresses[BM];
    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const uint4 record = __ldg(reinterpret_cast<const uint4*>(params.first.gemm.expert_ids_and_cumsum) + tile);
        const uint32_t row_base = record.w + (tile - record.z) * BM;
        const uint32_t valid_rows = min(BM, record.y - (tile - record.z) * BM);
        if constexpr (!Backward) {
            for (uint32_t row = threadIdx.x; row < BM; row += Threads) {
                const uint32_t source_row = __ldg(params.first.gemm.sorted_token_ids + row_base + row);
                uint64_t address = 0;
                if (source_row < uint64_t(params.first.num_ranks) * params.first.source_rows_per_rank) {
                    const uint32_t owner = source_row / params.first.source_rows_per_rank;
                    const uint32_t local_row = source_row % params.first.source_rows_per_rank;
                    address = __ldg(params.first.source_peers + owner) + uint64_t(local_row) * H * 2;
                }
                addresses[row] = address;
            }
            __syncthreads();
            pack_producer_input_rows<BM, H, Threads>(addresses, params.first, row_base);
            ffn_local_gemm<2 * I, H, E, BN, BK, WM, WN, Threads, Stages, false, false>(params.first, tile);
            __threadfence();
            __syncthreads();
            // Volatile scalar reads consume the gate/up values written in this
            // same launch; do not use the immutable read-only load path here.
            const auto* gate_up = reinterpret_cast<const volatile uint16_t*>(params.first.gemm.c_ptr);
            auto* hidden = reinterpret_cast<__ppu_bfloat16*>(const_cast<void*>(params.second.gemm.a_ptr));
            for (uint32_t item = threadIdx.x; item < valid_rows * I; item += Threads) {
                const uint64_t offset = uint64_t(row_base + item / I) * 2 * I + item % I;
                union Scalar { uint16_t bits; __ppu_bfloat16 value; };
                const Scalar gate{gate_up[offset]}, up{gate_up[offset + I]};
                const float g = __bfloat162float(gate.value), u = __bfloat162float(up.value);
                const float sigmoid = 1.f / (1.f + expf(-g));
                hidden[uint64_t(row_base) * I + item] = __float2bfloat16((g * sigmoid) * u);
            }
            __threadfence();
            __syncthreads();
            ffn_local_gemm<H, I, E, BN, BK, WM, WN, Threads, Stages, true, false>(params.second, tile);
        } else {
            ffn_local_gemm<I, H, E, BN, BK, WM, WN, Threads, Stages, false, true>(params.first, tile);
            __threadfence();
            __syncthreads();
            ffn_local_gemm<H, 2 * I, E, BN, BK, WM, WN, Threads, Stages, true, false>(params.second, tile);
        }
        __syncthreads();
    }
}
}  // namespace deep_gemm::training
