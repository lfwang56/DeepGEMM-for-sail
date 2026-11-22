#pragma once

// Included after peer_moe_gemm.cuh's normal implementation. All payloads retain
// their existing POD ABI. Helpers and Tensor Cell consumers share one CTA.
namespace deep_gemm {

template <uint32_t ShapeN, uint32_t ShapeK, uint32_t Groups,
          uint32_t BM, uint32_t BN, uint32_t BK, uint32_t WM, uint32_t WN,
          uint32_t ComputeThreads, int Stages, uint32_t HelperThreads,
          bool KeepGateUp, uint32_t CopyILP>
__device__ __forceinline__ void training_bf16_input_helpers_impl(
        const training::PeerGemmParams params) {
    static_assert(HelperThreads == 32 || HelperThreads == 64);
    static_assert(CopyILP == 1 || CopyILP == 2 || CopyILP == 4 || CopyILP == 8);
    static_assert(ShapeK % 8 == 0);
    constexpr uint32_t TotalThreads = ComputeThreads + HelperThreads;
    const uint32_t tiles = __ldg(params.gemm.aligned_num_m_blocks);
    if (threadIdx.x >= ComputeThreads) {
        const uint32_t helper = threadIdx.x - ComputeThreads;
        __shared__ uint64_t addresses[BM];
        for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
            const uint4 record = __ldg(
                reinterpret_cast<const uint4*>(params.gemm.expert_ids_and_cumsum) + tile);
            const uint32_t row_base = record.w + (tile - record.z) * BM;
            for (uint32_t row = helper; row < BM; row += HelperThreads) {
                const uint32_t source = __ldg(params.gemm.sorted_token_ids + tile * BM + row);
                uint64_t address = 0;
                if (source < uint64_t(params.num_ranks) * params.source_rows_per_rank) {
                    const uint32_t owner = source / params.source_rows_per_rank;
                    const uint32_t local = source % params.source_rows_per_rank;
                    address = __ldg(params.source_peers + owner)
                        + uint64_t(local) * ShapeK * sizeof(cutlass::bfloat16_t);
                }
                addresses[row] = address;
            }
            __ppu_barrier_sync(2, HelperThreads);
            auto* packed = reinterpret_cast<uint4*>(const_cast<void*>(params.gemm.a_ptr))
                + uint64_t(row_base) * (ShapeK / 8);
            for (uint32_t base = helper; base < BM * (ShapeK / 8);
                 base += HelperThreads * CopyILP) {
                uint4 values[CopyILP];
                #pragma unroll
                for (uint32_t j = 0; j < CopyILP; ++j) {
                    const uint32_t position = base + j * HelperThreads;
                    uint4 value{};
                    if (position < BM * (ShapeK / 8)) {
                        const uint32_t row = position / (ShapeK / 8);
                        const uint32_t column = position % (ShapeK / 8);
                        const uint64_t address = addresses[row];
                        if (address) value = reinterpret_cast<const uint4*>(address)[column];
                    }
                    values[j] = value;
                }
                #pragma unroll
                for (uint32_t j = 0; j < CopyILP; ++j) {
                    const uint32_t position = base + j * HelperThreads;
                    if (position < BM * (ShapeK / 8)) packed[position] = values[j];
                }
            }
            // Every helper completes its global stores before the team handoff.
            __threadfence();
            __ppu_barrier_sync(1, TotalThreads);
            // Consumers never access addresses[]. Join helper warps before
            // one can start replacing another's last-stage source addresses.
            __ppu_barrier_sync(2, HelperThreads);
        }
        return;
    }
    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        __ppu_barrier_sync(1, TotalThreads);
        training_bf16_peer_gemm_impl<
            cutlass::bfloat16_t, GemmType::MGroupedFused,
            ShapeN, ShapeK, Groups, BM, BN, BK, WM, WN, ComputeThreads, Stages,
            false, false,                      // peer reads/writes
            false, false, 0, false, false,      // coherent/activation/cache/staging
            0, 0, true, false,                 // all N, local AIU
            false, false, false, true,         // no pipeline; selected M tile
            4, true, true, KeepGateUp, CopyILP, // G8 FC1/SwiGLU
            false, 0, 0,                      // no external readiness waits
            false, false, false, false,        // no gradient/KN sidecars
            ComputeThreads                    // consumers-only barrier0
        >(params, training::PeerDispatchPipeline{}, tile);
    }
}

} // namespace deep_gemm
