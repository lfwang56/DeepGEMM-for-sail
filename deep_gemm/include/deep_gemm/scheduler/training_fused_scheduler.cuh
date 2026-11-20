#pragma once

#include <deep_gemm/scheduler/fused_scheduler.cuh>
#include <deep_gemm/tix/sys_atomics.cuh>

namespace deep_gemm::training {

// The phase barrier supplies release/acquire ordering. Volatile global loads
// keep metadata rewritten in this launch out of the immutable read-only path.
__device__ __forceinline__ int32_t load_pipeline_metadata(const int32_t* address) {
    return tix::load_volatile(address);
}
template <GemmType Type, uint32_t N, uint32_t K, uint32_t BM, uint32_t BN, uint32_t E>
struct CoherentFusedScheduler : FusedGemmScheduler<Type, N, K, BM, BN, E> {
    using Base = FusedGemmScheduler<Type, N, K, BM, BN, E>;
    __device__ __forceinline__ CoherentFusedScheduler(const int32_t* blocks, const int32_t* records)
        : Base(blocks, records) {
        this->num_aligned_m_blocks_ = load_pipeline_metadata(blocks);
        this->num_blocks_ = this->num_aligned_m_blocks_ * ((N + BN - 1) / BN);
    }

    __device__ __forceinline__ bool fetch_next_work(uint32_t& m, uint32_t& n) {
        constexpr uint32_t nblocks = (N + BN - 1) / BN;
        const uint32_t next = (this->current_iter++) * gridDim.x + blockIdx.x;
        if (next >= this->num_blocks_) return false;
        const auto* record = reinterpret_cast<const int32_t*>(this->expert_ids_and_cumsum_) + (next / nblocks) * 4;
        const uint32_t expert = load_pipeline_metadata(record);
        const uint32_t rows = load_pipeline_metadata(record + 1);
        const uint32_t prefix = load_pipeline_metadata(record + 2);
        const uint32_t offset = load_pipeline_metadata(record + 3);
        this->curr_group_idx = expert;
        this->curr_group_m = rows;
        this->get_swizzled_block_idx((rows + BM - 1) / BM, next - prefix * nblocks, m, n);
        this->cumsum_m_block_idx = m + prefix;
        this->curr_block_m_offset = offset + m * BM;
        this->valid_m_in_block = min(BM, rows - m * BM);
        return true;
    }
};

}  // namespace deep_gemm::training
