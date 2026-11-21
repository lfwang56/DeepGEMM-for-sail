#pragma once

#include <deep_gemm/scheduler/fused_scheduler.cuh>
#include <deep_gemm/tix/sys_atomics.cuh>

namespace deep_gemm::training {

// A CTA owns the complete N sweep of one M tile. The caller can run two
// dependent GEMMs on that tile without waiting for unrelated tiles or CTAs.
template <GemmType Type, uint32_t N, uint32_t K, uint32_t BM, uint32_t BN, uint32_t E>
struct SelectedMTileScheduler : FusedGemmScheduler<Type, N, K, BM, BN, E> {
    using Base = FusedGemmScheduler<Type, N, K, BM, BN, E>;
    uint32_t selected, selected_n;
    __device__ __forceinline__ SelectedMTileScheduler(
            const int32_t* blocks, const int32_t* records, uint32_t tile, uint32_t column = ~0u)
        : Base(blocks, records), selected(tile), selected_n(column) {}
    __device__ __forceinline__ bool fetch_next_work(uint32_t& m, uint32_t& n) {
        const uint32_t iteration = this->current_iter++;
        if (selected_n != ~0u && iteration != 0) return false;
        n = selected_n == ~0u ? iteration : selected_n;
        if (selected >= this->num_aligned_m_blocks_ || n >= (N + BN - 1) / BN) return false;
        const uint4 record = __ldg(this->expert_ids_and_cumsum_ + selected);
        m = selected - record.z;
        this->curr_group_idx = record.x;
        this->curr_group_m = record.y;
        this->cumsum_m_block_idx = selected;
        this->curr_block_m_offset = record.w + m * BM;
        this->valid_m_in_block = min(BM, record.y - m * BM);
        return true;
    }
};

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

// Resident communication CTAs publish one ready flag per packed M tile. Each
// compute CTA waits only for its next tile, never for the whole dispatch.
template <GemmType Type, uint32_t N, uint32_t K, uint32_t BM, uint32_t BN, uint32_t E,
          bool WaitForInput = true, bool InlineOwners = false, uint32_t PollYieldEvery = 0>
struct DispatchPipelineScheduler : FusedGemmScheduler<Type, N, K, BM, BN, E> {
    using Base = FusedGemmScheduler<Type, N, K, BM, BN, E>;
    PeerDispatchPipeline control;
    uint64_t wait_cycles = 0;

    __device__ __forceinline__ DispatchPipelineScheduler(
            const int32_t* blocks, const int32_t* records, PeerDispatchPipeline state)
        : Base(blocks, records), control(state) {}

    __device__ __forceinline__ bool fetch_next_work(uint32_t& m, uint32_t& n) {
        constexpr uint32_t nblocks = (N + BN - 1) / BN;
        const uint32_t next = (this->current_iter++) * (gridDim.x - control.communication_ctas)
                            + blockIdx.x - control.communication_ctas;
        if (next >= this->num_aligned_m_blocks_ * nblocks) return false;
        uint32_t record_index;
        if constexpr (InlineOwners) {
            // Expose every independent input owner before scheduling columns
            // that wait for it. Row-major ordering otherwise admits only
            // grid/N owners and turns most resident CTAs into polling work.
            if (next < this->num_aligned_m_blocks_) {
                record_index = next;
                n = 0;
            } else if constexpr (nblocks > 1) {
                const uint32_t consumer = next - this->num_aligned_m_blocks_;
                record_index = consumer / (nblocks - 1);
                n = 1 + consumer % (nblocks - 1);
            } else {
                return false;
            }
        } else {
            record_index = next / nblocks;
            n = next % nblocks;
        }
        const uint4 data = __ldg(this->expert_ids_and_cumsum_ + record_index);
        m = record_index - data.z;
        this->curr_group_idx = data.x;
        this->curr_group_m = data.y;
        this->cumsum_m_block_idx = record_index;
        this->curr_block_m_offset = data.w + m * BM;
        this->valid_m_in_block = min(BM, data.y - m * BM);
        if constexpr (!WaitForInput) return true;
        // N=0 owns this M tile's input. Every dependency points to a smaller
        // work index; with the verified fully resident grid, an owner can
        // always run. Other N tiles acquire only their own owner's signal.
        if constexpr (InlineOwners) if (n == 0) return true;
        __shared__ int failed;
        __syncthreads();
        if (threadIdx.x == 0) {
            const uint64_t begin = clock64();
            const uint32_t signal = control.serialize ? control.signal_capacity : record_index;
            const uint32_t target = control.serialize ? this->num_aligned_m_blocks_ : 1u;
            uint32_t polls = 0;
            while (tix::ld_acq(control.ready + signal) != target) {
                if ((tix::ld_acq(control.status) & 8u) || clock64() - begin > control.timeout_cycles) {
                    atomicOr(control.status, 8u);
                    break;
                }
                if constexpr (PollYieldEvery != 0)
                    if ((++polls & (PollYieldEvery - 1)) == 0) __ppu_yield();
            }
            wait_cycles += clock64() - begin;
            failed = (tix::ld_acq(control.status) & 8u) != 0;
        }
        __syncthreads();
        return failed == 0;
    }
};

// Keep a short run of N tiles on one CTA. Its private input cache is populated
// once for that run, without cross-CTA dependencies or residency barriers.
template <GemmType Type, uint32_t N, uint32_t K, uint32_t BM, uint32_t BN,
          uint32_t E, uint32_t CacheN>
struct CachedPeerScheduler : FusedGemmScheduler<Type, N, K, BM, BN, E> {
    using Base = FusedGemmScheduler<Type, N, K, BM, BN, E>;
    static constexpr uint32_t nblocks = (N + BN - 1) / BN;
    static constexpr uint32_t ngroups = (nblocks + CacheN - 1) / CacheN;
    uint32_t local_n = CacheN, group_n = 0, block_m = 0;
    bool first_in_group = false;

    __device__ __forceinline__ CachedPeerScheduler(const int32_t* blocks, const int32_t* records)
        : Base(blocks, records) {}

    __device__ __forceinline__ bool fetch_next_work(uint32_t& m, uint32_t& n) {
        first_in_group = local_n == CacheN || group_n + local_n == nblocks;
        if (first_in_group) {
            const uint32_t next = (this->current_iter++) * gridDim.x + blockIdx.x;
            if (next >= this->num_aligned_m_blocks_ * ngroups) return false;
            const uint32_t record_index = next / ngroups;
            const uint4 data = __ldg(this->expert_ids_and_cumsum_ + record_index);
            this->curr_group_idx = data.x;
            this->curr_group_m = data.y;
            this->cumsum_m_block_idx = record_index;
            block_m = record_index - data.z;
            this->curr_block_m_offset = data.w + block_m * BM;
            this->valid_m_in_block = min(BM, data.y - block_m * BM);
            group_n = (next % ngroups) * CacheN;
            local_n = 0;
        }
        m = block_m;
        n = group_n + local_n++;
        return true;
    }
};

// Schedule a contiguous N-tile slice without enumerating and discarding the
// other columns. A producer GEMM can materialize X while computing the first
// slice; a subsequent local GEMM exposes the remaining tiles to the whole GPU.
template <GemmType Type, uint32_t N, uint32_t K, uint32_t BM, uint32_t BN,
          uint32_t E, uint32_t FirstN, uint32_t CountN>
struct SlicedPeerScheduler : FusedGemmScheduler<Type, N, K, BM, BN, E> {
    using Base = FusedGemmScheduler<Type, N, K, BM, BN, E>;
    static constexpr uint32_t nblocks = (N + BN - 1) / BN;
    static constexpr uint32_t slice_n = CountN == 0 ? nblocks - FirstN : CountN;
    static_assert(FirstN < nblocks && slice_n > 0 && FirstN + slice_n <= nblocks);

    __device__ __forceinline__ SlicedPeerScheduler(const int32_t* blocks, const int32_t* records)
        : Base(blocks, records) {}

    __device__ __forceinline__ bool fetch_next_work(uint32_t& m, uint32_t& n) {
        const uint32_t next = (this->current_iter++) * gridDim.x + blockIdx.x;
        if (next >= this->num_aligned_m_blocks_ * slice_n) return false;
        const uint32_t record_index = next / slice_n;
        const uint4 data = __ldg(this->expert_ids_and_cumsum_ + record_index);
        m = record_index - data.z;
        n = FirstN + next % slice_n;
        this->curr_group_idx = data.x;
        this->curr_group_m = data.y;
        this->cumsum_m_block_idx = record_index;
        this->curr_block_m_offset = data.w + m * BM;
        this->valid_m_in_block = min(BM, data.y - m * BM);
        return true;
    }
};

}  // namespace deep_gemm::training
