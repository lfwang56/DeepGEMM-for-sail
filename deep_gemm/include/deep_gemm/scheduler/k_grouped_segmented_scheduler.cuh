#pragma once
#include <deep_gemm/common/k_grouped_segmented.cuh>
#include <deep_gemm/scheduler/k_grouped_scheduler.cuh>
namespace deep_gemm {
template <GemmType Type, uint32_t N, uint32_t BM, uint32_t BN,
          uint32_t Groups, uint32_t Segments, bool VisitEmpty>
struct KGroupedSegmentedScheduler
    : KGroupedScheduler<Type,N,BM,BN,Groups,math::ceil_div(N,BN),2,VisitEmpty> {
    using Base=KGroupedScheduler<Type,N,BM,BN,Groups,math::ceil_div(N,BN),2,VisitEmpty>;
    KGroupedSegmentArgs segments;
    __device__ __forceinline__ explicit KGroupedSegmentedScheduler(
        KGroupedSegmentArgs s, uint32_t m, uint32_t first=0, uint32_t end=~0u)
        : Base(nullptr,m,first,end),segments(s) {}
    __device__ __forceinline__ bool fetch_next_work(
        uint32_t& group, uint32_t& m, uint32_t& n) {
        const uint32_t per_group=this->num_m_blocks*Base::NUM_N_BLOCKS;
        while(true) {
            const uint32_t task=this->first_output_tile+(this->current_iter++)*gridDim.x+blockIdx.x;
            if(task>=this->num_blocks || task>=this->end_output_tile)return false;
            group=task/per_group;
            if(!this->group_loaded || this->curr_group_idx!=group) {
                this->current_shape_k=0;
                #pragma unroll
                for(uint32_t s=0;s<Segments;++s)
                    this->current_shape_k+=uint32_t(segments.offsets[s][group+1]-segments.offsets[s][group]);
                this->curr_group_idx=group;this->group_loaded=true;
            }
            if(!VisitEmpty && this->current_shape_k==0)continue;
            this->get_swizzled_block_idx(this->num_m_blocks,task-group*per_group,m,n);
            return true;
        }
    }
};
} // namespace deep_gemm
