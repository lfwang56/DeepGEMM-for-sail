#pragma once
#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
#include <deep_gemm/common/moe_fp8_wgrad.cuh>
namespace deep_gemm {
// Per-expert feature scales stay constant along K. The established A8W8
// mainloop performs raw E4M3 Tensor Cell accumulation and scales only once.
template<class ProblemShape_,class CollectiveMainloop_,class CollectiveEpilogue_,class TileScheduler_,bool kOverlap>
class TrainingFp8RowWgrad {
public:
    static_assert(!kOverlap);
    using Base=cutlass::gemm::kernel::DeepGemmUniversal<ProblemShape_,CollectiveMainloop_,CollectiveEpilogue_,TileScheduler_,false>;
    using Params=typename Base::Params;
    using TileShape=typename Base::TileShape;
    using TiledMma=typename Base::TiledMma;
    using MainloopParams=typename Base::MainloopParams;
    using ElementA=typename Base::ElementA;
    using ElementB=typename Base::ElementB;
    static constexpr int SharedStorageSize=Base::SharedStorageSize;
    static constexpr int MaxThreadsPerBlock=Base::MaxThreadsPerBlock;
    static constexpr int MinBlocksPerMultiprocessor=Base::MinBlocksPerMultiprocessor;
    CUTLASS_DEVICE void operator()(const Params& params,const training::Fp8WgradParams& dw,char* smem) {
        using namespace cute;
        constexpr int bm=size<0>(TileShape{}),bn=size<1>(TileShape{});
        const uint32_t mt=(dw.m+bm-1)/bm,nt=(dw.n+bn-1)/bn;
        const uint64_t per_group=uint64_t(mt)*nt;
        for(uint64_t work=blockIdx.x;work<uint64_t(dw.groups)*per_group;work+=gridDim.x) {
            const uint32_t group=work/per_group;
            const int32_t count=__ldg(dw.lengths+group),start=__ldg(dw.offsets+group),end=__ldg(dw.offsets+group+1);
            if(count<0 || start<0 || start%128 || end<start || end>dw.capacity ||
                    int64_t(start)+((int64_t(count)+127)/128)*128>end) {
                if(threadIdx.x==0)atomicOr(dw.status,2u);
                continue;
            }
            if(!count)continue;
            const int k=(count+127)/128*128;
            const uint32_t local=work%per_group,mc=local/nt,nc=local%nt;
            MainloopParams update=params.mainloop;
            update.ptr_A=reinterpret_cast<const ElementA*>(params.mainloop.ptr_A)+int64_t(start)*dw.m;
            update.ptr_B=reinterpret_cast<const ElementB*>(params.mainloop.ptr_B)+int64_t(start)*dw.n;
            get<0>(update.dA)=int64_t(k);get<2>(update.dA)=int64_t(dw.m)*k;
            get<0>(update.dB)=int64_t(k);get<2>(update.dB)=int64_t(dw.n)*k;
            update.ptr_scale_A=params.mainloop.ptr_scale_A+int64_t(group)*dw.m;
            update.ptr_scale_B=params.mainloop.ptr_scale_B+int64_t(group)*dw.n;
            auto problem=ProblemShape_{int(dw.m),int(dw.n),k,1};
            auto coord=make_coord(mc,nc,_,0);
            CollectiveMainloop_ mainloop(update,take<0,3>(problem));
            auto inputs=mainloop.load_init(problem,coord,update,int(dw.m),0,0,update.ptr_A,update.ptr_B);
            auto ga=get<0>(inputs);
            TiledMma mma;
            auto accum=make_fragment_like<float>(partition_fragment_C(mma,take<0,2>(TileShape{})));
            clear(accum);
            auto iterator=make_coord_iterator(shape<2>(ga));
            const int tiles=size<2>(ga);
            auto residue=make_tuple(int(dw.m)-int(mc)*bm,bn,0);
            mainloop(accum,inputs,accum,iterator,tiles,residue,int(threadIdx.x),int(threadIdx.x)/32,smem);
            auto coordinates=mma.get_thread_slice(int(threadIdx.x)).partition_C(make_identity_tensor(take<0,2>(TileShape{})));
            CUTE_UNROLL
            for(int item=0;item<size(accum);++item) {
                const auto point=coordinates(item);
                const uint32_t row=mc*bm+uint32_t(get<0>(point)),column=nc*bn+uint32_t(get<1>(point));
                if(row<dw.m && column<dw.n) {
                    float* ptr=dw.output+(uint64_t(group)*dw.m+row)*dw.n+column;
                    *ptr=*ptr+float(accum(item));
                }
            }
            __syncthreads();
        }
    }
};
}
