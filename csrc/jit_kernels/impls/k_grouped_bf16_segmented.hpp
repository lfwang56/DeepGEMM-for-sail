#pragma once
#include "k_grouped_bf16_gemm.hpp"
#include <deep_gemm/common/k_grouped_segmented.cuh>
namespace deep_gemm {
class Bf16KGroupedSegmentedRuntime final : public LaunchRuntime<Bf16KGroupedSegmentedRuntime> {
public:
    struct LaunchInfo {
        int m,n,groups,bm,bn,bk,wm,wn,stages,warp_map,aiu_first,pad,load_order,segments,max_continuous_k;
        bool accumulate,first_nonempty;
        std::string name;
    };
    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KGroupedGemmArgs kernel_params;
        KGroupedSegmentArgs segments;
        const int32_t* first_flags=nullptr;
    };
    static std::string generate_impl(const Args& a) {
        const auto& c=a.launch_info;
        return fmt::format(R"(
#include <deep_gemm/impls/k_grouped_bf16_segmented.cuh>
namespace deep_gemm {{
extern "C" __launch_bounds__({},1) __global__ void {}(
        const KGroupedGemmArgs args,const KGroupedSegmentArgs segments,const int32_t* first_flags) {{
    bf16_k_grouped_segmented_impl<cutlass::bfloat16_t,float,GemmType::KGroupedContiguous,
        {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},false,false,{},{},{}>(args,segments,0u,~0u,nullptr,nullptr,nullptr,first_flags);
}}
}}
)",32*(c.bm/c.wm)*(c.bn/c.wn),c.name,c.m,c.n,c.groups,c.bm,c.bn,c.bk,c.wm,c.wn,
    32*(c.bm/c.wm)*(c.bn/c.wn),c.stages,c.accumulate,c.warp_map,bool(c.aiu_first),c.pad,c.load_order,c.segments,c.max_continuous_k,c.first_nonempty);
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.kernel_params,a.segments,a.first_flags));
    }
};

static void k_grouped_bf16_segmented_impl_host(
        const std::vector<torch::Tensor>& a,const std::vector<torch::Tensor>& b,
        const std::vector<torch::Tensor>& offsets,const torch::Tensor& d,
        int config_k,bool accumulate,int max_continuous_k=0,const int32_t* first_flags=nullptr) {
    const int count=int(a.size()),g=int(d.size(0)),m=int(d.size(1)),n=int(d.size(2));
    const auto [sms,bm,bn,bk,wm,wn,stages,unused_smem]=deep_gemm_bf16_common::get_best_configs(
        m,n,config_k,g,get_num_sms(),GemmType::KGroupedContiguous);
    const int warp_map=get_env<int>("DG_BF16_WGRAD_WARP_MAP",0);
    const int aiu_first=get_env<int>("DG_BF16_WGRAD_AIU_BEFORE_TSM",1);
    const int pad=get_env<int>("DG_BF16_WGRAD_B_PAD_BYTES",0);
    const int load_order=get_env<int>("DG_BF16_WGRAD_TSM_LOAD_ORDER",0);
    DG_HOST_ASSERT(warp_map>=0 && warp_map<=3 && (aiu_first==0 || aiu_first==1));
    DG_HOST_ASSERT(pad>=0 && pad<=16384 && pad%1024==0 && load_order>=0 && load_order<=2);
    DG_HOST_ASSERT(get_env<int>("DG_BF16_WGRAD_TILE_CHUNK",0)==0);
    const int threads=32*(bm/wm)*(bn/wn);
    const int smem=int(gemm_smem_total_size(2,stages,bm,bn,bk))+pad;
    DG_HOST_ASSERT(smem<=256*1024);
    const std::string name=fmt::format("bf16_k_grouped_gemm_segments{}_{}_cap{}_first{}",count,accumulate?"add":"overwrite",max_continuous_k,int(first_flags!=nullptr));
    Bf16KGroupedSegmentedRuntime::Args args{
        .launch_info={m,n,g,bm,bn,bk,wm,wn,stages,warp_map,aiu_first,pad,load_order,count,max_continuous_k,accumulate,first_flags!=nullptr,name},
        .launch_args={get_grid_shape(sms),dim3(threads),smem},
        .kernel_params={.a_ptr=a[0].data_ptr(),.b_ptr=b[0].data_ptr(),.c_ptr=d.data_ptr(),
                        .grouped_layout=offsets[0].data_ptr<int32_t>()},
        .segments={},
        .first_flags=first_flags
    };
    for(int s=0;s<count;++s) {
        args.segments.a[s]=a[s].data_ptr();args.segments.b[s]=b[s].data_ptr();
        args.segments.offsets[s]=offsets[s].data_ptr<int32_t>();
    }
    const auto& code=Bf16KGroupedSegmentedRuntime::generate(args);
    const auto& runtime=compiler->build(name,code);int blocks=0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,runtime->kernel,threads,smem));
#else
    DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,runtime->kernel,threads,smem));
#endif
    DG_HOST_ASSERT(blocks>0);args.launch_args.grid_dim.x*=blocks;
    const int cus=get_env<int>("DG_BF16_WGRAD_COMPUTE_CUS",0);DG_HOST_ASSERT(cus>=0 && cus<=get_num_sms());
    if(cus)args.launch_args.grid_dim.x=std::min(args.launch_args.grid_dim.x,uint32_t(cus*blocks));
    Bf16KGroupedSegmentedRuntime::launch(runtime,args);
}
} // namespace deep_gemm
