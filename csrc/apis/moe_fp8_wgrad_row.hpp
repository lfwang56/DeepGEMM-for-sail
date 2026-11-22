#pragma once
#include <c10/cuda/CUDAGuard.h>
#include "../jit_kernels/impls/int8_gemm.hpp"
#include <deep_gemm/common/moe_fp8_wgrad.cuh>
namespace deep_gemm::training {
class Fp8RowWgradRuntime final:public LaunchRuntime<Fp8RowWgradRuntime> {
public:
    struct Args {
        INT8GemmCutlass3Runtime::LaunchInfo launch_info;
        LaunchArgs launch_args;
        INT8GemmCutlass3Runtime::GemmKernelParams kernel_params;
        Fp8WgradParams wgrad;
    };
    static auto compile_key(const Args& a) {
        const auto& i=a.launch_info;
        return std::make_tuple(i.block_m,i.block_n,i.block_k,i.warp_m,i.warp_n,i.num_groups,i.num_stages,
                              a.wgrad.n,a.wgrad.capacity,a.launch_args.smem_size);
    }
    static std::string generate_impl(const Args& a) {
        auto source=INT8GemmCutlass3Runtime::generate_impl({a.launch_info,a.launch_args,a.kernel_params,"cutlass::float_e4m3_t"});
        const auto replace=[&](const std::string& from,const std::string& to) {
            const auto p=source.find(from);
            TORCH_CHECK(p!=std::string::npos && source.find(from,p+from.size())==std::string::npos,from);
            source.replace(p,from.size(),to);
        };
        replace("#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>","#include <deep_gemm/impls/moe_fp8_wgrad_row.cuh>");
        replace("using GemmKernel = cutlass::gemm::kernel::DeepGemmUniversal<","using GemmKernel = TrainingFp8RowWgrad<");
        replace("typename GemmKernel::Params params","typename GemmKernel::Params params, const deep_gemm::training::Fp8WgradParams wgrad");
        replace("op(params, smem);","op(params, wgrad, smem);");
        return source;
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.kernel_params,a.wgrad));
    }
};
static int fp8_wgrad_row(const std::pair<torch::Tensor,torch::Tensor>& lhs,const std::pair<torch::Tensor,torch::Tensor>& rhs,
                         const torch::Tensor& output,const torch::Tensor& lengths,const torch::Tensor& offsets,
                         const torch::Tensor& status,bool compile_only,const std::vector<int>& config) {
    const auto& a=lhs.first;const auto& sa=lhs.second;const auto& b=rhs.first;const auto& sb=rhs.second;
    DG_HOST_ASSERT(a.is_cuda() && a.dim()==2 && b.dim()==2 && output.dim()==3);
    const c10::cuda::CUDAGuard guard(a.device());DG_HOST_ASSERT(is_ppu1v5_device());
    for(const auto& t:{a,sa,b,sb,output,lengths,offsets,status})
        DG_HOST_ASSERT(t.is_cuda() && t.device()==a.device() && t.is_contiguous());
    const auto [m,r]=get_shape<2>(a);const auto [n,rb]=get_shape<2>(b);const auto g=output.size(0);
    DG_HOST_ASSERT(m>0 && n>0 && r>0 && r==rb && m%128==0 && n%128==0 && r%128==0);
    DG_HOST_ASSERT(g>0 && g<128 && output.size(1)==m && output.size(2)==n);
    DG_HOST_ASSERT(a.numel()<=INT32_MAX && b.numel()<=INT32_MAX && output.numel()<=INT32_MAX);
    DG_HOST_ASSERT(a.scalar_type()==torch::kFloat8_e4m3fn && b.scalar_type()==torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sa.scalar_type()==torch::kFloat && sb.scalar_type()==torch::kFloat && output.scalar_type()==torch::kFloat);
    DG_HOST_ASSERT(sa.dim()==2 && sa.size(0)==g && sa.size(1)==m && sb.dim()==2 && sb.size(0)==g && sb.size(1)==n);
    DG_HOST_ASSERT(lengths.scalar_type()==torch::kInt && offsets.scalar_type()==torch::kInt && status.scalar_type()==torch::kInt);
    DG_HOST_ASSERT(lengths.numel()==g && offsets.numel()==g+1 && status.numel()==1);
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(a.data_ptr())%16==0 && reinterpret_cast<uintptr_t>(b.data_ptr())%16==0);
    const int sms=get_num_sms();int bm=128,bn=n%256==0?256:128,bk=128,wm=32,wn=64,stages=3;
    if(!config.empty()) {
        DG_HOST_ASSERT(config.size()==5);bn=config[0];bk=config[1];wm=config[2];wn=config[3];stages=config[4];
    }
    DG_HOST_ASSERT(bn>0 && bn%128==0 && n%bn==0 && bk==128 && wm>0 && bm%wm==0 && wn>0 && bn%wn==0 && stages>=2);
    const auto smem=deep_gemm_int8::get_smem_config(stages,r,bm,bn,bk,1);
    using Stride=cute::Stride<int64_t,cute::Int<1>,int64_t>;
    const auto da=cutlass::make_cute_packed_stride(Stride{},cute::make_shape(m,r,1));
    const auto db=cutlass::make_cute_packed_stride(Stride{},cute::make_shape(n,r,1));
    const auto dd=cutlass::make_cute_packed_stride(Stride{},cute::make_shape(m,n,1));
    cutlass::KernelHardwareInfo hw;hw.device_id=a.get_device();hw.cu_count=sms;
    const auto gemm=INT8GemmCutlass3Runtime::GemmArguments{
        .mode=cutlass::gemm::GemmUniversalMode::kGemm,.problem_shape={int(m),int(n),int(r),1},
        .mainloopargs={a.data_ptr(),da,b.data_ptr(),db,sa.data_ptr<float>(),sb.data_ptr<float>()},
        .epilogueargs={{1,0},nullptr,dd,reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr()),dd},
        .hw_info=hw,.scheduler={uint32_t(m),lengths.data_ptr<int32_t>()},.signal=nullptr};
    const std::string name="fp8_training_k_grouped_wgrad_row";
    auto args=Fp8RowWgradRuntime::Args{
        .launch_info={bm,bn,bk,wm,wn,int(g),stages,"MGroupedNoPad","Default",name,false,false},
        .launch_args={get_grid_shape(sms),dim3((bm/wm)*(bn/wn)*32),std::get<0>(smem)},
        .kernel_params=INT8GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm,nullptr),
        .wgrad={lengths.data_ptr<int32_t>(),offsets.data_ptr<int32_t>(),output.data_ptr<float>(),
                reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>()),uint32_t(g),uint32_t(m),uint32_t(n),uint32_t(r)}};
    const auto runtime=compiler->build(name,Fp8RowWgradRuntime::generate(args));int occupancy=0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,runtime->kernel,args.launch_args.block_dim.x,args.launch_args.smem_size));
    DG_HOST_ASSERT(occupancy>0);args.launch_args.grid_dim.x*=occupancy;
    if(!compile_only)Fp8RowWgradRuntime::launch(runtime,args);
    return args.launch_args.grid_dim.x;
}
static void register_fp8_wgrad_row_apis(pybind11::module_& m) {
    m.def("_moe_training_fp8_wgrad_row",&fp8_wgrad_row,py::arg("a"),py::arg("b"),py::arg("output"),py::arg("lengths"),
          py::arg("offsets"),py::arg("status"),py::arg("compile_only")=false,py::arg("config")=std::vector<int>{});
}
}
