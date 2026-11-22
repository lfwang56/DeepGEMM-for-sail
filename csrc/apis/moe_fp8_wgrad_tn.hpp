#pragma once
#include <c10/cuda/CUDAGuard.h>
#include "../jit_kernels/impls/k_grouped_bf16_gemm.hpp"
#include <deep_gemm/common/moe_fp8_wgrad.cuh>
namespace deep_gemm::training {
class Fp8TnWgradRuntime final:public LaunchRuntime<Fp8TnWgradRuntime> {
public:
    struct Args {
        Bf16KGroupedRuntime::LaunchInfo info;
        LaunchArgs launch_args;
        KGroupedGemmArgs params;
        Fp8WgradParams wgrad;
        const float* sa;
        const float* sb;
    };
    static auto compile_key(const Args& a) {
        const auto& i=a.info;
        return std::make_tuple(i.shape_m,i.shape_n,i.num_groups,i.block_m,i.block_n,i.block_k,i.warp_m,i.warp_n,i.num_stages);
    }
    static std::string generate_impl(const Args& a) {
        const auto& i=a.info;
        return fmt::format(R"(
#include <deep_gemm/impls/k_grouped_bf16_gemm.cuh>
#include <deep_gemm/common/moe_fp8_wgrad.cuh>
namespace deep_gemm {{
extern "C" __launch_bounds__({0},1)
__global__ void fp8_training_tn_wgrad(KGroupedGemmArgs args,const training::Fp8WgradParams dw,
                                    const float* sa,const float* sb) {{
    __shared__ int32_t rounded[{3}];
    for(uint32_t g=threadIdx.x;g<{3};g+=blockDim.x) {{
        const int32_t count=dw.lengths[g],start=dw.offsets[g],end=dw.offsets[g+1];
        const bool valid=count>=0 && start>=0 && start%128==0 && end>=start && end<=dw.capacity &&
            int64_t(start)+((int64_t(count)+127)/128)*128<=end;
        rounded[g]=valid?((count+127)/128)*128:0;
        if(!valid)atomicOr(dw.status,2u);
    }}
    __syncthreads();
    args.grouped_layout=rounded;
    bf16_k_grouped_gemm_impl<cutlass::float_e4m3_t,float,GemmType::KGroupedContiguous,
        {1},{2},{3},{4},{5},{6},{7},{8},{0},{9},true,0,true,0,0,true>(args,0,~0u,sa,sb,dw.offsets);
}}
}}
)",(i.block_m/i.warp_m)*(i.block_n/i.warp_n)*32,i.shape_m,i.shape_n,i.num_groups,
            i.block_m,i.block_n,i.block_k,i.warp_m,i.warp_n,i.num_stages);
    }
    static void launch_impl(const KernelHandle& kernel,const LaunchConfigHandle& config,Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,a.params,a.wgrad,a.sa,a.sb));
    }
};
static int fp8_wgrad_tn(const std::pair<torch::Tensor,torch::Tensor>& lhs,const std::pair<torch::Tensor,torch::Tensor>& rhs,
                       const torch::Tensor& output,const torch::Tensor& lengths,const torch::Tensor& offsets,
                       const torch::Tensor& status,bool compile_only,const std::vector<int>& config) {
    const auto& a=lhs.first;const auto& sa=lhs.second;const auto& b=rhs.first;const auto& sb=rhs.second;
    DG_HOST_ASSERT(a.is_cuda() && a.dim()==2 && b.dim()==2 && output.dim()==3);
    const c10::cuda::CUDAGuard guard(a.device());DG_HOST_ASSERT(get_ppu_arch()==AC_PPU0015);
    for(const auto& t:{a,sa,b,sb,output,lengths,offsets,status})
        DG_HOST_ASSERT(t.is_cuda() && t.device()==a.device() && t.is_contiguous());
    const auto [r,m]=get_shape<2>(a);const auto [rb,n]=get_shape<2>(b);const auto g=output.size(0);
    DG_HOST_ASSERT(m>0 && n>0 && r>0 && r==rb && m%128==0 && n%128==0 && r%128==0);
    DG_HOST_ASSERT(g>0 && g<128 && output.size(1)==m && output.size(2)==n);
    DG_HOST_ASSERT(a.numel()<=INT32_MAX && b.numel()<=INT32_MAX && output.numel()<=INT32_MAX);
    DG_HOST_ASSERT(a.scalar_type()==torch::kFloat8_e4m3fn && b.scalar_type()==torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sa.scalar_type()==torch::kFloat && sb.scalar_type()==torch::kFloat && output.scalar_type()==torch::kFloat);
    DG_HOST_ASSERT(sa.dim()==2 && sa.size(0)==g && sa.size(1)==m && sb.dim()==2 && sb.size(0)==g && sb.size(1)==n);
    DG_HOST_ASSERT(lengths.scalar_type()==torch::kInt && offsets.scalar_type()==torch::kInt && status.scalar_type()==torch::kInt);
    DG_HOST_ASSERT(lengths.numel()==g && offsets.numel()==g+1 && status.numel()==1);
    int bm=256,bn=256,bk=128,wm=64,wn=64,stages=3;
    if(!config.empty()) {DG_HOST_ASSERT(config.size()==6);bm=config[0];bn=config[1];bk=config[2];wm=config[3];wn=config[4];stages=config[5];}
    DG_HOST_ASSERT(bm>0 && bn>0 && bk>0 && 128%bk==0 && wm>0 && wn>0 && bm%wm==0 && bn%wn==0 && stages>=2);
    const int bytes=int(gemm_smem_total_size(1,stages,bm,bn,bk));DG_HOST_ASSERT(bytes+g*4<=256*1024);
    const auto name="fp8_training_tn_wgrad";
    auto args=Fp8TnWgradRuntime::Args{
        .info={int(m),int(n),int(g),bm,bn,bk,wm,wn,stages,0,true,name},
        .launch_args={get_grid_shape(get_num_sms()),dim3((bm/wm)*(bn/wn)*32),bytes},
        .params={a.data_ptr(),b.data_ptr(),output.data_ptr(),lengths.data_ptr<int32_t>()},
        .wgrad={lengths.data_ptr<int32_t>(),offsets.data_ptr<int32_t>(),output.data_ptr<float>(),
                reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>()),uint32_t(g),uint32_t(m),uint32_t(n),uint32_t(r)},
        .sa=sa.data_ptr<float>(),.sb=sb.data_ptr<float>()};
    const auto runtime=compiler->build(name,Fp8TnWgradRuntime::generate(args));int occ=0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occ,runtime->kernel,args.launch_args.block_dim.x,args.launch_args.smem_size));
    DG_HOST_ASSERT(occ>0);args.launch_args.grid_dim.x*=occ;
    if(!compile_only)Fp8TnWgradRuntime::launch(runtime,args);
    return args.launch_args.grid_dim.x;
}
static void register_fp8_wgrad_tn_apis(pybind11::module_& m) {
    m.def("_moe_training_fp8_wgrad_tn",&fp8_wgrad_tn,py::arg("a"),py::arg("b"),py::arg("output"),py::arg("lengths"),
          py::arg("offsets"),py::arg("status"),py::arg("compile_only")=false,py::arg("config")=std::vector<int>{});
}
}
