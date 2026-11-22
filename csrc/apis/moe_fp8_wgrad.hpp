#pragma once
#include <c10/cuda/CUDAGuard.h>
#include "../jit_kernels/impls/fp8_gemm.hpp"
#include <deep_gemm/common/moe_fp8_wgrad.cuh>

namespace deep_gemm::training {
class Fp8WgradRuntime final : public LaunchRuntime<Fp8WgradRuntime> {
public:
    struct Args {
        FP8GemmRuntime::LaunchInfo launch_info;
        LaunchArgs launch_args;
        FP8GemmRuntime::GemmKernelParams kernel_params;
        Fp8WgradParams wgrad;
        bool compact_panels;
    };
    static auto compile_key(const Args& a) {
        return std::make_tuple(FP8GemmRuntime::compile_key({a.launch_info, a.launch_args, a.kernel_params}), a.compact_panels);
    }
    static std::string generate_impl(const Args& a) {
        auto source = FP8GemmRuntime::generate_impl({a.launch_info, a.launch_args, a.kernel_params});
        const auto replace = [&](const std::string& from, const std::string& to) {
            auto p = source.find(from);
            TORCH_CHECK(p != std::string::npos && source.find(from, p + from.size()) == std::string::npos, from);
            source.replace(p, from.size(), to);
        };
        replace("#include <deep_gemm/impls/fp8_gemm.cuh>", "#include <deep_gemm/impls/moe_fp8_wgrad.cuh>");
        replace("using GemmKernel = DeepGemmUniversal<", "using GemmKernel = TrainingFp8Wgrad<");
        replace("typename GemmKernel::Params params", "typename GemmKernel::Params params, const deep_gemm::training::Fp8WgradParams wgrad");
        replace("op(params, smem);", "op(params, wgrad, smem);");
        return fmt::format("#define DG_FP8_DW_COMPACT {}\n", int(a.compact_panels)) + source;
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.kernel_params, a.wgrad));
    }
};

static int fp8_wgrad(const std::pair<torch::Tensor, torch::Tensor>& lhs,
                     const std::pair<torch::Tensor, torch::Tensor>& rhs,
                     const torch::Tensor& output, const torch::Tensor& lengths,
                     const torch::Tensor& offsets, const torch::Tensor& status,
                     bool compile_only, const std::vector<int>& config, bool compact_panels) {
    const auto& a = lhs.first; const auto& sa = lhs.second;
    const auto& b = rhs.first; const auto& sb = rhs.second;
    DG_HOST_ASSERT(a.is_cuda() && a.dim() == 2 && b.dim() == 2 && output.dim() == 3);
    const c10::cuda::CUDAGuard guard(a.device());
    for (const auto& t : {a,sa,b,sb,output,lengths,offsets,status})
        DG_HOST_ASSERT(t.is_cuda() && t.device() == a.device() && t.is_contiguous());
    const auto [m,r] = get_shape<2>(a);
    const auto [n,rb] = get_shape<2>(b);
    const auto g = output.size(0);
    DG_HOST_ASSERT(m > 0 && n > 0 && r > 0 && r == rb && m % 128 == 0 && n % 128 == 0 && r % 128 == 0);
    DG_HOST_ASSERT(g > 0 && g < 128 && output.size(1) == m && output.size(2) == n);
    DG_HOST_ASSERT(a.numel() <= INT32_MAX && b.numel() <= INT32_MAX && output.numel() <= INT32_MAX);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sa.scalar_type() == torch::kFloat && sb.scalar_type() == torch::kFloat && output.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sa.dim() == 2 && sa.size(0) == r/128 && sa.size(1) == m);
    DG_HOST_ASSERT(sb.dim() == 2 && sb.size(0) == n/128 && sb.size(1) == r/128);
    DG_HOST_ASSERT(lengths.scalar_type() == torch::kInt && offsets.scalar_type() == torch::kInt && status.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(lengths.numel() == g && offsets.numel() == g+1 && status.numel() == 1);
    DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0);
    const int sms = get_num_sms();
    int bn=128,bk=128,wm=32,wn=64,stages=3;
    if (!config.empty()) {
        DG_HOST_ASSERT(config.size() == 5);
        bn=config[0];bk=config[1];wm=config[2];wn=config[3];stages=config[4];
    }
    DG_HOST_ASSERT(bn > 0 && bn % 128 == 0 && bk == 128 && wm > 0 && 128 % wm == 0 && wn > 0 && bn % wn == 0 && stages >= 3);
    const int bm=128;
    auto smem = deep_gemm_fp8_common::get_smem_config(stages, r, bm, bn, bk);
    using Stride = cute::Stride<int64_t,cute::Int<1>,int64_t>;
    using Granularity = cute::Shape<cute::_1,cute::_128,cute::_128>;
    using ScaleConfig = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<Granularity,false,true>(Granularity{}));
    auto stride_a = cutlass::make_cute_packed_stride(Stride{},cute::make_shape(m,r,1));
    auto stride_b = cutlass::make_cute_packed_stride(Stride{},cute::make_shape(n,r,1));
    auto stride_d = cutlass::make_cute_packed_stride(Stride{},cute::make_shape(m,n,1));
    auto layout_a = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m,n,r,1));
    auto layout_b = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m,n,r,1));
    cutlass::KernelHardwareInfo hw;hw.device_id=a.get_device();hw.cu_count=sms;
    auto gemm = FP8GemmRuntime::GemmArguments{
        .mode=cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape={int(m),int(n),int(r),1},
        .mainloopargs={reinterpret_cast<cutlass::float_e4m3_t*>(a.data_ptr()),stride_a,
                       reinterpret_cast<cutlass::float_e4m3_t*>(b.data_ptr()),stride_b,4,
                       sa.data_ptr<float>(),layout_a,sb.data_ptr<float>(),layout_b},
        .epilogueargs={{1,0},nullptr,stride_d,reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr()),stride_d},
        .hw_info=hw,.scheduler={uint32_t(m),lengths.data_ptr<int32_t>()},.signal=nullptr};
    const std::string name="fp8_training_k_grouped_wgrad";
    auto args=Fp8WgradRuntime::Args{
        .launch_info={bm,bn,bk,wm,wn,int(g),stages,"MGroupedNoPad","Default",name,false},
        .launch_args={get_grid_shape(sms),dim3((bm/wm)*(bn/wn)*32),std::get<0>(smem)},
        .kernel_params=FP8GemmRuntime::to_underlying_arguments_rtc(gemm,nullptr),
        .wgrad={lengths.data_ptr<int32_t>(),offsets.data_ptr<int32_t>(),output.data_ptr<float>(),
                reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>()),uint32_t(g),uint32_t(m),uint32_t(n),uint32_t(r)},.compact_panels=compact_panels};
    auto runtime=compiler->build(name,Fp8WgradRuntime::generate(args));
    int occupancy=0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,runtime->kernel,args.launch_args.block_dim.x,args.launch_args.smem_size));
    DG_HOST_ASSERT(occupancy > 0);args.launch_args.grid_dim.x *= occupancy;
    if (!compile_only) Fp8WgradRuntime::launch(runtime,args);
    return args.launch_args.grid_dim.x;
}
static void register_fp8_wgrad_apis(pybind11::module_& m) {
    m.def("_moe_training_fp8_wgrad",&fp8_wgrad,py::arg("a"),py::arg("b"),py::arg("output"),py::arg("lengths"),
          py::arg("offsets"),py::arg("status"),py::arg("compile_only")=false,py::arg("config")=std::vector<int>{},py::arg("compact_panels")=false);
}
}
