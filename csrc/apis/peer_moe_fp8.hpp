#pragma once

#include <c10/cuda/CUDAGuard.h>
#include "../jit_kernels/impls/fp8_gemm.hpp"
#include <deep_gemm/common/peer_moe_training.cuh>

namespace deep_gemm::training {

class PeerFp8ReturnRuntime final : public LaunchRuntime<PeerFp8ReturnRuntime> {
public:
    struct Args {
        FP8GemmRuntime::LaunchInfo launch_info;
        LaunchArgs launch_args;
        FP8GemmRuntime::GemmKernelParams kernel_params;
        PeerGemmParams peers;
        bool keep_local;
    };
    static auto compile_key(const Args& args) {
        return std::make_tuple(FP8GemmRuntime::compile_key(
            {args.launch_info, args.launch_args, args.kernel_params}), args.keep_local);
    }
    static std::string generate_impl(const Args& args) {
        auto source = FP8GemmRuntime::generate_impl({args.launch_info, args.launch_args, args.kernel_params});
        const auto replace = [&](const std::string& from, const std::string& to) {
            const auto position = source.find(from);
            TORCH_CHECK(position != std::string::npos && source.find(from, position + from.size()) == std::string::npos,
                        "expected one native FP8 declaration: ", from);
            source.replace(position, from.size(), to);
        };
        replace("#include <deep_gemm/impls/fp8_gemm.cuh>", "#include <deep_gemm/impls/peer_moe_fp8.cuh>");
        replace("using GemmKernel = DeepGemmUniversal<", "using GemmKernel = TrainingFp8PeerReturn<");
        replace("typename GemmKernel::Params params", "typename GemmKernel::Params params, const deep_gemm::training::PeerGemmParams peers");
        replace("op(params, smem);", "op(params, peers, smem);");
        return fmt::format("#define DG_TRAINING_FP8_PEER_KEEP_LOCAL {}\n", int(args.keep_local)) + source;
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params, args.peers));
    }
};

// Blockwise E4M3 compute, with the same BF16 route return boundary as the
// BF16 training path. Local input packing/quantization precedes this call.
static int peer_fp8_return(const std::pair<torch::Tensor, torch::Tensor>& lhs,
                           const std::pair<torch::Tensor, torch::Tensor>& rhs,
                           const torch::Tensor& out, const torch::Tensor& lengths,
                           const torch::Tensor& permutation, const torch::Tensor& destinations,
                           int destination_rows_per_rank, bool keep_local, bool compile_only,
                           const std::vector<int>& config) {
    const auto& a = lhs.first;
    const auto& b = rhs.first;
    const auto& sa = lhs.second;
    const auto& sb = rhs.second;
    DG_HOST_ASSERT(a.is_cuda() && a.dim() == 2 && b.dim() == 3 && out.dim() == 2);
    const c10::cuda::CUDAGuard guard(a.device());
    for (const auto& t : {b, sa, sb, out, lengths, permutation, destinations})
        DG_HOST_ASSERT(t.is_cuda() && t.device() == a.device());
    for (const auto& t : {a, b, sb, out, lengths, permutation, destinations})
        DG_HOST_ASSERT(t.is_contiguous());
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn && b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sa.scalar_type() == torch::kFloat32 && sb.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(lengths.scalar_type() == torch::kInt32 && permutation.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(destinations.scalar_type() == torch::kInt64);
    const auto [m, k] = get_shape<2>(a);
    const auto [groups, n, weight_k] = get_shape<3>(b);
    DG_HOST_ASSERT(m > 0 && m % 128 == 0 && n > 0 && n % 128 == 0 && k > 0 && k % 128 == 0);
    DG_HOST_ASSERT(k == weight_k && groups > 0 && groups < 128);
    DG_HOST_ASSERT(a.numel() <= INT32_MAX && b.numel() <= INT32_MAX && out.numel() <= INT32_MAX);
    DG_HOST_ASSERT(out.size(0) == m && out.size(1) == n);
    DG_HOST_ASSERT(sa.dim() == 2 && sa.size(0) == m && sa.size(1) == k / 128);
    DG_HOST_ASSERT(sb.numel() == groups * (n / 128) * (k / 128));
    DG_HOST_ASSERT(lengths.numel() == groups && permutation.numel() == m);
    DG_HOST_ASSERT(destinations.numel() > 0 && destinations.numel() <= 64 && destination_rows_per_rank > 0);
    auto aligned_sa = get_mn_major_tma_aligned_tensor(sa);
    const int num_sms = get_num_sms();
    auto cfg = deep_gemm_fp8_common::get_best_configs(
        ceil_div(m, groups), n, k, groups, num_sms, GemmType::MGroupedNoPad);
    if (!config.empty()) {
        DG_HOST_ASSERT(config.size() == 5 && config[0] > 0 && config[1] == 128 &&
                       config[2] > 0 && config[3] > 0 && config[4] > 0);
        cfg = std::make_tuple(num_sms, 128, config[0], config[1], config[2], config[3], config[4],
            deep_gemm_fp8_common::get_smem_config(config[4], k, 128, config[0], config[1]));
    }
    // Small-K two-stage heuristics may select the native N-stage expansion.
    // This return wrapper currently preserves the regular mainloop only.
    if (k <= 512 && std::get<6>(cfg) == 2) {
        TORCH_CHECK(config.empty(), "FP8 peer return with K<=512 requires at least three stages");
        std::get<6>(cfg) = 3;
        std::get<7>(cfg) = deep_gemm_fp8_common::get_smem_config(
            3, k, std::get<1>(cfg), std::get<2>(cfg), std::get<3>(cfg));
    }
    const auto [sms, bm, bn, bk, wm, wn, stages, smem] = cfg;
    DG_HOST_ASSERT(bk == 128);
    using Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using Granularity = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<Granularity, false, true>(Granularity{}));
    const auto stride_a = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, k, 1));
    const auto stride_b = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(n, k, 1));
    const auto stride_d = cutlass::make_cute_packed_stride(Stride{}, cute::make_shape(m, n, 1));
    const auto layout_sa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    const auto layout_sb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
    cutlass::KernelHardwareInfo hw;
    hw.device_id = a.get_device();
    hw.cu_count = sms;
    const auto gemm = FP8GemmRuntime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {int(m), int(n), int(k), 1},
        .mainloopargs = {reinterpret_cast<cutlass::float_e4m3_t*>(a.data_ptr()), stride_a,
                         reinterpret_cast<cutlass::float_e4m3_t*>(b.data_ptr()), stride_b, 4,
                         aligned_sa.data_ptr<float>(), layout_sa, sb.data_ptr<float>(), layout_sb},
        .epilogueargs = {{1, 0}, nullptr, stride_d,
                         reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()), stride_d},
        .hw_info = hw,
        .scheduler = {uint32_t(m), lengths.data_ptr<int32_t>()},
        .signal = nullptr};
    PeerGemmParams peers{};
    peers.permutation = permutation.data_ptr<int32_t>();
    peers.destination_peers = reinterpret_cast<const uint64_t*>(destinations.data_ptr<int64_t>());
    peers.destination_rows_per_rank = destination_rows_per_rank;
    peers.num_ranks = destinations.numel();
    const std::string name = keep_local ? "fp8_training_peer_return_keep" : "fp8_training_peer_return_discard";
    auto args = PeerFp8ReturnRuntime::Args{
        .launch_info = {bm, bn, bk, wm, wn, int(groups), stages, "MGroupedNoPad", "Default", name, false},
        .launch_args = {get_grid_shape(sms), dim3((bm / wm) * (bn / wn) * 32), std::get<0>(smem)},
        .kernel_params = FP8GemmRuntime::to_underlying_arguments_rtc(gemm, nullptr),
        .peers = peers, .keep_local = keep_local};
    const auto runtime = compiler->build(name, PeerFp8ReturnRuntime::generate(args));
    int occupancy = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, runtime->kernel, args.launch_args.block_dim.x, args.launch_args.smem_size));
    DG_HOST_ASSERT(occupancy > 0);
    args.launch_args.grid_dim.x *= occupancy;
    if (!compile_only) PeerFp8ReturnRuntime::launch(runtime, args);
    return args.launch_args.grid_dim.x;
}

static void register_peer_fp8_apis(pybind11::module_& m) {
    m.def("_moe_training_peer_fp8_gemm", &peer_fp8_return,
          py::arg("a"), py::arg("weight"), py::arg("out"), py::arg("lengths"),
          py::arg("permutation"), py::arg("destination_peers"), py::arg("destination_rows_per_rank"),
          py::arg("keep_local") = true, py::arg("compile_only") = false,
          py::arg("config") = std::vector<int>{});
}

} // namespace deep_gemm::training
