#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <deep_gemm/common/moe_training_pipeline.cuh>
#include <deep_gemm/common/fused_gemm_common.cuh>
#include "../jit/compiler.hpp"
#include "../jit/kernel_runtime.hpp"
#include "../utils/utils.hpp"

namespace deep_gemm::training {

class TrainingPipelineRuntime final : public LaunchRuntime<TrainingPipelineRuntime> {
public:
    struct Args {
        int h, i, e, k, p, c;
        bool backward;
        LaunchArgs launch_args;
        PipelineParams params;
    };
    static std::string name(const Args& a) {
        return fmt::format("bf16_moe_pipeline_b{}_h{}_i{}_e{}_k{}_p{}_c{}",
                           a.backward, a.h, a.i, a.e, a.k, a.p, a.c);
    }
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/moe_training_pipeline.cuh>
extern "C" __launch_bounds__(128, 1)
__global__ void {}(const deep_gemm::training::PipelineParams params) {{
    deep_gemm::training::run_training_pipeline<{}, {}, {}, {}, {}, {}, {}>(params);
}}
)", name(a), a.h, a.i, a.e, a.k, a.p, a.c, a.backward ? "true" : "false");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args a) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, a.params));
    }
};

static void launch_training_pipeline(const std::vector<torch::Tensor>& tensors,
                                     const std::vector<int64_t>& dims,
                                     bool backward, bool compile_only) {
    DG_HOST_ASSERT(tensors.size() == NumPipelineBuffers && dims.size() == 6);
    const auto& x = tensors[X];
    DG_HOST_ASSERT(x.is_cuda() && x.dim() == 2 && tensors[W1].dim() == 3 && tensors[W2].dim() == 3);
    const c10::cuda::CUDAGuard guard(x.device());
    DG_HOST_ASSERT(get_ppu_arch() == AC_PPU0015);
    for (const auto& t : tensors) DG_HOST_ASSERT(t.device() == x.device() && t.is_contiguous());
    DG_HOST_ASSERT(tensors[Ids].dim() == 2 && tensors[LocalX].dim() == 2);
    const int64_t t = x.size(0), h = x.size(1), i = tensors[W1].size(1) / 2;
    const int64_t e = tensors[W1].size(0), k = tensors[Ids].size(1);
    const int64_t p = tensors[XPeers].numel(), c = tensors[LocalX].size(0);
    DG_HOST_ASSERT(t == dims[0] && dims[1] >= t && dims[2] >= 0 && dims[2] < p);
    const int64_t m = dims[3], alignment = dims[4], ctas = dims[5];
    DG_HOST_ASSERT(m > 0 && m <= INT32_MAX && m % 128 == 0);
    DG_HOST_ASSERT(alignment >= 128 && alignment % 128 == 0);
    DG_HOST_ASSERT(h > 0 && h % 256 == 0 && i > 0 && i % 128 == 0 && e > 0 && k > 0);
    DG_HOST_ASSERT(p > 0 && p <= 64 && c > 0 && int64_t(p) * c * k <= INT32_MAX);
    DG_HOST_ASSERT(ctas > 0 && ctas <= device_runtime->get_prop()->multiProcessorCount);
    const auto expect = [&](int index, std::initializer_list<int64_t> shape, torch::ScalarType dtype) {
        DG_HOST_ASSERT(tensors[index].sizes() == at::IntArrayRef(shape));
        DG_HOST_ASSERT(tensors[index].scalar_type() == dtype);
    };
    constexpr auto bf = torch::kBFloat16, f32 = torch::kFloat32, i32 = torch::kInt32, i64 = torch::kInt64;
    expect(X, {t, h}, bf); expect(Ids, {t, k}, i64); expect(Scores, {t, k}, f32);
    expect(W1, {e, 2 * i, h}, bf); expect(W2, {e, h, i}, bf);
    DG_HOST_ASSERT(tensors[Valid].numel() == 1 && tensors[Valid].scalar_type() == i32);
    if (backward) {
        expect(DY, {t, h}, bf); expect(DX, {t, h}, bf);
        expect(DW1, {e, 2 * i, h}, f32); expect(DW2, {e, h, i}, f32);
        expect(DScores, {t, k}, f32);
    } else {
        expect(Y, {t, h}, bf);
    }
    expect(LocalX, {c, h}, bf); expect(LocalIds, {c, k}, i64);
    for (int index : {LocalY, LocalDX, LocalDY}) expect(index, {c * k, h}, bf);
    expect(Counts, {e}, i32); expect(Offsets, {e + 1}, i32); expect(Lengths, {e}, i32);
    for (int index : {Permutation, SourceRows, IdentityRows}) expect(index, {m}, i32);
    for (int index : {RouteExpert, RouteRow}) expect(index, {p * c * k}, i32);
    for (int index : {PackedX, PackedDY}) expect(index, {m, h}, bf);
    expect(GateUp, {m, 2 * i}, bf); expect(Hidden, {m, i}, bf);
    expect(W1T, {e, h, 2 * i}, bf); expect(W2T, {e, i, h}, bf);
    expect(WorkInfo, {m / 128, 4}, i32); expect(NumBlocks, {1}, i32); expect(Status, {1}, i32);
    expect(PeerBarrierState, {4}, i32); expect(GridSync, {3}, i32);
    for (int index : {BarrierPeers, XPeers, IdPeers, YPeers, DXPeers, DYPeers}) expect(index, {p}, i64);
    PipelineParams params{};
    for (size_t index = 0; index < tensors.size(); ++index) params.ptr[index] = tensors[index].data_ptr();
    params.input_tokens = t;
    params.max_tokens = dims[1];
    params.rank = dims[2];
    params.rows = m;
    params.alignment = alignment;
    params.timeout_cycles = uint64_t(180) * 2'000'000'000;
    const int smem = gemm_smem_total_size(2, 3, 128, 128, 64);
    TrainingPipelineRuntime::Args args{int(h), int(i), int(e), int(k), int(p), int(c), backward,
                                       {dim3(ctas), dim3(128), smem}, params};
    const auto code = TrainingPipelineRuntime::generate(args);
    const auto runtime = compiler->build(TrainingPipelineRuntime::name(args), code);
    int resident = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&resident, runtime->kernel, 128, smem));
    DG_HOST_ASSERT(resident >= 1);
    if (!compile_only) TrainingPipelineRuntime::launch(runtime, args);
}

static void register_pipeline_apis(pybind11::module_& m) {
    m.def("_moe_training_pipeline", &launch_training_pipeline, py::arg("tensors"), py::arg("dims"),
          py::arg("backward"), py::arg("compile_only") = false);
}

}  // namespace deep_gemm::training
