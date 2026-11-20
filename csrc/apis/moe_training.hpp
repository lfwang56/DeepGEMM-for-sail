#pragma once

#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <deep_gemm/common/moe_training.cuh>
#include "../jit/compiler.hpp"
#include "../jit/kernel_runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/format.hpp"

namespace deep_gemm::training {

class Runtime final : public LaunchRuntime<Runtime> {
public:
    struct Args {
        int operation;
        LaunchArgs launch_args;
        Params params;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/moe_training.cuh>
extern "C" __launch_bounds__(256, 1)
__global__ void moe_training_{}(const deep_gemm::training::Params params) {{
    deep_gemm::training::run<static_cast<deep_gemm::training::Operation>({})>(params);
}}
)", args.operation, args.operation);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

// Internal, allocation-free tensor dispatcher. All kernels use the caller's
// current stream. It validates host-known shapes/types without reading device
// metadata, so warmed calls are legal during graph capture.
static void launch(const int operation, const std::vector<torch::Tensor>& tensors,
                   const std::vector<int64_t>& dims) {
    DG_HOST_ASSERT(operation >= 0 && operation <= int(Operation::ActivationBackwardQuantize));
    DG_HOST_ASSERT(!tensors.empty() && tensors.size() <= 12 && dims.size() <= 12);
    DG_HOST_ASSERT(tensors[0].is_cuda());
    const c10::cuda::CUDAGuard guard(tensors[0].device());
    for (const auto& tensor : tensors)
        DG_HOST_ASSERT(tensor.device() == tensors[0].device() && tensor.is_contiguous());
    for (const auto dimension : dims)
        DG_HOST_ASSERT(dimension >= 0 && dimension <= INT32_MAX);

    const auto schema = [&](size_t nt, size_t nd) {
        DG_HOST_ASSERT(tensors.size() == nt && dims.size() == nd);
    };
    const auto expect = [&](size_t i, int64_t elements, torch::ScalarType dtype) {
        DG_HOST_ASSERT(tensors[i].numel() == elements && tensors[i].scalar_type() == dtype);
    };
    constexpr auto bf = torch::kBFloat16, i32 = torch::kInt32, i64 = torch::kInt64;
    constexpr auto fp = torch::kFloat, f8 = torch::kFloat8_e4m3fn;
    int64_t work = 0, blocks = -1;
    switch (static_cast<Operation>(operation)) {
        case Operation::CountRoutes: {
            schema(5, 4);
            const auto r = dims[0], e = dims[1];
            DG_HOST_ASSERT(e > 0 && dims[3] > 0 && dims[2] < dims[3]);
            expect(0, r, i64); expect(1, e, i32); expect(2, r, i32);
            expect(3, r, i32); expect(4, 1, i32);
            work = r;
            break;
        }
        case Operation::PrefixRoutes:
            schema(3, 2);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 128 == 0);
            expect(0, dims[0], i32); expect(1, dims[0] + 1, i32); expect(2, dims[0], i32);
            blocks = 1;
            break;
        case Operation::InitializeLayout:
            schema(3, 2);
            DG_HOST_ASSERT(dims[1] > 0);
            expect(0, dims[1] + 1, i32); expect(1, dims[0], i32); expect(2, dims[0], i32);
            work = dims[0];
            break;
        case Operation::PackInput: {
            schema(7, 4);
            const auto r = dims[0], h = dims[1], k = dims[2], m = dims[3];
            DG_HOST_ASSERT(h > 0 && k > 0 && r % k == 0 && tensors[3].numel() > 1);
            expect(0, r / k * h, bf); expect(1, r, i32); expect(2, r, i32);
            DG_HOST_ASSERT(tensors[3].scalar_type() == i32);
            expect(4, m * h, bf); expect(5, m, i32); expect(6, 1, i32);
            blocks = r;
            break;
        }
        case Operation::UnpackRoutes:
        case Operation::PackRouteGrad: {
            schema(3, 3);
            const auto m = dims[0], h = dims[1], r = dims[2];
            DG_HOST_ASSERT(h > 0);
            expect(1, m, i32);
            if (static_cast<Operation>(operation) == Operation::UnpackRoutes) {
                expect(0, m * h, bf); expect(2, r * h, bf);
            } else {
                expect(0, r * h, bf); expect(2, m * h, bf);
            }
            work = m * h;
            break;
        }
        case Operation::Combine:
            schema(3, 6);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[5] <= 1);
            expect(0, dims[0] * dims[1] * dims[2], bf);
            expect(1, dims[3] * dims[2], fp); expect(2, dims[3] * dims[1], bf);
            work = dims[0] * dims[1];
            break;
        case Operation::Activation:
        case Operation::ActivationBackward:
            schema(static_cast<Operation>(operation) == Operation::Activation ? 2 : 3, 2);
            DG_HOST_ASSERT(dims[1] > 0);
            expect(0, dims[0] * dims[1] * 2, bf); expect(1, dims[0] * dims[1], bf);
            if (tensors.size() == 3) expect(2, dims[0] * dims[1] * 2, bf);
            work = dims[0] * dims[1];
            break;
        case Operation::QuantizeRows:
            schema(3, 2);
            DG_HOST_ASSERT(dims[1] > 0 && dims[1] % 128 == 0);
            expect(0, dims[0] * dims[1], bf); expect(1, dims[0] * dims[1], f8);
            expect(2, dims[0] * (dims[1] / 128), fp);
            blocks = (dims[0] * (dims[1] / 128) + 7) / 8;
            break;
        case Operation::ActivationQuantize:
        case Operation::ActivationBackwardQuantize: {
            const bool backward = static_cast<Operation>(operation) == Operation::ActivationBackwardQuantize;
            schema(backward ? 5 : 4, 2);
            const auto m = dims[0], width = dims[1];
            DG_HOST_ASSERT(width > 0 && width % 128 == 0);
            expect(0, m * 2 * width, bf); expect(1, m * width, bf);
            if (backward) expect(2, m * 2 * width, bf);
            const auto output_width = backward ? 2 * width : width;
            expect(backward ? 3 : 2, m * output_width, f8);
            expect(backward ? 4 : 3, m * (output_width / 128), fp);
            blocks = (m * (width / 128) + 7) / 8;
            break;
        }
        case Operation::QuantizeWeights:
            schema(3, 3);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0);
            DG_HOST_ASSERT(dims[1] % 128 == 0 && dims[2] % 128 == 0);
            expect(0, dims[0] * dims[1] * dims[2], bf);
            expect(1, dims[0] * dims[1] * dims[2], f8);
            blocks = dims[0] * (dims[1] / 128) * (dims[2] / 128);
            expect(2, blocks, fp);
            break;
        case Operation::WeightedGradient:
        case Operation::ScoreGradient:
            schema(3, 5);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0);
            if (static_cast<Operation>(operation) == Operation::WeightedGradient) {
                expect(0, dims[3] * dims[1], bf); expect(1, dims[3] * dims[2], fp);
                expect(2, dims[0] * dims[1] * dims[2], bf);
                work = dims[0] * dims[1] * dims[2];
            } else {
                expect(0, dims[0] * dims[1] * dims[2], bf);
                expect(1, dims[3] * dims[1], bf); expect(2, dims[3] * dims[2], fp);
                blocks = dims[0] * dims[2];
            }
            break;
        case Operation::ReadChunk:
            schema(6, 5);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[3] > 0);
            expect(0, dims[0] * dims[2], bf); expect(1, dims[0] * dims[3], i64);
            expect(2, dims[1] * dims[2], bf); expect(3, dims[1] * dims[3], i64);
            expect(4, 1, i32); expect(5, 1, i32);
            work = dims[1] * std::max(dims[2], dims[3]);
            break;
    }
    if (blocks < 0) blocks = (work + 255) / 256;
    if (!blocks) return;
    DG_HOST_ASSERT(blocks <= INT32_MAX);
    Params params{};
    for (size_t i = 0; i < tensors.size(); ++i) params.ptr[i] = tensors[i].data_ptr();
    for (size_t i = 0; i < dims.size(); ++i) params.dim[i] = dims[i];
    Runtime::Args args{operation, {dim3(blocks), dim3(256), 0}, params};
    const auto code = Runtime::generate(args);
    const auto runtime = compiler->build(fmt::format("moe_training_{}", operation), code);
    Runtime::launch(runtime, args);
}

// Compile the fixed pointwise variants without publishing EP payloads. GEMM
// preparation alone cannot protect peers from a later pointwise compiler error.
static void precompile(const int device, const std::vector<int>& operations) {
    const c10::cuda::CUDAGuard guard(device);
    for (const auto operation : operations) {
        DG_HOST_ASSERT(operation >= 0 && operation <= int(Operation::ActivationBackwardQuantize));
        Runtime::Args args{operation, {dim3(1), dim3(256), 0}, Params{}};
        compiler->build(fmt::format("moe_training_{}", operation), Runtime::generate(args));
    }
}

static void register_apis(pybind11::module_& m) {
    m.def("_moe_training_launch", &launch, py::arg("operation"), py::arg("tensors"), py::arg("dims"));
    m.def("_moe_training_precompile", &precompile, py::arg("device"), py::arg("operations"));
}

}  // namespace deep_gemm::training
