#pragma once

#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <deep_gemm/common/moe_training.cuh>
#include "../jit/compiler.hpp"
#include "../jit/kernel_runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/utils.hpp"

namespace deep_gemm::training {

class Runtime final : public LaunchRuntime<Runtime> {
public:
    struct Args {
        int operation;
        LaunchArgs launch_args;
        Params params;
    };

    static bool active_quantization(const int operation) {
        return operation >= int(Operation::QuantizeRowsDeviceRows) &&
               operation <= int(Operation::ActivationBackwardQuantizeDeviceRows) ||
               operation == int(Operation::QuantizeRowsVectorDeviceRows);
    }

    static std::string kernel_name(const Args& args) {
        const auto name = fmt::format("moe_training_{}", args.operation);
        if(args.operation==int(Operation::PackFp8WeightGradients) && args.params.dim[4]>=4)
            return name+fmt::format("_cache_{}x{}_reciprocal",args.params.dim[4]==4?32:16,args.params.dim[4]==4?3072:6144);
        if (args.operation == int(Operation::PackFp8WeightGradients) && args.params.dim[4])
            return name + (args.params.dim[4]==1 || args.params.dim[4]==2 ? "_quantize_before_transpose" : "") +
                (args.params.dim[4]>=2 ? "_reciprocal" : "");
        if (!active_quantization(args.operation)) return name;
        return name + (args.params.dim[2] ? "_mn_scales" : "_row_scales");
    }

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/moe_training.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void {}(const deep_gemm::training::Params params) {{
    deep_gemm::training::run<static_cast<deep_gemm::training::Operation>({}){}>(params);
}}
)", args.launch_args.block_dim.x,kernel_name(args), args.operation,
       args.operation==int(Operation::PackFp8WeightGradients) && args.params.dim[4]>=4 ?
           fmt::format(", false, false, true, {}, {}",args.params.dim[4]==4?32:16,args.params.dim[4]==4?3072:6144) :
       args.operation == int(Operation::PackFp8WeightGradients) && args.params.dim[4] ?
           fmt::format(", false, {}, {}",args.params.dim[4]==1 || args.params.dim[4]==2 ? "true" : "false",
                       args.params.dim[4]>=2 ? "true" : "false") :
       !active_quantization(args.operation) ? "" : args.params.dim[2] ? ", true" : ", false");
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
    DG_HOST_ASSERT(operation >= 0 && operation <= int(Operation::QuantizeWeightsAndTranspose));
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
        case Operation::QuantizeWeightsAndTranspose: {
            DG_HOST_ASSERT(tensors.size()==10 && (dims.size()==3 || dims.size()==4));
            const auto e=dims[0],h=dims[1],i=dims[2],ctas=dims.size()==4?dims[3]:4;
            DG_HOST_ASSERT(e>0 && e<128 && h>0 && i>0 && h%128==0 && i%128==0 && ctas>0 && ctas<=16);
            DG_HOST_ASSERT(e*h*2*i<=INT32_MAX);
            expect(0,e*2*i*h,bf);expect(1,e*h*i,bf);
            expect(2,e*2*i*h,f8);expect(3,e*(2*i/128)*(h/128),fp);
            expect(4,e*h*i,f8);expect(5,e*(h/128)*(i/128),fp);
            expect(6,e*h*2*i,f8);expect(7,e*(h/128)*(2*i/128),fp);
            expect(8,e*i*h,f8);expect(9,e*(i/128)*(h/128),fp);
            for(const auto index:{0,1,2,4,6,8})
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%16==0);
            blocks=std::min<int64_t>(e*((2*i/128)*(h/128)+(h/128)*(i/128)),int64_t(get_num_sms())*ctas);
            break;
        }
        case Operation::PackFp8WeightGradients: {
            DG_HOST_ASSERT(tensors.size()==9 && (dims.size()==4 || dims.size()==6));
            const auto mode=dims.size()==6?dims[4]:0;
            const auto ctas=dims.size()==6?dims[5]:4;
            DG_HOST_ASSERT(mode<=5 && ctas>0 && ctas<=16);
            const auto g=dims[0],r=dims[1],m=dims[2],n=dims[3];
            DG_HOST_ASSERT(g>0 && g<128 && r>0 && r%128==0 && m>0 && n>0 && m%128==0 && n%128==0);
            DG_HOST_ASSERT(r*m<=INT32_MAX && r*n<=INT32_MAX);
            expect(0,r*m,bf);expect(1,r*n,bf);expect(2,r*m,f8);expect(3,g*m,fp);
            expect(4,r*n,f8);expect(5,g*n,fp);expect(6,g,i32);expect(7,g+1,i32);expect(8,1,i32);
            for(const auto index:{0,1}) DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%8==0);
            for(const auto index:{2,4}) DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr())%4==0);
            blocks=std::min<int64_t>(g*((m+n)/(mode==4?32:mode==5?16:128)),int64_t(get_num_sms())*ctas);
            break;
        }
        case Operation::TransposeQuantizedWeightsSwizzled:
        case Operation::TransposeQuantizedWeights: {
            schema(8, 3);
            const auto e = dims[0], h = dims[1], i = dims[2];
            DG_HOST_ASSERT(e > 0 && h > 0 && i > 0 && h % 128 == 0 && i % 128 == 0);
            DG_HOST_ASSERT(e * h * 2 * i <= INT32_MAX);
            expect(0, e * 2 * i * h, f8); expect(1, e * (2 * i / 128) * (h / 128), fp);
            expect(2, e * h * i, f8); expect(3, e * (h / 128) * (i / 128), fp);
            expect(4, e * h * 2 * i, f8); expect(5, e * (h / 128) * (2 * i / 128), fp);
            expect(6, e * i * h, f8); expect(7, e * (i / 128) * (h / 128), fp);
            for (const auto index : {0, 2, 4, 6})
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[index].data_ptr()) % 16 == 0);
            const auto tiles = e * (2 * i / 128) * (h / 128) + e * (h / 128) * (i / 128);
            blocks = std::min<int64_t>(tiles, int64_t(get_num_sms()) * 4);
            break;
        }
        case Operation::PackQuantizedRouteRows:
            schema(7, 4);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 128 == 0 && dims[2] > 0 && dims[3] > 0);
            DG_HOST_ASSERT(dims[0] * dims[1] <= INT32_MAX && dims[2] * dims[1] <= INT32_MAX);
            expect(0, dims[2] * dims[1], f8); expect(1, dims[2] * (dims[1] / 128), fp);
            expect(2, dims[0], i32); expect(3, dims[0] * dims[1], f8);
            expect(4, dims[0] * (dims[1] / 128), fp); expect(5, 1, i32); expect(6, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
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
            if (dims[1] % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(tensors[0].data_ptr()) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(tensors[2].data_ptr()) % 16 == 0)
                work /= 8;
            break;
        case Operation::Activation:
        case Operation::ActivationBackward:
            DG_HOST_ASSERT(dims.size() == 2 || dims.size() == 3);
            schema(static_cast<Operation>(operation) == Operation::Activation ? 2 : 3, dims.size());
            if (dims.size() == 3) DG_HOST_ASSERT(dims[2] == 8);
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
        case Operation::QuantizeRowsVectorDeviceRows:
        case Operation::QuantizeRowsDeviceRows:
        case Operation::ActivationQuantizeDeviceRows:
        case Operation::ActivationBackwardQuantizeDeviceRows: {
            const bool row = static_cast<Operation>(operation) == Operation::QuantizeRowsDeviceRows ||
                             static_cast<Operation>(operation) == Operation::QuantizeRowsVectorDeviceRows;
            const bool backward = static_cast<Operation>(operation) == Operation::ActivationBackwardQuantizeDeviceRows;
            const size_t count_index = row ? 3 : backward ? 5 : 4;
            schema(count_index + 2, 3);
            const auto m = dims[0], width = dims[1];
            DG_HOST_ASSERT(m > 0 && width > 0 && width % 128 == 0 && dims[2] <= 1);
            expect(0, m * width * (row ? 1 : 2), bf);
            if (!row) expect(1, m * width, bf);
            if (backward) expect(2, m * 2 * width, bf);
            const auto output_width = backward ? 2 * width : width;
            if(static_cast<Operation>(operation)==Operation::QuantizeRowsVectorDeviceRows) {
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr())%16==0);
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[1].data_ptr())%8==0);
            }
            expect(row ? 1 : backward ? 3 : 2, m * output_width, f8);
            expect(row ? 2 : backward ? 4 : 3, m * (output_width / 128), fp);
            expect(count_index, 1, i32); expect(count_index + 1, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::QuantizeWeights:
        case Operation::QuantizeWeightsCachedFloat:
        case Operation::QuantizeWeightsCachedInteger:
            schema(3, 3);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0);
            DG_HOST_ASSERT(dims[1] % 128 == 0 && dims[2] % 128 == 0);
            expect(0, dims[0] * dims[1] * dims[2], bf);
            expect(1, dims[0] * dims[1] * dims[2], f8);
            if (static_cast<Operation>(operation) != Operation::QuantizeWeights) {
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[0].data_ptr()) % 16 == 0);
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[1].data_ptr()) % 8 == 0);
            }
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
        case Operation::ActivationDeviceRows:
            DG_HOST_ASSERT(dims.size() == 3 || dims.size() == 4);
            schema(4, dims.size());
            if (dims.size() == 4) DG_HOST_ASSERT(dims[3] == 8);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] <= 1);
            expect(0, dims[0] * 2 * dims[1], bf); expect(1, dims[0] * dims[1], bf);
            expect(2, 1, i32); expect(3, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        case Operation::TransformDeviceRows:
        case Operation::TransformDeviceRowsVector: {
            schema(5, 4);
            DG_HOST_ASSERT(dims[1] > 0 && dims[3] <= 1);
            const bool backward = dims[3];
            expect(0, (backward ? dims[2] : dims[0]) * dims[1], bf);
            expect(1, dims[0], i32);
            expect(2, (backward ? dims[0] : dims[2]) * dims[1], bf);
            expect(3, 1, i32); expect(4, 1, i32);
            if (static_cast<Operation>(operation) == Operation::TransformDeviceRowsVector)
                DG_HOST_ASSERT(dims[1] % 8 == 0);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::ZeroDeviceRows:
            schema(3, 2);
            DG_HOST_ASSERT(dims[1] > 0);
            expect(0, dims[0] * dims[1], bf); expect(1, 1, i32); expect(2, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        case Operation::BuildPermutationOnly:
            schema(5, 2);
            expect(0, dims[0], i32); expect(1, dims[0], i32);
            DG_HOST_ASSERT(tensors[2].numel() > 1 && tensors[2].scalar_type() == i32);
            expect(3, dims[1], i32); expect(4, 1, i32);
            work = dims[0];
            break;
        case Operation::PackInputDeviceRows:
            schema(5, 4);
            DG_HOST_ASSERT(dims[1] > 0 && dims[1] % 8 == 0 && dims[2] > 0 && dims[3] > 0);
            expect(0, dims[2] * dims[1], bf); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, 1, i32); expect(4, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        case Operation::DeepEPReduceRows: {
            schema(7, 6);
            const auto m = dims[0], h = dims[1], n = dims[2], k = dims[3];
            DG_HOST_ASSERT(h > 0 && h % 8 == 0 && k > 0 && dims[4] > 0 && dims[5] <= 1);
            expect(0, m * h, bf); expect(1, n * k, i32); expect(2, n * k, i32);
            expect(3, dims[4] + 1, i32); expect(4, n * k, fp);
            expect(5, n * h, bf); expect(6, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::DeepEPReduceWindowRows: {
            schema(8, 8);
            const auto m = dims[0], h = dims[1], n = dims[2], k = dims[3];
            DG_HOST_ASSERT(h > 0 && h % 8 == 0 && k > 0 && k <= 32 && dims[4] > 0
                           && dims[5] % 128 == 0 && dims[6] <= 1 && dims[7] >= m);
            expect(0, m * h, bf); expect(1, n * k, i32); expect(2, n * k, i32);
            expect(3, dims[4] + 1, i32); expect(4, n * k, fp); expect(5, n * h, fp);
            expect(6, 1, i32); expect(7, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::DeepEPBackwardRows: {
            schema(8, 4);
            const auto m = dims[0], h = dims[1], n = dims[2], k = dims[3];
            DG_HOST_ASSERT(h > 0 && k > 0);
            expect(0, n * h, bf); expect(1, n * k, fp); expect(2, m, i32);
            expect(3, m * h, bf); expect(4, m * h, bf); expect(5, n * k, fp);
            expect(6, 1, i32); expect(7, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        }
        case Operation::TransposeWeights: {
            schema(4, 3);
            const auto e = dims[0], h = dims[1], i = dims[2];
            DG_HOST_ASSERT(e > 0 && h > 0 && i > 0 && h % 128 == 0 && i % 128 == 0);
            expect(0, e * 2 * i * h, bf); expect(1, e * h * i, bf);
            expect(2, e * h * 2 * i, bf); expect(3, e * i * h, bf);
            for (const auto& tensor : tensors)
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0);
            blocks = e * (2 * i / 128) * (h / 128) + e * (h / 128) * (i / 128);
            break;
        }
    }
    if (blocks < 0) blocks = (work + 255) / 256;
    if (!blocks) return;
    DG_HOST_ASSERT(blocks <= INT32_MAX);
    Params params{};
    for (size_t i = 0; i < tensors.size(); ++i) params.ptr[i] = tensors[i].data_ptr();
    for (size_t i = 0; i < dims.size(); ++i) params.dim[i] = dims[i];
    Runtime::Args args{operation, {dim3(blocks), dim3(operation==int(Operation::PackFp8WeightGradients) && params.dim[4]>=4 ? 1024 : 256), 0}, params};
    const auto code = Runtime::generate(args);
    const auto runtime = compiler->build(Runtime::kernel_name(args), code);
    Runtime::launch(runtime, args);
}

// Compile the fixed pointwise variants without publishing EP payloads. GEMM
// preparation alone cannot protect peers from a later pointwise compiler error.
static void precompile(const int device, const std::vector<int>& operations) {
    const c10::cuda::CUDAGuard guard(device);
    for (const auto operation : operations) {
        DG_HOST_ASSERT(operation >= 0 && operation <= int(Operation::QuantizeWeightsAndTranspose));
        const int variants = Runtime::active_quantization(operation) ? 2 : 1;
        for (int layout = 0; layout < variants; ++layout) {
            Runtime::Args args{operation, {dim3(1), dim3(256), 0}, Params{}};
            args.params.dim[2] = layout;
            compiler->build(Runtime::kernel_name(args), Runtime::generate(args));
        }
    }
}

static void register_apis(pybind11::module_& m) {
    m.def("_moe_training_launch", &launch, py::arg("operation"), py::arg("tensors"), py::arg("dims"));
    m.def("_moe_training_precompile", &precompile, py::arg("device"), py::arg("operations"));
}

}  // namespace deep_gemm::training
