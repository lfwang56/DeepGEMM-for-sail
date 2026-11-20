#pragma once

#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <deep_gemm/common/peer_moe_training.cuh>
#include "../jit/compiler.hpp"
#include "../jit/kernel_runtime.hpp"
#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/layout.hpp"
#include "../utils/math.hpp"
#include "../utils/utils.hpp"

namespace deep_gemm::training {

class PeerRuntime final : public LaunchRuntime<PeerRuntime> {
public:
    struct Args {
        int operation;
        LaunchArgs launch_args;
        Params params;
    };
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/peer_moe_training.cuh>
extern "C" __launch_bounds__(256, 1)
__global__ void moe_training_peer_{}(const deep_gemm::training::Params params) {{
    deep_gemm::training::run_peer<static_cast<deep_gemm::training::PeerOperation>({})>(params);
}}
)", args.operation, args.operation);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static void launch_peer(const int operation, const std::vector<torch::Tensor>& tensors,
                        const std::vector<int64_t>& dims, bool compile_only) {
    DG_HOST_ASSERT(operation >= 0 && operation <= int(PeerOperation::Barrier));
    DG_HOST_ASSERT(!tensors.empty() && tensors.size() <= 12 && dims.size() <= 12);
    DG_HOST_ASSERT(tensors[0].is_cuda());
    const c10::cuda::CUDAGuard guard(tensors[0].device());
    for (const auto& tensor : tensors)
        DG_HOST_ASSERT(tensor.device() == tensors[0].device() && tensor.is_contiguous());
    for (const auto dimension : dims) DG_HOST_ASSERT(dimension >= 0);
    const auto schema = [&](size_t nt, size_t nd) {
        DG_HOST_ASSERT(tensors.size() == nt && dims.size() == nd);
    };
    const auto expect = [&](size_t i, int64_t elements, torch::ScalarType dtype) {
        DG_HOST_ASSERT(tensors[i].numel() == elements && tensors[i].scalar_type() == dtype);
    };
    constexpr auto bf = torch::kBFloat16, i32 = torch::kInt32, i64 = torch::kInt64;
    int64_t work = 0, blocks = -1;
    switch (static_cast<PeerOperation>(operation)) {
        case PeerOperation::CountRoutes:
            schema(5, 5);
            DG_HOST_ASSERT(dims[1] > 0 && dims[3] > 0 && dims[3] <= 64 && dims[2] < dims[3]);
            DG_HOST_ASSERT(dims[4] > 0 && dims[0] == dims[3] * dims[4]);
            expect(0, dims[3], i64); expect(1, dims[1], i32);
            expect(2, dims[0], i32); expect(3, dims[0], i32); expect(4, 1, i32);
            work = dims[0];
            break;
        case PeerOperation::InitializeLayout:
            schema(2, 2);
            expect(0, dims[0], i32); expect(1, dims[0], i32);
            work = dims[0];
            break;
        case PeerOperation::BuildPermutation:
            schema(6, 3);
            DG_HOST_ASSERT(dims[2] > 0 && tensors[2].numel() > 1);
            expect(0, dims[0], i32); expect(1, dims[0], i32);
            DG_HOST_ASSERT(tensors[2].scalar_type() == i32);
            expect(3, dims[1], i32); expect(4, dims[1], i32); expect(5, 1, i32);
            work = dims[0];
            break;
        case PeerOperation::BuildWorkInfo:
            schema(4, 3);
            DG_HOST_ASSERT(dims[0] > 0 && dims[2] > 0 && dims[1] % dims[2] == 0);
            expect(0, dims[0] + 1, i32); expect(1, dims[0], i32);
            expect(2, (dims[1] / dims[2]) * 4, i32); expect(3, 1, i32);
            work = dims[1] / dims[2];
            break;
        case PeerOperation::Pack:
            schema(4, 5);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[3] > 0 && dims[4] > 0 && dims[4] <= 64);
            expect(0, dims[4], i64); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, 1, i32);
            work = dims[0] * dims[1];
            break;
        case PeerOperation::Barrier:
            schema(3, 4);
            DG_HOST_ASSERT(dims[0] > 0 && dims[0] <= 64 && dims[1] < dims[0] && dims[2] > 0);
            expect(0, dims[0], i64); expect(1, 4, i32); expect(2, 1, i32);
            blocks = 1;
            break;
    }
    if (blocks < 0) blocks = (work + 255) / 256;
    if (!blocks) return;
    DG_HOST_ASSERT(blocks <= INT32_MAX);
    Params params{};
    for (size_t i = 0; i < tensors.size(); ++i) params.ptr[i] = tensors[i].data_ptr();
    for (size_t i = 0; i < dims.size(); ++i) params.dim[i] = dims[i];
    PeerRuntime::Args args{operation, {dim3(blocks), dim3(256), 0}, params};
    const auto code = PeerRuntime::generate(args);
    const auto runtime = compiler->build(fmt::format("moe_training_peer_{}", operation), code);
    if (!compile_only) PeerRuntime::launch(runtime, args);
}

class PeerGemmRuntime final : public LaunchRuntime<PeerGemmRuntime> {
public:
    struct Args {
        int n, k, groups;
        bool read_peer, write_peer;
        LaunchArgs launch_args;
        PeerGemmParams params;
    };
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/peer_moe_gemm.cuh>
extern "C" __launch_bounds__(128, 1)
__global__ void bf16_training_peer_gemm(const deep_gemm::training::PeerGemmParams params) {{
    deep_gemm::training_bf16_peer_gemm_impl<
        cutlass::bfloat16_t, deep_gemm::GemmType::MGroupedFused,
        {}, {}, {}, 128, 128, 64, 64, 64, 128, 3, {}, {}
    >(params);
}}
)", args.n, args.k, args.groups, args.read_peer ? "true" : "false", args.write_peer ? "true" : "false");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static void peer_gemm(const torch::Tensor& a, const torch::Tensor& weight, const torch::Tensor& out,
                      const torch::Tensor& source_peers, const torch::Tensor& destination_peers,
                      const torch::Tensor& input_rows, const torch::Tensor& permutation,
                      const torch::Tensor& work_info, const torch::Tensor& num_blocks,
                      int source_rows_per_rank, int destination_rows_per_rank,
                      bool read_peer, bool write_peer, bool compile_only) {
    DG_HOST_ASSERT(a.is_cuda() && a.dim() == 2 && weight.dim() == 3 && out.dim() == 2);
    const c10::cuda::CUDAGuard guard(a.device());
    for (const auto& tensor : {a, weight, out, source_peers, destination_peers, input_rows,
                               permutation, work_info, num_blocks})
        DG_HOST_ASSERT(tensor.device() == a.device() && tensor.is_contiguous());
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16);
    const auto [m, k] = get_shape<2>(a);
    const auto [groups, n, k_] = get_shape<3>(weight);
    DG_HOST_ASSERT(k == k_ && out.size(0) == m && out.size(1) == n && m > 0 && m % 128 == 0);
    DG_HOST_ASSERT(n > 0 && n % 128 == 0 && k > 0 && k % 128 == 0 && groups > 0);
    DG_HOST_ASSERT(source_peers.dim() == 1 && source_peers.numel() > 0 && source_peers.numel() <= 64);
    DG_HOST_ASSERT(source_peers.scalar_type() == torch::kInt64 && destination_peers.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(source_peers.sizes() == destination_peers.sizes());
    DG_HOST_ASSERT(input_rows.numel() == m && permutation.numel() == m);
    DG_HOST_ASSERT(input_rows.scalar_type() == torch::kInt32 && permutation.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(work_info.sizes() == at::IntArrayRef({m / 128, 4}) && work_info.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(num_blocks.numel() == 1 && num_blocks.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(source_rows_per_rank > 0 && destination_rows_per_rank > 0);
    DG_HOST_ASSERT(get_ppu_arch() == AC_PPU0015);
    const int ranks = source_peers.numel();
    DG_HOST_ASSERT(int64_t(ranks) * source_rows_per_rank <= INT32_MAX);
    PeerGemmParams params{};
    params.gemm.a_ptr = a.data_ptr();
    params.gemm.b_ptr = weight.data_ptr();
    params.gemm.c_ptr = out.data_ptr();
    params.gemm.expert_ids_and_cumsum = work_info.data_ptr<int32_t>();
    params.gemm.sorted_token_ids = input_rows.data_ptr<int32_t>();
    params.gemm.aligned_num_m_blocks = num_blocks.data_ptr<int32_t>();
    params.gemm.shape_m = read_peer ? ranks * source_rows_per_rank : m;
    params.source_peers = reinterpret_cast<uint64_t*>(source_peers.data_ptr<int64_t>());
    params.destination_peers = reinterpret_cast<uint64_t*>(destination_peers.data_ptr<int64_t>());
    params.permutation = permutation.data_ptr<int32_t>();
    params.source_rows_per_rank = source_rows_per_rank;
    params.destination_rows_per_rank = destination_rows_per_rank;
    params.num_ranks = ranks;
    const int smem = gemm_smem_total_size(2, 3, 128, 128, 64);
    PeerGemmRuntime::Args args{n, k, groups, read_peer, write_peer,
                              {get_grid_shape(get_num_sms()), dim3(128), smem}, params};
    const auto code = PeerGemmRuntime::generate(args);
    const auto runtime = compiler->build("bf16_training_peer_gemm", code);
    int occupancy = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, runtime->kernel, 128, smem));
    DG_HOST_ASSERT(occupancy > 0);
    args.launch_args.grid_dim.x *= occupancy;
    if (!compile_only) PeerGemmRuntime::launch(runtime, args);
}

static void register_peer_apis(pybind11::module_& m) {
    m.def("_moe_training_peer_launch", &launch_peer, py::arg("operation"), py::arg("tensors"),
          py::arg("dims"), py::arg("compile_only") = false);
    m.def("_moe_training_peer_gemm", &peer_gemm,
          py::arg("a"), py::arg("weight"), py::arg("out"), py::arg("source_peers"), py::arg("destination_peers"),
          py::arg("input_rows"), py::arg("permutation"), py::arg("work_info"), py::arg("num_blocks"),
          py::arg("source_rows_per_rank"), py::arg("destination_rows_per_rank"),
          py::arg("read_peer"), py::arg("write_peer"), py::arg("compile_only") = false);
}

}  // namespace deep_gemm::training
