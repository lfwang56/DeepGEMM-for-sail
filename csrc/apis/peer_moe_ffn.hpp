#pragma once

#include <deep_gemm/common/peer_moe_ffn.cuh>
#include "peer_moe_training.hpp"

namespace deep_gemm::training {

class PeerFfnRuntime final : public LaunchRuntime<PeerFfnRuntime> {
public:
    struct Args {
        int h, i, e, bn, bk, wm, wn, stages, threads;
        bool backward;
        LaunchArgs launch_args;
        PeerFfnParams params;
    };
    static std::string name(const Args& a) {
        return fmt::format("bf16_peer_ffn_h{}_i{}_e{}_bn{}_bk{}_wm{}_wn{}_s{}_b{}",
            a.h, a.i, a.e, a.bn, a.bk, a.wm, a.wn, a.stages, a.backward);
    }
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/peer_moe_ffn.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void {}(const deep_gemm::training::PeerFfnParams params) {{
    deep_gemm::training::peer_ffn_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>(params);
}}
)", a.threads, name(a), a.h, a.i, a.e, a.bn, a.bk, a.wm, a.wn, a.threads,
       a.stages, a.backward ? "true" : "false");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static int peer_ffn(const std::vector<torch::Tensor>& tensors, int source_rows_per_rank,
                    int destination_rows_per_rank, bool backward,
                    const std::vector<int>& config, bool compile_only) {
    DG_HOST_ASSERT(tensors.size() == 12 && tensors[0].is_cuda());
    const auto device = tensors[0].device();
    const c10::cuda::CUDAGuard guard(device);
    DG_HOST_ASSERT(get_ppu_arch() == AC_PPU0015);
    for (size_t index = 0; index < tensors.size(); ++index) {
        const auto& t = tensors[index];
        DG_HOST_ASSERT(t.device() == device && t.is_contiguous() && t.numel() <= INT32_MAX);
        DG_HOST_ASSERT(t.scalar_type() == (index < 6 ? torch::kBFloat16 :
                       index < 8 ? torch::kInt64 : torch::kInt32));
    }
    const auto& input = tensors[0];
    const auto& first_weight = tensors[1];
    const auto& gate = tensors[2];
    const auto& hidden = tensors[3];
    const auto& second_weight = tensors[4];
    const auto& output = tensors[5];
    DG_HOST_ASSERT(input.dim() == 2 && first_weight.dim() == 3 && second_weight.dim() == 3);
    const int m = input.size(0), h = input.size(1), e = first_weight.size(0);
    const int i = backward ? first_weight.size(1) : first_weight.size(1) / 2;
    DG_HOST_ASSERT(m > 0 && m % 128 == 0 && h > 0 && h % 256 == 0 && i > 0 && i % 128 == 0 && e > 0);
    DG_HOST_ASSERT(first_weight.sizes() == at::IntArrayRef({e, backward ? i : 2 * i, h}));
    DG_HOST_ASSERT(second_weight.sizes() == at::IntArrayRef({e, h, backward ? 2 * i : i}));
    DG_HOST_ASSERT(gate.sizes() == at::IntArrayRef({m, 2 * i}));
    DG_HOST_ASSERT(hidden.sizes() == at::IntArrayRef({m, i}));
    DG_HOST_ASSERT(output.sizes() == at::IntArrayRef({m, h}));
    const int ranks = tensors[6].numel();
    DG_HOST_ASSERT(tensors[6].dim() == 1 && ranks > 0 && ranks <= 64 && tensors[7].sizes() == tensors[6].sizes());
    DG_HOST_ASSERT(tensors[8].numel() == m && tensors[9].numel() == m);
    DG_HOST_ASSERT(tensors[10].sizes() == at::IntArrayRef({m / 128, 4}) && tensors[11].numel() == 1);
    DG_HOST_ASSERT(source_rows_per_rank > 0 && destination_rows_per_rank > 0);
    DG_HOST_ASSERT(int64_t(ranks) * source_rows_per_rank <= INT32_MAX);
    DG_HOST_ASSERT(int64_t(ranks) * destination_rows_per_rank <= INT32_MAX);
    DG_HOST_ASSERT(config.empty() || config.size() == 5);
    const int bn = config.empty() ? 256 : config[0], bk = config.empty() ? 64 : config[1];
    const int wm = config.empty() ? 64 : config[2], wn = config.empty() ? 64 : config[3];
    const int stages = config.empty() ? 2 : config[4];
    DG_HOST_ASSERT(bn == 64 || bn == 128 || bn == 256);
    DG_HOST_ASSERT(bk == 64 || bk == 128);
    DG_HOST_ASSERT(wm == 32 || wm == 64 || wm == 128);
    DG_HOST_ASSERT(wn == 32 || wn == 64 || wn == 128);
    DG_HOST_ASSERT(bn % wn == 0 && stages >= 2 && stages <= 4);
    const int threads = 32 * (128 / wm) * (bn / wn);
    const int smem = gemm_smem_total_size(2, stages, 128, bn, bk);
    DG_HOST_ASSERT(threads >= 32 && threads <= 1024 && smem <= 256 * 1024);
    PeerFfnParams params{};
    auto& first = params.first;
    first.gemm.a_ptr = input.data_ptr();
    first.gemm.b_ptr = first_weight.data_ptr();
    first.gemm.c_ptr = backward ? hidden.data_ptr() : gate.data_ptr();
    first.gemm.expert_ids_and_cumsum = tensors[10].data_ptr<int32_t>();
    first.gemm.sorted_token_ids = tensors[8].data_ptr<int32_t>();
    first.gemm.aligned_num_m_blocks = tensors[11].data_ptr<int32_t>();
    first.gemm.shape_m = m;
    first.source_peers = reinterpret_cast<const uint64_t*>(tensors[6].data_ptr<int64_t>());
    first.destination_peers = reinterpret_cast<const uint64_t*>(tensors[7].data_ptr<int64_t>());
    first.permutation = tensors[9].data_ptr<int32_t>();
    first.source_rows_per_rank = source_rows_per_rank;
    first.destination_rows_per_rank = destination_rows_per_rank;
    first.num_ranks = ranks;
    if (backward) first.activation_gate_up = gate.data_ptr();
    params.second = first;
    params.second.gemm.a_ptr = backward ? gate.data_ptr() : hidden.data_ptr();
    params.second.gemm.b_ptr = second_weight.data_ptr();
    params.second.gemm.c_ptr = output.data_ptr();
    params.second.activation_gate_up = nullptr;
    PeerFfnRuntime::Args args{h, i, e, bn, bk, wm, wn, stages, threads, backward,
                             {get_grid_shape(get_num_sms()), dim3(threads), smem}, params};
    const auto runtime = compiler->build(PeerFfnRuntime::name(args), PeerFfnRuntime::generate(args));
    int occupancy = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, runtime->kernel, threads, smem));
    DG_HOST_ASSERT(occupancy > 0);
    args.launch_args.grid_dim.x *= occupancy;
    if (!compile_only) PeerFfnRuntime::launch(runtime, args);
    return args.launch_args.grid_dim.x;
}

static void register_peer_ffn_apis(pybind11::module_& m) {
    m.def("_moe_training_peer_ffn", &peer_ffn, py::arg("tensors"), py::arg("source_rows_per_rank"),
          py::arg("destination_rows_per_rank"), py::arg("backward") = false,
          py::arg("config") = std::vector<int>{}, py::arg("compile_only") = false);
}
}
