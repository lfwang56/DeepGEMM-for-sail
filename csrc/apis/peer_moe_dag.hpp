#pragma once
#include <deep_gemm/common/peer_moe_dag.cuh>
#include "peer_moe_training.hpp"

namespace deep_gemm::training {

class PeerDagRuntime final : public LaunchRuntime<PeerDagRuntime> {
public:
    struct Args {
        int h, i, e, bm, bn, bk, wm, wn, stages, threads;
        bool return_output, trace, shared_pool, fc1_activation, input_prefetch, backward, direct_return;
        LaunchArgs launch_args;
        PeerFfnDagParams params;
        int input_copy_ilp = 1;
    };
    static std::string name(const Args& a) {
        return fmt::format("bf16_peer_dag_h{}_i{}_e{}_bm{}_bn{}_bk{}_wm{}_wn{}_s{}_r{}_t{}{}{}{}{}{}{}",
            a.h, a.i, a.e, a.bm, a.bn, a.bk, a.wm, a.wn, a.stages, a.return_output, a.trace,
            a.shared_pool ? "_shared_pool" : "", a.fc1_activation ? "_g8_swiglu" : "",
            a.input_prefetch ? "_input_prefetch" : "", a.backward ? "_backward_g8" : "",
            a.direct_return ? "_direct_return" : "",
            a.input_copy_ilp == 1 ? "" : fmt::format("_copy_ilp{}", a.input_copy_ilp));
    }
    static std::string generate_impl(const Args& a) {
        return fmt::format(R"(
#include <deep_gemm/impls/peer_moe_dag.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void {}(const deep_gemm::training::PeerFfnDagParams params) {{
    deep_gemm::training::{}<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}{}>(params);
}}
)", a.threads, name(a), a.shared_pool ? "peer_ffn_shared_pool_impl" : "peer_ffn_dag_impl",
       a.h, a.i, a.e, a.bm, a.bn, a.bk, a.wm, a.wn,
       a.threads, a.stages, a.return_output ? "true" : "false", a.trace ? "true" : "false",
       a.shared_pool ? fmt::format(", {}, {}, {}, {}, {}", a.fc1_activation ? "true" : "false",
                                  a.input_prefetch ? "true" : "false", a.backward ? "true" : "false",
                                  a.direct_return ? "true" : "false", a.input_copy_ilp) : "");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static int peer_ffn_dag(const std::vector<torch::Tensor>& tensors,
                        const std::vector<torch::Tensor>& controls,
                        int source_rows_per_rank, int destination_rows_per_rank,
                        int block_m, const std::vector<int>& config,
                        int input_ctas, int return_ctas, bool return_output,
                        bool serialize, bool trace, bool compile_only, bool shared_pool = false,
                        bool fc1_activation = false, bool input_prefetch = false,
                        bool backward = false, bool direct_return = false, int input_copy_ilp = 1) {
    DG_HOST_ASSERT(input_copy_ilp == 1 || input_copy_ilp == 2 || input_copy_ilp == 4 || input_copy_ilp == 8);
    DG_HOST_ASSERT(input_copy_ilp == 1 || (shared_pool && !backward));
    DG_HOST_ASSERT((!fc1_activation && !input_prefetch && !backward && !direct_return) || shared_pool);
    DG_HOST_ASSERT(!backward || (return_output && !fc1_activation));
    DG_HOST_ASSERT(tensors.size() == (backward ? 14 : 12) && controls.size() == 9 && tensors[0].is_cuda());
    const auto device = tensors[0].device();
    const c10::cuda::CUDAGuard guard(device);
    DG_HOST_ASSERT(get_ppu_arch() == AC_PPU0015);
    for (size_t j = 0; j < tensors.size(); ++j) {
        const auto& t = tensors[j];
        DG_HOST_ASSERT(t.device() == device && t.is_contiguous() && t.numel() <= INT32_MAX);
        DG_HOST_ASSERT(t.scalar_type() == (j < 6 ? torch::kBFloat16 : j < 8 || j >= 12 ? torch::kInt64 : torch::kInt32));
    }
    for (size_t j = 0; j < controls.size(); ++j) {
        const auto& t = controls[j];
        DG_HOST_ASSERT(t.device() == device && t.is_contiguous() && t.numel() <= INT32_MAX);
        DG_HOST_ASSERT(t.scalar_type() == ((j == 6 || j == 7) ? torch::kInt64 : torch::kInt32));
    }
    const auto& input = tensors[0];
    const auto& w1 = tensors[1];
    const auto& gate = tensors[2];
    const auto& hidden = tensors[3];
    const auto& w2 = tensors[4];
    const auto& output = tensors[5];
    DG_HOST_ASSERT(input.dim() == 2 && w1.dim() == 3 && w2.dim() == 3);
    const int m = input.size(0), h = input.size(1), e = w1.size(0), i = w1.size(1) / (backward ? 1 : 2);
    DG_HOST_ASSERT((block_m == 128 || block_m == 256) && m > 0 && m % block_m == 0);
    DG_HOST_ASSERT(h > 0 && h % 256 == 0 && i > 0 && i % 128 == 0 && e > 0);
    DG_HOST_ASSERT(w1.sizes() == at::IntArrayRef({e, backward ? i : 2*i, h})
                   && w2.sizes() == at::IntArrayRef({e, h, backward ? 2*i : i}));
    DG_HOST_ASSERT(gate.sizes() == at::IntArrayRef({m, 2*i}));
    DG_HOST_ASSERT(hidden.sizes() == at::IntArrayRef({m, i}) && output.sizes() == at::IntArrayRef({m, h}));
    const int ranks = tensors[6].numel();
    DG_HOST_ASSERT(tensors[6].dim() == 1 && ranks > 0 && ranks <= 64 && tensors[7].sizes() == tensors[6].sizes());
    DG_HOST_ASSERT(tensors[8].numel() == m && tensors[9].numel() == m);
    DG_HOST_ASSERT(tensors[10].sizes() == at::IntArrayRef({m / block_m, 4}) && tensors[11].numel() == 1);
    DG_HOST_ASSERT(source_rows_per_rank > 0 && destination_rows_per_rank > 0);
    DG_HOST_ASSERT(!backward || destination_rows_per_rank % source_rows_per_rank == 0);
    if (backward) {
        DG_HOST_ASSERT(tensors[12].sizes() == tensors[6].sizes()
                       && tensors[13].sizes() == tensors[6].sizes());
    }
    DG_HOST_ASSERT(int64_t(ranks) * source_rows_per_rank <= INT32_MAX);
    DG_HOST_ASSERT(int64_t(ranks) * destination_rows_per_rank <= INT32_MAX);
    DG_HOST_ASSERT(config.size() == 5);
    const int bn = config[0], bk = config[1], wm = config[2], wn = config[3], stages = config[4];
    DG_HOST_ASSERT((bn == 128 || bn == 256) && bk == 64 && wm == 64 && wn == 64 && stages >= 2 && stages <= 4);
    DG_HOST_ASSERT(h % bn == 0 && i % bn == 0 && input_ctas > 0 && return_ctas >= 0);
    DG_HOST_ASSERT(!return_output || return_ctas > 0);
    const int max_m = m / block_m, n2 = h / bn;
    DG_HOST_ASSERT(controls[0].numel() == DagStateSize
                   && controls[1].numel() == max_m * (shared_pool ? (1 + 2*i/bn) : 1));
    DG_HOST_ASSERT(controls[2].numel() == max_m * (i / bn) && controls[3].numel() == max_m);
    DG_HOST_ASSERT(controls[4].numel() == max_m * n2
                   && controls[5].numel() == max_m * n2 * (shared_pool ? 2 : 1));
    DG_HOST_ASSERT(controls[6].numel() == 12 && controls[8].numel() == 1);
    DG_HOST_ASSERT(controls[7].dim() == 2 && controls[7].size(1) == 6 && controls[7].size(0) > 0);
    const int threads = 32 * (block_m / wm) * (bn / wn);
    const int smem = gemm_smem_total_size(2, stages, block_m, bn, bk);
    DG_HOST_ASSERT(threads >= 64 && threads <= 1024 && smem <= 256*1024);
    PeerFfnDagParams p{};
    auto& first = p.ffn.first;
    first.gemm.a_ptr = input.data_ptr();
    first.gemm.b_ptr = w1.data_ptr();
    first.gemm.c_ptr = backward ? hidden.data_ptr() : gate.data_ptr();
    first.gemm.expert_ids_and_cumsum = tensors[10].data_ptr<int32_t>();
    first.gemm.sorted_token_ids = tensors[8].data_ptr<int32_t>();
    first.gemm.aligned_num_m_blocks = tensors[11].data_ptr<int32_t>();
    first.gemm.shape_m = m;
    if (fc1_activation || backward) first.activation_gate_up = backward ? gate.data_ptr() : hidden.data_ptr();
    first.source_peers = reinterpret_cast<const uint64_t*>(tensors[6].data_ptr<int64_t>());
    first.destination_peers = reinterpret_cast<const uint64_t*>(tensors[7].data_ptr<int64_t>());
    first.permutation = tensors[9].data_ptr<int32_t>();
    first.source_rows_per_rank = source_rows_per_rank;
    first.destination_rows_per_rank = destination_rows_per_rank;
    first.num_ranks = ranks;
    p.ffn.second = first;
    p.ffn.second.gemm.a_ptr = backward ? gate.data_ptr() : hidden.data_ptr();
    p.ffn.second.gemm.b_ptr = w2.data_ptr();
    p.ffn.second.gemm.c_ptr = output.data_ptr();
    if (backward) {
        // These scratch pointers are unused by local-AIU GEMM tasks. Keep
        // score/dScore pointer tables here without changing the GPU ABI.
        first.dispatch_scratch = tensors[12].data_ptr();
        p.ffn.second.dispatch_scratch = tensors[13].data_ptr();
    }
    p.state = reinterpret_cast<uint32_t*>(controls[0].data_ptr<int32_t>());
    p.input_ready = reinterpret_cast<uint32_t*>(controls[1].data_ptr<int32_t>());
    p.pair_done = reinterpret_cast<uint32_t*>(controls[2].data_ptr<int32_t>());
    p.activated_pairs = reinterpret_cast<uint32_t*>(controls[3].data_ptr<int32_t>());
    p.queue = reinterpret_cast<uint32_t*>(controls[4].data_ptr<int32_t>());
    p.output_ready = reinterpret_cast<uint32_t*>(controls[5].data_ptr<int32_t>());
    p.monitor = reinterpret_cast<uint64_t*>(controls[6].data_ptr<int64_t>());
    p.samples = reinterpret_cast<uint64_t*>(controls[7].data_ptr<int64_t>());
    p.sample_capacity = controls[7].size(0);
    p.global_status = reinterpret_cast<uint32_t*>(controls[8].data_ptr<int32_t>());
    p.input_ctas = input_ctas;
    p.return_ctas = return_output ? return_ctas : 0;
    p.serialize = serialize;
    PeerDagRuntime::Args a{h,i,e,block_m,bn,bk,wm,wn,stages,threads,return_output,trace,shared_pool,fc1_activation,input_prefetch,backward,direct_return,
        {get_grid_shape(get_num_sms()),dim3(threads),smem},p};
    a.input_copy_ilp = input_copy_ilp;
    const auto runtime = compiler->build(PeerDagRuntime::name(a), PeerDagRuntime::generate(a));
    int occupancy = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, runtime->kernel, threads, smem));
    DG_HOST_ASSERT(occupancy > 0 && get_num_sms() > 1);
    a.launch_args.grid_dim.x = (get_num_sms() - 1) * occupancy;
    DG_HOST_ASSERT((shared_pool ? 0 : input_ctas + (return_output ? return_ctas : 0)) + int(trace)
                   < a.launch_args.grid_dim.x);
    if (get_env<int>("show_log"))
        std::printf("[PeerDag] BM%d BN%d threads%d smem%d occupancy%d grid%u input%d return%d trace%d\n",
            block_m,bn,threads,smem,occupancy,a.launch_args.grid_dim.x,input_ctas,p.return_ctas,trace);
    if (!compile_only) PeerDagRuntime::launch(runtime,a);
    return a.launch_args.grid_dim.x;
}

static void register_peer_dag_apis(pybind11::module_& m) {
    m.def("_moe_training_peer_dag", &peer_ffn_dag, py::arg("tensors"), py::arg("controls"),
        py::arg("source_rows_per_rank"), py::arg("destination_rows_per_rank"),
        py::arg("block_m"), py::arg("config"), py::arg("input_ctas"), py::arg("return_ctas"),
        py::arg("return_output"), py::arg("serialize") = false, py::arg("trace") = false,
        py::arg("compile_only") = false, py::arg("shared_pool") = false,
        py::arg("fc1_activation") = false, py::arg("input_prefetch") = false,
        py::arg("backward") = false, py::arg("direct_return") = false,
        py::arg("input_copy_ilp") = 1);
}

}  // namespace deep_gemm::training
