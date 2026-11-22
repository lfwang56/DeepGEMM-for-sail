#pragma once

#include <algorithm>
#include <cstdio>
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
        const int specialize = get_env<int>("DG_INPUT_ASSIST_STATIC_GEOMETRY", 1);
        DG_HOST_ASSERT(specialize == 0 || specialize == 1);
        const int warp_rows = get_env<int>("DG_INPUT_ASSIST_WARP_ROWS", 1);
        DG_HOST_ASSERT(warp_rows == 0 || warp_rows == 1);
        const bool assist = args.operation == int(PeerOperation::AssistInputTiles) && specialize;
        const bool tiled_quant=args.operation==int(PeerOperation::ExpertLocalGradientsQuantizedTiles) ||
                               args.operation==int(PeerOperation::ExpertLocalGradientsQuantizedTilesCached);
        const bool fused_quant = args.operation == int(PeerOperation::PackActiveQuantized) ||
                                 args.operation == int(PeerOperation::ExpertLocalGradientsQuantized) ||
                                 tiled_quant;
        const bool cache_fp8_scales = args.operation == int(PeerOperation::StageFp8InputTiles) && args.params.dim[7];
        return fmt::format(R"(
#include <deep_gemm/impls/peer_moe_training.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void moe_training_peer_{}(const deep_gemm::training::Params params) {{
    deep_gemm::training::run_peer<static_cast<deep_gemm::training::PeerOperation>({}), {}, {}, {}{}>(params);
}}
)", args.launch_args.block_dim.x, args.operation, args.operation,
       fused_quant || assist || cache_fp8_scales || (args.operation == int(PeerOperation::StageFp8InputTiles) && specialize) ? args.params.dim[1] : 0, assist ? args.params.dim[4] : tiled_quant ? args.params.dim[2] : 0,
       args.operation == int(PeerOperation::AssistInputTiles) && warp_rows ? "true" : "false",
       tiled_quant ? fmt::format(", false, {}",args.params.dim[3]) : cache_fp8_scales ? ", true" : "");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static void launch_peer(const int operation, const std::vector<torch::Tensor>& tensors,
                        const std::vector<int64_t>& dims, bool compile_only) {
    DG_HOST_ASSERT(operation >= 0 && operation <= int(PeerOperation::ExpertLocalGradientsQuantizedTilesCached));
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
    int threads = 256;
    switch (static_cast<PeerOperation>(operation)) {
        case PeerOperation::ExpertLocalGradientsQuantizedTilesCached:
        case PeerOperation::ExpertLocalGradientsQuantizedTiles:
            schema(11,6);
            DG_HOST_ASSERT(dims[0]>0 && dims[0]%128==0 && dims[1]>0 && dims[1]%128==0 &&
                           dims[2]>0 && dims[3]>0 && dims[4]>0 && dims[4]<=64 && dims[5]==128);
            expect(0,dims[4],i64);expect(1,dims[4],i64);expect(2,dims[4],i64);
            expect(3,dims[0],i32);expect(4,dims[0]*dims[1],bf);expect(5,dims[0]*dims[1],bf);
            expect(6,1,i32);expect(7,1,i32);expect(8,dims[0]*dims[1],torch::kFloat8_e4m3fn);
            expect(9,dims[0]*(dims[1]/128),torch::kFloat32);
            expect(10,dims[0]*(dims[1]/128),torch::kFloat32);
            DG_HOST_ASSERT(dims[0]*dims[1]<=INT32_MAX &&
                reinterpret_cast<uintptr_t>(tensors[4].data_ptr())%16==0 &&
                reinterpret_cast<uintptr_t>(tensors[5].data_ptr())%16==0 &&
                reinterpret_cast<uintptr_t>(tensors[8].data_ptr())%8==0);
            blocks=int64_t(get_num_sms())*4;
            break;
        case PeerOperation::PackActiveQuantized:
            schema(7,6);
            DG_HOST_ASSERT(dims[0]>0 && dims[1]>0 && dims[1]%128==0 && dims[2]>0 && dims[3]>0 &&
                           dims[4]>0 && dims[4]<=64 && dims[5]==128 && dims[0]%128==0);
            expect(0,dims[4],i64);expect(1,dims[0],i32);expect(2,dims[0]*dims[1],bf);
            expect(3,1,i32);expect(4,1,i32);expect(5,dims[0]*dims[1],torch::kFloat8_e4m3fn);
            expect(6,dims[0]*(dims[1]/128),torch::kFloat32);
            DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[2].data_ptr())%16==0 &&
                           reinterpret_cast<uintptr_t>(tensors[5].data_ptr())%8==0);
            blocks=int64_t(get_num_sms())*4;
            break;
        case PeerOperation::PrepareContextReplay:
            schema(11, 3);
            DG_HOST_ASSERT(dims[0] > 0 && dims[0] < 128 && dims[1] > 0 && dims[2] % dims[1] == 0);
            expect(0, dims[0] + 1, i32); expect(1, dims[0], i32); expect(2, 1, i32);
            expect(3, dims[0] + 1, i32); expect(4, dims[0] + 1, i32);
            expect(5, dims[0], i32); expect(6, 1, i32); expect(7, 1, i32);
            expect(8, 1, i32); expect(9, 1, i32); expect(10, 1, i64);
            blocks = 1;
            break;
        case PeerOperation::StageFp8InputTiles:
            // Three source pointer tables (FP8, row scales, original BF16),
            // global source rows, local FP8/MN scales/BF16, count, ready, status.
            DG_HOST_ASSERT(dims.size() >= 6 && dims.size() <= 8);
            schema(10, dims.size());
            DG_HOST_ASSERT(dims[0] > 0 && dims[0] % 128 == 0 && dims[1] > 0 && dims[1] % 128 == 0
                           && dims[2] > 0 && dims[3] > 0 && dims[3] <= 64
                           && dims[4] > 0 && dims[4] < get_num_sms() && dims[5] <= 1);
            expect(0, dims[3], i64); expect(1, dims[3], i64); expect(2, dims[3], i64);
            expect(3, dims[0], i32); expect(4, dims[0] * dims[1], torch::kFloat8_e4m3fn);
            expect(5, dims[0] * (dims[1] / 128), torch::kFloat32);
            expect(6, dims[0] * dims[1], bf); expect(7, 1, i32);
            expect(8, dims[0] / 128, i32); expect(9, 1, i32);
            DG_HOST_ASSERT(dims[0] * dims[1] <= INT32_MAX);
            if (dims.size() == 8) DG_HOST_ASSERT(dims[7] <= 1 && (!dims[7] || dims[1] <= 8192));
            threads = dims.size() >= 7 ? dims[6] : 128;
            DG_HOST_ASSERT(threads == 128 || threads == 256 || threads == 512);
            blocks = dims[4];
            break;
        case PeerOperation::AssistInputTiles:
            schema(8, 7);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 8 == 0
                           && dims[2] > 0 && dims[3] > 0 && dims[3] <= 64
                           && (dims[4] == 128 || dims[4] == 256)
                           && dims[0] % dims[4] == 0 && dims[5] > 0 && dims[5] <= 156
                           && (dims[6] == 1 || dims[6] == 2 || dims[6] == 4 || dims[6] == 8));
            expect(0, dims[3], i64); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, dims[0] / dims[4] * 4, i32);
            expect(4, 1, i32); expect(5, dims[0] / dims[4] + 1, i32);
            expect(6, 1, i32); expect(7, 1, i32);
            threads = 128;
            blocks = dims[5];
            break;
        case PeerOperation::PrefetchRawDy:
            schema(5, 6);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 8 == 0
                           && dims[2] > 0 && dims[3] > 0 && dims[4] > 0 && dims[4] <= 64
                           && (dims[5] == 128 || dims[5] == 256));
            expect(0, dims[4], i64); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, 1, i32); expect(4, 1, i32);
            threads = 128;
            blocks = int64_t(get_num_sms()) * 2;
            break;
        case PeerOperation::SetRouteWindow:
            schema(8, 6);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0 && dims[2] % 128 == 0
                           && dims[3] % 128 == 0 && dims[4] > 0 && dims[5] > 0);
            expect(0, dims[0] + 1, i32); expect(1, dims[1], i32);
            expect(2, dims[0] + 1, i32); expect(3, dims[0], i32);
            expect(4, dims[2], i32); expect(5, dims[2], i32); expect(6, 1, i32);
            expect(7, dims[2], i32);
            blocks = int64_t(get_num_sms()) * 2;
            break;
        case PeerOperation::ReduceScorePartials:
            schema(5, 6);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0 && dims[3] > 0
                           && dims[4] > 0 && dims[4] <= 64 && (dims[5] == 128 || dims[5] == 256));
            expect(0, dims[0] * dims[1], torch::kFloat32); expect(1, dims[0], i32);
            expect(2, 1, i32); expect(3, dims[4], i64); expect(4, 1, i32);
            blocks = int64_t(get_num_sms()) * 2;
            break;
        case PeerOperation::StageRawWeightedDy:
            schema(8, 6);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[1] % 8 == 0
                           && dims[2] > 0 && dims[3] > 0 && dims[4] > 0 && dims[4] <= 64
                           && (dims[5] == 128 || dims[5] == 256) && dims[0] % dims[5] == 0);
            expect(0, dims[4], i64); expect(1, dims[4], i64); expect(2, dims[0], i32);
            expect(3, dims[0] * dims[1], bf); expect(4, dims[0] * dims[1], bf);
            expect(5, dims[0], torch::kFloat32); expect(6, 1, i32); expect(7, 1, i32);
            DG_HOST_ASSERT(!tensors[3].is_alias_of(tensors[4]));
            blocks = int64_t(get_num_sms()) * 2;
            break;
        case PeerOperation::ExpertLocalGradientsQuantized:
        case PeerOperation::ExpertLocalGradients: {
            DG_HOST_ASSERT(dims.size() == 5 || dims.size() == 6);
            const bool quantized = static_cast<PeerOperation>(operation)==PeerOperation::ExpertLocalGradientsQuantized;
            DG_HOST_ASSERT(quantized ? tensors.size()==10 : (tensors.size() == 8 || tensors.size() == 9));
            if (dims.size() == 6) DG_HOST_ASSERT(dims[5] == 128 || dims[5] == 256);
            DG_HOST_ASSERT(dims[1] > 0 && dims[1] % 8 == 0 && dims[2] > 0 && dims[3] > 0
                           && dims[4] > 0 && dims[4] <= 64);
            expect(0, dims[4], i64); expect(1, dims[4], i64); expect(2, dims[4], i64);
            expect(3, dims[0], i32); expect(4, dims[0] * dims[1], bf);
            expect(5, dims[0] * dims[1], bf); expect(6, 1, i32); expect(7, 1, i32);
            if (quantized) {
                DG_HOST_ASSERT(dims.size()==6 && dims[5]==128 && dims[1]%128==0);
                expect(8,dims[0]*dims[1],torch::kFloat8_e4m3fn);
                expect(9,dims[0]*(dims[1]/128),torch::kFloat32);
                DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(tensors[5].data_ptr())%16==0 &&
                               reinterpret_cast<uintptr_t>(tensors[8].data_ptr())%8==0);
            } else if (tensors.size() == 9) expect(8, dims[0] * dims[1], bf);
            blocks = int64_t(get_num_sms()) * 2;
            break;
        }
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
        case PeerOperation::Scatter:
        case PeerOperation::ScatterVector:
            DG_HOST_ASSERT(dims.size() == 4 || dims.size() == 5);
            schema(5, dims.size());
            if (dims.size() == 5) DG_HOST_ASSERT(dims[4] == 128 || dims[4] == 256);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[3] > 0 && dims[3] <= 64);
            expect(0, dims[0] * dims[1], bf); expect(1, dims[0], i32);
            expect(2, dims[3], i64); expect(3, 1, i32);
            expect(4, 1, i32);
            if (static_cast<PeerOperation>(operation) == PeerOperation::ScatterVector)
                DG_HOST_ASSERT(dims[1] % 8 == 0);
            work = dims[0] * dims[1];
            blocks = std::min((work + 255) / 256, int64_t(get_num_sms()) * 2);
            break;
        case PeerOperation::PackActive:
        case PeerOperation::PackActiveVector:
            DG_HOST_ASSERT(dims.size() == 5 || dims.size() == 6);
            schema(5, dims.size());
            if (dims.size() == 6) DG_HOST_ASSERT(dims[5] == 128 || dims[5] == 256);
            DG_HOST_ASSERT(dims[1] > 0 && dims[2] > 0 && dims[3] > 0 && dims[4] > 0 && dims[4] <= 64);
            expect(0, dims[4], i64); expect(1, dims[0], i32);
            expect(2, dims[0] * dims[1], bf); expect(3, 1, i32); expect(4, 1, i32);
            if (static_cast<PeerOperation>(operation) == PeerOperation::PackActiveVector)
                DG_HOST_ASSERT(dims[1] % 8 == 0);
            work = dims[0] * dims[1];
            blocks = std::min((work + 255) / 256, int64_t(get_num_sms()) * 2);
            break;
        case PeerOperation::ActivationActive:
        case PeerOperation::ActivationBackwardActive:
            DG_HOST_ASSERT(dims.size() >= 2 && dims.size() <= 4);
            schema(4, dims.size());
            if (dims.size() >= 3) DG_HOST_ASSERT(dims[2] == 128 || dims[2] == 256);
            if (dims.size() == 4) DG_HOST_ASSERT(dims[3] == 8);
            DG_HOST_ASSERT(dims[0] >= 0 && dims[1] > 0);
            expect(0, dims[0] * 2 * dims[1], bf); expect(1, dims[0] * dims[1], bf);
            expect(2, 1, i32); expect(3, 1, i32);
            blocks = int64_t(get_num_sms()) * 4;
            break;
        case PeerOperation::LocalGradients:
            schema(5, 5);
            DG_HOST_ASSERT(dims[0] > 0 && dims[1] > 0 && dims[2] > 0);
            expect(0, dims[0] * dims[2] * dims[1], bf);
            expect(1, dims[3] * dims[1], bf); expect(2, dims[3] * dims[2], torch::kFloat);
            expect(3, dims[3] * dims[2], torch::kFloat); expect(4, dims[0] * dims[2] * dims[1], bf);
            blocks = dims[0] * dims[2];
            break;
    }
    if(operation==int(PeerOperation::PackActiveVector) || operation==int(PeerOperation::ExpertLocalGradients) ||
       operation==int(PeerOperation::StageRawWeightedDy)) {
        const int ctas=get_env<int>("DG_MOE_PEER_STAGE_CTAS_PER_CU",0);
        DG_HOST_ASSERT(ctas==0 || ctas==1 || ctas==2 || ctas==4 || ctas==8 || ctas==16);
        if(ctas)blocks=int64_t(get_num_sms())*ctas;
    }
    if(operation==int(PeerOperation::PackActiveQuantized) || operation==int(PeerOperation::ExpertLocalGradientsQuantized) || operation==int(PeerOperation::ExpertLocalGradientsQuantizedTiles) || operation==int(PeerOperation::ExpertLocalGradientsQuantizedTilesCached)) {
        const int ctas=get_env<int>("DG_MOE_FP8_FUSED_QUANT_CTAS_PER_CU",0);
        DG_HOST_ASSERT(ctas==0 || ctas==1 || ctas==2 || ctas==4 || ctas==8 || ctas==16);
        if(ctas)blocks=int64_t(get_num_sms())*ctas;
    }
    if (blocks < 0) blocks = (work + 255) / 256;
    if (!blocks) return;
    DG_HOST_ASSERT(blocks <= INT32_MAX);
    Params params{};
    for (size_t i = 0; i < tensors.size(); ++i) params.ptr[i] = tensors[i].data_ptr();
    for (size_t i = 0; i < dims.size(); ++i) params.dim[i] = dims[i];
    PeerRuntime::Args args{operation, {dim3(blocks), dim3(threads), 0}, params};
    const auto code = PeerRuntime::generate(args);
    const auto runtime = compiler->build(fmt::format("moe_training_peer_{}", operation), code);
    if (!compile_only) PeerRuntime::launch(runtime, args);
}

class PeerGemmRuntime final : public LaunchRuntime<PeerGemmRuntime> {
public:
    struct Args {
        int n, k, groups;
        bool read_peer, write_peer;
        bool activation_backward;
        int dispatch_cache_n;
        bool dispatch_cache_aiu;
        bool staged_return;
        int block_n, block_k, warp_m, warp_n, stages, threads;
        int dispatch_cache_ctas;
        int first_n_tile, num_n_tiles;
        bool local_input_aiu;
        bool producer_input_aiu;
        LaunchArgs launch_args;
        PeerGemmParams params;
        PeerDispatchPipeline pipeline{};
        bool cache_return_addresses = false;
        int block_m = 128;
        int return_vector_bytes = 4;
        bool interleaved_gate_up = false;
        bool fc1_activation = false;
        bool keep_gate_up = true;
        int input_copy_ilp = 1;
        bool external_input = false;
        int poll_yield_every = 0;
        int poll_pause_cycles = 0;
        bool replay_gradients = false;
        bool replay_warp_reduce = false;
        PeerReplayGradients gradients{};
        bool dhidden_scores = false;
        PeerDHiddenScores scores{};
        bool weight_kn = false;
        int fc1_helper_threads = 0;
        int metadata_block_m = 0;
    };
    static std::string name(const Args& args) {
        auto name = fmt::format("bf16_peer_r{}_w{}_n{}_k{}_bn{}_bk{}_wm{}_wn{}_s{}",
                           args.read_peer, args.write_peer, args.n, args.k, args.block_n,
                           args.block_k, args.warp_m, args.warp_n, args.stages);
        if (args.activation_backward) name += "_activation_backward";
        if (args.dispatch_cache_n) name += fmt::format("_cache{}", args.dispatch_cache_n);
        if (args.dispatch_cache_aiu) name += "_aiu";
        if (args.staged_return) name += "_staged_return";
        if (args.dispatch_cache_n && args.dispatch_cache_ctas != 2)
            name += fmt::format("_ctas{}", args.dispatch_cache_ctas);
        if (args.first_n_tile || args.num_n_tiles)
            name += fmt::format("_nslice{}_{}", args.first_n_tile, args.num_n_tiles);
        if (args.local_input_aiu) name += "_local_aiu";
        if (args.producer_input_aiu) name += "_producer_aiu";
        if (args.pipeline.ready) name += args.write_peer ? "_return_pipeline" : "_dispatch_pipeline";
        if (args.pipeline.ready && args.pipeline.serialize) name += "_serialized";
        if (args.poll_yield_every) name += fmt::format("_yield{}", args.poll_yield_every);
        if (args.poll_pause_cycles) name += fmt::format("_pause{}", args.poll_pause_cycles);
        if (args.replay_gradients) name += "_replay_grad";
        if (args.dhidden_scores) name += "_dhidden_scores";
        if (args.weight_kn) name += "_weight_kn";
        if (args.fc1_helper_threads) name += fmt::format("_input_helpers{}", args.fc1_helper_threads);
        if (args.metadata_block_m && args.metadata_block_m != args.block_m)
            name += fmt::format("_meta_bm{}", args.metadata_block_m);
        if (args.replay_warp_reduce) name += "_warp_reduce";
        if (args.pipeline.ready && !args.write_peer && args.pipeline.communication_ctas == 0)
            name += args.external_input ? "_external_dispatch" : "_owned_dispatch";
        if (args.cache_return_addresses) name += "_cached_return";
        if (args.block_m != 128) name += fmt::format("_bm{}", args.block_m);
        if (args.return_vector_bytes != 4) name += fmt::format("_return{}", args.return_vector_bytes);
        if (args.interleaved_gate_up) name += "_g8";
        if (args.fc1_activation) name += args.keep_gate_up ? "_swiglu_keep" : "_swiglu_discard";
        if (args.input_copy_ilp != 1) name += fmt::format("_copy_ilp{}", args.input_copy_ilp);
        return name;
    }
    static std::string generate_impl(const Args& args) {
        if (args.fc1_helper_threads) {
            return fmt::format(R"(
#include <deep_gemm/impls/peer_moe_gemm.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void {}(const deep_gemm::training::PeerGemmParams params) {{
    deep_gemm::training_bf16_input_helpers_impl<
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    >(params);
}}
)", args.threads + args.fc1_helper_threads, name(args),
            args.n, args.k, args.groups, args.block_m, args.block_n, args.block_k,
            args.warp_m, args.warp_n, args.threads, args.stages,
            args.fc1_helper_threads, args.keep_gate_up ? "true" : "false", args.input_copy_ilp,
            args.metadata_block_m ? args.metadata_block_m : args.block_m);
        }
        return fmt::format(R"(
#include <deep_gemm/impls/peer_moe_gemm.cuh>
extern "C" __launch_bounds__({}, 1)
__global__ void {}(const deep_gemm::training::PeerGemmParams params{}) {{
    deep_gemm::training_bf16_peer_gemm_impl<
        cutlass::bfloat16_t, deep_gemm::GemmType::MGroupedFused,
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, false, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, false, {}, {}, {}, {}, {}, {}, {}, {}{}
    >(params{});
}}
)", args.threads, name(args),
       args.pipeline.ready ? ", const deep_gemm::training::PeerDispatchPipeline pipeline"
                           : args.replay_gradients ? ", const deep_gemm::training::PeerReplayGradients gradients"
                           : args.dhidden_scores ? ", const deep_gemm::training::PeerDHiddenScores scores" : "",
       args.n, args.k, args.groups, args.block_m, args.block_n, args.block_k,
       args.warp_m, args.warp_n, args.threads, args.stages,
       args.read_peer ? "true" : "false", args.write_peer ? "true" : "false",
       args.activation_backward ? "true" : "false", args.dispatch_cache_n,
       args.dispatch_cache_aiu ? "true" : "false", args.staged_return ? "true" : "false",
       args.first_n_tile, args.num_n_tiles, args.local_input_aiu ? "true" : "false",
       args.producer_input_aiu ? "true" : "false",
       args.pipeline.ready ? "true" : "false",
       args.pipeline.ready && !args.write_peer && args.pipeline.communication_ctas == 0 && !args.external_input ? "true" : "false",
       args.cache_return_addresses ? "true" : "false",
       args.return_vector_bytes,
       args.interleaved_gate_up ? "true" : "false",
       args.fc1_activation ? "true" : "false", args.keep_gate_up ? "true" : "false",
       args.input_copy_ilp, args.external_input ? "true" : "false", args.poll_yield_every, args.poll_pause_cycles,
       args.weight_kn ? (args.dhidden_scores ? ", false, false, true, true" : ", false, false, false, true")
                           : args.dhidden_scores ? ", false, false, true"
                           : args.replay_gradients ? (args.replay_warp_reduce ? ", true, true" : ", true") : "",
       args.pipeline.ready ? ", pipeline"
                           : args.replay_gradients ? ", deep_gemm::training::PeerDispatchPipeline{}, 0, ~0u, gradients"
                           : args.dhidden_scores ? ", deep_gemm::training::PeerDispatchPipeline{}, 0, ~0u, deep_gemm::training::PeerReplayGradients{}, scores" : "");
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if (args.dhidden_scores)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params, args.scores));
        else if (args.replay_gradients)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params, args.gradients));
        else if (args.pipeline.ready)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params, args.pipeline));
        else
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.params));
    }
};

static int peer_gemm(const torch::Tensor& a, const torch::Tensor& weight, const torch::Tensor& out,
                      const torch::Tensor& source_peers, const torch::Tensor& destination_peers,
                      const torch::Tensor& input_rows, const torch::Tensor& permutation,
                      const torch::Tensor& work_info, const torch::Tensor& num_blocks,
                      int source_rows_per_rank, int destination_rows_per_rank,
                      bool read_peer, bool write_peer, bool compile_only,
                      bool pack_input, const std::vector<int>& config,
                      const c10::optional<torch::Tensor>& activation_gate_up,
                      int dispatch_cache_n, const c10::optional<torch::Tensor>& dispatch_scratch,
                      bool dispatch_cache_aiu, bool staged_return, int dispatch_cache_ctas,
                      int first_n_tile, int num_n_tiles, bool local_input_aiu, bool producer_input_aiu,
                      const c10::optional<torch::Tensor>& pipeline_ready,
                      const c10::optional<torch::Tensor>& pipeline_status,
                      const c10::optional<torch::Tensor>& pipeline_statistics,
                      int pipeline_comm_ctas, bool pipeline_serialize, bool cache_return_addresses,
                      int block_m = 128, int return_vector_bytes = 4,
                      bool interleaved_gate_up = false, bool fc1_activation = false, bool keep_gate_up = true,
                      int input_copy_ilp = 1, bool external_input = false, int compute_cus = 0,
                      const std::vector<torch::Tensor>& replay_gradient_tensors = {},
                      const std::vector<torch::Tensor>& dhidden_score_tensors = {}, bool weight_kn = false,
                      int fc1_helper_threads = 0, int fc1_compute_block_m = 0) {
    DG_HOST_ASSERT(fc1_helper_threads == 0 || fc1_helper_threads == 32 || fc1_helper_threads == 64);
    DG_HOST_ASSERT(fc1_compute_block_m == 0 || (fc1_helper_threads && (fc1_compute_block_m == 128 || fc1_compute_block_m == 256)));
    const int compute_block_m = fc1_compute_block_m ? fc1_compute_block_m : block_m;
    DG_HOST_ASSERT(compute_block_m <= block_m && block_m % compute_block_m == 0);
    DG_HOST_ASSERT(!fc1_helper_threads || (read_peer && !write_peer && pack_input
        && fc1_activation && interleaved_gate_up && !pipeline_ready.has_value()
        && !dispatch_cache_n && !staged_return && !local_input_aiu && !producer_input_aiu
        && !first_n_tile && !num_n_tiles && !weight_kn && !external_input
        && replay_gradient_tensors.empty() && dhidden_score_tensors.empty()));
    DG_HOST_ASSERT(!external_input || (pipeline_ready.has_value() && pipeline_comm_ctas == 0 && !read_peer && !write_peer));
    DG_HOST_ASSERT(input_copy_ilp == 1 || input_copy_ilp == 2 || input_copy_ilp == 4 || input_copy_ilp == 8);
    DG_HOST_ASSERT(input_copy_ilp == 1 || producer_input_aiu || fc1_helper_threads
                   || (pipeline_ready.has_value() && pipeline_comm_ctas == 0 && !write_peer));
    DG_HOST_ASSERT(block_m == 128 || block_m == 256);
    DG_HOST_ASSERT(return_vector_bytes == 4 || return_vector_bytes == 8);
    DG_HOST_ASSERT(return_vector_bytes == 4 || (write_peer && !pipeline_ready.has_value() && !staged_return));
    DG_HOST_ASSERT(!fc1_activation || (interleaved_gate_up && !write_peer
                                     && (!pipeline_ready.has_value() || pipeline_comm_ctas == 0)
                                     && activation_gate_up.has_value()));
    DG_HOST_ASSERT(!interleaved_gate_up || activation_gate_up.has_value());
    DG_HOST_ASSERT(a.is_cuda() && a.dim() == 2 && weight.dim() == 3 && out.dim() == 2);
    const c10::cuda::CUDAGuard guard(a.device());
    DG_HOST_ASSERT(compute_cus >= 0 && compute_cus <= get_num_sms());
    for (const auto& tensor : {a, weight, out, source_peers, destination_peers, input_rows,
                               permutation, work_info, num_blocks})
        DG_HOST_ASSERT(tensor.device() == a.device() && tensor.is_contiguous());
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(a.numel() <= INT32_MAX && weight.numel() <= INT32_MAX && out.numel() <= INT32_MAX);
    const auto [m, k] = get_shape<2>(a);
    const auto [groups, weight_rows, weight_columns] = get_shape<3>(weight);
    const int n = weight_kn ? weight_columns : weight_rows;
    const int k_ = weight_kn ? weight_rows : weight_columns;
    DG_HOST_ASSERT(k == k_ && out.size(0) == m && out.size(1) == n && m > 0 && m % block_m == 0);
    DG_HOST_ASSERT(n > 0 && n % 128 == 0 && k > 0 && k % 128 == 0 && groups > 0);
    DG_HOST_ASSERT(source_peers.dim() == 1 && source_peers.numel() > 0 && source_peers.numel() <= 64);
    DG_HOST_ASSERT(source_peers.scalar_type() == torch::kInt64 && destination_peers.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(source_peers.sizes() == destination_peers.sizes());
    DG_HOST_ASSERT(input_rows.numel() == m && permutation.numel() == m);
    DG_HOST_ASSERT(input_rows.scalar_type() == torch::kInt32 && permutation.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(work_info.sizes() == at::IntArrayRef({m / block_m, 4}) && work_info.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(num_blocks.numel() == 1 && num_blocks.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(source_rows_per_rank > 0 && destination_rows_per_rank > 0);
    DG_HOST_ASSERT(!pack_input || read_peer);
    DG_HOST_ASSERT(block_m == 128 || ((!pipeline_ready.has_value() || (pipeline_comm_ctas == 0 && !write_peer))
                                     && !dispatch_cache_n && !staged_return));
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
    params.pack_input = pack_input;
    DG_HOST_ASSERT(dispatch_cache_n == 0 || dispatch_cache_n == 2 || dispatch_cache_n == 4
                   || dispatch_cache_n == 8 || dispatch_cache_n == 16);
    if (!read_peer) dispatch_cache_n = 0;
    DG_HOST_ASSERT(!dispatch_cache_aiu || (read_peer && dispatch_cache_n));
    DG_HOST_ASSERT(!staged_return || write_peer);
    DG_HOST_ASSERT(dispatch_cache_ctas == 1 || dispatch_cache_ctas == 2 || dispatch_cache_ctas == 4);
    if (dispatch_cache_n) {
        DG_HOST_ASSERT(dispatch_scratch.has_value());
        const auto& scratch = dispatch_scratch.value();
        DG_HOST_ASSERT(scratch.is_contiguous() && scratch.device() == a.device()
                       && scratch.scalar_type() == torch::kBFloat16 && scratch.dim() == 2);
        DG_HOST_ASSERT(scratch.size(1) == k
                       && scratch.size(0) >= get_num_sms() * dispatch_cache_ctas * 128);
        DG_HOST_ASSERT(scratch.numel() <= INT32_MAX);
        params.dispatch_scratch = scratch.data_ptr();
    }
    if (activation_gate_up.has_value()) {
        const auto& activation = activation_gate_up.value();
        DG_HOST_ASSERT(!write_peer && activation.device() == a.device() && activation.is_contiguous());
        DG_HOST_ASSERT(activation.scalar_type() == torch::kBFloat16 && activation.dim() == 2);
        DG_HOST_ASSERT(activation.size(0) == m && activation.size(1) == (fc1_activation ? n / 2 : 2 * n));
        DG_HOST_ASSERT(!fc1_activation || n % 16 == 0);
        DG_HOST_ASSERT(activation.numel() <= INT32_MAX);
        params.activation_gate_up = activation.data_ptr();
    }
    DG_HOST_ASSERT(config.empty() || config.size() == 5);
    const int bn = config.empty() ? 128 : config[0];
    const int bk = config.empty() ? 64 : config[1];
    const int wm = config.empty() ? 64 : config[2];
    const int wn = config.empty() ? 64 : config[3];
    const int stages = config.empty() ? 3 : config[4];
    DG_HOST_ASSERT(return_vector_bytes == 4 || (wm == 64 && wn == 64 && n % 16 == 0));
    DG_HOST_ASSERT(bn == 64 || bn == 128 || bn == 256 || (fc1_helper_threads && bn == 512 && compute_block_m == 128));
    DG_HOST_ASSERT(bk == 64 || bk == 128);
    DG_HOST_ASSERT(wm == 32 || wm == 64 || wm == 128);
    DG_HOST_ASSERT(wn == 32 || wn == 64 || wn == 128);
    DG_HOST_ASSERT(bn % wn == 0 && stages >= 2 && stages <= 4);
    DG_HOST_ASSERT(first_n_tile >= 0 && first_n_tile < (n + bn - 1) / bn && num_n_tiles >= 0);
    DG_HOST_ASSERT(num_n_tiles == 0 || first_n_tile + num_n_tiles <= (n + bn - 1) / bn);
    DG_HOST_ASSERT((first_n_tile == 0 && num_n_tiles == 0) || dispatch_cache_n == 0);
    DG_HOST_ASSERT(!local_input_aiu || (!read_peer && !write_peer && first_n_tile > 0));
    DG_HOST_ASSERT(!producer_input_aiu || (read_peer && !write_peer && pack_input
                   && first_n_tile == 0 && num_n_tiles == 1 && dispatch_cache_n == 0));
    DG_HOST_ASSERT(!fc1_helper_threads || ((bn == 256 || bn == 512) && bk == 64
        && ((wm == 64 && wn == 64) || (wm == 32 && wn == 128)) && stages == 3));
    const int threads = (compute_block_m / wm) * (bn / wn) * 32;
    const int launch_threads = threads + fc1_helper_threads;
    DG_HOST_ASSERT(launch_threads <= 1024);
    DG_HOST_ASSERT(threads >= 32 && threads <= 1024);
    const int smem = gemm_smem_total_size(2, stages, compute_block_m, bn, bk);
    DG_HOST_ASSERT(smem <= 256 * 1024);
    PeerGemmRuntime::Args args{n, k, groups, read_peer, write_peer, activation_gate_up.has_value(), dispatch_cache_n,
                              dispatch_cache_aiu,
                              staged_return,
                              bn, bk, wm, wn, stages, threads,
                              dispatch_cache_ctas,
                              first_n_tile, num_n_tiles,
                              local_input_aiu,
                              producer_input_aiu,
                              {get_grid_shape(get_num_sms()), dim3(launch_threads), smem}, params};
    DG_HOST_ASSERT(!cache_return_addresses || (write_peer && !staged_return && !pipeline_ready.has_value()));
    args.cache_return_addresses = cache_return_addresses;
    args.block_m = compute_block_m;
    if (fc1_helper_threads) args.metadata_block_m = block_m;
    args.return_vector_bytes = return_vector_bytes;
    args.activation_backward = activation_gate_up.has_value() && !fc1_activation;
    args.interleaved_gate_up = interleaved_gate_up;
    args.fc1_activation = fc1_activation;
    args.keep_gate_up = keep_gate_up;
    args.input_copy_ilp = input_copy_ilp;
    args.external_input = external_input;
    DG_HOST_ASSERT(!weight_kn || (!read_peer && !fc1_activation && !pack_input
                   && !dispatch_cache_n && !pipeline_ready.has_value()
                   && replay_gradient_tensors.empty() && !local_input_aiu && !producer_input_aiu
                   && !first_n_tile && !num_n_tiles && !staged_return));
    args.weight_kn = weight_kn;
    args.fc1_helper_threads = fc1_helper_threads;
    if (fc1_helper_threads) args.params.gemm.shape_m = m;
    if (!dhidden_score_tensors.empty()) {
        const auto& g = dhidden_score_tensors;
        DG_HOST_ASSERT(g.size() == 3 && replay_gradient_tensors.empty()
                       && args.activation_backward && interleaved_gate_up && !read_peer && !write_peer
                       && !pipeline_ready.has_value() && !dispatch_cache_n && !pack_input
                       && !local_input_aiu && !producer_input_aiu && !first_n_tile
                       && !num_n_tiles && !staged_return && !fc1_activation);
        for (const auto& tensor : g)
            DG_HOST_ASSERT(tensor.device() == a.device() && tensor.is_contiguous());
        const uint32_t ntiles = (n + bn - 1) / bn;
        DG_HOST_ASSERT(g[0].scalar_type() == torch::kBFloat16 && g[0].sizes() == out.sizes());
        DG_HOST_ASSERT(g[1].scalar_type() == torch::kFloat32 && g[1].sizes() == at::IntArrayRef({m}));
        DG_HOST_ASSERT(g[2].scalar_type() == torch::kFloat32
                       && g[2].sizes() == at::IntArrayRef({m, int64_t(ntiles)}));
        // out aliases the retained hidden tensor by design: this epilogue
        // writes gate/up gradients and score partials, never out/hidden.
        DG_HOST_ASSERT(!a.is_alias_of(g[0]) && !g[2].is_alias_of(g[1]));
        args.dhidden_scores = true;
        args.scores = {g[0].data_ptr(), g[1].data_ptr<float>(), g[2].data_ptr<float>(), ntiles};
    }
    if (!replay_gradient_tensors.empty()) {
        const auto& g = replay_gradient_tensors;
        DG_HOST_ASSERT(g.size() == 5 && !read_peer && !write_peer && !activation_gate_up.has_value()
                       && !pipeline_ready.has_value() && !dispatch_cache_n && !pack_input
                       && !local_input_aiu && !producer_input_aiu && !first_n_tile
                       && !num_n_tiles && !staged_return);
        for (const auto& tensor : g)
            DG_HOST_ASSERT(tensor.device() == a.device() && tensor.is_contiguous());
        for (int index : {0, 1})
            DG_HOST_ASSERT(g[index].scalar_type() == torch::kInt64 && g[index].dim() == 1
                           && g[index].numel() == source_peers.numel());
        DG_HOST_ASSERT(g[2].scalar_type() == torch::kBFloat16 && g[2].sizes() == out.sizes());
        const uint32_t ntiles = (n + bn - 1) / bn;
        DG_HOST_ASSERT(g[3].scalar_type() == torch::kFloat32
                       && g[3].sizes() == at::IntArrayRef({m, int64_t(ntiles)}));
        DG_HOST_ASSERT(g[4].scalar_type() == torch::kInt32 && g[4].numel() == 1);
        DG_HOST_ASSERT(destination_rows_per_rank % source_rows_per_rank == 0);
        args.replay_gradients = true;
        const int warp_reduce = get_env<int>("DG_REPLAY_GRAD_WARP_REDUCTION", 0);
        DG_HOST_ASSERT(warp_reduce == 0 || warp_reduce == 1);
        args.replay_warp_reduce = warp_reduce;
        args.gradients = {
            reinterpret_cast<const uint64_t*>(g[0].data_ptr<int64_t>()),
            reinterpret_cast<const uint64_t*>(g[1].data_ptr<int64_t>()),
            g[2].data_ptr(), g[3].data_ptr<float>(),
            reinterpret_cast<uint32_t*>(g[4].data_ptr<int32_t>()),
            uint32_t(source_rows_per_rank), uint32_t(destination_rows_per_rank / source_rows_per_rank),
            uint32_t(source_peers.numel()), ntiles};
    }
    if (pipeline_ready.has_value()) {
        args.poll_pause_cycles = get_env<int>("DG_INPUT_READY_PAUSE_CYCLES", 0);
        DG_HOST_ASSERT(args.poll_pause_cycles >= 0 && args.poll_pause_cycles <= 16384);
        args.poll_yield_every = get_env<int>("DG_INPUT_READY_YIELD_EVERY", 0);
        DG_HOST_ASSERT(args.poll_yield_every >= 0 && args.poll_yield_every <= 1024
                       && (args.poll_yield_every == 0 || (args.poll_yield_every & (args.poll_yield_every - 1)) == 0));
        DG_HOST_ASSERT(!read_peer && (!activation_gate_up.has_value() || fc1_activation)
                       && !dispatch_cache_n && !local_input_aiu && !producer_input_aiu
                       && !first_n_tile && !num_n_tiles && !pack_input);
        const auto& ready = pipeline_ready.value();
        DG_HOST_ASSERT(ready.device() == a.device() && ready.is_contiguous()
                       && ready.scalar_type() == torch::kInt32
                       && ready.numel() > (m / block_m) * (write_peer ? ((n + bn - 1) / bn) : 1)
                       && ready.numel() <= INT32_MAX);
        DG_HOST_ASSERT(pipeline_status.has_value());
        const auto& status = pipeline_status.value();
        DG_HOST_ASSERT(status.device() == a.device() && status.is_contiguous()
                       && status.scalar_type() == torch::kInt32 && status.numel() == 1);
        DG_HOST_ASSERT(get_num_sms() >= 2 && pipeline_comm_ctas >= 0);
        DG_HOST_ASSERT(pipeline_comm_ctas != 0 || (!write_peer && !pipeline_serialize));
        args.pipeline.ready = reinterpret_cast<uint32_t*>(ready.data_ptr<int32_t>());
        args.pipeline.status = reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>());
        args.pipeline.communication_ctas = pipeline_comm_ctas;
        args.pipeline.signal_capacity = ready.numel() - 1;
        args.pipeline.serialize = pipeline_serialize;
        if (pipeline_statistics.has_value()) {
            const auto& statistics = pipeline_statistics.value();
            DG_HOST_ASSERT(statistics.device() == a.device() && statistics.is_contiguous()
                           && statistics.scalar_type() == torch::kInt64
                           && statistics.dim() == 2 && statistics.size(1) == 4);
            args.pipeline.statistics = reinterpret_cast<uint64_t*>(statistics.data_ptr<int64_t>());
        }
    } else {
        DG_HOST_ASSERT(!pipeline_status.has_value() && !pipeline_statistics.has_value());
    }
    const auto code = PeerGemmRuntime::generate(args);
    const auto runtime = compiler->build(PeerGemmRuntime::name(args), code);
    int occupancy = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, runtime->kernel, launch_threads, smem));
    DG_HOST_ASSERT(occupancy > 0);
    if (args.pipeline.ready) {
        // A CTA is not a CU: the actual kernel can have multiple resident
        // CTAs per CU. Fill that resident pool while reserving one CU's worth
        // of slots for framework progress. No grid-level oversubscription.
        int compute_cus = get_num_sms() - (pipeline_comm_ctas == 0 && !external_input ? 0 : 1);
        if (external_input) {
            // This caps the persistent compute grid; it does not pin CTAs to
            // particular CUs or promise hardware partitioning across streams.
            const int requested = get_env<int>("DG_EXTERNAL_INPUT_COMPUTE_CUS", 0);
            DG_HOST_ASSERT(requested >= 0 && requested < get_num_sms());
            if (requested) compute_cus = requested;
        }
        args.launch_args.grid_dim.x = compute_cus * occupancy;
        DG_HOST_ASSERT(pipeline_comm_ctas < args.launch_args.grid_dim.x);
        if (pipeline_statistics.has_value())
            DG_HOST_ASSERT(pipeline_statistics.value().numel() >= args.launch_args.grid_dim.x * 4);
    } else
        args.launch_args.grid_dim.x *= dispatch_cache_n ? std::min(occupancy, dispatch_cache_ctas) : occupancy;
    // The scheduler strides by gridDim.x, so reducing its persistent CTA
    // budget preserves complete work coverage. This is not CU affinity.
    if (compute_cus) {
        args.launch_args.grid_dim.x = std::min(args.launch_args.grid_dim.x,
                                              uint32_t(compute_cus * occupancy));
        if (args.pipeline.ready)
            DG_HOST_ASSERT(pipeline_comm_ctas < args.launch_args.grid_dim.x);
    }
    if (get_env<int>("show_log")) {
        int registers = 0, local_bytes = 0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
        hggcFuncAttributes attributes;
        DG_HGGC_RUNTIME_CHECK(hggcFuncGetAttributes(&attributes, runtime->kernel));
        registers = attributes.numRegs;
        local_bytes = static_cast<int>(attributes.localSizeBytes);
#else
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&registers, HG_FUNC_ATTRIBUTE_NUM_REGS, runtime->kernel));
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&local_bytes, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, runtime->kernel));
#endif
        std::printf("[PeerTraining_BF16:]\n");
        std::printf("kernel:%s, vreg:%d, local_size_bytes:%d\n",
                    PeerGemmRuntime::name(args).c_str(), registers, local_bytes);
        std::printf("groups:%d, n:%d, k:%d, read_peer:%d, write_peer:%d, capacity_rows:%d\n",
                    int(groups), int(n), int(k), read_peer, write_peer, int(m));
        std::printf("tile:[%d,%d,%d], warp:[%d,%d], stages:%d, threads:%d, "
                    "dynamic_smem_bytes:%d, blocks_per_cu:%d, grid:%u\n",
                    block_m, bn, bk, wm, wn, stages, launch_threads, smem, occupancy, args.launch_args.grid_dim.x);
    }
    if (!compile_only) PeerGemmRuntime::launch(runtime, args);
    return int(args.launch_args.grid_dim.x);
}

static void register_peer_apis(pybind11::module_& m) {
    m.def("_moe_training_peer_launch", &launch_peer, py::arg("operation"), py::arg("tensors"),
          py::arg("dims"), py::arg("compile_only") = false);
    m.def("_moe_training_peer_gemm", &peer_gemm,
          py::arg("a"), py::arg("weight"), py::arg("out"), py::arg("source_peers"), py::arg("destination_peers"),
          py::arg("input_rows"), py::arg("permutation"), py::arg("work_info"), py::arg("num_blocks"),
          py::arg("source_rows_per_rank"), py::arg("destination_rows_per_rank"),
          py::arg("read_peer"), py::arg("write_peer"), py::arg("compile_only") = false,
          py::arg("pack_input") = false, py::arg("config") = std::vector<int>{},
          py::arg("activation_gate_up") = py::none(), py::arg("dispatch_cache_n") = 0,
          py::arg("dispatch_scratch") = py::none(), py::arg("dispatch_cache_aiu") = false,
          py::arg("staged_return") = false, py::arg("dispatch_cache_ctas") = 2,
          py::arg("first_n_tile") = 0, py::arg("num_n_tiles") = 0,
          py::arg("local_input_aiu") = false, py::arg("producer_input_aiu") = false,
          py::arg("pipeline_ready") = py::none(), py::arg("pipeline_status") = py::none(),
          py::arg("pipeline_statistics") = py::none(), py::arg("pipeline_comm_ctas") = 2,
          py::arg("pipeline_serialize") = false, py::arg("cache_return_addresses") = false,
          py::arg("block_m") = 128, py::arg("return_vector_bytes") = 4,
          py::arg("interleaved_gate_up") = false, py::arg("fc1_activation") = false,
          py::arg("keep_gate_up") = true, py::arg("input_copy_ilp") = 1,
          py::arg("external_input") = false, py::arg("compute_cus") = 0,
          py::arg("replay_gradient_tensors") = std::vector<torch::Tensor>{},
          py::arg("dhidden_score_tensors") = std::vector<torch::Tensor>{},
          py::arg("weight_kn") = false, py::arg("fc1_helper_threads") = 0,
          py::arg("fc1_compute_block_m") = 0);
}

}  // namespace deep_gemm::training
