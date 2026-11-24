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
        PeerDispatchPipeline input;
        bool input_mode;
        PeerDispatchPipeline output;
        bool output_mode;
        bool copy_only;
        int copy_threads;
        int return_store_bytes;
        bool activation_backward;
        uint32_t* input_work_counter;
        int input_ready_count;
        bool tile_producer;
        int producer_topk;
        PeerFp8ProducerGradient producer_gradient;
        bool gradient_producer;
        bool k_slice_input;
        int input_sub_m;
        int expert_begin=0,expert_end=0;
        bool preprocessed_experts=false;
    };
    static auto compile_key(const Args& args) {
        return std::make_tuple(FP8GemmRuntime::compile_key(
            {args.launch_info, args.launch_args, args.kernel_params}), args.keep_local, args.input_mode, args.output_mode, args.copy_only,
            args.copy_only ? args.copy_threads : 0, args.return_store_bytes, args.activation_backward,
            args.input_work_counter!=nullptr, args.input_ready_count, args.tile_producer, args.producer_topk, args.gradient_producer, args.k_slice_input, args.input_sub_m, args.expert_begin, args.expert_end, args.preprocessed_experts,
            args.copy_only || args.return_store_bytes==16 ? args.peers.destination_rows_per_rank : 0);
    }
    static std::string generate_impl(const Args& args) {
        auto source = FP8GemmRuntime::generate_impl({args.launch_info, args.launch_args, args.kernel_params});
        if(args.preprocessed_experts)return source;
        const auto replace = [&](const std::string& from, const std::string& to) {
            const auto position = source.find(from);
            TORCH_CHECK(position != std::string::npos && source.find(from, position + from.size()) == std::string::npos,
                        "expected one native FP8 declaration: ", from);
            source.replace(position, from.size(), to);
        };
        replace("#include <deep_gemm/impls/fp8_gemm.cuh>", "#include <deep_gemm/impls/peer_moe_fp8.cuh>");
        if(args.expert_end) {
            // Preserve the actual native universal mainloop and epilogue.
            // Full-range control uses the original scheduler too.
            replace("  TileScheduler,\n  kEnableSboOverlap,",
                "  std::conditional_t<DG_TRAINING_FP8_EXPERT_BEGIN==0 && DG_TRAINING_FP8_EXPERT_END==NUM_GROUPS, TileScheduler, TrainingFp8ExpertBatchScheduler<TileScheduler>>,\n  kEnableSboOverlap,");
        } else replace("using GemmKernel = DeepGemmUniversal<", args.copy_only ? "using GemmKernel = TrainingFp8PeerTileCopy<" : "using GemmKernel = TrainingFp8PeerReturn<");
        if(args.copy_only) {
            replace("__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmOcc::kMinBlocksPerMultiprocessor)",
                fmt::format("__launch_bounds__({}, 1)",args.copy_threads));
            // The copy kernel uses static row-address scratch and no MMA or
            // dynamic TSM. Query its real function occupancy at launch below.
            const auto begin=source.find("using GemmOcc = GemmOccModel<");
            TORCH_CHECK(begin!=std::string::npos,"expected native GEMM occupancy declaration");
            const auto end=source.find(';',begin);
            TORCH_CHECK(end!=std::string::npos,"expected end of GEMM occupancy declaration");
            source.erase(begin,end-begin+1);
        }
        if(!args.expert_end) {
            replace("typename GemmKernel::Params params", "typename GemmKernel::Params params, const deep_gemm::training::PeerGemmParams peers, const deep_gemm::training::PeerDispatchPipeline input, const deep_gemm::training::PeerDispatchPipeline output, uint32_t* input_work_counter, const deep_gemm::training::PeerFp8ProducerGradient producer_gradient");
            replace("op(params, smem);", "op(params, peers, input, output, input_work_counter, producer_gradient, smem);");
        }
        source = fmt::format("#define DG_TRAINING_FP8_INPUT_K_SLICE {}\n", int(args.k_slice_input)) + source;
        source=fmt::format("#define DG_TRAINING_FP8_INPUT_SUBTILES {}\n",128/args.input_sub_m)+source;
        source = fmt::format("#define DG_TRAINING_FP8_TILE_PRODUCER {}\n#define DG_TRAINING_FP8_PRODUCER_TOPK {}\n#define DG_TRAINING_FP8_GRADIENT_PRODUCER {}\n", int(args.tile_producer), args.producer_topk, int(args.gradient_producer)) + source;
        source=fmt::format("#define DG_TRAINING_FP8_EXPERT_BEGIN {}\n#define DG_TRAINING_FP8_EXPERT_END {}\n",args.expert_begin,args.expert_end)+source;
        return fmt::format("#define DG_TRAINING_FP8_PEER_KEEP_LOCAL {}\n#define DG_TRAINING_FP8_PEER_INPUT {}\n#define DG_TRAINING_FP8_RETURN_PIPELINE {}\n#define DG_TRAINING_FP8_RETURN_STRIDE {}\n#define DG_TRAINING_FP8_RETURN_STORE_BYTES {}\n#define DG_TRAINING_FP8_ACTIVATION_BACKWARD {}\n#define DG_TRAINING_FP8_INPUT_DYNAMIC {}\n#define DG_TRAINING_FP8_INPUT_READY_COUNT {}\n",
                           int(args.keep_local), int(args.input_mode), int(args.output_mode), args.peers.destination_rows_per_rank, args.return_store_bytes, int(args.activation_backward), int(args.input_work_counter!=nullptr), args.input_ready_count) + source;
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if(args.expert_end || args.preprocessed_experts)DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel,config,args.kernel_params));
        else DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params, args.peers, args.input, args.output,args.input_work_counter,args.producer_gradient));
    }
};

// Blockwise E4M3 compute, with the same BF16 route return boundary as the
// BF16 training path. Local input packing/quantization precedes this call.
static int peer_fp8_return(const std::pair<torch::Tensor, torch::Tensor>& lhs,
                           const std::pair<torch::Tensor, torch::Tensor>& rhs,
                           const torch::Tensor& out, const torch::Tensor& lengths,
                           const torch::Tensor& permutation, const torch::Tensor& destinations,
                           int destination_rows_per_rank, bool keep_local, bool compile_only,
                           const std::vector<int>& config,
                           const std::vector<torch::Tensor>& input_control, int reserved_input_ctas,
                           const std::vector<torch::Tensor>& return_control, int reserved_return_ctas, bool copy_only, int copy_threads,
                           int return_store_bytes, const std::optional<torch::Tensor>& activation_gate_up,
                           bool co_resident_input, int input_ready_count,
                           const std::vector<torch::Tensor>& producer_sources, int producer_tokens, int producer_topk,
                           const std::vector<torch::Tensor>& producer_gradient_sources, bool k_slice_input, int input_sub_m, int expert_begin, int expert_end, const std::optional<torch::Tensor>& expert_metadata, int metadata_block_m, int metadata_ctas) {
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
    const bool preprocessed_mode=expert_metadata.has_value();
    DG_HOST_ASSERT(metadata_ctas>=0 && metadata_ctas<=2*get_num_sms() && (preprocessed_mode || metadata_ctas==0));
    DG_HOST_ASSERT(preprocessed_mode ? (metadata_block_m==32 || metadata_block_m==64 || metadata_block_m==128 || metadata_block_m==192 || metadata_block_m==256) : metadata_block_m==0);
    DG_HOST_ASSERT(!preprocessed_mode || (keep_local && input_control.empty() && return_control.empty()
        && !copy_only && return_store_bytes==4 && !activation_gate_up.has_value()
        && producer_sources.empty() && producer_gradient_sources.empty() && !k_slice_input && expert_begin==0 && expert_end==0));
    DG_HOST_ASSERT(expert_begin>=0 && expert_end>=0 && (expert_end==0 ? expert_begin==0 : expert_begin<expert_end && expert_end<=groups));
    DG_HOST_ASSERT(!expert_end || (keep_local && input_control.empty() && return_control.empty()
        && !copy_only && return_store_bytes==4 && !activation_gate_up.has_value()
        && producer_sources.empty() && producer_gradient_sources.empty() && !k_slice_input));
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
    const bool tile_producer = !producer_sources.empty();
    if (tile_producer) {
        TORCH_CHECK(producer_sources.size() == 3 && keep_local && input_control.empty() &&
                    return_control.empty() && !copy_only && !activation_gate_up && !co_resident_input && return_store_bytes == 4,
                    "tile producer requires local FP8 GEMM with source pointers, BF16 cache and status");
        const auto& table = producer_sources[0];const auto& packed = producer_sources[1];const auto& status = producer_sources[2];
        for (const auto& t : producer_sources) DG_HOST_ASSERT(t.is_cuda() && t.device() == a.device() && t.is_contiguous());
        DG_HOST_ASSERT(table.scalar_type() == torch::kInt64 && table.numel() == destinations.numel());
        DG_HOST_ASSERT(packed.scalar_type() == torch::kBFloat16 && packed.dim() == 2 && packed.size(0) == m && packed.size(1) == k);
        DG_HOST_ASSERT(status.scalar_type() == torch::kInt32 && status.numel() == 1);
        DG_HOST_ASSERT(producer_tokens > 0 && producer_topk > 0 && producer_topk <= 64 &&
                       int64_t(producer_tokens) * producer_topk == destination_rows_per_rank);
    } else DG_HOST_ASSERT(producer_tokens == 0 && producer_topk == 0);
    const bool gradient_producer = !producer_gradient_sources.empty();
    PeerFp8ProducerGradient producer_gradient{};
    if (gradient_producer) {
        DG_HOST_ASSERT(tile_producer && producer_gradient_sources.size() == 3);
        for (const auto& t : producer_gradient_sources) DG_HOST_ASSERT(t.is_cuda() && t.device() == a.device() && t.is_contiguous());
        const auto& scores = producer_gradient_sources[0];const auto& y = producer_gradient_sources[1];const auto& partials = producer_gradient_sources[2];
        DG_HOST_ASSERT(scores.scalar_type() == torch::kInt64 && scores.numel() == destinations.numel());
        DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 && y.dim() == 2 && y.size(0) == m && y.size(1) == k);
        DG_HOST_ASSERT(partials.scalar_type() == torch::kFloat32 && partials.dim() == 2 && partials.size(0) == m && partials.size(1) == k / 128);
        producer_gradient = {reinterpret_cast<const uint64_t*>(scores.data_ptr<int64_t>()), y.data_ptr(), partials.data_ptr<float>()};
    }
    const bool input_mode = !input_control.empty();
    DG_HOST_ASSERT(input_ready_count>=1 && input_ready_count<=16 &&
        (input_ready_count&(input_ready_count-1))==0 &&
        (input_mode ? k%(128*input_ready_count)==0 : input_ready_count==1));
    TORCH_CHECK(!k_slice_input || (input_mode && input_ready_count>=1 && !tile_producer && k%(128*input_ready_count)==0),
                "K-slice input requires explicit bitmap control and aligned K stripes");
    DG_HOST_ASSERT(input_sub_m==16 || input_sub_m==32 || input_sub_m==64 || input_sub_m==128);
    DG_HOST_ASSERT(input_ready_count*(128/input_sub_m)<=32 && (input_sub_m==128 || k_slice_input));
    const bool dynamic_input=input_control.size()==3;
    PeerDispatchPipeline input{}, output{};
    const bool output_mode = !return_control.empty();
    DG_HOST_ASSERT(!(input_mode && output_mode) && (!copy_only || output_mode));
    DG_HOST_ASSERT(!co_resident_input || (input_mode && !output_mode && !copy_only && !activation_gate_up));
    DG_HOST_ASSERT(!dynamic_input || co_resident_input);
    DG_HOST_ASSERT(return_store_bytes==4 || return_store_bytes==16);
    DG_HOST_ASSERT(return_store_bytes==4 || (!input_mode && !output_mode));
    const bool activation_backward=activation_gate_up.has_value();
    if(activation_backward) {
        const auto& gu=*activation_gate_up;
        DG_HOST_ASSERT(!input_mode && !output_mode && !copy_only && !keep_local && return_store_bytes==4);
        DG_HOST_ASSERT(gu.is_cuda() && gu.device()==a.device() && gu.is_contiguous() &&
                       gu.scalar_type()==torch::kBFloat16 && gu.dim()==2 && gu.size(0)==m && gu.size(1)==2*n);
        DG_HOST_ASSERT(gu.numel()<=INT32_MAX);
    }
    DG_HOST_ASSERT(copy_threads==256 || copy_threads==512 || copy_threads==1024);
    if (input_mode) {
        TORCH_CHECK((input_control.size() == 2 || input_control.size()==3) && keep_local,
                    "FP8 input pipeline requires ready/status and a local output");
        for (const auto& tensor : input_control)
            DG_HOST_ASSERT(tensor.is_cuda() && tensor.device() == a.device() && tensor.is_contiguous()
                           && tensor.scalar_type() == torch::kInt32);
        DG_HOST_ASSERT(input_control[0].numel() == m / 128 && input_control[1].numel() == 1);
        DG_HOST_ASSERT(co_resident_input ? (dynamic_input ? reserved_input_ctas>=num_sms && reserved_input_ctas<=(input_sub_m<128?16:4)*num_sms && reserved_input_ctas%num_sms==0 : reserved_input_ctas==num_sms) :
                       reserved_input_ctas>0 && reserved_input_ctas<num_sms);
        if(dynamic_input)DG_HOST_ASSERT(input_control[2].numel()==1);
        input.ready = reinterpret_cast<uint32_t*>(input_control[0].data_ptr<int32_t>());
        input.status = reinterpret_cast<uint32_t*>(input_control[1].data_ptr<int32_t>());
        input.signal_capacity = m / 128;
    } else {
        DG_HOST_ASSERT(reserved_input_ctas == 0);
    }
    if (output_mode) {
        DG_HOST_ASSERT(return_control.size()==2 && keep_local && n%256==0 &&
                       reserved_return_ctas>0 && reserved_return_ctas<num_sms);
        for(const auto& tensor:return_control)
            DG_HOST_ASSERT(tensor.is_cuda() && tensor.device()==a.device() && tensor.is_contiguous()
                           && tensor.scalar_type()==torch::kInt32);
        DG_HOST_ASSERT(return_control[0].numel()==(m/128)*(n/256) && return_control[1].numel()==1);
        output.ready=reinterpret_cast<uint32_t*>(return_control[0].data_ptr<int32_t>());
        output.status=reinterpret_cast<uint32_t*>(return_control[1].data_ptr<int32_t>());
        output.signal_capacity=return_control[0].numel();
    } else DG_HOST_ASSERT(reserved_return_ctas==0);
    auto cfg = deep_gemm_fp8_common::get_best_configs(
        ceil_div(m, groups), n, k, groups, num_sms, GemmType::MGroupedNoPad);
    if (!config.empty()) {
        DG_HOST_ASSERT(config.size() == 5 && config[0] > 0 && config[1] == 128 &&
                       config[2] > 0 && config[3] > 0 && config[4] > 0);
        const int config_m=preprocessed_mode ? metadata_block_m : 128;
        cfg = std::make_tuple(num_sms, config_m, config[0], config[1], config[2], config[3], config[4],
            deep_gemm_fp8_common::get_smem_config(config[4], k, config_m, config[0], config[1]));
    }
    // Small-K two-stage heuristics may select the native N-stage expansion.
    // This return wrapper currently preserves the regular mainloop only.
    if (!preprocessed_mode && k <= 512 && std::get<6>(cfg) == 2) {
        TORCH_CHECK(config.empty(), "FP8 peer return with K<=512 requires at least three stages");
        std::get<6>(cfg) = 3;
        std::get<7>(cfg) = deep_gemm_fp8_common::get_smem_config(
            3, k, std::get<1>(cfg), std::get<2>(cfg), std::get<3>(cfg));
    }
    const auto [sms, bm, bn, bk, wm, wn, stages, smem] = cfg;
    if(preprocessed_mode) {
        const auto& metadata=*expert_metadata;
        DG_HOST_ASSERT(metadata.device()==a.device() && metadata.is_contiguous() && metadata.dim()==1 && metadata.scalar_type()==torch::kInt32);
        DG_HOST_ASSERT(metadata_block_m==bm && metadata.numel()>=std::max<int64_t>(128,4*(1+ceil_div(m,bm)+groups)));
    }
    DG_HOST_ASSERT(bk == 128);
    if (tile_producer) DG_HOST_ASSERT(bm == 128 && ((bm/wm)*(bn/wn)*32 == 256 || (bm/wm)*(bn/wn)*32 == 512));
    if(output_mode)DG_HOST_ASSERT(bm==128 && bn==256);
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
        .scheduler = {uint32_t(m), preprocessed_mode ? expert_metadata->data_ptr<int32_t>() : lengths.data_ptr<int32_t>()},
        .signal = nullptr};
    PeerGemmParams peers{};
    peers.permutation = permutation.data_ptr<int32_t>();
    peers.destination_peers = reinterpret_cast<const uint64_t*>(destinations.data_ptr<int64_t>());
    peers.destination_rows_per_rank = destination_rows_per_rank;
    peers.num_ranks = destinations.numel();
    if(activation_backward)peers.activation_gate_up=activation_gate_up->data_ptr();
    if (tile_producer) {
        peers.source_peers = reinterpret_cast<const uint64_t*>(producer_sources[0].data_ptr<int64_t>());
        peers.source_rows_per_rank = producer_tokens;peers.dispatch_scratch = producer_sources[1].data_ptr();
        input.status = reinterpret_cast<uint32_t*>(producer_sources[2].data_ptr<int32_t>());
    }
    const std::string name = preprocessed_mode ? "fp8_grouped_deep_gemm_NoPad" : expert_end ? "fp8_training_expert_batch_native" : tile_producer ? "fp8_training_peer_tileproducer" : activation_backward ? "fp8_training_dhidden_activation" : return_store_bytes==16 ? "fp8_training_peer_return_wide16" : copy_only ? "fp8_training_peer_return_copy" : output_mode ? "fp8_training_peer_return_pipeline" : dynamic_input ? "fp8_training_peer_input_elastic" : co_resident_input ? "fp8_training_peer_input_coresident" : input_mode ? "fp8_training_peer_input" :
        keep_local ? "fp8_training_peer_return_keep" : "fp8_training_peer_return_discard";
    auto args = PeerFp8ReturnRuntime::Args{
        .launch_info = {bm, bn, bk, wm, wn, preprocessed_mode ? 128 : int(groups), stages, "MGroupedNoPad", "Default", name, false},
        .launch_args = {get_grid_shape(sms), dim3((bm / wm) * (bn / wn) * 32), std::get<0>(smem)},
        .kernel_params = FP8GemmRuntime::to_underlying_arguments_rtc(gemm, nullptr),
        .peers = peers, .keep_local = keep_local, .input = input, .input_mode = input_mode,
        .output = output, .output_mode = output_mode, .copy_only = copy_only, .copy_threads = copy_threads,
        .return_store_bytes = return_store_bytes, .activation_backward = activation_backward,
        .input_work_counter = dynamic_input ? reinterpret_cast<uint32_t*>(input_control[2].data_ptr<int32_t>()) : nullptr,
        .input_ready_count = input_ready_count, .tile_producer = tile_producer, .producer_topk = producer_topk,
        .producer_gradient = producer_gradient, .gradient_producer = gradient_producer, .k_slice_input = k_slice_input, .input_sub_m=input_sub_m,.expert_begin=expert_begin,.expert_end=expert_end,.preprocessed_experts=preprocessed_mode};
    if(return_store_bytes==16) {
        args.launch_args.smem_size=std::max<size_t>(args.launch_args.smem_size,size_t(bm)*(bn+16)*sizeof(cutlass::bfloat16_t));
        DG_HOST_ASSERT(args.launch_args.smem_size+size_t(bm)*sizeof(uint64_t)+8<=256*1024);
    }
    if(copy_only) { args.launch_args.block_dim.x=copy_threads; args.launch_args.smem_size=0; }
    const auto runtime = compiler->build(name, PeerFp8ReturnRuntime::generate(args));
    int occupancy = 0;
    DG_HGGC_UNIFIED_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, runtime->kernel, args.launch_args.block_dim.x, args.launch_args.smem_size));
    DG_HOST_ASSERT(occupancy > 0);
    TORCH_CHECK(!co_resident_input || occupancy>=2,
                "co-resident input GEMM must permit at least two resident CTAs per CU");
    // Waiting consumers may never fill the CUs reserved for their producer.
    // In particular, multiplying this grid by occupancy can deadlock input.
    if(copy_only)args.launch_args.grid_dim.x=reserved_return_ctas;
    else if (input_mode) args.launch_args.grid_dim.x = co_resident_input ? num_sms : num_sms-reserved_input_ctas;
    else if(output_mode)args.launch_args.grid_dim.x=num_sms-reserved_return_ctas;
    else args.launch_args.grid_dim.x *= occupancy;
    // No polling consumer exists in metadata mode. Limiting work CTAs is
    // a resource experiment, not CU affinity or a joint-residency promise.
    if(preprocessed_mode && metadata_ctas)args.launch_args.grid_dim.x=std::min<uint32_t>(args.launch_args.grid_dim.x,metadata_ctas);
    if (!compile_only) PeerFp8ReturnRuntime::launch(runtime, args);
    return args.launch_args.grid_dim.x;
}


// Read-only native heuristic query for preallocating a graph-stable schedule.
static std::vector<int> peer_fp8_auto_config(int device,int m,int n,int k,int groups) {
    const c10::cuda::CUDAGuard guard(device);
    DG_HOST_ASSERT(m>0 && n>0 && k>0 && groups>0 && groups<128);
    const auto cfg=deep_gemm_fp8_common::get_best_configs(ceil_div(m,groups),n,k,groups,get_num_sms(),GemmType::MGroupedNoPad);
    return {std::get<1>(cfg),std::get<2>(cfg),std::get<3>(cfg),std::get<4>(cfg),std::get<5>(cfg),std::get<6>(cfg)};
}

static void register_peer_fp8_apis(pybind11::module_& m) {
    m.def("_moe_training_peer_fp8_auto_config",&peer_fp8_auto_config,py::arg("device"),py::arg("m"),py::arg("n"),py::arg("k"),py::arg("groups"));
    m.def("_moe_training_peer_fp8_gemm", &peer_fp8_return,
          py::arg("a"), py::arg("weight"), py::arg("out"), py::arg("lengths"),
          py::arg("permutation"), py::arg("destination_peers"), py::arg("destination_rows_per_rank"),
          py::arg("keep_local") = true, py::arg("compile_only") = false,
          py::arg("config") = std::vector<int>{},
          py::arg("input_control") = std::vector<torch::Tensor>{},
          py::arg("reserved_input_ctas") = 0,
          py::arg("return_control") = std::vector<torch::Tensor>{},
          py::arg("reserved_return_ctas") = 0,
          py::arg("copy_only") = false, py::arg("copy_threads") = 512,
          py::arg("return_store_bytes") = 4,
          py::arg("activation_gate_up") = std::nullopt,
          py::arg("co_resident_input") = false, py::arg("input_ready_count") = 1,
          py::arg("producer_sources") = std::vector<torch::Tensor>{}, py::arg("producer_tokens") = 0, py::arg("producer_topk") = 0, py::arg("producer_gradient_sources") = std::vector<torch::Tensor>{}, py::arg("k_slice_input") = false, py::arg("input_sub_m")=128, py::arg("expert_begin")=0, py::arg("expert_end")=0, py::arg("expert_metadata")=std::nullopt, py::arg("metadata_block_m")=0, py::arg("metadata_ctas")=0);
}

} // namespace deep_gemm::training
