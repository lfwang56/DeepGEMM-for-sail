#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/utils.hpp"
#include "../heuristics/common_bf16.hpp"
#include <deep_gemm/common/fused_gemm_common.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm {

class Bf16KGroupedRuntime final : public LaunchRuntime<Bf16KGroupedRuntime> {
public:
    struct LaunchInfo {
        int shape_m, shape_n, num_groups;
        int block_m, block_n, block_k;
        int warp_m, warp_n;
        int num_stages;
        int warp_map;
        bool aiu_before_tsm;
        std::string kernel_name;
        bool cta_probe = false;
        bool output_tile_range = false;
        int b_padding_bytes = 0;
    };

    struct Args {
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        KGroupedGemmArgs kernel_params;
        void* cta_probe_ptr = nullptr;
        uint2 output_tile_range{};
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(
            R"(
#include <deep_gemm/impls/k_grouped_bf16_gemm.cuh>

namespace deep_gemm {{

constexpr uint32_t SHAPE_M = {};
constexpr uint32_t SHAPE_N = {};
constexpr uint32_t NUM_GROUPS = {};
constexpr uint32_t BLOCK_M = {};
constexpr uint32_t BLOCK_N = {};
constexpr uint32_t BLOCK_K = {};
constexpr uint32_t WARP_M = {};
constexpr uint32_t WARP_N = {};
constexpr uint32_t NUM_STAGES = {};
constexpr uint32_t WARP_MAP = {};
constexpr bool AIU_BEFORE_TSMLD = {};
constexpr uint32_t BLOCK_SIZE = 32 * (BLOCK_M / WARP_M) * (BLOCK_N / WARP_N);

extern "C"
__launch_bounds__(BLOCK_SIZE, 1)
__global__ void {}(const KGroupedGemmArgs args{}) {{
    {}
    bf16_k_grouped_gemm_impl<
        cutlass::bfloat16_t, float, GemmType::KGroupedContiguous,
        SHAPE_M, SHAPE_N, NUM_GROUPS,
        BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, BLOCK_SIZE, NUM_STAGES, true,
        WARP_MAP, AIU_BEFORE_TSMLD{}
    >(args{});
    {}
}}

}}  // namespace deep_gemm
)",
            args.launch_info.shape_m,
            args.launch_info.shape_n,
            args.launch_info.num_groups,
            args.launch_info.block_m,
            args.launch_info.block_n,
            args.launch_info.block_k,
            args.launch_info.warp_m,
            args.launch_info.warp_n,
            args.launch_info.num_stages,
            args.launch_info.warp_map,
            args.launch_info.aiu_before_tsm,
            args.launch_info.b_padding_bytes ? fmt::format(", {}", args.launch_info.b_padding_bytes) : "",
            args.launch_info.kernel_name,
            args.launch_info.cta_probe ? ", unsigned long long* cta_probe"
                : args.launch_info.output_tile_range ? ", const uint2 tile_range" : "",
            args.launch_info.cta_probe ? "const unsigned long long cta_begin = __ppu_read_clock64();" : "",
            args.launch_info.output_tile_range ? ", tile_range.x, tile_range.y" : "",
            args.launch_info.cta_probe ?
                "__syncthreads(); if (threadIdx.x == 0) { "
                "cta_probe[2 * blockIdx.x] = cta_begin; "
                "cta_probe[2 * blockIdx.x + 1] = __ppu_read_clock64(); }" : "");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if (args.launch_info.output_tile_range)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params, args.output_tile_range));
        else if (args.launch_info.cta_probe)
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params, args.cta_probe_ptr));
        else
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

static void k_grouped_bf16_gemm_tn_contiguous_impl(
    const torch::Tensor& a, const torch::Tensor& b,
    const std::optional<torch::Tensor>& c, const torch::Tensor& d,
    const int& m, const int& n, const int& sum_k,
    const std::vector<int>& ks_cpu, const torch::Tensor& grouped_layout,
    const bool device_lengths = false,
    const std::optional<torch::Tensor>& cta_probe = std::nullopt) {
    const auto num_groups = static_cast<int>(ks_cpu.size());
    const int b_padding_bytes = get_env<int>("DG_BF16_WGRAD_B_PAD_BYTES", 0);
    DG_HOST_ASSERT(b_padding_bytes >= 0 && b_padding_bytes <= 16384 && b_padding_bytes % 1024 == 0);
    const int tile_chunk = device_lengths ? get_env<int>("DG_BF16_WGRAD_TILE_CHUNK", 0) : 0;
    DG_HOST_ASSERT(tile_chunk >= 0 && tile_chunk <= 1 << 20);
    // A single probe buffer cannot represent several launches. Reject before
    // any output mutation; the original diagnostic remains unchanged at0.
    DG_HOST_ASSERT(tile_chunk == 0 || !cta_probe.has_value());
    // Follow upstream: choose the config using the largest group's K.
    const auto max_k = *std::max_element(ks_cpu.begin(), ks_cpu.end());
    const auto [num_sms, block_m, block_n, block_k, warp_m, warp_n, num_stages, _smem_config] =
        deep_gemm_bf16_common::get_best_configs(m, n, max_k, num_groups, get_num_sms(),
                                               GemmType::KGroupedContiguous);
    const int warp_map = get_env<int>("DG_BF16_WGRAD_WARP_MAP", 0);
    const int aiu_before_tsm = get_env<int>("DG_BF16_WGRAD_AIU_BEFORE_TSM", 1);
    DG_HOST_ASSERT(warp_map >= 0 && warp_map <= 3);
    DG_HOST_ASSERT(aiu_before_tsm == 0 || aiu_before_tsm == 1);
    const int warp_rows = block_m / warp_m, warp_cols = block_n / warp_n;
    if (warp_map == 2) DG_HOST_ASSERT((warp_cols & (warp_cols - 1)) == 0);
    if (warp_map == 3) DG_HOST_ASSERT((warp_rows & (warp_rows - 1)) == 0);
    const int block_size = 32 * (block_m / warp_m) * (block_n / warp_n);
    const int smem_size = static_cast<int>(gemm_smem_total_size(
        2, num_stages, block_m, block_n, block_k)) + b_padding_bytes;
    DG_HOST_ASSERT(smem_size <= 256 * 1024);
    std::string kernel_name = tile_chunk ? "bf16_k_grouped_gemm_tile_range"
        : cta_probe.has_value() ? "bf16_k_grouped_gemm_cta_probe" : "bf16_k_grouped_gemm";
    if (b_padding_bytes) kernel_name += fmt::format("_bpad{}", b_padding_bytes);

    auto args = Bf16KGroupedRuntime::Args{
        .launch_info = {m, n, num_groups, block_m, block_n, block_k,
                        warp_m, warp_n, num_stages, warp_map, bool(aiu_before_tsm), kernel_name, cta_probe.has_value(), tile_chunk != 0, b_padding_bytes},
        .launch_args = {get_grid_shape(num_sms), dim3(block_size), smem_size},
        .kernel_params = {
            .a_ptr = a.data_ptr(),
            .b_ptr = b.data_ptr(),
            .c_ptr = d.data_ptr(),
            .grouped_layout = grouped_layout.data_ptr<int32_t>(),
        }
    };

    const auto& code = Bf16KGroupedRuntime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    const auto& kernel = runtime->kernel;

    int blocks_per_cu = 0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
    DG_HGGC_RUNTIME_CHECK(hggcOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, kernel, block_size, smem_size));
#else
    DG_HGGC_DRIVER_CHECK(hgOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu, kernel, block_size, smem_size));
#endif
    DG_HOST_ASSERT(blocks_per_cu > 0 and "K-grouped tile exceeds the device's launch resources");
    args.launch_args.grid_dim.x *= blocks_per_cu;
    const int compute_cus = get_env<int>("DG_BF16_WGRAD_COMPUTE_CUS", 0);
    DG_HOST_ASSERT(compute_cus >= 0 && compute_cus <= get_num_sms());
    if (compute_cus)
        args.launch_args.grid_dim.x = std::min(args.launch_args.grid_dim.x,
                                              uint32_t(compute_cus * blocks_per_cu));
    if (cta_probe.has_value()) {
        DG_HOST_ASSERT(cta_probe->numel() >= 2 * args.launch_args.grid_dim.x);
        args.cta_probe_ptr = cta_probe->data_ptr();
    }

    DgProfParam dg_prof_params;
    if (deep_gemm::profiling_enabled() && !device_lengths) {
        dg_prof_params.set_k_grouped_params(m, n, sum_k, ks_cpu,
                                           c->data_ptr() != d.data_ptr(), current_stream());
    }
    if (device_lengths) {
        // The internal profiler's CPU K list cannot describe a dynamic device
        // layout. External profilers still observe the actual kernel normally.
        if (tile_chunk) {
            const uint64_t tiles = uint64_t(num_groups) * math::ceil_div(m, block_m) * math::ceil_div(n, block_n);
            DG_HOST_ASSERT(tiles <= UINT32_MAX);
            for (uint64_t first = 0; first < tiles; first += tile_chunk) {
                args.output_tile_range = make_uint2(uint32_t(first), uint32_t(std::min(tiles, first + tile_chunk)));
                Bf16KGroupedRuntime::launch(runtime, args);
            }
        } else {
            Bf16KGroupedRuntime::launch(runtime, args);
        }
    } else {
        auto profiling_guard = deep_gemm::show_gemm_ut(dg_prof_params);
        Bf16KGroupedRuntime::launch(runtime, args);
    }

    const char* p_env_params = std::getenv("show_log");
    if (p_env_params && std::isdigit(static_cast<unsigned char>(*p_env_params))) {
        int num_regs = 0, local_size = 0;
#if HGGCRT_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)
        hggcFuncAttributes attributes;
        DG_HGGC_RUNTIME_CHECK(hggcFuncGetAttributes(&attributes, kernel));
        num_regs = attributes.numRegs;
        local_size = static_cast<int>(attributes.localSizeBytes);
#else
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&num_regs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel));
        DG_HGGC_DRIVER_CHECK(lazy_hgFuncGetAttribute(&local_size, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel));
#endif

        std::printf("[%s_BF16:]\n", device_lengths ? "KGroupedDeviceLengths" : "KGroupedContiguous");
        if (device_lengths)
            std::printf("groups:%d, m:%d, n:%d, capacity_rows:%d, tile_bound_k:%d\n",
                        num_groups, m, n, sum_k, max_k);
        else
            std::printf("groups:%d, m:%d, n:%d, sum_k:%d, max_k:%d\n", num_groups, m, n, sum_k, max_k);
        std::printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%u, threads:%d\n",
                    num_sms, blocks_per_cu, args.launch_args.grid_dim.x, block_size);
        std::printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n",
                    block_m, block_n, block_k, warp_m, warp_n, block_k, num_stages);
        std::printf("dynamic_smem_bytes:%d, vreg:%d, local_size_bytes:%d\n", smem_size, num_regs, local_size);
        std::printf("warp_map:%d, aiu_before_tsm:%d\n", warp_map, aiu_before_tsm);
    }
}

}  // namespace deep_gemm
