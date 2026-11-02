#pragma once
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include "../../jit/device_runtime.hpp"
#include <deep_gemm/common/w4a8_arguments.cuh>
#include <tuple>
#include <vector>
#include <variant>
#include <utility>
#include "../../utils/exception.hpp"
#include "../../utils/utils.hpp"

namespace deep_gemm::w4fa8 {

struct W4A8Config {
    int num_sms;
    int block_m, block_n, block_k;
    int warp_m, warp_n, warp_k;
    int num_stages, n_expand;
};
// Execution choices are independent; only supported combinations are admitted.
enum class ChunkSchedule { Fixed, Ramp };
enum class GatherPolicy { None, Full, Chunk };
enum class EmptyExpertPolicy { Dense, Nonempty };

struct NativeKernelConfig {
    W4A8Config tile;
    FragmentPolicy fragment;
    ALoadPolicy a_load;
    SchedulerPolicy scheduler;
};
struct NativePlan {
    std::vector<NativeKernelConfig> kernels;
};
struct StreamedConfig {
    W4A8Config core;
    int decode_n, expert_chunk;
    ChunkSchedule schedule;
    GatherPolicy gather;
    EmptyExpertPolicy empty_experts;
};
using ExecutionPlan = std::variant<NativePlan, StreamedConfig>;

// Routing storage and compute tiles have different roles in multi-kernel plans.
static int get_token_block_m(const ExecutionPlan& plan, GemmType kind) {
    if (kind != GemmType::MGroupedFused)
        return 0;
    if (const auto* native = std::get_if<NativePlan>(&plan)) {
        DG_HOST_ASSERT(native->kernels.size() == 1 || native->kernels.size() == 2);
        return native->kernels.size() == 1 ? native->kernels.front().tile.block_m : 64;
    }
    return 64;
}

using W4A8ConfigTuple = std::tuple<int, int, int, int, int, int, int, int, int>;

static W4A8Config unpack_config(const W4A8ConfigTuple& tuple) {
    const auto& [sms, bm, bn, bk, wm, wn, wk, stages, expand] = tuple;
    return {sms, bm, bn, bk, wm, wn, wk, stages, expand};
}
static W4A8ConfigTuple pack_config(const W4A8Config& c) {
    return {c.num_sms, c.block_m, c.block_n, c.block_k, c.warp_m, c.warp_n, c.warp_k, c.num_stages, c.n_expand};
}
static int get_num_threads(const W4A8Config& c) {
    const int64_t threads = int64_t(c.block_m / c.warp_m) * (c.block_n / c.warp_n) * (c.block_k / c.warp_k) * 32;
    DG_HOST_ASSERT(threads > 0 && threads <= 1024);
    return int(threads);
}

static void check_config(const W4A8Config& c, int n, int k, GemmType kind) {
    DG_HOST_ASSERT(c.num_sms > 0 and c.num_sms <= get_num_sms());
    DG_HOST_ASSERT(c.block_m == 16 or c.block_m == 32 or c.block_m == 64 or c.block_m == 128);
    DG_HOST_ASSERT(c.warp_m == 16 or c.warp_m == 32 or c.warp_m == 64 or c.warp_m == 128);
    DG_HOST_ASSERT(c.block_m % c.warp_m == 0 and (c.warp_n == 64 or c.warp_n == 32));
    DG_HOST_ASSERT(c.block_n >= 64 and c.block_n % 64 == 0);
    if (c.warp_n == 32 or c.warp_m == 128) {
        DG_HOST_ASSERT(kind == GemmType::MGroupedNoPad and c.warp_m == 128 and c.warp_n == 32);
        DG_HOST_ASSERT(c.block_m == 128 and c.block_k == c.warp_k and c.n_expand == 1);
    }
    // PPU1.5 AIU minimum 64B swizzle requires 128 packed-FP4 K elements.
    DG_HOST_ASSERT(c.block_k == 128 and (c.warp_k == 64 or c.warp_k == 128));
    DG_HOST_ASSERT(k % c.block_k == 0 and c.num_stages >= 2 and c.num_stages <= 4);
    DG_HOST_ASSERT(c.n_expand >= 1 and c.n_expand <= 4);
    if (c.n_expand > 1) {
        DG_HOST_ASSERT(kind == GemmType::MGroupedNoPad and c.num_stages == 2);
        DG_HOST_ASSERT(c.block_k == c.warp_k);
        DG_HOST_ASSERT(n % (int64_t(c.block_n) * c.n_expand) == 0);
    }
    DG_HOST_ASSERT(c.block_m == c.warp_m or c.block_k == c.warp_k);
    DG_HOST_ASSERT(get_num_threads(c) <= 1024);
}

static int get_smem_size(const W4A8Config& c, int k, SchedulerPolicy scheduler = SchedulerPolicy::Standard) {
    const int threads = get_num_threads(c);
    const int copy_rows = threads / std::min(threads, std::min(c.block_k, 128) / 16);
    const int a_rows = scheduler == SchedulerPolicy::Queue ? ceil_div(c.block_m, copy_rows) * copy_rows : c.block_m;
    const bool resident_a = c.n_expand > 1 && c.warp_m == 64 && k <= 384;
    const int a_slots = resident_a ? std::max(c.num_stages, k / c.block_k) : c.num_stages;
    const int64_t mainloop = int64_t(a_rows) * c.block_k * a_slots +
                            int64_t(c.block_n) * (c.block_k / 2 + c.block_k / 32) * c.num_stages;
    const int64_t reduction = int64_t(16 * 64) * (c.block_n / c.warp_n) * (c.block_k / c.warp_k) * sizeof(float);
    int64_t bytes = std::max(mainloop, reduction);
    if (scheduler == SchedulerPolicy::Prefix && c.block_m == 32)
        bytes += 2 * 64 * sizeof(uint32_t);
    if (scheduler == SchedulerPolicy::Queue)
        bytes = ceil_div(bytes, int64_t(128)) * 128 + ceil_div(int64_t(2 * c.block_m * sizeof(float)), int64_t(128)) * 128;
    DG_HOST_ASSERT(bytes <= INT32_MAX);
    return int(bytes);
}

// Bound M before rounding so even a large tuning hint cannot overflow.
static W4A8Config get_default_tile(int expected_m, int n, int k, int groups, GemmType kind) {
    DG_HOST_ASSERT(expected_m >= 0 && groups > 0 && n > 0 && n % 64 == 0 && k > 0);
    DG_HOST_ASSERT(kind == GemmType::MGroupedFused || kind == GemmType::MGroupedNoPad ||
                   kind == GemmType::MGroupedMasked);
    // Wider M reuses each native FP4 decode over eight M16 fragments.
    // Keep the small-M, short-K and other grouped paths on their existing tiles.
    if (kind == GemmType::MGroupedNoPad and expected_m >= 128 and n >= 1024 and n % 128 == 0 and k >= 1024) {
        W4A8Config c{get_num_sms(), 128, 128, 128, 128, 32, 128, 2, 1};
        check_config(c, n, k, kind);
        return c;
    }
    int bm = 16;
    while (bm < expected_m && bm < 64)
        bm *= 2;
    const int bn = n >= 256 ? 256 : n >= 128 ? 128 : 64;
    // M32/N256 has enough shared memory for three stages at four CTAs/SM
    // (64,512 bytes per CTA). M16 and M64 lose residency with a third stage.
    const int stages = kind == GemmType::MGroupedMasked && bm == 32 && bn == 256 ? 3 : 2;
    W4A8Config c{get_num_sms(), bm, bn, 128, bm, 64, 128, stages, 1};
    if (kind == GemmType::MGroupedNoPad and k <= 512 and k % c.block_k == 0 and c.num_stages == 2 and
        c.block_k == c.warp_k) {
        for (const int candidate : {4, 3, 2}) {
            if (n % (bn * candidate) == 0) {
                c.n_expand = candidate;
                break;
            }
        }
    }
    check_config(c, n, k, kind);
    return c;
}
static const char* get_gemm_type_name(GemmType kind) {
    switch (kind) {
    case GemmType::MGroupedFused:
        return "MGroupedFused";
    case GemmType::MGroupedNoPad:
        return "MGroupedNoPad";
    case GemmType::MGroupedMasked:
        return "MGroupedMasked";
    default:
        DG_HOST_ASSERT(false);
    }
    return "";
}

// Strategy domains follow the scheduler capacity, alignment and minimum work
// needed to amortize prefix/metadata construction. They contain no model shapes.
static bool supports_native_strategy() { return is_ppu1v5_device() && get_num_sms() > 0; }

static W4A8Config small_m_config(int expected_m, int n = 256, GemmType kind = GemmType::MGroupedNoPad, int groups = 1) {
    const int bm = expected_m <= 16 ? 16 : expected_m <= 32 ? 32 : 64;
    const bool fused = kind == GemmType::MGroupedFused;
    const int base_n = n >= 256 ? 256 : 128;
    const bool underfilled = int64_t(groups) * ceil_div(n, base_n) < get_num_sms();
    const int bn = fused && bm == 32 ? 128 : base_n;
    const int wn = bm == 32 || (fused && bm == 16 && underfilled) ? 32 : 64;
    return {get_num_sms(), bm, bn, 128, bm, wn, 128, 2, 1};
}

static int specialized_smem_size(const W4A8Config& c, int k, bool routed, bool prefix = true) {
    return get_smem_size(c, k, routed ? SchedulerPolicy::Queue : prefix ? SchedulerPolicy::Prefix : SchedulerPolicy::Standard);
}

static bool fits_strategy(const W4A8Config& c, int k, bool routed, bool prefix = true) {
    const auto prop = device_runtime->get_prop();
    return get_num_threads(c) <= prop->maxThreadsPerBlock &&
           specialized_smem_size(c, k, routed, prefix) <= int(prop->sharedMemPerBlockOptin);
}

static bool fits_compiled_strategy(int local_bytes) {
    // Allow a small compiler frame, but reject spilled operand fragments.
    return local_bytes <= 64;
}

static bool small_m_eligible(int groups, int m, int n, int k, GemmType kind = GemmType::MGroupedNoPad) {
    const bool prefix = kind != GemmType::MGroupedFused && groups <= 64;
    if (groups < 1 || groups > 1024 || ceil_div(m, groups) >= 512 || n < 128 || n % 64 || k < 512 ||
        k % 128)
        return false;
    return fits_strategy(small_m_config(ceil_div(m, groups), n, kind, groups), k, false, prefix);
}

static W4A8Config routed_config(bool small, GemmType kind = GemmType::MGroupedNoPad) {
    const int bm = small ? 160 : kind == GemmType::MGroupedFused ? 176 : 192;
    const int stages = kind == GemmType::MGroupedFused ? (small ? 3 : 2) : (small ? 2 : 3);
    return {get_num_sms(), bm, 256, 128, bm, 32, 128, stages, 1};
}

// A second persistent launch must have enough K-tile work to amortize its
// empty-queue cost. This bound is independent of the current routing values.
static bool routed_partition_eligible(const W4A8Config& c, int m, int n, int k) {
    constexpr int min_k_steps_per_cu = 256;
    const int64_t steps = int64_t(ceil_div(m, c.block_m)) * ceil_div(n, c.block_n) * (k / c.block_k);
    return steps >= int64_t(c.num_sms) * min_k_steps_per_cu;
}

static bool routed_eligible(int groups, int m, int n, int k, GemmType kind, const W4A8Config& original) {
    if (groups < 1 || groups > 1024 || (groups <= 64 && kind != GemmType::MGroupedFused) || n < 256 || n % 256 ||
        k < 512 || k % 128)
        return false;
    const int expected_m = ceil_div(m, groups);
    if (expected_m < 96 || expected_m > 256 || int64_t(ceil_div(m, 192)) * (n / 256) < get_num_sms())
        return false;
    // Small expert sets need several waves to repay the device metadata pass.
    if (groups <= 64 && int64_t(ceil_div(m, 192)) * (n / 256) < int64_t(get_num_sms()) * 4)
        return false;
    // Cost both queues that runtime can select, retaining the original padding bound.
    if (kind == GemmType::MGroupedFused) {
        const auto lower = routed_config(true, kind), upper = routed_config(false, kind);
        int routed_rows = ceil_div(expected_m, lower.block_m) * lower.block_m;
        if (routed_partition_eligible(lower, m, n, k) && fits_strategy(upper, k, true))
            routed_rows = std::min(routed_rows, ceil_div(expected_m, upper.block_m) * upper.block_m);
        if (routed_rows * 4 > ceil_div(expected_m, original.block_m) * original.block_m * 5)
            return false;
    }
    return fits_strategy(routed_config(true, kind), k, true) &&
           (kind == GemmType::MGroupedFused || fits_strategy(routed_config(false), k, true));
}

inline constexpr int64_t kStreamedWeightBudget = 384LL * 1024 * 1024;

// Keep at least one complete N256 core tile in each partial buffer. Expert
// chunks can shrink before N is split; a single oversized expert uses N slices.
static int streamed_tile_n(int n, int k, int groups) {
    const auto prop = device_runtime->get_prop();
    if (prop->maxThreadsPerBlock < 512 || prop->sharedMemPerBlockOptin < 256 * 1024 || groups < 8 || groups > 1024 ||
        n < 1024 || n % 64 || k < 1024 || k % 128)
        return 0;
    const int64_t capacity = (kStreamedWeightBudget - 8) / k / 256 * 256;
    return capacity >= 256 ? int(std::min(int64_t(n), capacity)) : 0;
}
static int streamed_chunk_limit(int tile_n, int k, int groups) {
    // One warp of expert counts stays below the stock scheduler's 128-expert
    // preprocessing boundary, and bounds decoder specialization/code size.
    return std::min({groups / 2, 32, int((kStreamedWeightBudget - 8) / (int64_t(tile_n) * k))});
}

// Estimate complete waves for a bounded expert chunk, including its tail.
static int64_t streamed_waves(int m, int n, int groups, int num_sms, int chunk) {
    const int64_t row_tiles = (int64_t(m) + int64_t(groups) * 256 - 1) / (int64_t(groups) * 256);
    const int64_t n_tiles = (int64_t(n) + 255) / 256;
    auto waves = [&](int experts) {
        const int64_t tiles = int64_t(experts) * row_tiles * n_tiles;
        return ceil_div(tiles, int64_t(std::max(1, num_sms)));
    };
    return int64_t(groups / chunk) * waves(chunk) + waves(groups % chunk);
}

// Short calls need enough N/K tile work to repay additional decode/core
// launches. Use tile counts rather than a list of projection dimensions.
static bool short_streamed_eligible(int m, int n, int k, int groups, int num_sms) {
    const int64_t nk_tiles = int64_t(ceil_div(n, 256)) * (k / 128);
    const int tile_n = streamed_tile_n(n, k, groups);
    if (!tile_n || nk_tiles < 128 || k / 128 < 16 || int64_t(m) < int64_t(groups) * 512 ||
        int64_t(m) >= int64_t(groups) * 1024)
        return false;
    const int maximum = streamed_chunk_limit(tile_n, k, groups);
    return streamed_waves(m, tile_n, groups, num_sms, std::min(4, maximum)) <=
           streamed_waves(m, tile_n, groups, num_sms, maximum);
}

static int streamed_min_rows(int n, int k, int groups, int m, int num_sms) {
    if (!streamed_tile_n(n, k, groups))
        return 2048;
    return short_streamed_eligible(m, n, k, groups, num_sms) ? 512 : 1024;
}

static bool streamed_fused_eligible(int m, int n, int k, int groups, int num_sms) {
    // Bound the additional activation workspace independently of weight chunks.
    constexpr int64_t activation_budget = 1024LL * 1024 * 1024;
    return supports_native_strategy() && streamed_tile_n(n, k, groups) &&
           int64_t(m) * (k + int(sizeof(float))) <= activation_budget &&
           int64_t(m) >= int64_t(groups) * streamed_min_rows(n, k, groups, m, num_sms);
}

struct StreamedPlan {
    std::vector<std::pair<int, int>> groups;
    int max_groups;
    bool ramp, short_call;
};
static StreamedPlan make_streamed_plan(int m, int n, int k, int groups, int num_sms, bool fused = false) {
    const int tile_n = streamed_tile_n(n, k, groups);
    const int maximum = streamed_chunk_limit(tile_n, k, groups);
    const bool short_call = short_streamed_eligible(m, n, k, groups, num_sms);
    int selected = maximum;
    int64_t best = INT64_MAX;
    for (int candidate = 1; candidate <= maximum; ++candidate) {
        const int64_t waves = streamed_waves(m, tile_n, groups, num_sms, candidate);
        if (short_call) {
            // Minimize decode startup without increasing complete compute waves.
            if (waves <= streamed_waves(m, tile_n, groups, num_sms, maximum)) {
                selected = candidate;
                break;
            }
        } else {
            const int64_t score = waves + ceil_div(groups, candidate);
            if (score <= best) {
                best = score;
                selected = candidate;
            }
        }
    }
    const bool ramp = int64_t(tile_n) * k >= 8LL * 1024 * 1024 && int64_t(m) >= int64_t(groups) * 1024 &&
                      (fused || int64_t(m) < int64_t(groups) * 4096);
    std::vector<std::pair<int, int>> plan;
    int chunk = 0;
    for (int start = 0, target = ramp ? std::min(2, selected) : selected; start < groups;) {
        const int count = std::min({target, selected, groups - start});
        plan.emplace_back(start, count);
        chunk = std::max(chunk, count);
        start += count;
        if (ramp)
            target = std::min(selected, target * 2);
    }
    return {std::move(plan), chunk, ramp, short_call};
}

static void check_streamed_config(const StreamedConfig& c, int m, int n, int k, int groups, GemmType kind) {
    const auto& t = c.core;
    DG_HOST_ASSERT(kind == GemmType::MGroupedNoPad || kind == GemmType::MGroupedFused);
    DG_HOST_ASSERT(groups >= 8 && groups <= 1024 && n >= 1024 && n % 64 == 0 && k >= 1024 && k % 128 == 0);
    DG_HOST_ASSERT(t.num_sms > 0 && t.num_sms <= get_num_sms());
    DG_HOST_ASSERT(pack_config(t) == W4A8ConfigTuple(t.num_sms, 256, 256, 128, 64, 64, 128, 4, 1));
    const auto prop = device_runtime->get_prop();
    DG_HOST_ASSERT(prop->maxThreadsPerBlock >= 512 && prop->sharedMemPerBlockOptin >= 256 * 1024);
    DG_HOST_ASSERT(c.decode_n > 0 && c.decode_n <= n && (c.decode_n == n || c.decode_n % 256 == 0));
    DG_HOST_ASSERT(c.expert_chunk >= 1 && c.expert_chunk <= std::min(32, groups / 2));
    DG_HOST_ASSERT(int64_t(c.expert_chunk) * c.decode_n * k + 8 <= kStreamedWeightBudget);
    DG_HOST_ASSERT(c.empty_experts != EmptyExpertPolicy::Nonempty || c.decode_n == n);
    if (kind == GemmType::MGroupedFused) {
        DG_HOST_ASSERT(c.gather != GatherPolicy::None);
        DG_HOST_ASSERT(c.gather != GatherPolicy::Chunk || c.decode_n == n);
        DG_HOST_ASSERT(int64_t(m) * (k + sizeof(float)) <= 1024LL * 1024 * 1024);
    } else {
        DG_HOST_ASSERT(c.gather == GatherPolicy::None);
    }
}

static StreamedPlan make_streamed_plan(const StreamedConfig& c, int groups, bool short_call = false) {
    std::vector<std::pair<int, int>> plan;
    const bool ramp = c.schedule == ChunkSchedule::Ramp;
    int maximum = 0;
    for (int start = 0, target = ramp ? std::min(2, c.expert_chunk) : c.expert_chunk; start < groups;) {
        const int count = std::min({target, c.expert_chunk, groups - start});
        plan.emplace_back(start, count);
        maximum = std::max(maximum, count);
        start += count;
        if (ramp) target = std::min(c.expert_chunk, target * 2);
    }
    return {std::move(plan), maximum, ramp, short_call};
}

static StreamedConfig default_streamed_config(int m, int n, int k, int groups, int num_sms,
                                               bool fused, bool allow_sparse) {
    const int tile_n = streamed_tile_n(n, k, groups);
    const auto schedule = make_streamed_plan(m, n, k, groups, num_sms, fused);
    return {{num_sms, 256, 256, 128, 64, 64, 128, 4, 1}, tile_n, schedule.max_groups,
            schedule.ramp ? ChunkSchedule::Ramp : ChunkSchedule::Fixed,
            !fused ? GatherPolicy::None : !schedule.short_call && tile_n == n ? GatherPolicy::Chunk : GatherPolicy::Full,
            allow_sparse && tile_n == n ? EmptyExpertPolicy::Nonempty : EmptyExpertPolicy::Dense};
}

static void check_native_kernel(const NativeKernelConfig& spec, int n, int k, int groups, GemmType kind) {
    const auto& c = spec.tile;
    DG_HOST_ASSERT(groups > 0 && groups <= 1024 && c.num_sms > 0 && c.num_sms <= get_num_sms());
    if (spec.scheduler == SchedulerPolicy::Queue) {
        DG_HOST_ASSERT(kind != GemmType::MGroupedMasked);
        DG_HOST_ASSERT(c.block_m == 160 || c.block_m == 176 || c.block_m == 192);
        DG_HOST_ASSERT(c.warp_m == c.block_m && c.block_n == 256 && c.warp_n == 32);
        DG_HOST_ASSERT(c.block_k == 128 && c.warp_k == 128 && c.n_expand == 1);
        DG_HOST_ASSERT(c.num_stages >= 2 && c.num_stages <= 4 && n % 256 == 0 && k >= 512 && k % 128 == 0);
        DG_HOST_ASSERT(spec.fragment == FragmentPolicy::RollingK32 && spec.a_load == ALoadPolicy::Ring2);
    } else if (spec.fragment == FragmentPolicy::FullK && spec.scheduler == SchedulerPolicy::Standard) {
        check_config(c, n, k, kind);
        DG_HOST_ASSERT(spec.a_load == ALoadPolicy::Bulk);
    } else {
        DG_HOST_ASSERT(kind != GemmType::MGroupedMasked && spec.a_load == ALoadPolicy::Bulk && n >= 128);
        DG_HOST_ASSERT(c.warp_n == (c.block_m == 32 ? 32 : c.block_m == 16 && spec.fragment == FragmentPolicy::RollingK32 ? 32 : 64));
        DG_HOST_ASSERT(c.block_m == 16 || c.block_m == 32 || c.block_m == 64);
        DG_HOST_ASSERT(c.block_m == c.warp_m && (c.block_n == 128 || c.block_n == 256));
        DG_HOST_ASSERT(c.warp_n == 32 || c.warp_n == 64);
        DG_HOST_ASSERT(c.block_k == 128 && c.warp_k == 128 && c.n_expand == 1);
        DG_HOST_ASSERT(k >= 512 && k % 128 == 0 && c.num_stages == 2);
        if (spec.scheduler == SchedulerPolicy::Prefix) {
            DG_HOST_ASSERT(kind == GemmType::MGroupedNoPad && groups <= 64);
            DG_HOST_ASSERT(spec.fragment == (c.block_m == 16 ? FragmentPolicy::FullK : FragmentPolicy::RollingK32));
        } else {
            DG_HOST_ASSERT(spec.fragment == FragmentPolicy::RollingK32);
        }
    }
    const auto prop = device_runtime->get_prop();
    DG_HOST_ASSERT(get_num_threads(c) <= prop->maxThreadsPerBlock &&
                   get_smem_size(c, k, spec.scheduler) <= int(prop->sharedMemPerBlockOptin));
}
static void check_plan(const ExecutionPlan& plan, int m, int n, int k, int groups, GemmType kind) {
    if (const auto* p = std::get_if<NativePlan>(&plan)) {
        TORCH_CHECK(p->kernels.size() == 1 || p->kernels.size() == 2, "native plan requires one or two kernels");
        for (const auto& c : p->kernels) check_native_kernel(c, n, k, groups, kind);
        if (p->kernels.size() == 2) {
            const auto& a = p->kernels[0]; const auto& b = p->kernels[1];
            DG_HOST_ASSERT(a.scheduler == SchedulerPolicy::Queue && b.scheduler == SchedulerPolicy::Queue);
            DG_HOST_ASSERT(a.tile.block_m < b.tile.block_m && a.tile.block_n == b.tile.block_n &&
                           a.tile.num_sms == b.tile.num_sms);
        }
    } else check_streamed_config(std::get<StreamedConfig>(plan), m, n, k, groups, kind);
}

static ExecutionPlan get_best_configs(int m, int n, int k, int groups, GemmType kind, int expected_m = -1) {
    const int em = expected_m >= 0 ? expected_m : ceil_div(m, groups);
    const auto base = get_default_tile(em, n, k, groups, kind);
    auto full = [](const W4A8Config& c) { return NativeKernelConfig{c, FragmentPolicy::FullK, ALoadPolicy::Bulk, SchedulerPolicy::Standard}; };
    if (kind != GemmType::MGroupedMasked && supports_native_strategy()) {
        const bool fused = kind == GemmType::MGroupedFused;
        if (em <= 256) {
            const bool routed = routed_eligible(groups, m, n, k, kind, base);
            if (small_m_eligible(groups, m, n, k, kind) && !(kind == GemmType::MGroupedNoPad && base.block_m == 128) &&
                (!routed || em <= 64)) {
                const auto tile = small_m_config(em, n, kind, groups);
                const bool rolling = !fused || tile.block_m != 16 || tile.warp_n != 64;
                return NativePlan{{{tile, rolling && (fused || tile.block_m != 16) ? FragmentPolicy::RollingK32 : FragmentPolicy::FullK,
                                    ALoadPolicy::Bulk, !fused && groups <= 64 ? SchedulerPolicy::Prefix : SchedulerPolicy::Standard}}};
            }
            if (routed) {
                NativePlan plan{{{routed_config(true, kind), FragmentPolicy::RollingK32, ALoadPolicy::Ring2, SchedulerPolicy::Queue}}};
                const auto upper = routed_config(false, kind);
                if ((!fused || routed_partition_eligible(plan.kernels[0].tile, m, n, k)) && fits_strategy(upper, k, true))
                    plan.kernels.push_back({upper, FragmentPolicy::RollingK32, ALoadPolicy::Ring2, SchedulerPolicy::Queue});
                return plan;
            }
        }
        if (streamed_tile_n(n, k, groups) && int64_t(m) >= int64_t(groups) * streamed_min_rows(n, k, groups, m, base.num_sms) &&
            (!fused || streamed_fused_eligible(m, n, k, groups, base.num_sms)))
            return default_streamed_config(m, n, k, groups, base.num_sms, fused, true);
    }
    return NativePlan{{full(base)}};
}

// Public tuples describe complete plans; other GEMM dtypes retain their formats.
namespace py = pybind11;
static NativeKernelConfig parse_native_kernel(const py::handle& value) {
    const auto t = py::cast<py::tuple>(value);
    TORCH_CHECK(t.size() == 4, "native kernel must be (tile, fragment, a_load, scheduler)");
    NativeKernelConfig c{unpack_config(py::cast<W4A8ConfigTuple>(t[0])), FragmentPolicy::FullK,
                         ALoadPolicy::Bulk, SchedulerPolicy::Standard};
    const auto fragment = py::cast<std::string>(t[1]), load = py::cast<std::string>(t[2]), scheduler = py::cast<std::string>(t[3]);
    TORCH_CHECK(fragment == "full_k" || fragment == "rolling_k32", "unknown W4FA8 fragment policy");
    TORCH_CHECK(load == "bulk" || load == "ring2", "unknown W4FA8 A-load policy");
    TORCH_CHECK(scheduler == "standard" || scheduler == "prefix" || scheduler == "queue", "unknown W4FA8 scheduler policy");
    c.fragment = fragment == "full_k" ? FragmentPolicy::FullK : FragmentPolicy::RollingK32;
    c.a_load = load == "bulk" ? ALoadPolicy::Bulk : ALoadPolicy::Ring2;
    c.scheduler = scheduler == "standard" ? SchedulerPolicy::Standard : scheduler == "prefix" ? SchedulerPolicy::Prefix : SchedulerPolicy::Queue;
    return c;
}
static ExecutionPlan parse_plan(const py::tuple& t) {
    TORCH_CHECK(t.size() >= 1, "empty W4FA8 configs");
    const auto mode = py::cast<std::string>(t[0]);
    if (mode == "native") {
        TORCH_CHECK(t.size() == 2, "native configs must be ('native', kernels)");
        NativePlan p;
        for (const auto kernel : py::cast<py::tuple>(t[1])) p.kernels.push_back(parse_native_kernel(kernel));
        return p;
    }
    TORCH_CHECK(mode == "streamed" && t.size() == 5,
                "W4FA8 configs must be a native or streamed execution plan");
    const auto chunk = py::cast<std::tuple<int, int, std::string>>(t[2]);
    const auto schedule = std::get<2>(chunk), gather = py::cast<std::string>(t[3]), empty = py::cast<std::string>(t[4]);
    TORCH_CHECK(schedule == "fixed" || schedule == "ramp", "unknown W4FA8 chunk schedule");
    TORCH_CHECK(gather == "none" || gather == "full" || gather == "chunk", "unknown W4FA8 gather policy");
    TORCH_CHECK(empty == "dense" || empty == "nonempty", "unknown W4FA8 empty-expert policy");
    return StreamedConfig{unpack_config(py::cast<W4A8ConfigTuple>(t[1])), std::get<0>(chunk), std::get<1>(chunk),
                          schedule == "fixed" ? ChunkSchedule::Fixed : ChunkSchedule::Ramp,
                          gather == "none" ? GatherPolicy::None : gather == "full" ? GatherPolicy::Full : GatherPolicy::Chunk,
                          empty == "dense" ? EmptyExpertPolicy::Dense : EmptyExpertPolicy::Nonempty};
}
static py::tuple pack_plan(const ExecutionPlan& p) {
    if (const auto* native = std::get_if<NativePlan>(&p)) {
        py::tuple kernels(native->kernels.size());
        for (size_t i = 0; i < native->kernels.size(); ++i) {
            const auto& c = native->kernels[i];
            kernels[i] = py::make_tuple(pack_config(c.tile), c.fragment == FragmentPolicy::FullK ? "full_k" : "rolling_k32",
                                        c.a_load == ALoadPolicy::Bulk ? "bulk" : "ring2",
                                        c.scheduler == SchedulerPolicy::Standard ? "standard" : c.scheduler == SchedulerPolicy::Prefix ? "prefix" : "queue");
        }
        return py::make_tuple("native", kernels);
    }
    const auto& c = std::get<StreamedConfig>(p);
    return py::make_tuple("streamed", pack_config(c.core), py::make_tuple(c.decode_n, c.expert_chunk,
                          c.schedule == ChunkSchedule::Fixed ? "fixed" : "ramp"),
                          c.gather == GatherPolicy::None ? "none" : c.gather == GatherPolicy::Full ? "full" : "chunk",
                          c.empty_experts == EmptyExpertPolicy::Dense ? "dense" : "nonempty");
}

} // namespace deep_gemm::w4fa8
