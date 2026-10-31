#pragma once
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/common/w4a8_arguments.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
namespace deep_gemm::w4fa8 {
using namespace cute;
// Reserved words in the private routed metadata header. Rebuilt on every call.
inline constexpr int kRoutedAffineRows = 2;
inline constexpr int kRoutedAffineWitness = 3;
struct Triple {
  int r, p, b;
};
struct Add {
  CUTLASS_DEVICE Triple operator()(Triple a, Triple b) const { return {a.r + b.r, a.p + b.p, a.b + b.b}; }
};
template <int N, int BM, int BN, bool InterleaveM = false> struct RoutedWorkScheduler {
  const uint4* records;
  int next = 0, begin, end, start;
  int expert, rows, out_row, token_base, valid;
  int affine_first = 0, affine_stride = 0;
  __device__ RoutedWorkScheduler(const uint8_t* meta, int chunkid, int start_)
      : records(reinterpret_cast<const uint4*>(meta + kRoutedMetadataBytes)), start(start_) {
    auto p = reinterpret_cast<const int32_t*>(meta);
    begin = __ldg(p + 2 * chunkid);
    end = __ldg(p + 2 * chunkid + 1);
  }
  __device__ bool fetch(uint32_t& mb, uint32_t& nb) {
    constexpr int NC = N / BN;
    int work = (next++) * gridDim.x + blockIdx.x + begin * NC;
    if (work >= end * NC)
      return false;
    auto d = __ldg(records + 2 * (work / NC));
    auto t = __ldg(records + 2 * (work / NC) + 1);
    int count = (d.y + BM - 1) / BM, local = work - int(d.z) * NC;
    int first = (local / (NC * 2)) * 2, in = local % (NC * 2), width = min(2, count - first);
    mb = first + in % width;
    nb = in / width;
    // Keep a persistent worker from repeatedly receiving only full tiles or
    // only tiny tails when the grid stride preserves the M-pair parity.
    if constexpr (InterleaveM)
      if (width == 2)
        mb ^= (nb + int(d.x)) & 1;
    expert = int(d.x) - start;
    rows = d.y;
    out_row = d.w + mb * BM;
    token_base = t.x + mb * BM;
    valid = min(BM, rows - int(mb) * BM);
    if constexpr (InterleaveM) {
      // M interleaving can select the other record in this expert's pair.
      const auto a = __ldg(records + 2 * (int(d.z) + int(mb)) + 1);
      affine_stride = int(a.y);
      affine_first = int(a.z);
    }
    return true;
  }
};
template <int N, int K, int BM, int BN, bool InterleaveM = false> struct W4A8RoutedScheduler {
  struct Params {
    const uint8_t* meta;
  };
  RoutedWorkScheduler<N, BM, BN, InterleaveM> inner;
  int curr_group_idx, curr_group_m, curr_block_m_offset, valid_m_in_block, token_base, group_base;
  int affine_first = 0, affine_stride = 0;
  CUTLASS_DEVICE explicit W4A8RoutedScheduler(Params p) : inner(p.meta, 0, 0) {}
  CUTLASS_DEVICE bool fetch_next_work(uint32_t& m, uint32_t& n) {
    if (!inner.fetch(m, n))
      return false;
    curr_group_idx = inner.expert;
    curr_group_m = inner.rows;
    curr_block_m_offset = inner.out_row;
    valid_m_in_block = inner.valid;
    token_base = inner.token_base;
    group_base = inner.out_row - int(m) * BM;
    affine_first = inner.affine_first;
    affine_stride = inner.affine_stride;
    return true;
  }
  CUTLASS_DEVICE int64_t curr_offset_a() { return int64_t(group_base) * K; }
  CUTLASS_DEVICE int64_t curr_offset_c() { return int64_t(group_base) * N; }
};
// Prove each routed tile's current token progression once per invocation.
// Other metadata fields and the public padded token layout remain unchanged.
template<int BM, int K>
CUTLASS_DEVICE void build_w4a8_affine_tiles(uint8_t* meta, const int32_t* tokens, int num_tokens) {
  // The witness slot is written only by the preceding metadata kernel.
  if (__ldg(reinterpret_cast<const int32_t*>(meta) + kRoutedAffineWitness) == 0)
    return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int total = __ldg(reinterpret_cast<int32_t*>(meta) + 1);
  auto* records = reinterpret_cast<uint4*>(meta + kRoutedMetadataBytes);
  bool any_bad = false;
  for (int index = blockIdx.x * 8 + warp; index < total; index += gridDim.x * 8) {
    const auto d = __ldg(records + 2 * index);
    const auto t = __ldg(records + 2 * index + 1);
    const int relative = index - int(d.z);
    const int valid = min(BM, int(d.y) - relative * BM);
    const int begin = int(t.x) + relative * BM;
    const int first = __ldg(tokens + begin);
    const int64_t delta = valid > 1 ? int64_t(__ldg(tokens + begin + 1)) - first : 1;
    bool affine = first >= 0 && first < num_tokens && delta > 0 && delta <= 0x7fffffff / K;
    for (int i = lane; i < valid; i += 32) {
      const int token = __ldg(tokens + begin + i);
      affine = affine && token >= 0 && token < num_tokens && int64_t(token) == int64_t(first) + int64_t(i) * delta;
    }
    const bool all_affine = __all_sync(0xffffffffu, affine);
    any_bad = any_bad || !all_affine;
    if (lane == 0)
      records[2 * index + 1] = make_uint4(t.x, all_affine ? uint32_t(delta) : 0, all_affine ? uint32_t(first) : 0, 0);
  }
  const int block_bad = __syncthreads_or(int(any_bad));
  if (threadIdx.x == 0 && block_bad) {
    atomicExch(reinterpret_cast<int32_t*>(meta) + kRoutedAffineRows, 0);
  }
}

// A failed witness is sufficient to retain the ordinary copy body. Passing
// three tokens never enables AIU: the next kernel still checks every row.
template<int BM, int K>
CUTLASS_DEVICE void probe_w4a8_affine_witness(uint8_t* meta, const int32_t* tokens, int num_tokens) {
  __syncthreads();
  if (threadIdx.x == 0) {
    auto* header = reinterpret_cast<int32_t*>(meta);
    if (header[1] <= header[0])
      return;
    auto* records = reinterpret_cast<uint4*>(meta + kRoutedMetadataBytes);
    const auto d = records[2 * header[0]];
    const auto t = records[2 * header[0] + 1];
    const int valid = min(BM, int(d.y));
    const int first = __ldg(tokens + t.x);
    const int64_t delta = valid > 1 ? int64_t(__ldg(tokens + t.x + 1)) - first : 1;
    bool possible = first >= 0 && first < num_tokens && delta > 0 && delta <= 0x7fffffff / K;
    if (valid > 2) {
      const int third = __ldg(tokens + t.x + 2);
      possible = possible && third >= 0 && third < num_tokens && int64_t(third) == int64_t(first) + 2 * delta;
    }
    header[kRoutedAffineWitness] = int(possible);
    if (!possible) {
      header[kRoutedAffineRows] = 0;
    }
  }
}

// Keep the narrower A tile whenever a larger tile would not save weight
// work. Prefixes for output rows and the public padded tokens stay global;
// each work queue receives its own tile prefix.
struct RoutedCounts {
  int rows, padded, lower, upper;
};
struct AddRoutedCounts {
  CUTLASS_DEVICE RoutedCounts operator()(RoutedCounts a, RoutedCounts b) const {
    return {a.rows + b.rows, a.padded + b.padded, a.lower + b.lower, a.upper + b.upper};
  }
};
template <int Groups, int SmallM, int LargeM, int TokenBlockM>
CUTLASS_DEVICE void build_w4a8_partitioned_metadata(const int32_t* rows, uint8_t* lower, uint8_t* upper) {
  constexpr int Items = (Groups + 255) / 256;
  const int tid = threadIdx.x;
  RoutedCounts counts[Items], sum{0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int expert = tid * Items + j;
    const int r = expert < Groups ? __ldg(rows + expert) : 0;
    const int small = (r + SmallM - 1) / SmallM, large = (r + LargeM - 1) / LargeM;
    const bool use_large = large < small;
    counts[j] = {r, (r + TokenBlockM - 1) / TokenBlockM * TokenBlockM,
                 use_large ? 0 : small, use_large ? large : 0};
    sum = AddRoutedCounts{}(sum, counts[j]);
  }
  using Scan = cub::BlockScan<RoutedCounts, 256>;
  __shared__ typename Scan::TempStorage scan;
  RoutedCounts prefix;
  Scan(scan).ExclusiveScan(sum, prefix, RoutedCounts{0, 0, 0, 0}, AddRoutedCounts{});
  auto* lower_range = reinterpret_cast<int32_t*>(lower);
  auto* upper_range = reinterpret_cast<int32_t*>(upper);
  if (tid == 0)
    lower_range[0] = upper_range[0] = 0;
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int expert = tid * Items + j;
    const auto count = counts[j];
    if (expert < Groups) {
      if (expert == Groups - 1) {
        lower_range[1] = prefix.lower + count.lower;
        upper_range[1] = prefix.upper + count.upper;
      }
      const bool use_large = count.upper != 0;
      const int begin = use_large ? prefix.upper : prefix.lower;
      const int tiles = count.lower + count.upper;
      auto* records = reinterpret_cast<uint4*>((use_large ? upper : lower) + kRoutedMetadataBytes);
      for (int tile = 0; tile < tiles; ++tile) {
        records[2 * (begin + tile)] = make_uint4(expert, count.rows, begin, prefix.rows);
        records[2 * (begin + tile) + 1] = make_uint4(prefix.padded, 0, 0, 0);
      }
    }
    prefix = AddRoutedCounts{}(prefix, count);
  }
}
template <int Groups, int SmallM, int LargeM, int TokenBlockM>
CUTLASS_DEVICE void build_w4a8_routed_metadata(const int32_t* rows, uint8_t* lower, uint8_t* upper) {
  const int tid = threadIdx.x;
  constexpr int Items = (Groups + 255) / 256;
  int row_count[Items], extra = 0;
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int e = tid * Items + j;
    const int r = e < Groups ? __ldg(rows + e) : 0;
    row_count[j] = r;
    extra += (r + SmallM - 1) / SmallM - (r + LargeM - 1) / LargeM;
  }
  using Reduce = cub::BlockReduce<int, 256>;
  __shared__ typename Reduce::TempStorage reduction;
  __shared__ int small;
  const int total = Reduce(reduction).Sum(extra);
  if (tid == 0)
    small = total == 0;
  __syncthreads();
  if (tid == 0) {
    auto* inactive = reinterpret_cast<int32_t*>(small ? upper : lower);
    inactive[0] = inactive[1] = 0;
  }
  Triple counts[Items], sum{0, 0, 0};
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int r = row_count[j];
    counts[j] = {r, (r + TokenBlockM - 1) / TokenBlockM * TokenBlockM,
                 small ? (r + SmallM - 1) / SmallM : (r + LargeM - 1) / LargeM};
    sum = Add{}(sum, counts[j]);
  }
  using Scan = cub::BlockScan<Triple, 256>;
  __shared__ typename Scan::TempStorage scan;
  Triple prefix;
  Scan(scan).ExclusiveScan(sum, prefix, Triple{0, 0, 0}, Add{});
  auto* range = reinterpret_cast<int32_t*>(small ? lower : upper);
  auto* records = reinterpret_cast<uint4*>((small ? lower : upper) + kRoutedMetadataBytes);
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int e = tid * Items + j;
    const auto count = counts[j];
    if (e < Groups) {
      if (e == 0)
        range[0] = prefix.b;
      if (e == Groups - 1)
        range[1] = prefix.b + count.b;
      for (int tile = 0; tile < count.b; ++tile) {
        records[2 * (prefix.b + tile)] = make_uint4(e, count.r, prefix.b, prefix.r);
        records[2 * (prefix.b + tile) + 1] = make_uint4(prefix.p, 0, 0, 0);
      }
    }
    prefix = Add{}(prefix, count);
  }
}
// Evaluate live padding/reuse, then build only the bounded wide work records.
// Rates are calibrated kernel time per M tile; no input or routing values persist.
template <int Groups, int NativeM, int WideM, int TokenBlockM>
CUTLASS_DEVICE void build_w4a8_adaptive_metadata(AdaptiveMetadataArguments p) {
  static_assert(NativeM == TokenBlockM, "adaptive native/public row tiles must match");
  constexpr int Items = (Groups + 255) / 256;
  const int tid = threadIdx.x;
  Triple counts[Items], sum{0, 0, 0};
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int e = tid * Items + j;
    const int r = e < Groups ? __ldg(p.rows + e) : 0;
    counts[j] = {r, (r + TokenBlockM - 1) / TokenBlockM * TokenBlockM,
                 (r + WideM - 1) / WideM};
    sum = Add{}(sum, counts[j]);
  }
  using Scan = cub::BlockScan<Triple, 256>;
  __shared__ typename Scan::TempStorage scan;
  Triple prefix;
  Scan(scan).ExclusiveScan(sum, prefix, Triple{0, 0, 0}, Add{});
  auto* range = reinterpret_cast<int32_t*>(p.work);
  auto* records = reinterpret_cast<uint4*>(p.work + kRoutedMetadataBytes);
  if (tid == 0)
    range[0] = 0;
#pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int e = tid * Items + j;
    const auto count = counts[j];
    if (e < Groups) {
      if (e == Groups - 1) {
        const int native_tiles = (prefix.p + count.p) / NativeM, wide_tiles = prefix.b + count.b;
        const bool native = float(native_tiles) * p.native_rate <= float(wide_tiles) * p.wide_rate;
        range[1] = native ? 0 : wide_tiles;
        p.control[0] = int(native);
        p.control[1] = native_tiles;
        p.control[2] = wide_tiles;
        if (p.native_layout)
          p.native_layout[0] = native_tiles;
      }
      // Reuse the row prefix for the stock NoPad scheduler table, preserving
      // the native ABI without a second preprocessing pass.
      if (p.native_layout) {
        auto* native_records = reinterpret_cast<uint4*>(p.native_layout) + 1;
        const int native_prefix = prefix.p / NativeM;
        for (int tile = 0; tile < count.p / NativeM; ++tile)
          native_records[native_prefix + tile] = make_uint4(e, count.r, native_prefix, prefix.r);
      }
      for (int tile = 0; tile < count.b; ++tile) {
        records[2 * (prefix.b + tile)] = make_uint4(e, count.r, prefix.b, prefix.r);
        records[2 * (prefix.b + tile) + 1] = make_uint4(prefix.p, 0, 0, 0);
      }
    }
    prefix = Add{}(prefix, count);
  }
}

} // namespace deep_gemm::w4fa8
