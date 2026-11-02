#pragma once
#include <deep_gemm/scheduler/scheduler_cutlass3.cuh>
#include <deep_gemm/common/w4a8_arguments.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
namespace deep_gemm::w4fa8 {
using namespace cute;
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
    return true;
  }
};
template <int N, int K, int BM, int BN, bool InterleaveM = false> struct W4A8RoutedScheduler {
  struct Params {
    const uint8_t* meta;
  };
  RoutedWorkScheduler<N, BM, BN, InterleaveM> inner;
  int curr_group_idx, curr_group_m, curr_block_m_offset, valid_m_in_block, token_base, group_base;
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
    return true;
  }
  CUTLASS_DEVICE int64_t curr_offset_a() { return int64_t(group_base) * K; }
  CUTLASS_DEVICE int64_t curr_offset_c() { return int64_t(group_base) * N; }
};
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

} // namespace deep_gemm::w4fa8
