#pragma once
#include <cstdint>
namespace deep_gemm::w4fa8 {
enum class FragmentPolicy { FullK, RollingK32 };
enum class ALoadPolicy { Bulk, Ring2 };
enum class SchedulerPolicy { Standard, Queue };

template <class Storage, int Rows> struct RowScaleStorage : Storage {
  alignas(128) float row_scales[2][Rows];
};
template <class Storage> struct RowScaleStorage<Storage, 0> : Storage {};
} // namespace deep_gemm::w4fa8

namespace deep_gemm::w4fa8_streamed {
inline constexpr int kBlockM = 256;
struct GroupedRowRangeArguments {
  uint32_t shape_m;
  int32_t* grouped_layout;
  const int32_t* row_range;
};
struct GatherParams {
  const uint8_t *a;
  const float *a_scale;
  uint8_t *grouped_a;
  float *grouped_scale;
  const int32_t *experts;
  const int32_t *tokens;
  const int32_t *aligned;
  int num_tokens;
};
struct DecodeParams {
  const uint8_t* packed;
  const uint8_t* scales;
  uint8_t* output;
  int n_begin;
  const int32_t* all_rows;
  int group_start;
};
} // namespace deep_gemm::w4fa8_streamed

namespace deep_gemm::w4fa8 {
// Queue header precedes the per-tile task records.
inline constexpr int kRoutedMetadataBytes = 1024;
struct RoutedMetadataArguments {
  const int32_t* rows;
  uint8_t* metadata;
};


} // namespace deep_gemm::w4fa8
