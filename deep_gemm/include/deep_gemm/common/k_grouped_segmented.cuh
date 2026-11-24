#pragma once
#include <cstdint>
namespace deep_gemm {
// Host/device launch parameter: fixed-size, no device allocation or pointer chase.
struct KGroupedSegmentArgs {
    const void* a[8]{};
    const void* b[8]{};
    const int32_t* offsets[8]{};
};
} // namespace deep_gemm
