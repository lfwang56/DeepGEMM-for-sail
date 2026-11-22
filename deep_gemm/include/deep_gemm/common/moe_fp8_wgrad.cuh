#pragma once
#include <cstdint>
namespace deep_gemm::training {
struct Fp8WgradParams {
    const int32_t* lengths;
    const int32_t* offsets;
    float* output;
    uint32_t* status;
    uint32_t groups, m, n, capacity;
};
}
