#pragma once

#include <cstdint>

namespace deep_gemm::training {

// Fixed POD ABI shared by the host dispatcher and the JIT device entry points.
// Tensor storage remains caller-owned for the entire asynchronous operation.
struct Params {
    void* ptr[12]{};
    int64_t dim[12]{};
};

enum class Operation : int {
    CountRoutes,
    PrefixRoutes,
    InitializeLayout,
    PackInput,
    UnpackRoutes,
    PackRouteGrad,
    Combine,
    Activation,
    ActivationBackward,
    QuantizeRows,
    QuantizeWeights,
    WeightedGradient,
    ScoreGradient,
    ReadChunk,
    ActivationQuantize,
    ActivationBackwardQuantize,
    ActivationDeviceRows,
    TransformDeviceRows,
    ZeroDeviceRows,
    BuildPermutationOnly,
    PackInputDeviceRows,
    TransformDeviceRowsVector,
    TransposeWeights,
    DeepEPReduceRows,
    DeepEPBackwardRows,
    DeepEPReduceWindowRows,
    QuantizeRowsDeviceRows,
    ActivationQuantizeDeviceRows,
    ActivationBackwardQuantizeDeviceRows,
    QuantizeWeightsCachedFloat,
    QuantizeWeightsCachedInteger,
    PackQuantizedRouteRows,
    TransposeQuantizedWeights,
    PackFp8WeightGradients,
};

}  // namespace deep_gemm::training
