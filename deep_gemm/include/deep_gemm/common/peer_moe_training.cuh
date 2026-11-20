#pragma once

#include <deep_gemm/common/moe_training.cuh>
#include <deep_gemm/common/fused_gemm_common.cuh>

namespace deep_gemm::training {

enum class PeerOperation : int {
    CountRoutes,
    InitializeLayout,
    BuildPermutation,
    BuildWorkInfo,
    Pack,
    Barrier,
    Scatter,
    PackActive,
};

struct PeerGemmParams {
    GemmArgs gemm;
    const uint64_t* source_peers;
    const uint64_t* destination_peers;
    const int32_t* permutation;
    uint32_t source_rows_per_rank;
    uint32_t destination_rows_per_rank;
    uint32_t num_ranks;
    bool pack_input;
};

}  // namespace deep_gemm::training
