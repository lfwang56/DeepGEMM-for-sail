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
    ActivationActive,
    ActivationBackwardActive,
    LocalGradients,
    PackActiveVector,
    ScatterVector,
    ExpertLocalGradients,
    SetRouteWindow,
    PrefetchRawDy,
    AssistInputTiles,
    ReduceScorePartials,
    StageRawWeightedDy,
    StageFp8InputTiles,
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
    void* activation_gate_up = nullptr;
    void* dispatch_scratch = nullptr;
};

// Separate launch argument preserves the existing PeerGemmParams ABI.
struct PeerDispatchPipeline {
    uint32_t* ready = nullptr;
    uint32_t* status = nullptr;
    uint64_t* statistics = nullptr;
    uint32_t communication_ctas = 0;
    uint64_t timeout_cycles = uint64_t(180) * 2'000'000'000;
    uint32_t signal_capacity = 0;
    bool serialize = false;
};

// Separate optional launch argument: preserve PeerGemmParams and old kernels.
struct PeerReplayGradients {
    const uint64_t* dy_peers = nullptr;
    const uint64_t* score_peers = nullptr;
    void* packed_dy = nullptr;
    float* score_partials = nullptr;
    uint32_t* status = nullptr;
    uint32_t source_tokens = 0;
    uint32_t topk = 0;
    uint32_t num_ranks = 0;
    uint32_t num_n_tiles = 0;
};

// Optional dHidden epilogue argument, separate from existing GPU ABI structs.
struct PeerDHiddenScores {
    const void* hidden = nullptr;
    const float* row_scores = nullptr;
    float* score_partials = nullptr;
    uint32_t num_n_tiles = 0;
};

}  // namespace deep_gemm::training
