#pragma once

#include <cstdint>

namespace deep_gemm::training {

enum PipelineBuffer : int {
    X, Ids, Scores, W1, W2, Valid, Y, DY, DX, DW1, DW2, DScores,
    LocalX, LocalIds, LocalY, LocalDX, LocalDY,
    Counts, Offsets, Lengths, Permutation, RouteExpert, RouteRow,
    PackedX, GateUp, Hidden, PackedDY, W1T, W2T,
    SourceRows, IdentityRows, WorkInfo, NumBlocks,
    Status, PeerBarrierState, BarrierPeers, XPeers, IdPeers, YPeers, DXPeers,
    DYPeers, GridSync, NumPipelineBuffers,
};

struct PipelineParams {
    void* ptr[NumPipelineBuffers];
    int64_t input_tokens, max_tokens, rank, rows, alignment;
    uint64_t timeout_cycles;
};

}  // namespace deep_gemm::training
