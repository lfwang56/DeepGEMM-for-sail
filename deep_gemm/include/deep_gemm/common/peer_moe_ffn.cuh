#pragma once
#include <deep_gemm/common/peer_moe_training.cuh>

namespace deep_gemm::training {
// Separate argument type leaves existing peer GEMM launch ABIs unchanged.
struct PeerFfnParams {
    PeerGemmParams first;
    PeerGemmParams second;
};
}
