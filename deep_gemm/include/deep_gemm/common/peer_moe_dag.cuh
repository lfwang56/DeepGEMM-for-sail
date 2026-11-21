#pragma once
#include <deep_gemm/common/peer_moe_ffn.cuh>

namespace deep_gemm::training {

enum DagState : uint32_t {
    NextFC1, HeadFC2, TailFC2, DoneFC2, DoneInput, DoneActivation, DoneReturn,
    DagError, ActiveInput, ActiveFC1, ActiveActivation, ActiveFC2, ActiveReturn,
    NextInput, HeadFC1, TailFC1, HeadReturn, TailReturn,
    DagStateSize
};

// A separate launch structure leaves existing peer GEMM/FFN argument ABIs intact.
struct PeerFfnDagParams {
    PeerFfnParams ffn;
    uint32_t* state;
    uint32_t* input_ready;
    uint32_t* pair_done;
    uint32_t* activated_pairs;
    uint32_t* queue;
    uint32_t* output_ready;
    uint32_t* global_status;
    uint64_t* monitor;
    uint64_t* samples;
    uint32_t sample_capacity;
    uint32_t input_ctas, return_ctas;
    bool serialize;
    uint64_t timeout_cycles = 30ull * 2'000'000'000;
};

}  // namespace deep_gemm::training
