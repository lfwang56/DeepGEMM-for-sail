#pragma once

#include <deep_gemm/common/moe_training_pipeline.cuh>
#include <deep_gemm/impls/moe_training.cuh>
#include <deep_gemm/impls/peer_moe_training.cuh>
#include <deep_gemm/impls/peer_moe_gemm.cuh>
#include <deep_gemm/impls/k_grouped_bf16_gemm.cuh>

namespace deep_gemm::training {

template <typename T>
__device__ __forceinline__ T* pipeline_pointer(const PipelineParams& s, PipelineBuffer index) {
    return reinterpret_cast<T*>(s.ptr[index]);
}

template <PipelineBuffer... Buffers, typename... Dims>
__device__ __forceinline__ Params pipeline_arguments(const PipelineParams& s, Dims... dims) {
    return Params{{s.ptr[Buffers]...}, {int64_t(dims)...}};
}

template <Operation Op, PipelineBuffer... Buffers, typename... Dims>
__device__ __forceinline__ void pipeline_op(const PipelineParams& s, int64_t blocks, Dims... dims) {
    const auto p = pipeline_arguments<Buffers...>(s, dims...);
    for (int64_t block = blockIdx.x; block < blocks; block += gridDim.x) {
        run<Op>(p, block);
        if constexpr (Op == Operation::ScoreGradient || Op == Operation::QuantizeWeights)
            __syncthreads();
    }
}

template <PeerOperation Op, PipelineBuffer... Buffers, typename... Dims>
__device__ __forceinline__ void pipeline_peer_op(const PipelineParams& s, int64_t blocks, Dims... dims) {
    const auto p = pipeline_arguments<Buffers...>(s, dims...);
    for (int64_t block = blockIdx.x; block < blocks; block += gridDim.x)
        run_peer<Op>(p, block);
}

template <typename T>
__device__ __forceinline__ void pipeline_zero(const PipelineParams& s, PipelineBuffer buffer, int64_t count) {
    auto* data = pipeline_pointer<T>(s, buffer);
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(gridDim.x) * blockDim.x)
        data[i] = T{};
}

__device__ __forceinline__ bool pipeline_grid_barrier(const PipelineParams& s, uint32_t& sequence) {
    auto* signals = pipeline_pointer<uint32_t>(s, GridSync);
    auto* status = pipeline_pointer<uint32_t>(s, Status);
    const uint32_t phase = sequence & 1u, sign = (sequence >> 1) & 1u;
    __shared__ int failed;
    __syncthreads();
    if (threadIdx.x == 0) {
        tix::red_add_rel(signals + phase, sign ? uint32_t(-1) : 1u);
        const uint32_t target = sign ? 0 : gridDim.x;
        const uint64_t begin = clock64();
        while (tix::ld_acq(signals + phase) != target) {
            if ((tix::ld_acq(status) & 8) || clock64() - begin > s.timeout_cycles) {
                atomicOr(status, 8u);
                break;
            }
        }
        failed = (tix::ld_acq(status) & 8) != 0;
    }
    __syncthreads();
    ++sequence;
    return failed == 0;
}

template <int P>
__device__ __forceinline__ void pipeline_rank_barrier(const PipelineParams& s, int64_t tag) {
    if (blockIdx.x == 0) {
        const auto p = pipeline_arguments<BarrierPeers, PeerBarrierState, Status>(s, P, s.rank, s.timeout_cycles, tag);
        run_peer<PeerOperation::Barrier>(p, 0);
    }
}

template <int N, int Reduction, int E, int P, int C, int TopK, bool Read, bool Write>
__device__ __forceinline__ void pipeline_gemm(const PipelineParams& s,
        PipelineBuffer a, PipelineBuffer weight, PipelineBuffer output,
        PipelineBuffer input_rows, PipelineBuffer sources, PipelineBuffer destinations, bool pack = false) {
    PeerGemmParams p{};
    p.gemm.a_ptr = s.ptr[a];
    p.gemm.b_ptr = s.ptr[weight];
    p.gemm.c_ptr = s.ptr[output];
    p.gemm.expert_ids_and_cumsum = pipeline_pointer<int32_t>(s, WorkInfo);
    p.gemm.sorted_token_ids = pipeline_pointer<int32_t>(s, input_rows);
    p.gemm.aligned_num_m_blocks = pipeline_pointer<int32_t>(s, NumBlocks);
    const int source_rows = sources == XPeers ? C : C * TopK;
    p.source_rows_per_rank = source_rows;
    p.gemm.shape_m = Read ? P * source_rows : s.rows;
    p.source_peers = pipeline_pointer<uint64_t>(s, sources);
    p.destination_peers = pipeline_pointer<uint64_t>(s, destinations);
    p.permutation = pipeline_pointer<int32_t>(s, Permutation);
    p.destination_rows_per_rank = C * TopK;
    p.num_ranks = P;
    p.pack_input = pack;
    training_bf16_peer_gemm_impl<cutlass::bfloat16_t, GemmType::MGroupedFused,
        N, Reduction, E, 128, 128, 64, 64, 64, 128, 3, Read, Write, true>(p);
    cp_async_wait<0>();
    __syncthreads();
}

template <int M, int N, int E>
__device__ __forceinline__ void pipeline_wgrad(const PipelineParams& s,
        PipelineBuffer a, PipelineBuffer b, PipelineBuffer output) {
    const KGroupedGemmArgs args{s.ptr[a], s.ptr[b], s.ptr[output], pipeline_pointer<int32_t>(s, Lengths)};
    bf16_k_grouped_gemm_impl<cutlass::bfloat16_t, float, GemmType::KGroupedContiguous,
        M, N, E, 128, 128, 64, 64, 64, 128, 3, true>(args);
    cp_async_wait<0>();
    __syncthreads();
}

template <int H, int I, int E, int K, int P, int C, bool Backward>
__device__ __forceinline__ void run_training_pipeline(const PipelineParams s) {
    static_assert(H % 256 == 0 && I % 128 == 0 && P <= 64);
    auto* status = pipeline_pointer<uint32_t>(s, Status);
    auto* grid_state = pipeline_pointer<uint32_t>(s, GridSync);
    uint32_t sequence = tix::ld_acq(grid_state + 2);
    if (tix::ld_acq(status) & 8) return;
    constexpr int64_t R = int64_t(P) * C * K;
    const int64_t rows = s.rows;
    const auto blocks = [](int64_t elements) { return (elements + 127) / 128; };

#define DG_TRAINING_GRID_SYNC() do { if (!pipeline_grid_barrier(s, sequence)) return; } while (false)
    for (int64_t start = 0; start < s.max_tokens; start += C) {
        pipeline_op<Operation::ReadChunk, X, Ids, LocalX, LocalIds, Valid, Status>(
            s, blocks(int64_t(C) * (H > K ? H : K)), s.input_tokens, C, H, K, start);
        pipeline_zero<__ppu_bfloat16>(s, LocalY, int64_t(C) * K * H);
        if constexpr (Backward) pipeline_zero<__ppu_bfloat16>(s, LocalDX, int64_t(C) * K * H);
        pipeline_zero<int32_t>(s, Counts, E);
        DG_TRAINING_GRID_SYNC();
        pipeline_rank_barrier<P>(s, start * 4);
        DG_TRAINING_GRID_SYNC();
        pipeline_peer_op<PeerOperation::CountRoutes, IdPeers, Counts, RouteExpert, RouteRow, Status>(
            s, blocks(R), R, E, s.rank, P, int64_t(C) * K);
        DG_TRAINING_GRID_SYNC();
        pipeline_op<Operation::PrefixRoutes, Counts, Offsets, Lengths>(s, 1, E, s.alignment);
        pipeline_peer_op<PeerOperation::InitializeLayout, Permutation, SourceRows>(s, blocks(rows), rows, P * C);
        DG_TRAINING_GRID_SYNC();
        pipeline_peer_op<PeerOperation::BuildPermutation, RouteExpert, RouteRow, Offsets, Permutation, SourceRows, Status>(
            s, blocks(R), R, rows, K);
        pipeline_peer_op<PeerOperation::BuildWorkInfo, Offsets, Lengths, WorkInfo, NumBlocks>(
            s, blocks(rows / 128), E, rows, 128);
        DG_TRAINING_GRID_SYNC();
        pipeline_gemm<2 * I, H, E, P, C, K, true, false>(s, PackedX, W1, GateUp, SourceRows, XPeers, YPeers, Backward);
        DG_TRAINING_GRID_SYNC();
        const int active_rows = load_pipeline_metadata(pipeline_pointer<int32_t>(s, Offsets) + E);
        pipeline_op<Operation::Activation, GateUp, Hidden>(s, blocks(int64_t(active_rows) * I), active_rows, I);
        DG_TRAINING_GRID_SYNC();
        pipeline_gemm<H, I, E, P, C, K, false, true>(s, Hidden, W2, PackedDY, IdentityRows, XPeers, YPeers);
        DG_TRAINING_GRID_SYNC();
        pipeline_rank_barrier<P>(s, start * 4 + 1);
        DG_TRAINING_GRID_SYNC();
        if constexpr (!Backward) {
            pipeline_op<Operation::Combine, LocalY, Scores, Y>(s, blocks(int64_t(C) * H), C, H, K, s.input_tokens, start, 1);
        } else {
            pipeline_op<Operation::ScoreGradient, LocalY, DY, DScores>(s, int64_t(C) * K, C, H, K, s.input_tokens, start);
            pipeline_op<Operation::WeightedGradient, DY, Scores, LocalDY>(
                s, blocks(int64_t(C) * K * H), C, H, K, s.input_tokens, start);
            DG_TRAINING_GRID_SYNC();
            pipeline_rank_barrier<P>(s, start * 4 + 2);
            DG_TRAINING_GRID_SYNC();
            pipeline_peer_op<PeerOperation::Pack, DYPeers, Permutation, PackedDY, Status>(
                s, blocks(int64_t(active_rows) * H), active_rows, H, int64_t(C) * K, 1, P);
            DG_TRAINING_GRID_SYNC();
            pipeline_wgrad<H, I, E>(s, PackedDY, Hidden, DW2);
            DG_TRAINING_GRID_SYNC();
            pipeline_gemm<I, H, E, P, C, K, true, false>(s, PackedDY, W2T, Hidden, Permutation, DYPeers, YPeers);
            DG_TRAINING_GRID_SYNC();
            pipeline_op<Operation::ActivationBackward, GateUp, Hidden, GateUp>(
                s, blocks(int64_t(active_rows) * I), active_rows, I);
            DG_TRAINING_GRID_SYNC();
            pipeline_wgrad<2 * I, H, E>(s, GateUp, PackedX, DW1);
            DG_TRAINING_GRID_SYNC();
            pipeline_gemm<H, 2 * I, E, P, C, K, false, true>(s, GateUp, W1T, PackedDY, IdentityRows, XPeers, DXPeers);
            DG_TRAINING_GRID_SYNC();
            pipeline_rank_barrier<P>(s, start * 4 + 3);
            DG_TRAINING_GRID_SYNC();
            pipeline_op<Operation::Combine, LocalDX, Scores, DX>(s, blocks(int64_t(C) * H), C, H, K, s.input_tokens, start, 0);
        }
        DG_TRAINING_GRID_SYNC();
    }
#undef DG_TRAINING_GRID_SYNC
    if (blockIdx.x == 0 && threadIdx.x == 0) grid_state[2] = sequence;
}

}  // namespace deep_gemm::training
