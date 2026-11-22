#pragma once

#include <deep_gemm/impls/fp8_gemm.cuh>
#include <deep_gemm/impls/peer_moe_gemm.cuh>

namespace deep_gemm {

// Native blockwise FP8 mainloop and scheduler, with direct BF16 peer stores.
// Input is already locally packed/quantized; this primitive fuses the return.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool kEnableSboOverlap, bool kUseNStageKernel>
class TrainingFp8PeerReturn {
public:
    static_assert(!kEnableSboOverlap && !kUseNStageKernel);
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    using Base = DeepGemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, false, false>;
    using ProblemShape = typename Base::ProblemShape;
    using TileShape = typename Base::TileShape;
    using TiledMma = typename Base::TiledMma;
    using TileScheduler = typename Base::TileScheduler;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementScale = typename Base::ElementScale;
    using LayoutSFA = typename Base::LayoutSFA;
    using MainloopParams = typename Base::MainloopParams;
    using SharedStorage = typename Base::SharedStorage;
    using Params = typename Base::Params;
    using StrideA = typename Base::StrideA;
    using StrideB = typename Base::StrideB;
    using StrideC = typename Base::StrideC;
    using StrideD = typename Base::StrideD;
    static_assert(TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad);
    static constexpr int N_PREFETCH_CACHELINE = Base::N_PREFETCH_CACHELINE;
    static constexpr bool kUseReplicaWeights = false;
    static constexpr int SharedStorageSize = Base::SharedStorageSize;
    static constexpr int MaxThreadsPerBlock = Base::MaxThreadsPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = Base::MinBlocksPerMultiprocessor;
    CUTLASS_DEVICE void
  operator()(Params const& params, const training::PeerGemmParams& peers,
             const training::PeerDispatchPipeline& input, char* smem_buf) {
    using namespace cute;
    using X = Underscore;
    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
      // group is small 8|16, just prefetch one cacheline
      __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout));
    } else if (TileScheduler::GEMM_TYPE == GemmType::MGroupedNoPad) {
      // each warp prefetch one cacheline
      if (warp_idx < N_PREFETCH_CACHELINE) {
        __ppu_prefetch_KSD((void*)(params.scheduler.grouped_layout + (warp_idx<<5)));
      }
    }

    TileScheduler deep_scheduler{params.scheduler};

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);


    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)
    auto strides = make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, M));
    uint32_t m_block_idx, n_block_idx;
    #pragma clang loop licm(disable)
    while (deep_scheduler.fetch_next_work(m_block_idx, n_block_idx)) {
      auto m_coord = m_block_idx;
      auto n_coord = n_block_idx;
      auto l_coord = 0;
      M = deep_scheduler.curr_problem_m();
      auto offset_scalea = deep_scheduler.curr_offset_scalea();
      auto scale_offset_b = deep_scheduler.curr_offset_b(m_block_idx);
      const auto offset_a = deep_scheduler.curr_offset_a(get<2>(params.mainloop.dA));
#if DG_TRAINING_FP8_PEER_INPUT
      const uint32_t input_row_base = uint32_t(offset_a / K) + uint32_t(m_coord) * size<0>(TileShape{});
      const uint32_t input_valid_rows = uint32_t(cute::min(int(size<0>(TileShape{})), int(M) - int(m_coord) * int(size<0>(TileShape{}))));
      __shared__ int input_failed;
      if (threadIdx.x == 0) {
        input_failed = 0;
        const uint64_t begin = clock64();
        // Native M192 can cover two route-aligned M128 publication tiles.
        for (uint32_t tile = input_row_base / 128;
             tile < (input_row_base + input_valid_rows + 127) / 128; ++tile) {
          if (tile >= input.signal_capacity) { atomicOr(input.status, 2u); input_failed = 1; break; }
          while (tix::ld_acq(input.ready + tile) != 1u) {
            if (tix::ld_acq(input.status) || clock64() - begin > input.timeout_cycles) {
              atomicOr(input.status, 8u); input_failed = 1; break;
            }
          }
          if (input_failed) break;
        }
      }
      __syncthreads();
      if (input_failed) return;
#endif
      auto offset_b = deep_scheduler.curr_offset_b(m_block_idx, get<2>(params.mainloop.dB));
      const ElementA* ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + offset_a;
      const ElementB* weight_base = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B);
      const ElementScale* scale_base = reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_B);
      if constexpr (kUseReplicaWeights) {
        const uint32_t expert = deep_scheduler.curr_group_idx;
        if (expert >= params.num_master_groups) {
          offset_b = int64_t(expert - params.num_master_groups) * N * K;
          scale_offset_b = offset_b;
          weight_base = params.rhs_replicas;
          scale_base = params.rhs_replica_scales;
        }
      }
      const ElementB* ptr_B = weight_base + offset_b;
      const ElementScale* ptr_scale_A = reinterpret_cast<const ElementScale*>(params.mainloop.ptr_scale_A) + offset_scalea;
      // Each expert owns ceil(N/128) * ceil(K/128) scale blocks.
      const auto expert_idx = scale_offset_b / (int64_t(N) * int64_t(K));
      const auto scale_stride_b = int64_t(cute::ceil_div(N, 128)) * cute::ceil_div(K, 128);
      const ElementScale* ptr_scale_B = scale_base + expert_idx * scale_stride_b;
      auto mk_layout = make_layout(
        make_shape(make_shape(Int<1>{}, M),
                  make_shape(Int<128>{}, cute::ceil_div(K, 128))),
        strides
      );
      LayoutSFA layout_SFA = make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));

      auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
      CollectiveMainloop collective_mainloop;
      // update actual global ptr offset
      MainloopParams update_params = {
        ptr_A, params.mainloop.dA, ptr_B, params.mainloop.dB, 4,
        ptr_scale_A, layout_SFA,
        ptr_scale_B, params.mainloop.layout_SFB
      };
      problem_shape_MNKL = ProblemShape{M, N, K, L};
      auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, blk_coord_mnkl, update_params);
      static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

      Tensor gA = get<0>(load_inputs);
      Tensor gB = get<1>(load_inputs);
      // Compute tile residues for predication
      auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
      auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
      auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);
      // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
      TiledMma tiled_mma;
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
      clear(accumulators);

      auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
      int  k_tile_count = size<2>(gA);

      // Perform the collective scoped MMA
      collective_mainloop(
        update_params,
        load_inputs,
        accumulators,
        k_tile_iter, k_tile_count,
        thread_idx,
        smem_buf
      );
      auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
      const auto coordinates = thread_mma.partition_C(make_identity_tensor(take<0, 2>(blk_shape)));
      const uint32_t row_base = uint32_t(offset_a / K) + uint32_t(m_coord) * size<0>(TileShape{});
      const uint32_t valid_rows = uint32_t(cute::min(int(size<0>(TileShape{})), int(M) - int(m_coord) * int(size<0>(TileShape{}))));
#if !DG_TRAINING_FP8_PEER_INPUT
      training::scatter_peer_output<cutlass::bfloat16_t, TileScheduler::SHAPE_N, false>(
          accumulators, coordinates, peers, row_base, valid_rows,
          uint32_t(n_coord) * size<1>(TileShape{}));
#endif
#if DG_TRAINING_FP8_PEER_KEEP_LOCAL
      // update params.epilogue for ptrC and ptrD
      auto params_epilogue_local = params.epilogue;
      params_epilogue_local.ptr_D += deep_scheduler.curr_offset_c(cute::get<2>(params_epilogue_local.dD));
      // Epilogue and write to gD
      CollectiveEpilogue epilogue{params_epilogue_local, shared_storage.tensors.epilogue};
      epilogue(
        problem_shape_MNKL,
        blk_shape,
        blk_coord_mnkl,
        accumulators,
        tiled_mma,
        residue_mnk,
        thread_idx,
        (char*)&shared_storage.tensors.epilogue
      );
#endif
      if constexpr(kEnableSboOverlap && TileScheduler::GEMM_TYPE == GemmType::MGroupedMasked) {
        cp_async_wait<0>();
        __syncthreads();

        if (threadIdx.x == 0) {
          tix::atomic_add_release_global(params.signal + deep_scheduler.curr_group_idx
                  * math::ceil_div(deep_scheduler.params.shape_m, TileScheduler::BLOCK_M) + m_block_idx, 1);
        }
      }
    } // Scheduler work fetch loop
  }
};

} // namespace deep_gemm
