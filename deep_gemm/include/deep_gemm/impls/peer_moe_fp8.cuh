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
             const training::PeerDispatchPipeline& input, const training::PeerDispatchPipeline& output, char* smem_buf) {
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

#if DG_TRAINING_FP8_RETURN_PIPELINE
    __shared__ int return_layout_bad;
    if(threadIdx.x==0) {
      return_layout_bad=0;int64_t total_rows=0;
      for(uint32_t g=0;g<TileScheduler::kNumGroups;++g) {
        const int rows=params.scheduler.grouped_layout[g];
        if(rows<0 || rows%128)return_layout_bad=1;
        total_rows+=rows;
      }
      if(total_rows>params.scheduler.shape_m)return_layout_bad=1;
      if(return_layout_bad)atomicOr(output.status,2u);
    }
    __syncthreads();if(return_layout_bad)return;
#endif
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
#if !DG_TRAINING_FP8_PEER_INPUT && !DG_TRAINING_FP8_RETURN_PIPELINE
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
#if DG_TRAINING_FP8_RETURN_PIPELINE
      static_assert(size<0>(TileShape{})==128 && size<1>(TileShape{})==256);
      __threadfence();
      __syncthreads();
      if(threadIdx.x==0) {
        const uint32_t tile=(row_base/128)*(TileScheduler::SHAPE_N/256)+uint32_t(n_coord);
        if(tile>=output.signal_capacity)atomicOr(output.status,2u);
        else tix::red_add_rel(output.ready+tile,1u);
      }
      __syncthreads();
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

// Copy consumers share the producer scheduler, including expert groups and
// its two-M-tile swizzle. They retain row destinations while N tiles advance.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool EnableSbo, bool UseNStage>
class TrainingFp8PeerTileCopy {
public:
    using Producer=TrainingFp8PeerReturn<ProblemShape_,CollectiveMainloop_,CollectiveEpilogue_,TileScheduler_,EnableSbo,UseNStage>;
    using Params=typename Producer::Params;
    using TileScheduler=TileScheduler_;
    static constexpr int MaxThreadsPerBlock=1024;
    static constexpr int SharedStorageSize=0; // Dynamic TSM; row addresses are static shared.
    static constexpr int MinBlocksPerMultiprocessor=1;
    CUTLASS_DEVICE void operator()(Params const& params,const training::PeerGemmParams& peers,
        const training::PeerDispatchPipeline&,const training::PeerDispatchPipeline& output,char*) {
        static_assert(TileScheduler::BLOCK_M==128 && TileScheduler::BLOCK_N==256);
        static_assert(TileScheduler::GEMM_TYPE==GemmType::MGroupedNoPad);
        constexpr uint32_t width=TileScheduler::SHAPE_N,stride=DG_TRAINING_FP8_RETURN_STRIDE;
        static_assert(width%256==0 && stride>0);
        __shared__ int failed;
        __shared__ uint64_t destinations[128];
        if(threadIdx.x==0) {
            failed=0;int64_t total=0;
            for(uint32_t g=0;g<TileScheduler::kNumGroups;++g) {
                const int rows=params.scheduler.grouped_layout[g];
                if(rows<0 || rows%128)failed=1;
                total+=rows;
            }
            if(total>params.scheduler.shape_m)failed=1;
            if(failed)atomicOr(output.status,2u);
        }
        __syncthreads();
        if(failed)return;
        TileScheduler scheduler{params.scheduler};
        uint32_t mi,ni;int last_row=-1;
        while(scheduler.fetch_next_work(mi,ni)) {
            const uint32_t row_base=uint32_t(scheduler.curr_offset_m())+mi*128;
            if(int(row_base)!=last_row) {
                if(threadIdx.x<128) {
                    const int route=__ldg(peers.permutation+row_base+threadIdx.x);
                    const bool valid=route>=0 && uint64_t(route)<uint64_t(stride)*peers.num_ranks;
                    if(route>=0 && !valid)atomicOr(output.status,2u);
                    destinations[threadIdx.x]=valid?__ldg(peers.destination_peers+uint32_t(route)/stride)+
                        uint64_t(uint32_t(route)%stride)*width*sizeof(cutlass::bfloat16_t):0;
                }
                __syncthreads();last_row=row_base;
            }
            if(threadIdx.x==0) {
                failed=0;const uint32_t tile=(row_base/128)*(width/256)+ni;
                const uint64_t begin=clock64();
                if(tile>=output.signal_capacity){failed=1;atomicOr(output.status,2u);}
                else while(tix::ld_acq(output.ready+tile)!=1u) {
                    if(tix::ld_acq(output.status) || clock64()-begin>output.timeout_cycles) {
                        failed=1;atomicOr(output.status,8u);break;
                    }
                }
            }
            __syncthreads();if(failed)return;
            for(int base=threadIdx.x;base<128*(256/8);base+=blockDim.x*4) {
                uint4 values[4];uint64_t addresses[4];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const int item=base+j*blockDim.x,row=item/(256/8),column=(item%(256/8))*8;
                    uint64_t address=0;uint4 value{};
                    if(item<128*(256/8) && destinations[row]) {
                        address=destinations[row]+uint64_t(ni*256+column)*sizeof(cutlass::bfloat16_t);
                        value=*reinterpret_cast<const uint4*>(params.epilogue.ptr_D+
                            int64_t(row_base+row)*width+ni*256+column);
                    }
                    values[j]=value;addresses[j]=address;
                }
                #pragma unroll
                for(int j=0;j<4;++j)if(addresses[j])*reinterpret_cast<uint4*>(addresses[j])=values[j];
            }
            __syncthreads();
        }
        __threadfence();
    }
};

} // namespace deep_gemm
