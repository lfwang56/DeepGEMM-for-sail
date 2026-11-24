#pragma once

#include <deep_gemm/impls/fp8_gemm.cuh>
#include <deep_gemm/impls/peer_moe_gemm.cuh>

#if DG_TRAINING_FP8_TILE_PRODUCER
#include <deep_gemm/impls/peer_moe_fp8_dispatch_mainloop.cuh>
#endif
#if DG_TRAINING_FP8_PEER_INPUT && DG_TRAINING_FP8_INPUT_K_SLICE
#include <deep_gemm/impls/peer_moe_fp8_kready_mainloop.cuh>
#endif
namespace deep_gemm {


#if DG_TRAINING_FP8_EXPERT_END
// Native expert-major/two-M swizzle over a range. Global row/weight offsets
// remain native; only work ordinals restart at zero for this independent batch.
template<class NativeScheduler>
struct TrainingFp8ExpertBatchScheduler : NativeScheduler {
    CUTLASS_DEVICE explicit TrainingFp8ExpertBatchScheduler(typename NativeScheduler::Params const& p):NativeScheduler(p) {
        static_assert(!NativeScheduler::kIsNoPadPreprocessLayout);
        this->curr_group_idx=DG_TRAINING_FP8_EXPERT_BEGIN;
        this->curr_cumsum=this->curr_cumsum_m=0;
        for(uint32_t e=0;e<DG_TRAINING_FP8_EXPERT_BEGIN;++e)this->curr_cumsum_m+=__ldg(p.grouped_layout+e);
    }
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi,uint32_t& ni) {
        constexpr uint32_t columns=(NativeScheduler::SHAPE_N+NativeScheduler::BLOCK_N-1)/NativeScheduler::BLOCK_N;
        const uint32_t work=(this->current_iter++)*gridDim.x+blockIdx.x;
        while(this->curr_group_idx<DG_TRAINING_FP8_EXPERT_END) {
            this->curr_group_m=__ldg(this->params.grouped_layout+this->curr_group_idx);
            const uint32_t blocks=(this->curr_group_m+NativeScheduler::BLOCK_M-1)/NativeScheduler::BLOCK_M;
            if(work<(this->curr_cumsum+blocks)*columns) {
                this->get_swizzled_block_idx(blocks,work-this->curr_cumsum*columns,mi,ni);return true;
            }
            this->curr_cumsum+=blocks;this->curr_cumsum_m+=this->curr_group_m;++this->curr_group_idx;
        }
        mi=0;ni=columns;return false;
    }
};
#endif

template <class NativeScheduler>
struct TrainingFp8TileOwnerScheduler : NativeScheduler {
    uint32_t owned_m = blockIdx.x, next_n = 0;
    CUTLASS_DEVICE explicit TrainingFp8TileOwnerScheduler(typename NativeScheduler::Params const& p) : NativeScheduler(p) {}
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi, uint32_t& ni) {
      constexpr uint32_t columns = (NativeScheduler::SHAPE_N + NativeScheduler::BLOCK_N - 1) / NativeScheduler::BLOCK_N;
      if (next_n == columns) { owned_m += gridDim.x;next_n = 0; }
      while (this->curr_group_idx < NativeScheduler::kNumGroups) {
        this->curr_group_m = __ldg(this->params.grouped_layout + this->curr_group_idx);
        const uint32_t blocks = (this->curr_group_m + NativeScheduler::BLOCK_M - 1) / NativeScheduler::BLOCK_M;
        if (owned_m < this->curr_cumsum + blocks) {
          mi = owned_m - this->curr_cumsum;ni = next_n++;return true;
        }
        this->curr_cumsum += blocks;this->curr_cumsum_m += this->curr_group_m;++this->curr_group_idx;
      }
      return false;
    }
};

// Native expert-major/two-M swizzle with a shared atomic work ordinal. Two
// independently launched worker grids can join/leave without losing tiles.
template<class NativeScheduler>
struct TrainingFp8InputWorkScheduler : NativeScheduler {
    uint32_t* counter;
    CUTLASS_DEVICE explicit TrainingFp8InputWorkScheduler(typename NativeScheduler::Params const& params,
                                                         uint32_t* work_counter)
        : NativeScheduler(params),counter(work_counter) {}
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& mi,uint32_t& ni) {
        static_assert(NativeScheduler::GEMM_TYPE==GemmType::MGroupedNoPad &&
                      !NativeScheduler::kIsNoPadPreprocessLayout);
        constexpr uint32_t columns=(NativeScheduler::SHAPE_N+NativeScheduler::BLOCK_N-1)/NativeScheduler::BLOCK_N;
        __shared__ uint32_t ordinal;
        __syncthreads();
        if(threadIdx.x==0)ordinal=atomicAdd(counter,1u);
        __syncthreads();
        const uint32_t work=ordinal;
        uint32_t mblocks=0;
        while(this->curr_group_idx<NativeScheduler::kNumGroups) {
            this->curr_group_m=uint32_t(__ldg(this->params.grouped_layout+this->curr_group_idx));
            mblocks=(this->curr_group_m+NativeScheduler::BLOCK_M-1)/NativeScheduler::BLOCK_M;
            const uint32_t end=this->curr_cumsum+mblocks;
            if(work<end*columns) {
                this->get_swizzled_block_idx(mblocks,work-this->curr_cumsum*columns,mi,ni);
                return true;
            }
            ++this->curr_group_idx;
            this->curr_cumsum=end;
            this->curr_cumsum_m+=this->curr_group_m;
        }
        mi=0;ni=columns;
        return false;
    }
};

// Native blockwise FP8 mainloop and scheduler, with direct BF16 peer stores.
// Input is already locally packed/quantized; this primitive fuses the return.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool kEnableSboOverlap, bool kUseNStageKernel>
class TrainingFp8PeerReturn {
public:
    static_assert(!kEnableSboOverlap && !kUseNStageKernel);
#if DG_TRAINING_FP8_TILE_PRODUCER
    using CollectiveMainloop = TrainingFp8DispatchMainloop<CollectiveMainloop_>;
#elif DG_TRAINING_FP8_PEER_INPUT && DG_TRAINING_FP8_INPUT_K_SLICE
    using CollectiveMainloop = TrainingFp8KReadyMainloop<CollectiveMainloop_, TileScheduler_::SHAPE_K>;
#else
    using CollectiveMainloop = CollectiveMainloop_;
#endif
    using CollectiveEpilogue = CollectiveEpilogue_;
    using Base = DeepGemmUniversal<ProblemShape_, CollectiveMainloop, CollectiveEpilogue_, TileScheduler_, false, false>;
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
             const training::PeerDispatchPipeline& input, const training::PeerDispatchPipeline& output,
             uint32_t* input_work_counter,const training::PeerFp8ProducerGradient& producer_gradient,char* smem_buf) {
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

#if DG_TRAINING_FP8_RETURN_PIPELINE || DG_TRAINING_FP8_INPUT_DYNAMIC || DG_TRAINING_FP8_TILE_PRODUCER || DG_TRAINING_FP8_INPUT_K_SLICE
    __shared__ int return_layout_bad;
    if(threadIdx.x==0) {
      return_layout_bad=0;int64_t total_rows=0;
      for(uint32_t g=0;g<TileScheduler::kNumGroups;++g) {
        const int rows=params.scheduler.grouped_layout[g];
        if(rows<0 || rows%128)return_layout_bad=1;
        total_rows+=rows;
      }
      if(total_rows>params.scheduler.shape_m)return_layout_bad=1;
      if(return_layout_bad)atomicOr((DG_TRAINING_FP8_INPUT_DYNAMIC || DG_TRAINING_FP8_TILE_PRODUCER || DG_TRAINING_FP8_INPUT_K_SLICE) ? input.status : output.status,2u);
    }
    __syncthreads();if(return_layout_bad)return;
#endif
#if DG_TRAINING_FP8_EXPERT_END
    TrainingFp8ExpertBatchScheduler<TileScheduler> deep_scheduler{params.scheduler};
#elif DG_TRAINING_FP8_TILE_PRODUCER
    TrainingFp8TileOwnerScheduler<TileScheduler> deep_scheduler{params.scheduler};
#elif DG_TRAINING_FP8_INPUT_DYNAMIC
    TrainingFp8InputWorkScheduler<TileScheduler> deep_scheduler{params.scheduler,input_work_counter};
#else
    TileScheduler deep_scheduler{params.scheduler};
#endif

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
#if DG_TRAINING_FP8_PEER_INPUT && !DG_TRAINING_FP8_INPUT_K_SLICE
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
          while (tix::ld_acq(input.ready + tile) != uint32_t(DG_TRAINING_FP8_INPUT_READY_COUNT)) {
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
#if DG_TRAINING_FP8_PEER_INPUT && DG_TRAINING_FP8_INPUT_K_SLICE
      collective_mainloop.configure(input, uint32_t(offset_a/K)+uint32_t(m_coord)*size<0>(TileShape{}),
          cute::min(int(size<0>(TileShape{})),int(M)-int(m_coord)*int(size<0>(TileShape{}))),K);
#endif
#if DG_TRAINING_FP8_TILE_PRODUCER
      __shared__ uint64_t producer_addresses[128];
      __shared__ float producer_scores[128];
      collective_mainloop.configure(peers, producer_gradient, producer_scores, input.status, producer_addresses,
          uint32_t(offset_a / K) + uint32_t(m_coord) * 128, uint32_t(m_coord) * 128,
          params.scheduler.shape_m, K, n_coord == 0);
#endif
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
#if DG_TRAINING_FP8_TILE_PRODUCER
      if (n_coord == 0) { __threadfence();__syncthreads(); }
#endif
      auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
      const auto coordinates = thread_mma.partition_C(make_identity_tensor(take<0, 2>(blk_shape)));
      const uint32_t row_base = uint32_t(offset_a / K) + uint32_t(m_coord) * size<0>(TileShape{});
      const uint32_t valid_rows = uint32_t(cute::min(int(size<0>(TileShape{})), int(M) - int(m_coord) * int(size<0>(TileShape{}))));
#if DG_TRAINING_FP8_ACTIVATION_BACKWARD
      // Reuse the BF16 dHidden boundary and established split-layout SwiGLU
      // derivative. Quantization follows on the same stream, so its output
      // can safely alias a consumed dY quantization buffer.
      training::activation_backward_epilogue<TileScheduler::SHAPE_N,false>(
          accumulators,coordinates,peers,row_base,valid_rows,uint32_t(n_coord)*size<1>(TileShape{}));
#endif
#if !DG_TRAINING_FP8_EXPERT_END && !DG_TRAINING_FP8_PEER_INPUT && !DG_TRAINING_FP8_RETURN_PIPELINE && !DG_TRAINING_FP8_ACTIVATION_BACKWARD && !DG_TRAINING_FP8_TILE_PRODUCER
#if DG_TRAINING_FP8_RETURN_STORE_BYTES == 16
      // The Tensor Cell mainloop is finished. Reuse its TSM for one rounded
      // BF16 tile, then issue contiguous local/peer 128-bit stores. No extra
      // global output staging buffer or copy kernel is needed.
      constexpr uint32_t BM=size<0>(TileShape{}),BN=size<1>(TileShape{}),StageN=BN+16;
      constexpr uint32_t Width=TileScheduler::SHAPE_N,Stride=DG_TRAINING_FP8_RETURN_STRIDE;
      static_assert(BN%8==0 && Width%8==0 && Stride>0);
      auto* stage=reinterpret_cast<cutlass::bfloat16_t*>(smem_buf);
      __shared__ uint64_t return_destinations[BM];
      __syncthreads();
      for(uint32_t row=threadIdx.x;row<valid_rows;row+=blockDim.x) {
        const int route=__ldg(peers.permutation+row_base+row);
        uint64_t address=0;
        if(route>=0 && uint64_t(route)<uint64_t(Stride)*peers.num_ranks) {
          address=__ldg(peers.destination_peers+uint32_t(route)/Stride)+
              (uint64_t(uint32_t(route)%Stride)*Width+uint32_t(n_coord)*BN)*sizeof(cutlass::bfloat16_t);
        }
        return_destinations[row]=address;
      }
      CUTLASS_PRAGMA_UNROLL
      for(int wm=0;wm<cute::size<1>(coordinates);++wm) {
        CUTLASS_PRAGMA_UNROLL
        for(int mm=0;mm<cute::size<0,1>(coordinates);++mm) {
          const uint32_t row=cute::get<0>(coordinates(cute::make_coord(cute::_0{},mm,cute::_0{}),wm,cute::_0{}));
          if(row>=valid_rows)continue;
          CUTLASS_PRAGMA_UNROLL
          for(int wn=0;wn<cute::size<2>(coordinates);++wn) {
            CUTLASS_PRAGMA_UNROLL
            for(int mn=0;mn<cute::size<0,2>(coordinates);++mn) {
              const uint32_t column=cute::get<1>(coordinates(cute::make_coord(cute::_0{},mm,mn),wm,wn));
              const auto* values=cute::raw_pointer_cast(accumulators(cute::make_coord(cute::_,mm,mn),wm,wn).data());
              *reinterpret_cast<uint32_t*>(stage+uint64_t(row)*StageN+column)=tix::cvt_rte_bf16x2_f32(values[1],values[0]);
            }
          }
        }
      }
      __syncthreads();
      for(uint32_t index=threadIdx.x;index<valid_rows*(BN/8);index+=blockDim.x) {
        const uint32_t row=index/(BN/8),column=(index%(BN/8))*8;
        if(uint32_t(n_coord)*BN+column>=Width)continue;
        const uint4 value=*reinterpret_cast<const uint4*>(stage+uint64_t(row)*StageN+column);
        if(return_destinations[row])
          *reinterpret_cast<uint4*>(return_destinations[row]+column*sizeof(cutlass::bfloat16_t))=value;
#if DG_TRAINING_FP8_PEER_KEEP_LOCAL
        *reinterpret_cast<uint4*>(params.epilogue.ptr_D+uint64_t(row_base+row)*Width+uint32_t(n_coord)*BN+column)=value;
#endif
      }
      __syncthreads();
#else
      training::scatter_peer_output<cutlass::bfloat16_t, TileScheduler::SHAPE_N, false>(
          accumulators, coordinates, peers, row_base, valid_rows,
          uint32_t(n_coord) * size<1>(TileShape{}));
#endif
#endif
#if DG_TRAINING_FP8_PEER_KEEP_LOCAL && DG_TRAINING_FP8_RETURN_STORE_BYTES != 16 && !DG_TRAINING_FP8_ACTIVATION_BACKWARD
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
        const training::PeerDispatchPipeline&,const training::PeerDispatchPipeline& output,uint32_t*,const training::PeerFp8ProducerGradient&,char*) {
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
