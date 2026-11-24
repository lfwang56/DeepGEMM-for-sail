/***************************************************************************************************
 * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved.
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#include <deep_gemm/impls/peer_moe_training.cuh>

namespace deep_gemm {

// Native blockwise mainloop, with peer BF16 gather/quant replacing A's AIU
// load in the first N tile. The other warps continue native MMA while these
// producers fill the next K128 stage. All later N tiles read the local cache.
template <class Native>
struct TrainingFp8DispatchMainloop : Native {
    using DispatchPolicy = typename Native::DispatchPolicy;
    using TileShape = typename Native::TileShape;
    using TiledMma = typename Native::TiledMma;
    using ElementAccumulator = typename Native::ElementAccumulator;
    using SmemCopyAtomA = typename Native::SmemCopyAtomA;
    using SmemCopyAtomB = typename Native::SmemCopyAtomB;
    using TransformA = typename Native::TransformA;
    using TransformB = typename Native::TransformB;
    using ElementScale = typename Native::ElementScale;
    using SmemLayoutA = typename Native::SmemLayoutA;
    using SmemLayoutB = typename Native::SmemLayoutB;
    using SmemLayoutSFA = typename Native::SmemLayoutSFA;
    using SmemLayoutSFB = typename Native::SmemLayoutSFB;
    using RealSmemLayoutSFA = typename Native::RealSmemLayoutSFA;
    using RealSmemLayoutSFB = typename Native::RealSmemLayoutSFB;
    using Params = typename Native::Params;
    static constexpr auto NumThreadsPerCTA = Native::NumThreadsPerCTA;
    static constexpr auto WarpInterleaving = Native::WarpInterleaving;
    static constexpr auto ScaleGranularityK = Native::ScaleGranularityK;
    static constexpr auto ScalePromotionInterval = Native::ScalePromotionInterval;
    static constexpr auto ScaleMsPerTile = Native::ScaleMsPerTile;
    static constexpr auto ScaleNsPerTile = Native::ScaleNsPerTile;
    static constexpr auto ScaleKsPerTile = Native::ScaleKsPerTile;
    using Native::gmem_tiled_copy_A;
    using Native::gmem_tiled_copy_B;
    using Native::gmem_tiled_copy_SFA;
    using Native::gmem_tiled_copy_SFB;
    using ElementA = typename Native::ElementA;
    using SharedStorage = typename Native::SharedStorage;
    uint64_t* addresses_ = nullptr;
    __ppu_bfloat16* packed_ = nullptr;
    int global_row_ = 0, local_row_ = 0, capacity_ = 0, width_ = 0;
    bool produce_ = false;
    training::PeerFp8ProducerGradient gradient_{};
    float* scores_ = nullptr;

    CUTLASS_DEVICE void configure(const training::PeerGemmParams& peers, const training::PeerFp8ProducerGradient& gradient, float* scores, uint32_t* status,
        uint64_t* addresses, int global_row, int local_row, int capacity, int width, bool produce) {
      gradient_ = gradient;scores_ = scores;
      addresses_ = addresses;packed_ = reinterpret_cast<__ppu_bfloat16*>(peers.dispatch_scratch);
      global_row_ = global_row;local_row_ = local_row;capacity_ = capacity;width_ = width;produce_ = produce;
      if (produce) {
        if (threadIdx.x < 128) {
          const int route = peers.permutation[global_row + threadIdx.x];
          const uint64_t token = route >= 0 ? uint64_t(route) / DG_TRAINING_FP8_PRODUCER_TOPK : 0;
          const uint32_t owner = token / peers.source_rows_per_rank;
          const bool valid = route >= 0 && owner < peers.num_ranks;
          if (route >= 0 && !valid) atomicOr(status, 2u);
#if DG_TRAINING_FP8_GRADIENT_PRODUCER
          scores[threadIdx.x] = valid ? reinterpret_cast<const float*>(gradient.score_peers[owner])[
              uint64_t(route) % (uint64_t(peers.source_rows_per_rank) * DG_TRAINING_FP8_PRODUCER_TOPK)] : 0.f;
#endif
          addresses[threadIdx.x] = valid ? peers.source_peers[owner] +
              (token % peers.source_rows_per_rank) * uint64_t(width) * sizeof(__ppu_bfloat16) : 0;
        }
        __syncthreads();
      }
    }
  template <
    class... Ts,
    class FrgTensorC,
    class KTileIterator
  >
  CUTLASS_DEVICE void
  operator() (
      Params const& mainloop_params,
      cute::tuple<Ts...> const& load_inputs,
      FrgTensorC &accum,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      char *smem_buf)
  {
    using namespace cute;
    static_assert((NumThreadsPerCTA == 256 || NumThreadsPerCTA == 512) && size<0>(TileShape{}) == 128 && size<2>(TileShape{}) == 128);
    if (!produce_) {
      Native::operator()(mainloop_params, load_inputs, accum, k_tile_iter, k_tile_count, thread_idx, smem_buf);
      return;
    }

    // static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");
    static_assert(rank(SmemLayoutB{}) == 3,
      "MainloopPPUCpAsync must have a pipeline mode in the smem layout.");

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int aiu_warp_group_thread_idx = warp_idx * 32;
    int warp_group_id = warp_idx / 8;

    Tensor gA = get<0>(load_inputs);
    Tensor gB = get<1>(load_inputs);

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<0>(gA) == size<0>(sA));                          // BLK_M
    CUTE_STATIC_ASSERT_V(size<1>(gA) == size<1>(sA));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<0>(gB) == size<0>(sB));                          // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(gB) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));                          // BLK_K
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));        // PIPE

    // Partition the copying of A and B tiles across the threads
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx & 31);

    Tensor tAgA = gmem_thr_copy_A.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_A.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)
    Tensor tBgB = gmem_thr_copy_B.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBsB = gmem_thr_copy_B.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    Tensor gSFA = get<2>(load_inputs);
    Tensor gSFB = get<3>(load_inputs);

    auto sSFA = make_tensor(cute::make_smem_ptr(storage.smem_SFA.data()), RealSmemLayoutSFA{});
    auto sSFB = make_tensor(cute::make_smem_ptr(storage.smem_SFB.data()), RealSmemLayoutSFB{});

    auto gmem_thr_copy_SFA = gmem_tiled_copy_SFA.get_slice((thread_idx & 31) + 64);
    Tensor tSFAgSFA = gmem_thr_copy_SFA.partition_S(gSFA);
    Tensor tSFAsSFA = gmem_thr_copy_SFA.partition_D(sSFA);

    auto gmem_thr_copy_SFB = gmem_tiled_copy_SFB.get_slice((thread_idx & 31) + 32);
    Tensor tSFBgSFB = gmem_thr_copy_SFB.partition_S(gSFB);
    Tensor tSFBsSFB = gmem_thr_copy_SFB.partition_D(sSFB);

    auto copy_to_tsm = [&](auto producer_tag, int k_pipe_write, int k_idx) {
      constexpr int ProducerWarpIdx = decltype(producer_tag)::value;
      if constexpr (ProducerWarpIdx >= 0 && ProducerWarpIdx < 4) {
        // Four producer warps own disjoint rows. Quantize each K128 block
        // directly into the native swizzled A/SFA shared-memory stage.
        const int lane = thread_idx & 15;
        for (int row = ProducerWarpIdx * 2 + ((thread_idx & 31) >> 4); row < 128; row += 8) {
          const auto* source = reinterpret_cast<const __ppu_bfloat16*>(addresses_[row]);
          const int column = k_idx * 128 + lane * 8;
          uint4 raw = source ? *reinterpret_cast<const uint4*>(source + column) : uint4{};
#if DG_TRAINING_FP8_GRADIENT_PRODUCER
          const uint4 raw_y = source ? *reinterpret_cast<const uint4*>(
              reinterpret_cast<const __ppu_bfloat16*>(gradient_.unweighted_y) + int64_t(global_row_ + row) * width_ + column) : uint4{};
          const auto* dy = reinterpret_cast<const __ppu_bfloat16*>(&raw);
          const auto* y = reinterpret_cast<const __ppu_bfloat16*>(&raw_y);
          uint4 weighted_raw{};auto* weighted = reinterpret_cast<__ppu_bfloat16*>(&weighted_raw);
          float dot = 0.f;
          #pragma unroll
          for (int j = 0; j < 8; ++j) {
            const float value = __bfloat162float(dy[j]);
            weighted[j] = __float2bfloat16(value * scores_[row]);
            dot += value * __bfloat162float(y[j]);
          }
          const unsigned mask = 0xffffu << ((thread_idx & 31) >= 16 ? 16 : 0);
          #pragma unroll
          for (int delta = 8; delta > 0; delta >>= 1) dot += __shfl_down_sync(mask, dot, delta, 16);
          if (lane == 0) gradient_.score_partials[int64_t(global_row_ + row) * (width_ / 128) + k_idx] = dot;
          raw = weighted_raw;
#endif
          float scale;
          const uint2 quantized = training::peer_quantize_eight(raw, scale);
          if (lane == 0) {
            const_cast<ElementScale*>(mainloop_params.ptr_scale_A)[int64_t(k_idx) * capacity_ + local_row_ + row] = scale;
          }
          const int64_t at = int64_t(local_row_ + row) * width_ + column;
          *reinterpret_cast<uint2*>(const_cast<ElementA*>(mainloop_params.ptr_A) + at) = quantized;
          *reinterpret_cast<uint4*>(packed_ + int64_t(global_row_ + row) * width_ + column) = raw;
        }
        // Native AIU and scale-copy atoms preserve their physical TSM layout.
        // This is still K128 pipelined: no complete-input dispatch barrier.
        __threadfence();
        __ppu_barrier_sync(7, 128, 0);
        if constexpr (ProducerWarpIdx == 0) {
          copy(gmem_tiled_copy_A, tAgA(_,_,_,k_idx), tAsA(_,_,_,k_pipe_write));
          copy(gmem_tiled_copy_SFA, tSFAgSFA(_,_,_,k_idx), tSFAsSFA(_,_,_,k_pipe_write));
        }
      } else if constexpr (ProducerWarpIdx == 4) {
        copy(gmem_tiled_copy_B, tBgB(_,_,_,k_idx), tBsB(_,_,_,k_pipe_write));
        copy(gmem_tiled_copy_SFB, tSFBgSFB(_,_,_,k_idx), tSFBsSFB(_,_,_,k_pipe_write));
      }
    };

    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < DispatchPolicy::Stages-1; ++k_pipe) {
      if (k_tile_count > 0) {
        // All three producers belong to physical group 0; preserve prologue dispatch.
        switch (warp_idx) {
          case 0: copy_to_tsm(Int<0>{}, k_pipe, *k_tile_iter); break;
          case 1: copy_to_tsm(Int<1>{}, k_pipe, *k_tile_iter); break;
          case 2: copy_to_tsm(Int<2>{}, k_pipe, *k_tile_iter); break;
          case 3: copy_to_tsm(Int<3>{}, k_pipe, *k_tile_iter); break;
          case 4: copy_to_tsm(Int<4>{}, k_pipe, *k_tile_iter); break;
        }
        ++k_tile_iter;
      }
      cp_async_fence();
      --k_tile_count;
    }

    //
    // MMA Atom partitioning
    //

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,_0{}));                     // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,_0{}));                     // (MMA,MMA_N,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    using EngineAccum =  typename FrgTensorC::engine_type;
    using LayoutAccum =  typename FrgTensorC::layout_type;
    cutlass::gemm::collective::MixAccumulation<EngineAccum, LayoutAccum, ElementAccumulator> accumulation(accum, ScalePromotionInterval, size<2>(tCrA));

    //
    // Copy Atom retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(make_mix_tensor_like(sA));                  // (CPY,CPY_M,CPY_K,PIPE)

    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                   // (CPY,CPY_M,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));            // CPY_K

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(aiu_warp_group_thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(make_mix_tensor_like(sB));                  // (CPY,CPY_N,CPY_K,PIPE)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                   // (CPY,CPY_N,CPY_K)
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    // Block scaling
    auto layout_sSFA_copy = make_layout(
      make_shape(shape<0>(SmemLayoutSFA{}), get<1>(TileShape{}),
                 make_shape(shape<1>(SmemLayoutSFA{}), shape<2>(SmemLayoutSFA{}))),
      make_stride(stride<0>(SmemLayoutSFA{}), _0{},
                  make_stride(stride<1>(SmemLayoutSFA{}), stride<2>(SmemLayoutSFA{}))));
    auto layout_sSFB_copy = make_layout(
      make_shape(get<0>(TileShape{}), shape<0>(SmemLayoutSFB{}),
                 make_shape(shape<1>(SmemLayoutSFB{}), shape<2>(SmemLayoutSFB{}))),
      make_stride(_0{}, stride<0>(SmemLayoutSFB{}),
                  make_stride(stride<1>(SmemLayoutSFB{}), stride<2>(SmemLayoutSFB{}))));
    Tensor sSFA_copy = make_tensor(cute::make_smem_ptr(storage.smem_SFA.data()), layout_sSFA_copy);
    Tensor sSFB_copy = make_tensor(cute::make_smem_ptr(storage.smem_SFB.data()), layout_sSFB_copy);

    Tensor tCsSFA = tiled_mma.get_slice(thread_idx).partition_C(sSFA_copy);
    Tensor tCsSFB = tiled_mma.get_slice(thread_idx).partition_C(sSFB_copy);
    Tensor tCrSFA = make_fragment_like<ElementScale>(tCsSFA(_, _, _, _0{}));
    Tensor tCrSFB = make_fragment_like<ElementScale>(tCsSFB(_, _, _, _0{}));

    //
    // PIPELINED MAIN LOOP
    //

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = DispatchPolicy::Stages-1;

    int smem_scale_pipe_read = smem_pipe_read;

    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA_copy_view);
    auto K_ATOM_PER_COPY = size<2>(tCrA) / size<2>(tCrA_copy_view);
    auto K_BLOCK_GROUP = K_BLOCK_MAX / Int<ScaleKsPerTile>{};

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
      // Wait until our first prefetched tile is loaded in
      cp_async_wait<DispatchPolicy::Stages-2>();
      __syncthreads();

      // Prefetch the first rmem from the first k-tile
      copy(smem_tiled_copy_A, tCsA_p(_,_,_0{}), tCrA_copy_view(_,_,_0{}));
      copy(smem_tiled_copy_B, tCsB_p(_,_,_0{}), tCrB_copy_view(_,_,_0{}));
      copy(tCsSFA(_,_,_,make_coord(_0{}, _0{})), tCrSFA);
      copy(tCsSFB(_,_,_,make_coord(_0{}, _0{})), tCrSFB);
    }
    if constexpr (WarpInterleaving) {
      if (warp_group_id == 1) {
        __ppu_barrier_arrive(5, NumThreadsPerCTA, 0);
      }
    }

    // Specialize barrier IDs by group while preserving the shared loop body.
    auto run_group = [&](auto group_tag, auto producer_tag) {
      constexpr int WarpGroupId = decltype(group_tag)::value;
      auto preprocess_scale = [&]() {
        if constexpr (WarpInterleaving) {
          __ppu_barrier_sync(5 + WarpGroupId, NumThreadsPerCTA, 0); // group 0 wait bar 5, group 1 wait bar 6
        }
        clear(accumulation());
        if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
          tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
        }
        if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
          ElementScale scale_b = tCrSFB(_0{});
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
            filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
          }
        }
        if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
          ElementScale scale_a = tCrSFA(_0{});
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
            filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
          }
        }
      };

      auto process_scale_and_copy = [&](int k_pipe_read, int k_idx = _0{}) {
        if constexpr (WarpInterleaving) {
          __ppu_barrier_arrive(6 - WarpGroupId, NumThreadsPerCTA, 0);
        }
        // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
        if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
          ElementScale scale_ab = tCrSFA(_0{});
          accumulation.scale(scale_ab);
        }
        if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
          accumulation.scale(tCrSFA);
        }
        if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
          accumulation.scale(tCrSFB);
        }
        if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
          accumulation.scale(tCrSFA, tCrSFB);
        }

        copy(tCsSFB(_,_,_,make_coord(k_idx, k_pipe_read)), tCrSFB);
        copy(tCsSFA(_,_,_,make_coord(k_idx, k_pipe_read)), tCrSFA);
      };

      CUTLASS_PRAGMA_NO_UNROLL
      while (k_tile_count > -(DispatchPolicy::Stages-1)) {
        // Pipeline the outer products with a static for loop.
        //
        // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
          if (k_block == K_BLOCK_MAX - 1) {
            // Slice the smem_pipe_read smem
            tCsA_p = tCsA(_,_,_,smem_pipe_read);
            tCsB_p = tCsB(_,_,_,smem_pipe_read);

            // Commit the smem for smem_pipe_read
            cp_async_wait<DispatchPolicy::Stages-2>();
            if constexpr (WarpInterleaving) {
              __ppu_barrier_sync(3 + WarpGroupId, NumThreadsPerCTA / 2, 0);   // warp group sync
            } else {
              __syncthreads();
            }
          }
          if (k_block % K_BLOCK_GROUP == 0) {
            preprocess_scale();
          }
          // Load A, B shmem->regs for k_block+1
          auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
          copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
          copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

          CUTLASS_PRAGMA_UNROLL
          for (int k_loop = 0; k_loop < K_ATOM_PER_COPY; k_loop++) {
            auto atom_idx = k_block * K_ATOM_PER_COPY + k_loop;
            // Transform before compute
            cute::transform(tCrA(_,_,atom_idx), TransformA{});
            cute::transform(tCrB(_,_,atom_idx), TransformB{});
            // gemm for one tiled_mma atom on K
            cute::gemm(tiled_mma, tCrA(_,_,atom_idx), tCrB(_,_,atom_idx), accumulation());
          }
          // Issue the next tile after the first MMA block.
          if (k_block == 0) {
            copy_to_tsm(producer_tag, smem_pipe_write, *k_tile_iter);
            cp_async_fence();

            // Advance the tile
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_iter; }

            // Advance the pipe before the last-block wait.
            smem_pipe_write = smem_pipe_read;
            ++smem_pipe_read;
            smem_pipe_read = (smem_pipe_read == DispatchPolicy::Stages) ? 0 : smem_pipe_read;
          }
          if constexpr (ScaleKsPerTile > 1) {
            if (k_block_next % K_BLOCK_GROUP == 0) {    // ctaK = 256
              process_scale_and_copy(smem_scale_pipe_read, k_block_next / K_BLOCK_GROUP * ScaleGranularityK);
              smem_scale_pipe_read = smem_pipe_read;
            }
          } else {                                      // ctaK = 128
            if (k_block_next == 0) {
              process_scale_and_copy(smem_pipe_read);
            }
          }
        }); // for_each
      }
    };

    // Non-producer warps keep their original physical group; -1 disables gmem copies.
    auto run_non_producer = [&]() {
      if constexpr (WarpInterleaving) {
        if (warp_group_id == 0) {
          run_group(Int<0>{}, Int<-1>{});
        } else {
          run_group(Int<1>{}, Int<-1>{});
        }
      } else {
        run_group(Int<0>{}, Int<-1>{});
      }
    };

    // Select a producer role once, before entering the steady-state loop.
    switch (warp_idx) {
      case 0: run_group(Int<0>{}, Int<0>{}); break;  // A/B
      case 1: run_group(Int<0>{}, Int<1>{}); break;  // SFB
      case 2: run_group(Int<0>{}, Int<2>{}); break;
      case 3: run_group(Int<0>{}, Int<3>{}); break;
      case 4: run_group(Int<0>{}, Int<4>{}); break;
      default: run_non_producer(); break;
    }
    if constexpr (WarpInterleaving) {
      if (warp_group_id == 0) {
        __ppu_barrier_sync(5, NumThreadsPerCTA, 0);
      }
    }

    cp_async_wait<0>();
    __syncthreads();
  }
};
} // namespace deep_gemm
