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
#include <deep_gemm/tix/sys_atomics.cuh>

namespace deep_gemm {

// Keep the native TC mainloop free of readiness polls. A CTA starts as soon
// as one K stripe is published and coalesces adjacent published stripes into
// a native run. Later stripes are gathered/quantized concurrently with that run.
template<class Native, uint32_t FullK>
struct TrainingFp8KReadyMainloop : Native {
  using Params=typename Native::Params;
  using TileShape=typename Native::TileShape;
  uint32_t* ready_=nullptr;
  uint32_t* status_=nullptr;
  uint64_t timeout_=0;
  CUTLASS_DEVICE void configure(const training::PeerDispatchPipeline& input,
      uint32_t row,uint32_t valid,uint32_t width) {
    static_assert(cute::size<0>(TileShape{})==128);
    ready_=input.ready+row/128;status_=input.status;timeout_=input.timeout_cycles;
    static_cast<void>(valid);static_cast<void>(width);
  }
  struct BoundedKIterator {
    int index,last;
    CUTLASS_DEVICE int operator*() const {return index<last?index:last;}
    CUTLASS_DEVICE BoundedKIterator& operator++(){++index;return *this;}
  };
  template<class...Ts,class FrgTensorC,class KTileIterator>
  CUTLASS_DEVICE void operator()(Params const& params,cute::tuple<Ts...> const& inputs,
      FrgTensorC& accum,KTileIterator,int count,int thread,char* smem) {
    constexpr uint32_t stripes=DG_TRAINING_FP8_INPUT_READY_COUNT;
    constexpr uint32_t subtiles=DG_TRAINING_FP8_INPUT_SUBTILES;
    static_assert(stripes*subtiles<=32);
    constexpr uint32_t submask=(1u<<subtiles)-1u;
    static_assert(FullK%(128*stripes)==0);
    constexpr uint32_t tiles_per_stripe=FullK/(128*stripes);
    __shared__ uint32_t ready_end,ready_failed;
    int first=0;
    CUTLASS_PRAGMA_NO_UNROLL
    while(first<count) {
      if(threadIdx.x==0) {
        ready_failed=0;
        const uint32_t start_stripe=uint32_t(first)/tiles_per_stripe;
        const uint32_t required=submask<<(start_stripe*subtiles);
        const uint64_t begin=clock64();
        uint32_t mask;
        while(true) {
          mask=tix::ld_acq(ready_);
          if((mask&required)==required)break;
          if(tix::ld_acq(status_) || clock64()-begin>timeout_) {
            atomicOr(status_,8u);ready_failed=1;break;
          }
        }
        uint32_t last_stripe=start_stripe+1;
        while(last_stripe<stripes && ((mask>>(last_stripe*subtiles))&submask)==submask)++last_stripe;
        ready_end=last_stripe*tiles_per_stripe;
      }
      __syncthreads();
      if(ready_failed)return;
      const int end=int(ready_end);
      // Native drain may issue unused prefetches. Clamp them to the final
      // acquired tile so it never reads a not-yet-published stripe.
      Native::operator()(params,inputs,accum,BoundedKIterator{first,end-1},
          end-first,thread,smem);
      first=end;
    }
  }
};
} // namespace deep_gemm
