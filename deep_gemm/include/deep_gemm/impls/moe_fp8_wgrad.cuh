#pragma once
#include <deep_gemm/impls/fp8_gemm.cuh>
#include <deep_gemm/common/moe_fp8_wgrad.cuh>

namespace deep_gemm {
// K-grouped weight gradients: each expert owns a contiguous, padded token
// interval in feature-major FP8 panels. Counts and offsets remain on device.
// Native SFA/SFB layouts keep their full-capacity physical pitch while each
// mainloop receives the expert's actual rounded K. Every output tile has one
// writer, and accumulates FP32 directly into the caller's FP32 gradient.
template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_,
          bool kEnableSboOverlap, bool kUseNStageKernel>
class TrainingFp8Wgrad {
public:
    static_assert(!kEnableSboOverlap && !kUseNStageKernel);
    using Base = DeepGemmUniversal<ProblemShape_, CollectiveMainloop_, CollectiveEpilogue_, TileScheduler_, false, false>;
    using Params = typename Base::Params;
    using TileShape = typename Base::TileShape;
    using TiledMma = typename Base::TiledMma;
    using CollectiveMainloop = CollectiveMainloop_;
    using MainloopParams = typename Base::MainloopParams;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    static constexpr int SharedStorageSize = Base::SharedStorageSize;
    static constexpr int MaxThreadsPerBlock = Base::MaxThreadsPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = Base::MinBlocksPerMultiprocessor;
    CUTLASS_DEVICE void operator()(const Params& params, const training::Fp8WgradParams& dw, char* smem) {
        using namespace cute;
        constexpr int bm = size<0>(TileShape{}), bn = size<1>(TileShape{});
        const uint32_t mt = (dw.m + bm - 1) / bm, nt = (dw.n + bn - 1) / bn;
        const uint64_t per_group = uint64_t(mt) * nt;
        for (uint64_t work = blockIdx.x; work < uint64_t(dw.groups) * per_group; work += gridDim.x) {
            const uint32_t group = work / per_group;
            const int32_t count = __ldg(dw.lengths + group);
            const int32_t start = __ldg(dw.offsets + group);
            const int32_t end = __ldg(dw.offsets + group + 1);
            if (count < 0 || start < 0 || start % 128 || end < start || end > dw.capacity
                    || int64_t(start) + ((int64_t(count) + 127) / 128) * 128 > end) {
                if (threadIdx.x == 0) atomicOr(dw.status, 2u);
                continue;
            }
            if (!count) continue;
            const uint32_t k = (uint32_t(count) + 127) / 128 * 128;
            const uint32_t local = work % per_group;
            const uint32_t mc = local / nt, nc = local % nt;
            // qA=[M,R], SFA=[R/128,M]; qB=[N,R], SFB=[N/128,R/128].
            MainloopParams update = {
                reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + start, params.mainloop.dA,
                reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + start, params.mainloop.dB, 4,
                params.mainloop.ptr_scale_A + int64_t(start / 128) * dw.m, params.mainloop.layout_SFA,
                params.mainloop.ptr_scale_B + start / 128, params.mainloop.layout_SFB
            };
#if DG_FP8_DW_COMPACT
            // Expert-local compact K eliminates the full-ring row stride.
            update.ptr_A = reinterpret_cast<const ElementA*>(params.mainloop.ptr_A) + int64_t(start) * dw.m;
            update.ptr_B = reinterpret_cast<const ElementB*>(params.mainloop.ptr_B) + int64_t(start) * dw.n;
            get<0>(update.dA) = int64_t(k); get<2>(update.dA) = int64_t(dw.m) * k;
            get<0>(update.dB) = int64_t(k); get<2>(update.dB) = int64_t(dw.n) * k;
            using Granularity = cute::Shape<cute::_1,cute::_128,cute::_128>;
            using ScaleConfig = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<Granularity,false,true>(Granularity{}));
            update.ptr_scale_B = params.mainloop.ptr_scale_B + int64_t(start / 128) * (dw.n / 128);
            update.layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(int(dw.m), int(dw.n), int(k), 1));
#endif
            auto problem = typename Base::ProblemShape{int(dw.m), int(dw.n), int(k), 1};
            auto coordinate = make_coord(mc, nc, _, 0);
            CollectiveMainloop mainloop;
            auto inputs = mainloop.load_init(problem, coordinate, update);
            auto ga = get<0>(inputs);
            TiledMma mma;
            auto accum = partition_fragment_C(mma, take<0,2>(TileShape{}));
            clear(accum);
            auto iterator = make_coord_iterator(shape<2>(ga));
            const int tiles = size<2>(ga);
            mainloop(update, inputs, accum, iterator, tiles, int(threadIdx.x), smem);
            auto coordinates = mma.get_thread_slice(int(threadIdx.x)).partition_C(
                make_identity_tensor(take<0,2>(TileShape{})));
            CUTE_UNROLL
            for (int item = 0; item < size(accum); ++item) {
                const auto point = coordinates(item);
                const uint32_t row = mc * bm + uint32_t(get<0>(point));
                const uint32_t column = nc * bn + uint32_t(get<1>(point));
                if (row < dw.m && column < dw.n) {
                    auto* destination = dw.output + (uint64_t(group) * dw.m + row) * dw.n + column;
                    *destination = *destination + float(accum(item));
                }
            }
            __syncthreads();
        }
    }
};
}
