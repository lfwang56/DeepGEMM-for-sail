#pragma once
#include <deep_gemm/common/peer_moe_dag.cuh>
#include <deep_gemm/impls/peer_moe_gemm.cuh>

namespace deep_gemm::training {

template <bool Trace>
__device__ __forceinline__ void dag_activity(uint32_t* state, uint32_t role, bool begin) {
    if constexpr (Trace)
        if (threadIdx.x == 0) atomicAdd(state + role, begin ? 1u : ~0u);
}

__device__ __forceinline__ bool dag_wait(
        const PeerFfnDagParams& p, const uint32_t* address, uint32_t target) {
    __shared__ uint32_t failed;
    if (threadIdx.x == 0) {
        const uint64_t start = clock64();
        while (tix::ld_acq(address) < target) {
            if (tix::ld_acq(p.state + DagError) || clock64() - start > p.timeout_cycles) {
                atomicOr(p.state + DagError, 8u);
                atomicOr(p.global_status, 8u);
                break;
            }
        }
        failed = tix::ld_acq(p.state + DagError);
    }
    __syncthreads();
    return !failed;
}

template <uint32_t N, uint32_t K, uint32_t E, uint32_t BM, uint32_t BN,
          uint32_t BK, uint32_t WM, uint32_t WN, uint32_t Threads, uint32_t Stages,
          bool Fc1Activation = false, bool KeepGateUp = true,
          bool BackwardActivation = false, bool PeerWrite = false>
__device__ __forceinline__ void dag_gemm(const PeerGemmParams& p, uint32_t m, uint32_t n) {
    training_bf16_peer_gemm_impl<cutlass::bfloat16_t, GemmType::MGroupedFused,
        N, K, E, BM, BN, BK, WM, WN, Threads, Stages, false, PeerWrite,
        false, BackwardActivation, 0, false, false, 0, 0, true, false,
        false, false, false, true, 4, Fc1Activation || BackwardActivation,
        Fc1Activation, KeepGateUp>(p, {}, m, n);
}

// Same vector, accumulation and BF16 rounding order as ExpertLocalGradients.
// The input task consumes this M tile's replayed Y before publishing dHidden;
// later DX jobs may then reuse the same local output workspace safely.
template <uint32_t H, uint32_t BM, uint32_t Threads>
__device__ __forceinline__ void dag_gradient_input(const PeerFfnDagParams& p, uint32_t base) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr uint32_t width = H / 8, warps = Threads / 32;
    const uint32_t stride = p.ffn.first.destination_rows_per_rank;
    const uint32_t topk = stride / p.ffn.first.source_rows_per_rank;
    const auto* score_peers = reinterpret_cast<const uint64_t*>(p.ffn.first.dispatch_scratch);
    const auto* dscore_peers = reinterpret_cast<const uint64_t*>(p.ffn.second.dispatch_scratch);
    auto* packed_dy = reinterpret_cast<uint4*>(const_cast<void*>(p.ffn.first.gemm.a_ptr));
    const auto* packed_y = reinterpret_cast<const uint4*>(p.ffn.second.gemm.c_ptr);
    for (uint32_t local_row = warp; local_row < BM; local_row += warps) {
        const uint32_t row = base + local_row;
        const int route = __ldg(p.ffn.first.permutation + row);
        const bool valid = route >= 0 && uint64_t(route) < uint64_t(p.ffn.first.num_ranks) * stride;
        if (route >= 0 && !valid && lane == 0) atomicOr(p.global_status, 2u);
        const uint32_t owner = valid ? uint32_t(route) / stride : 0;
        const uint32_t local_route = valid ? uint32_t(route) % stride : 0;
        const auto* dy = valid ? reinterpret_cast<const uint4*>(__ldg(p.ffn.first.source_peers + owner))
            + uint64_t(local_route / topk) * width : nullptr;
        const float score = valid ? reinterpret_cast<const float*>(__ldg(score_peers + owner))[local_route] : 0.f;
        float dot = 0.f;
        for (uint32_t column = lane; column < width; column += 32) {
            const uint4 raw_dy = valid ? dy[column] : uint4{};
            const uint4 raw_y = valid ? packed_y[uint64_t(row) * width + column] : uint4{};
            const auto* d = reinterpret_cast<const __ppu_bfloat16*>(&raw_dy);
            const auto* y = reinterpret_cast<const __ppu_bfloat16*>(&raw_y);
            uint4 raw_weighted{};
            auto* weighted = reinterpret_cast<__ppu_bfloat16*>(&raw_weighted);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float value = __bfloat162float(d[j]);
                weighted[j] = __float2bfloat16(value * score);
                dot += value * __bfloat162float(y[j]);
            }
            packed_dy[uint64_t(row) * width + column] = raw_weighted;
        }
        #pragma unroll
        for (int delta = 16; delta; delta >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, delta);
        if (valid && lane == 0)
            reinterpret_cast<float*>(__ldg(dscore_peers + owner))[local_route] = dot;
    }
    __threadfence();
    __syncthreads();
}

// One CTA records all time samples using its own clock. The samples describe
// task activity windows, not physical wire utilization or exact MMA occupancy.
// No timestamps from different CUs are compared.
template <bool Return>
__device__ __forceinline__ void dag_monitor(const PeerFfnDagParams& p, uint32_t mtiles, uint32_t ntasks) {
    if (threadIdx.x) return;
    constexpr uint64_t period = 6500;
    const uint64_t start = clock64();
    uint64_t previous = start, sum[9]{};
    uint32_t active[5]{};
    uint64_t samples = 0, dropped = 0;
    while (true) {
        const uint64_t now = clock64(), delta = now - previous;
        const bool communication = active[0] || active[4];
        const bool computation = active[1] || active[2] || active[3];
        sum[0] += delta;
        if (active[0]) sum[1] += delta;
        if (computation) sum[2] += delta;
        if (communication && computation) sum[3] += delta;
        if (!communication && !computation) sum[4] += delta;
        if (active[1]) sum[5] += delta;
        if (active[2]) sum[6] += delta;
        if (active[3]) sum[7] += delta;
        if (active[4]) sum[8] += delta;
        for (uint32_t role = 0; role < 5; ++role)
            active[role] = tix::ld_acq(p.state + ActiveInput + role);
        if (samples < p.sample_capacity) {
            uint64_t* row = p.samples + samples * 6;
            row[0] = now - start;
            for (uint32_t role = 0; role < 5; ++role) row[1 + role] = active[role];
            ++samples;
        } else ++dropped;
        const bool done = tix::ld_acq(p.state + DoneInput) == mtiles
            && tix::ld_acq(p.state + DoneFC2) == ntasks
            && (!Return || tix::ld_acq(p.state + DoneReturn) == ntasks);
        if (done || tix::ld_acq(p.state + DagError)) break;
        if (now - start > p.timeout_cycles) {
            atomicOr(p.state + DagError, 8u); atomicOr(p.global_status, 8u); break;
        }
        previous = now;
        while (clock64() - now < period) {}
    }
    for (uint32_t i = 0; i < 9; ++i) p.monitor[i] = sum[i];
    p.monitor[9] = samples;
    p.monitor[10] = dropped;
    p.monitor[11] = tix::ld_acq(p.state + DagError);
}

template <uint32_t H, uint32_t I, uint32_t E, uint32_t BM, uint32_t BN,
          uint32_t BK, uint32_t WM, uint32_t WN, uint32_t Threads, uint32_t Stages,
          bool Return, bool Trace>
__device__ __forceinline__ void peer_ffn_dag_impl(const PeerFfnDagParams p) {
    static_assert(H % BN == 0 && I % BN == 0);
    __shared__ uint32_t entry_status;
    if (threadIdx.x == 0) entry_status = tix::ld_acq(p.global_status);
    __syncthreads();
    if (entry_status) return;
    constexpr uint32_t pairs = I / BN, n1 = 2 * pairs, n2 = H / BN;
    const uint32_t mtiles = __ldg(p.ffn.first.gemm.aligned_num_m_blocks);
    const uint32_t fc1tasks = mtiles * n1, fc2tasks = mtiles * n2;
    if constexpr (Trace) {
        if (blockIdx.x == gridDim.x - 1) {dag_monitor<Return>(p, mtiles, fc2tasks); return;}
    }
    __shared__ uint64_t addresses[BM];
    __shared__ uint32_t task, kind, last_pair, queue_start;
    const auto* records = reinterpret_cast<const uint4*>(p.ffn.first.gemm.expert_ids_and_cumsum);

    if (blockIdx.x < p.input_ctas) {
        for (uint32_t tile = blockIdx.x; tile < mtiles; tile += p.input_ctas) {
            if (!dag_wait(p, p.state + DoneInput, 0u)) return;
            dag_activity<Trace>(p.state, ActiveInput, true);
            const uint4 record = __ldg(records + tile);
            const uint32_t base = record.w + (tile - record.z) * BM;
            for (uint32_t row = threadIdx.x; row < BM; row += Threads) {
                const uint32_t source = __ldg(p.ffn.first.gemm.sorted_token_ids + base + row);
                uint64_t address = 0;
                if (source < uint64_t(p.ffn.first.num_ranks) * p.ffn.first.source_rows_per_rank) {
                    const uint32_t owner = source / p.ffn.first.source_rows_per_rank;
                    const uint32_t local = source % p.ffn.first.source_rows_per_rank;
                    address = __ldg(p.ffn.first.source_peers + owner) + uint64_t(local) * H * 2;
                }
                addresses[row] = address;
            }
            __syncthreads();
            pack_producer_input_rows<BM, H, Threads>(addresses, p.ffn.first, base);
            dag_activity<Trace>(p.state, ActiveInput, false);
            if (threadIdx.x == 0) {
                tix::red_add_rel(p.input_ready + tile, 1u);
                tix::red_add_rel(p.state + DoneInput, 1u);
            }
            __syncthreads();
        }
        return;
    }
    if (blockIdx.x < p.input_ctas + p.return_ctas) {
        if constexpr (Return) {
            if (p.serialize && !dag_wait(p, p.state + DoneFC2, fc2tasks)) return;
            const auto* output = reinterpret_cast<const uint4*>(p.ffn.second.gemm.c_ptr);
            for (uint32_t item = blockIdx.x - p.input_ctas; item < fc2tasks; item += p.return_ctas) {
                if (!dag_wait(p, p.output_ready + item, 1)) return;
                dag_activity<Trace>(p.state, ActiveReturn, true);
                const uint32_t tile = item / n2, column_base = (item % n2) * BN;
                const uint4 record = __ldg(records + tile);
                const uint32_t base = record.w + (tile - record.z) * BM;
                const uint32_t valid = min(BM, record.y - (tile - record.z) * BM);
                for (uint32_t row = threadIdx.x; row < valid; row += Threads) {
                    const int route = __ldg(p.ffn.second.permutation + base + row);
                    uint64_t address = 0;
                    if (route >= 0) {
                        const uint32_t owner = uint32_t(route) / p.ffn.second.destination_rows_per_rank;
                        if (owner < p.ffn.second.num_ranks) {
                            const uint32_t local = uint32_t(route) % p.ffn.second.destination_rows_per_rank;
                            address = __ldg(p.ffn.second.destination_peers + owner)
                                + (uint64_t(local) * H + column_base) * 2;
                        }
                    }
                    addresses[row] = address;
                }
                __syncthreads();
                for (uint32_t index = threadIdx.x; index < valid * (BN / 8); index += Threads) {
                    const uint32_t row = index / (BN / 8), col = index % (BN / 8);
                    const uint64_t address = addresses[row];
                    if (address) reinterpret_cast<uint4*>(address)[col] =
                        output[uint64_t(base + row) * (H / 8) + column_base / 8 + col];
                }
                __threadfence();
                __syncthreads();
                dag_activity<Trace>(p.state, ActiveReturn, false);
                if (threadIdx.x == 0) tix::red_add_rel(p.state + DoneReturn, 1u);
                __syncthreads();
            }
        }
        return;
    }

    const uint64_t begin = clock64();
    while (true) {
        if (threadIdx.x == 0) {
            kind = 0;
            if (tix::ld_acq(p.state + DoneFC2) == fc2tasks || tix::ld_acq(p.state + DagError)) kind = 3;
            else {
                // Prioritize ready FC2 work so completed M tiles can return
                // while other CTAs still compute FC1 on later input tiles.
                if (!p.serialize || tix::ld_acq(p.state + DoneActivation) == mtiles) {
                    uint32_t head = atomicAdd(p.state + HeadFC2, 0u);
                    const uint32_t tail = tix::ld_acq(p.state + TailFC2);
                    if (head < tail && atomicCAS(p.state + HeadFC2, head, head + 1) == head) {
                        task = head;
                        kind = 2;
                    }
                }
                if (!kind && atomicAdd(p.state + NextFC1, 0u) < fc1tasks) {
                    task = atomicAdd(p.state + NextFC1, 1u);
                    if (task < fc1tasks) kind = 1;
                }
                if (!kind && clock64() - begin > p.timeout_cycles) {
                    atomicOr(p.state + DagError, 8u);
                    atomicOr(p.global_status, 8u);
                    kind = 3;
                }
            }
        }
        __syncthreads();
        if (kind == 3) break;
        if (kind == 1) {
            const uint32_t tile = task / n1, column = task % n1;
            if (!dag_wait(p, p.serialize ? p.state + DoneInput : p.input_ready + tile,
                          p.serialize ? mtiles : 1u)) break;
            dag_activity<Trace>(p.state, ActiveFC1, true);
            dag_gemm<2*I, H, E, BM, BN, BK, WM, WN, Threads, Stages>(p.ffn.first, tile, column);
            __threadfence();
            __syncthreads();
            dag_activity<Trace>(p.state, ActiveFC1, false);
            const uint32_t pair = column % pairs;
            if (threadIdx.x == 0) last_pair = atomicAdd(p.pair_done + tile * pairs + pair, 1u) == 1;
            __syncthreads();
            if (last_pair) {
                if (!dag_wait(p, p.pair_done + tile * pairs + pair, 2u)) break;
                dag_activity<Trace>(p.state, ActiveActivation, true);
                const uint4 record = __ldg(records + tile);
                const uint32_t base = record.w + (tile - record.z) * BM;
                const uint32_t valid = min(BM, record.y - (tile - record.z) * BM);
                const auto* gate = reinterpret_cast<const volatile uint16_t*>(p.ffn.first.gemm.c_ptr);
                auto* hidden = reinterpret_cast<__ppu_bfloat16*>(const_cast<void*>(p.ffn.second.gemm.a_ptr));
                for (uint32_t index = threadIdx.x; index < valid * BN; index += Threads) {
                    const uint32_t row = index / BN, col = pair * BN + index % BN;
                    const uint64_t offset = uint64_t(base + row) * 2 * I + col;
                    union Scalar {uint16_t bits; __ppu_bfloat16 value;};
                    const Scalar g{gate[offset]}, u{gate[offset + I]};
                    const float gf = __bfloat162float(g.value), uf = __bfloat162float(u.value);
                    const float sigmoid = 1.f / (1.f + expf(-gf));
                    hidden[uint64_t(base + row) * I + col] = __float2bfloat16((gf * sigmoid) * uf);
                }
                __threadfence();
                __syncthreads();
                dag_activity<Trace>(p.state, ActiveActivation, false);
                if (threadIdx.x == 0) {
                    last_pair = atomicAdd(p.activated_pairs + tile, 1u) == pairs - 1;
                    if (last_pair) {
                        queue_start = atomicAdd(p.state + TailFC2, n2);
                        for (uint32_t n = 0; n < n2; ++n)
                            tix::red_add_rel(p.queue + queue_start + n, tile * n2 + n + 1);
                        tix::red_add_rel(p.state + DoneActivation, 1u);
                    }
                }
                __syncthreads();
            }
        } else if (kind == 2) {
            if (!dag_wait(p, p.queue + task, 1u)) break;
            const uint32_t item = tix::ld_acq(p.queue + task) - 1;
            dag_activity<Trace>(p.state, ActiveFC2, true);
            dag_gemm<H, I, E, BM, BN, BK, WM, WN, Threads, Stages>(p.ffn.second, item / n2, item % n2);
            __threadfence();
            __syncthreads();
            dag_activity<Trace>(p.state, ActiveFC2, false);
            if (threadIdx.x == 0) {
                if constexpr (Return) tix::red_add_rel(p.output_ready + item, 1u);
                tix::red_add_rel(p.state + DoneFC2, 1u);
            }
        }
        __syncthreads();
    }
}

// Called only by a CTA's leader. A successful claim may briefly precede the
// publisher's release store; that publisher is already resident and never
// waits for a consumer, so acquiring the claimed entry cannot form a cycle.
__device__ __forceinline__ bool dag_claim_queue(
        uint32_t* state, uint32_t head_index, uint32_t tail_index, uint32_t& slot) {
    const uint32_t head = tix::ld_acq(state + head_index);
    if (head < tix::ld_acq(state + tail_index)
            && atomicCAS(state + head_index, head, head + 1) == head) {
        slot = head;
        return true;
    }
    return false;
}

// All worker CTAs share runnable copy and GEMM jobs. A completed input tile
// publishes FC1 jobs; activation publishes FC2 jobs; FC2 publishes return jobs.
// Workers never claim FC1 work whose input still needs an unscheduled producer.
template <uint32_t H, uint32_t I, uint32_t E, uint32_t BM, uint32_t BN,
          uint32_t BK, uint32_t WM, uint32_t WN, uint32_t Threads, uint32_t Stages,
          bool Return, bool Trace, bool Fc1Activation = false, bool InputPrefetch = false,
          bool Backward = false, bool DirectReturn = false>
__device__ __forceinline__ void peer_ffn_shared_pool_impl(const PeerFfnDagParams p) {
    static_assert(H % BN == 0 && I % BN == 0);
    static_assert(!Backward || (Return && !Fc1Activation));
    __shared__ uint32_t entry_status;
    if (threadIdx.x == 0) entry_status = tix::ld_acq(p.global_status);
    __syncthreads();
    if (entry_status) return;
    constexpr uint32_t pairs = I / BN, n1 = (Backward ? 1 : 2) * pairs, n2 = H / BN;
    const uint32_t mtiles = __ldg(p.ffn.first.gemm.aligned_num_m_blocks);
    const uint32_t max_m = p.ffn.first.gemm.shape_m / BM;
    const uint32_t fc2tasks = mtiles * n2;
    auto* fc1_queue = p.input_ready + max_m;
    auto* return_queue = p.output_ready + max_m * n2;
    if constexpr (Trace) {
        if (blockIdx.x == gridDim.x - 1) {dag_monitor<Return>(p, mtiles, fc2tasks); return;}
    }
    __shared__ uint64_t addresses[BM];
    __shared__ uint32_t task, kind, last_pair;
    const auto* records = reinterpret_cast<const uint4*>(p.ffn.first.gemm.expert_ids_and_cumsum);
    const uint64_t begin = clock64();
    while (true) {
        if (threadIdx.x == 0) {
            kind = 0;
            const bool done = tix::ld_acq(p.state + DoneFC2) == fc2tasks
                && tix::ld_acq(p.state + DoneInput) == mtiles
                && (!Return || tix::ld_acq(p.state + DoneReturn) == fc2tasks);
            if (done || tix::ld_acq(p.state + DagError)) kind = 5;
            else {
                if constexpr (Return && !DirectReturn) {
                    if ((!p.serialize || tix::ld_acq(p.state + DoneFC2) == fc2tasks)
                            && dag_claim_queue(p.state, HeadReturn, TailReturn, task)) kind = 4;
                }
                if constexpr (InputPrefetch) {
                    // Bound outstanding input copies without reserving CTA
                    // identities. A worker reverts to compute after copying;
                    // the tail therefore retains the entire resource pool.
                    if (!kind) {
                        const uint32_t completed = tix::ld_acq(p.state + DoneInput);
                        const uint32_t next = tix::ld_acq(p.state + NextInput);
                        if (next < mtiles && next >= completed && next - completed < p.input_ctas
                                && atomicCAS(p.state + NextInput, next, next + 1) == next) {
                            task = next;
                            kind = 1;
                        }
                    }
                }
                if (!kind && (!p.serialize || tix::ld_acq(p.state + DoneActivation) == mtiles)
                        && dag_claim_queue(p.state, HeadFC2, TailFC2, task)) kind = 3;
                if (!kind && (!p.serialize || tix::ld_acq(p.state + DoneInput) == mtiles)
                        && dag_claim_queue(p.state, HeadFC1, TailFC1, task)) kind = 2;
                if constexpr (!InputPrefetch) {
                    if (!kind && tix::ld_acq(p.state + NextInput) < mtiles) {
                        task = atomicAdd(p.state + NextInput, 1u);
                        if (task < mtiles) kind = 1;
                    }
                }
                if (!kind && clock64() - begin > p.timeout_cycles) {
                    atomicOr(p.state + DagError, 8u);
                    atomicOr(p.global_status, 8u);
                    kind = 5;
                }
            }
        }
        __syncthreads();
        if (kind == 5) break;
        if (kind == 1) {
            const uint32_t tile = task;
            dag_activity<Trace>(p.state, ActiveInput, true);
            const uint4 record = __ldg(records + tile);
            const uint32_t base = record.w + (tile - record.z) * BM;
            if constexpr (Backward) {
                dag_gradient_input<H, BM, Threads>(p, base);
            } else {
            for (uint32_t row = threadIdx.x; row < BM; row += Threads) {
                const uint32_t source = __ldg(p.ffn.first.gemm.sorted_token_ids + base + row);
                uint64_t address = 0;
                if (source < uint64_t(p.ffn.first.num_ranks) * p.ffn.first.source_rows_per_rank) {
                    const uint32_t owner = source / p.ffn.first.source_rows_per_rank;
                    const uint32_t local = source % p.ffn.first.source_rows_per_rank;
                    address = __ldg(p.ffn.first.source_peers + owner) + uint64_t(local) * H * 2;
                }
                addresses[row] = address;
            }
            __syncthreads();
            pack_producer_input_rows<BM, H, Threads>(addresses, p.ffn.first, base);
            }
            dag_activity<Trace>(p.state, ActiveInput, false);
            if (threadIdx.x == 0) {
                tix::red_add_rel(p.input_ready + tile, 1u);
                const uint32_t slot = atomicAdd(p.state + TailFC1, n1);
                for (uint32_t n = 0; n < n1; ++n)
                    tix::red_add_rel(fc1_queue + slot + n, tile * n1 + n + 1);
                tix::red_add_rel(p.state + DoneInput, 1u);
            }
        } else if (kind == 2) {
            if (!dag_wait(p, fc1_queue + task, 1u)) break;
            const uint32_t item = tix::ld_acq(fc1_queue + task) - 1;
            const uint32_t tile = item / n1, column = item % n1, pair = column % pairs;
            dag_activity<Trace>(p.state, ActiveFC1, true);
            dag_gemm<Backward ? I : 2*I, H, E, BM, BN, BK, WM, WN, Threads, Stages,
                     Fc1Activation, !Return, Backward>(p.ffn.first, tile, column);
            __threadfence();
            __syncthreads();
            dag_activity<Trace>(p.state, ActiveFC1, false);
            if constexpr (Fc1Activation || Backward) {
                // Every G8 FC1 tile already writes its compact hidden slice.
                // Publish FC2 only after all slices are visible. Replay keeps
                // BF16 gate/up as well, for the later activation gradient.
                if (threadIdx.x == 0 && atomicAdd(p.activated_pairs + tile, 1u) == n1 - 1) {
                    const uint32_t slot = atomicAdd(p.state + TailFC2, n2);
                    for (uint32_t n = 0; n < n2; ++n)
                        tix::red_add_rel(p.queue + slot + n, tile * n2 + n + 1);
                    tix::red_add_rel(p.state + DoneActivation, 1u);
                }
            } else {
                if (threadIdx.x == 0) last_pair = atomicAdd(p.pair_done + tile * pairs + pair, 1u) == 1;
                __syncthreads();
                if (last_pair) {
                    if (!dag_wait(p, p.pair_done + tile * pairs + pair, 2u)) break;
                    dag_activity<Trace>(p.state, ActiveActivation, true);
                    const uint4 record = __ldg(records + tile);
                    const uint32_t base = record.w + (tile - record.z) * BM;
                    const uint32_t valid = min(BM, record.y - (tile - record.z) * BM);
                    const auto* gate = reinterpret_cast<const volatile uint16_t*>(p.ffn.first.gemm.c_ptr);
                    auto* hidden = reinterpret_cast<__ppu_bfloat16*>(const_cast<void*>(p.ffn.second.gemm.a_ptr));
                    for (uint32_t index = threadIdx.x; index < valid * BN; index += Threads) {
                        const uint32_t row = index / BN, col = pair * BN + index % BN;
                        const uint64_t offset = uint64_t(base + row) * 2 * I + col;
                        union Scalar {uint16_t bits; __ppu_bfloat16 value;};
                        const Scalar g{gate[offset]}, u{gate[offset + I]};
                        const float gf = __bfloat162float(g.value), uf = __bfloat162float(u.value);
                        const float sigmoid = 1.f / (1.f + expf(-gf));
                        hidden[uint64_t(base + row) * I + col] = __float2bfloat16((gf * sigmoid) * uf);
                    }
                    __threadfence();
                    __syncthreads();
                    dag_activity<Trace>(p.state, ActiveActivation, false);
                    if (threadIdx.x == 0 && atomicAdd(p.activated_pairs + tile, 1u) == pairs - 1) {
                        const uint32_t slot = atomicAdd(p.state + TailFC2, n2);
                        for (uint32_t n = 0; n < n2; ++n)
                            tix::red_add_rel(p.queue + slot + n, tile * n2 + n + 1);
                        tix::red_add_rel(p.state + DoneActivation, 1u);
                    }
                }
            }
        } else if (kind == 3) {
            if (!dag_wait(p, p.queue + task, 1u)) break;
            const uint32_t item = tix::ld_acq(p.queue + task) - 1;
            dag_activity<Trace>(p.state, ActiveFC2, true);
            dag_gemm<H, Backward ? 2*I : I, E, BM, BN, BK, WM, WN, Threads, Stages,
                     false, true, false, Return && DirectReturn>(p.ffn.second, item / n2, item % n2);
            __threadfence();
            __syncthreads();
            dag_activity<Trace>(p.state, ActiveFC2, false);
            if (threadIdx.x == 0) {
                if constexpr (Return) {
                    if constexpr (DirectReturn) {
                        // Return traffic is embedded in this GEMM epilogue.
                        // Do not label its entire span independent communication.
                        tix::red_add_rel(p.state + DoneReturn, 1u);
                    } else {
                    tix::red_add_rel(p.output_ready + item, 1u);
                    const uint32_t slot = atomicAdd(p.state + TailReturn, 1u);
                    tix::red_add_rel(return_queue + slot, item + 1);
                    }
                }
                tix::red_add_rel(p.state + DoneFC2, 1u);
            }
        } else if (kind == 4) {
            if constexpr (Return && !DirectReturn) {
            if (!dag_wait(p, return_queue + task, 1u)) break;
            const uint32_t item = tix::ld_acq(return_queue + task) - 1;
            const uint32_t tile = item / n2, column_base = (item % n2) * BN;
            dag_activity<Trace>(p.state, ActiveReturn, true);
            const uint4 record = __ldg(records + tile);
            const uint32_t base = record.w + (tile - record.z) * BM;
            const uint32_t valid = min(BM, record.y - (tile - record.z) * BM);
            const auto* output = reinterpret_cast<const uint4*>(p.ffn.second.gemm.c_ptr);
            for (uint32_t row = threadIdx.x; row < valid; row += Threads) {
                const int route = __ldg(p.ffn.second.permutation + base + row);
                uint64_t address = 0;
                if (route >= 0) {
                    const uint32_t owner = uint32_t(route) / p.ffn.second.destination_rows_per_rank;
                    if (owner < p.ffn.second.num_ranks) {
                        const uint32_t local = uint32_t(route) % p.ffn.second.destination_rows_per_rank;
                        address = __ldg(p.ffn.second.destination_peers + owner)
                            + (uint64_t(local) * H + column_base) * 2;
                    }
                }
                addresses[row] = address;
            }
            __syncthreads();
            for (uint32_t index = threadIdx.x; index < valid * (BN / 8); index += Threads) {
                const uint32_t row = index / (BN / 8), col = index % (BN / 8);
                const uint64_t address = addresses[row];
                if (address) reinterpret_cast<uint4*>(address)[col] =
                    output[uint64_t(base + row) * (H / 8) + column_base / 8 + col];
            }
            __threadfence();
            __syncthreads();
            dag_activity<Trace>(p.state, ActiveReturn, false);
            if (threadIdx.x == 0) tix::red_add_rel(p.state + DoneReturn, 1u);
            }
        }
        __syncthreads();
    }
}

}  // namespace deep_gemm::training
