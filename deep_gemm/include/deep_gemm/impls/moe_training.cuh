#pragma once

#include <cstdint>
#include <type_traits>
#include <hggc_runtime.h>
#include <hggc_bf16.h>
#include <cutlass/numeric_types.h>
#include <deep_gemm/common/moe_training.cuh>
#include <deep_gemm/epilogue/activation_quant_ops.cuh>

namespace deep_gemm::training {

using Bf16 = __ppu_bfloat16;

template <typename T>
__device__ __forceinline__ T* pointer(const Params& p, int index) {
    return reinterpret_cast<T*>(p.ptr[index]);
}

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, delta);
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
    return __shfl_sync(0xffffffffu, value, 0);
}

template <Operation Op, bool MnMajorScales = false, bool QuantizeDwBeforeTranspose = false, bool ReciprocalDwScale = false, uint32_t DwCachedFeatures = 0, uint32_t DwCachedRows = 0>
__device__ __forceinline__ void run(const Params& p, int64_t block_index = -1) {
    if (block_index < 0) block_index = blockIdx.x;
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    const int warp = thread >> 5;
    const int64_t index = block_index * blockDim.x + thread;

    if constexpr (Op == Operation::QuantizeWeightsAndTranspose) {
        __shared__ float maxima[8];
        __shared__ uint32_t tile[128*32];
        auto slot=[](int row,int word){return row*32+(word^(row&3)^(row>>4));};
        const int64_t first_tiles=p.dim[0]*(2*p.dim[2]/128)*(p.dim[1]/128);
        const int64_t total_tiles=first_tiles+p.dim[0]*(p.dim[1]/128)*(p.dim[2]/128);
        for(int64_t work=block_index;work<total_tiles;work+=gridDim.x) {
            const bool first=work<first_tiles;
            const int64_t rows=first?2*p.dim[2]:p.dim[1],columns=first?p.dim[1]:p.dim[2];
            const int64_t local=first?work:work-first_tiles;
            const int64_t column_tiles=columns/128,row_tiles=rows/128;
            const int64_t group=local/(row_tiles*column_tiles);
            const int64_t tile_row=(local/column_tiles)%row_tiles,tile_column=local%column_tiles;
            const auto* source=pointer<Bf16>(p,first?0:1);
            auto* quantized=pointer<uint8_t>(p,first?2:4);
            auto* transposed=pointer<uint4>(p,first?6:8);
            uint4 cached[8];
            unsigned maximum=0;
            #pragma unroll
            for(int j=0;j<8;++j) {
                const int vector=thread+j*256;
                const int64_t at=((group*rows+tile_row*128+vector/16)*columns+tile_column*128+(vector%16)*8);
                cached[j]=*reinterpret_cast<const uint4*>(source+at);
                const unsigned words[4]={cached[j].x,cached[j].y,cached[j].z,cached[j].w};
                #pragma unroll
                for(int pair=0;pair<4;++pair) {
                    const unsigned lo=words[pair]&0x7fffu,hi=(words[pair]>>16)&0x7fffu;
                    if(lo<=0x7f80u && lo>maximum)maximum=lo;
                    if(hi<=0x7f80u && hi>maximum)maximum=hi;
                }
            }
            float value=warp_max(__uint_as_float(maximum<<16));
            if(lane==0)maxima[warp]=value;
            __syncthreads();
            if(warp==0) {
                value=warp_max(lane<8?maxima[lane]:0.f);
                if(lane==0)maxima[0]=fmaxf(value,1e-4f)/448.f;
            }
            __syncthreads();
            const float scale=maxima[0];
            if(thread==0) {
                pointer<float>(p,first?3:5)[local]=scale;
                pointer<float>(p,first?7:9)[group*row_tiles*column_tiles+tile_column*row_tiles+tile_row]=scale;
            }
            #pragma unroll
            for(int j=0;j<8;++j) {
                const unsigned words[4]={cached[j].x,cached[j].y,cached[j].z,cached[j].w};
                unsigned converted[4];
                #pragma unroll
                for(int pair=0;pair<4;++pair)
                    converted[pair]=epilogue::quant::fp8_pair(__uint_as_float(words[pair]<<16)/scale,
                        __uint_as_float(words[pair]&0xffff0000u)/scale);
                const uint2 packed=make_uint2(converted[0]|(converted[1]<<16),converted[2]|(converted[3]<<16));
                const int vector=thread+j*256,row=vector/16,column=(vector%16)*8;
                const int64_t at=((group*rows+tile_row*128+row)*columns+tile_column*128+column);
                *reinterpret_cast<uint2*>(quantized+at)=packed;
                tile[slot(row,column/4)]=packed.x;
                tile[slot(row,column/4+1)]=packed.y;
            }
            __syncthreads();
            #pragma unroll
            for(int vector=thread;vector<1024;vector+=256) {
                const int row=vector/8,column=(vector%8)*16;
                uint32_t words[4]{};
                #pragma unroll
                for(int word=0;word<4;++word) {
                    #pragma unroll
                    for(int byte=0;byte<4;++byte)
                        words[word]|=((tile[slot(column+word*4+byte,row/4)]>>(8*(row&3)))&255u)<<(8*byte);
                }
                const int64_t at=((group*columns+tile_column*128+row)*rows+tile_row*128+column)/16;
                transposed[at]=make_uint4(words[0],words[1],words[2],words[3]);
            }
            __syncthreads();
        }
        return;
    }

    if constexpr (Op == Operation::QuantizeRowsVectorDeviceRows) {
        const int rows=__ldg(pointer<int32_t>(p,3));
        if(rows<0 || int64_t(rows)>p.dim[0]) {
            if(thread==0 && blockIdx.x==0)atomicOr(pointer<int32_t>(p,4),2);
            return;
        }
        const int within=thread&15;
        const unsigned mask=0xffffu<<((thread&31)>=16?16:0);
        const int64_t segments=p.dim[1]/128,groups=int64_t(rows)*segments;
        for(int64_t group=block_index*16+thread/16;group<groups;group+=int64_t(gridDim.x)*16) {
            const int64_t base=group*128;
            const uint4 cached=*reinterpret_cast<const uint4*>(pointer<Bf16>(p,0)+base+within*8);
            const unsigned words[4]={cached.x,cached.y,cached.z,cached.w};
            unsigned maximum=0;
            #pragma unroll
            for(int j=0;j<4;++j) {
                const unsigned lo=words[j]&0x7fffu,hi=(words[j]>>16)&0x7fffu;
                if(lo<=0x7f80u)maximum=max(maximum,lo);
                if(hi<=0x7f80u)maximum=max(maximum,hi);
            }
            #pragma unroll
            for(int delta=8;delta>0;delta>>=1)
                maximum=max(maximum,__shfl_down_sync(mask,maximum,delta,16));
            maximum=__shfl_sync(mask,maximum,0,16);
            const float scale=fmaxf(__uint_as_float(maximum<<16),1e-4f)/448.f;
            if(within==0) {
                const int64_t at=MnMajorScales?(group%segments)*p.dim[0]+group/segments:group;
                pointer<float>(p,2)[at]=scale;
            }
            unsigned packed[4];
            #pragma unroll
            for(int j=0;j<4;++j)packed[j]=epilogue::quant::fp8_pair(
                __uint_as_float(words[j]<<16)/scale,__uint_as_float(words[j]&0xffff0000u)/scale);
            *reinterpret_cast<uint2*>(pointer<uint8_t>(p,1)+base+within*8)=
                make_uint2(packed[0]|(packed[1]<<16),packed[2]|(packed[3]<<16));
        }
        return;
    }
    if constexpr (Op == Operation::PackFp8WeightGradients && DwCachedFeatures != 0) {
        constexpr int F=DwCachedFeatures,KCap=DwCachedRows,V=F/4,Warps=32;
        static_assert((F==16 || F==32) && KCap%128==0);
        __shared__ uint32_t cached[KCap/2][F];
        __shared__ uint32_t maxima[Warps][F];
        __shared__ float inverse[F];
        const int capacity=p.dim[1],m=p.dim[2],n=p.dim[3],per_group=(m+n)/F;
        const auto slot=[](int pair,int feature) {
            return (feature^(pair&(F-1))^((pair/F)&(F-1)))&(F-1);
        };
        for(int work=block_index;work<p.dim[0]*per_group;work+=gridDim.x) {
            const int g=work/per_group,local=work%per_group;
            const bool left=local<m/F;
            const int width=left?m:n,feature_base=(left?local:local-m/F)*F;
            const int count=__ldg(pointer<int32_t>(p,6)+g),start=__ldg(pointer<int32_t>(p,7)+g);
            const int end=__ldg(pointer<int32_t>(p,7)+g+1);
            if(count<0 || start<0 || start%128 || end<start || end>capacity ||
                    int64_t(start)+((int64_t(count)+127)/128)*128>end) {
                if(thread==0)atomicOr(pointer<int32_t>(p,8),2);
                continue;
            }
            const int padded=(count+127)/128*128,feature=(thread%V)*4;
            const bool retain=count<=KCap;
            const auto* source=pointer<Bf16>(p,left?0:1);
            auto* destination=pointer<uint8_t>(p,left?2:4);
            auto* sf=pointer<float>(p,left?3:5);
            uint32_t maximum[4]{};
            for(int pair=thread/V;pair<(count+1)/2;pair+=blockDim.x/V) {
                const int token=pair*2;
                const auto a=*reinterpret_cast<const uint2*>(source+(int64_t(start)+token)*width+feature_base+feature);
                uint2 b{};
                if(token+1<count)b=*reinterpret_cast<const uint2*>(source+(int64_t(start)+token+1)*width+feature_base+feature);
                const uint32_t va[4]={a.x&65535u,a.x>>16,a.y&65535u,a.y>>16};
                const uint32_t vb[4]={b.x&65535u,b.x>>16,b.y&65535u,b.y>>16};
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const auto lo=va[j]&0x7fffu,hi=vb[j]&0x7fffu;
                    if(lo<=0x7f80u)maximum[j]=max(maximum[j],lo);
                    if(hi<=0x7f80u)maximum[j]=max(maximum[j],hi);
                    if(retain)cached[pair][slot(pair,feature+j)]=va[j]|(vb[j]<<16);
                }
            }
            #pragma unroll
            for(int j=0;j<4;++j) {
                #pragma unroll
                for(int delta=16;delta>=V;delta>>=1)
                    maximum[j]=max(maximum[j],__shfl_down_sync(0xffffffffu,maximum[j],delta));
                if(lane<V)maxima[warp][feature+j]=maximum[j];
            }
            __syncthreads();
            if(thread<V) {
                float scales[4];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    uint32_t value=0;
                    #pragma unroll
                    for(int w=0;w<Warps;++w)value=max(value,maxima[w][feature+j]);
                    scales[j]=fmaxf(__uint_as_float(value<<16),1e-4f)/448.f;
                    inverse[feature+j]=1.f/scales[j];
                }
                *reinterpret_cast<float4*>(sf+int64_t(g)*width+feature_base+feature)=
                    make_float4(scales[0],scales[1],scales[2],scales[3]);
            }
            __syncthreads();
            for(int base=0;base<padded;base+=128) {
                // Oversized experts retain the same recipe using ordinary
                // two-pass loads through the first128rows of this scratch.
                if(!retain) {
                    for(int pair=thread/V;pair<64;pair+=blockDim.x/V) {
                        uint2 a{},b{};
                        if(base+2*pair<count)a=*reinterpret_cast<const uint2*>(source+(int64_t(start)+base+2*pair)*width+feature_base+feature);
                        if(base+2*pair+1<count)b=*reinterpret_cast<const uint2*>(source+(int64_t(start)+base+2*pair+1)*width+feature_base+feature);
                        const uint32_t va[4]={a.x&65535u,a.x>>16,a.y&65535u,a.y>>16};
                        const uint32_t vb[4]={b.x&65535u,b.x>>16,b.y&65535u,b.y>>16};
                        #pragma unroll
                        for(int j=0;j<4;++j)cached[pair][slot(pair,feature+j)]=va[j]|(vb[j]<<16);
                    }
                    __syncthreads();
                }
                if(warp<F) {
                    const int at=base+lane*4,pair=retain?at/2:lane*2;
                    uint32_t a=at<count?cached[pair][slot(pair,warp)]:0;
                    uint32_t b=at+2<count?cached[pair+1][slot(pair+1,warp)]:0;
                    if(at+1>=count)a&=65535u;
                    if(at+3>=count)b&=65535u;
                    const float scale=inverse[warp];
                    const uint32_t lo=epilogue::quant::fp8_pair(__uint_as_float(a<<16)*scale,__uint_as_float(a&0xffff0000u)*scale);
                    const uint32_t hi=epilogue::quant::fp8_pair(__uint_as_float(b<<16)*scale,__uint_as_float(b&0xffff0000u)*scale);
                    const int64_t offset=int64_t(start)*width+int64_t(feature_base+warp)*padded+at;
                    *reinterpret_cast<uint32_t*>(destination+offset)=lo|(hi<<16);
                }
                if(!retain)__syncthreads();
            }
            __syncthreads();
        }
        return;
    }
    if constexpr (Op == Operation::PackFp8WeightGradients) {
        // Per-expert feature scales over actual token K. First pass reduces
        // coalesced token-major BF16 reads. Second pass transposes through
        // swizzled TSM into the native FP8 non-transposed MMA panels.
        __shared__ uint32_t maxima[8][128];
        __shared__ float scales[128];
        __shared__ uint32_t tile[128][QuantizeDwBeforeTranspose ? 32 : 64];
        const int gcount=p.dim[0],capacity=p.dim[1],m=p.dim[2],n=p.dim[3];
        const int per_group=(m+n)/128;
        for(int work=block_index;work<gcount*per_group;work+=gridDim.x) {
            const int g=work/per_group,local=work%per_group;
            const bool left=local<m/128;
            const int width=left?m:n,feature_base=(left?local:local-m/128)*128;
            const int count=__ldg(pointer<int32_t>(p,6)+g);
            const int start=__ldg(pointer<int32_t>(p,7)+g),end=__ldg(pointer<int32_t>(p,7)+g+1);
            if(count<0 || start<0 || start%128 || end<start || end>capacity ||
                    int64_t(start)+((int64_t(count)+127)/128)*128>end) {
                if(thread==0)atomicOr(pointer<int32_t>(p,8),2);
                continue;
            }
            const int padded=(count+127)/128*128;
            const auto* source=pointer<Bf16>(p,left?0:1);
            auto* destination=pointer<uint8_t>(p,left?2:4);
            auto* sf=pointer<float>(p,left?3:5);
            unsigned maximum[4]{};
            for(int token=warp;token<count;token+=8) {
                const int64_t at=(int64_t(start)+token)*width+feature_base+lane*4;
                const uint2 values=*reinterpret_cast<const uint2*>(source+at);
                const unsigned bits[4]={values.x&65535u,values.x>>16,values.y&65535u,values.y>>16};
                #pragma unroll
                for(int j=0;j<4;++j) {
                    const unsigned positive=bits[j]&0x7fffu;
                    if(positive<=0x7f80u && positive>maximum[j])maximum[j]=positive;
                }
            }
            #pragma unroll
            for(int j=0;j<4;++j)maxima[warp][lane*4+j]=maximum[j];
            __syncthreads();
            if(warp==0) {
                float values[4];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    unsigned maximum=0;
                    #pragma unroll
                    for(int w=0;w<8;++w) maximum=max(maximum,maxima[w][lane*4+j]);
                    values[j]=fmaxf(__uint_as_float(maximum<<16),1e-4f)/448.f;
                    scales[lane*4+j]=ReciprocalDwScale ? 1.f/values[j] : values[j];
                }
                *reinterpret_cast<float4*>(sf+int64_t(g)*width+feature_base+lane*4)=
                    make_float4(values[0],values[1],values[2],values[3]);
            }
            __syncthreads();
            for(int base=0;base<padded;base+=128) {
                #pragma unroll
                for(int v=thread;v<4096;v+=256) {
                    const int token=v/32,feature=(v%32)*4;
                    uint2 values=make_uint2(0,0);
                    if(base+token<count) {
                        const int64_t at=(int64_t(start)+base+token)*width+feature_base+feature;
                        values=*reinterpret_cast<const uint2*>(source+at);
                    }
                    if constexpr (QuantizeDwBeforeTranspose) {
                        const float4 sf4=*reinterpret_cast<const float4*>(scales+feature);
                        const auto normalize=[](float value,float factor) {
                            if constexpr(ReciprocalDwScale)return value*factor;
                            else return value/factor;
                        };
                        const unsigned lo=epilogue::quant::fp8_pair(
                            normalize(__uint_as_float(values.x<<16),sf4.x),
                            normalize(__uint_as_float(values.x&0xffff0000u),sf4.y));
                        const unsigned hi=epilogue::quant::fp8_pair(
                            normalize(__uint_as_float(values.y<<16),sf4.z),
                            normalize(__uint_as_float(values.y&0xffff0000u),sf4.w));
                        tile[token][(feature/4)^(token>>2)]=lo|(hi<<16);
                    } else {
                        tile[token][(feature/2)^(token>>2)]=values.x;
                        tile[token][(feature/2+1)^(token>>2)]=values.y;
                    }
                }
                __syncthreads();
                #pragma unroll
                for(int j=0;j<16;++j) {
                    const int feature=warp*16+j;
                    unsigned packed=0;
                    if constexpr (QuantizeDwBeforeTranspose) {
                        #pragma unroll
                        for(int byte=0;byte<4;++byte) {
                            const int token=lane*4+byte;
                            const unsigned word=tile[token][(feature/4)^(token>>2)];
                            packed|=((word>>(8*(feature&3)))&255u)<<(byte*8);
                        }
                    } else {
                        float values[4];
                        #pragma unroll
                        for(int byte=0;byte<4;++byte) {
                            const int token=lane*4+byte;
                            const unsigned word=tile[token][(feature/2)^(token>>2)];
                            const unsigned raw=(word>>(16*(feature&1)))&65535u;
                            if constexpr(ReciprocalDwScale)values[byte]=__uint_as_float(raw<<16)*scales[feature];
                            else values[byte]=__uint_as_float(raw<<16)/scales[feature];
                        }
                        const unsigned lo=epilogue::quant::fp8_pair(values[0],values[1]);
                        const unsigned hi=epilogue::quant::fp8_pair(values[2],values[3]);
                        packed=lo|(hi<<16);
                    }
                    const int64_t at=int64_t(start)*width+int64_t(feature_base+feature)*padded+base+lane*4;
                    *reinterpret_cast<unsigned*>(destination+at)=packed;
                }
                __syncthreads();
            }
        }
        return;
    }
    if constexpr (Op == Operation::TransposeQuantizedWeights || Op == Operation::TransposeQuantizedWeightsSwizzled) {
        // The 128x128 quantization grid is closed under transpose. Transfer
        // E4M3 bytes and FP32 scale bits without re-quantizing BF16 weights.
        constexpr bool swizzled=Op==Operation::TransposeQuantizedWeightsSwizzled;
        __shared__ uint32_t tile[128*(swizzled?32:33)];
        auto slot=[](int row,int word) {
            if constexpr(swizzled)return row*32+(word^(row&3)^(row>>4));
            else return row*33+word;
        };
        const int64_t first_tiles = p.dim[0] * (2 * p.dim[2] / 128) * (p.dim[1] / 128);
        const int64_t total_tiles = first_tiles + p.dim[0] * (p.dim[1] / 128) * (p.dim[2] / 128);
        for (int64_t t = block_index; t < total_tiles; t += gridDim.x) {
            const bool first = t < first_tiles;
            const int64_t rows = first ? 2 * p.dim[2] : p.dim[1];
            const int64_t columns = first ? p.dim[1] : p.dim[2];
            const int64_t local = first ? t : t - first_tiles;
            const int64_t column_tiles = columns / 128, row_tiles = rows / 128;
            const int64_t group = local / (row_tiles * column_tiles);
            const int64_t tile_row = (local / column_tiles) % row_tiles;
            const int64_t tile_column = local % column_tiles;
            const auto* source = pointer<uint4>(p, first ? 0 : 2);
            auto* destination = pointer<uint4>(p, first ? 4 : 6);
            #pragma unroll
            for (int vector = thread; vector < 1024; vector += 256) {
                const int row = vector / 8, column = vector % 8 * 16;
                const int64_t offset = ((group * rows + tile_row * 128 + row) * columns
                                       + tile_column * 128 + column) / 16;
                const uint4 raw = source[offset];
                tile[slot(row,column/4)] = raw.x;
                tile[slot(row,column/4+1)] = raw.y;
                tile[slot(row,column/4+2)] = raw.z;
                tile[slot(row,column/4+3)] = raw.w;
            }
            if (thread == 0) {
                const auto raw = pointer<uint32_t>(p, first ? 1 : 3)[local];
                pointer<uint32_t>(p, first ? 5 : 7)[group * row_tiles * column_tiles
                                                + tile_column * row_tiles + tile_row] = raw;
            }
            __syncthreads();
            #pragma unroll
            for (int vector = thread; vector < 1024; vector += 256) {
                const int row = vector / 8, column = vector % 8 * 16;
                uint32_t words[4]{};
                #pragma unroll
                for (int word = 0; word < 4; ++word) {
                    #pragma unroll
                    for (int byte = 0; byte < 4; ++byte)
                        words[word] |= ((tile[slot(column+word*4+byte,row/4)]
                                          >> (8 * (row & 3))) & 255u) << (8 * byte);
                }
                const int64_t offset = ((group * columns + tile_column * 128 + row) * rows
                                       + tile_row * 128 + column) / 16;
                destination[offset] = make_uint4(words[0], words[1], words[2], words[3]);
            }
            __syncthreads();
        }
        return;
    }
    if constexpr (Op == Operation::PackQuantizedRouteRows) {
        const int64_t capacity = p.dim[0], width = p.dim[1], source_rows = p.dim[2], topk = p.dim[3];
        const int32_t active = __ldg(pointer<int32_t>(p, 5));
        if (active < 0 || active > capacity) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 6), 2);
            return;
        }
        const int64_t vectors = width / 16;
        for (int64_t position = index; position < int64_t(active) * vectors;
             position += int64_t(gridDim.x) * blockDim.x) {
            const int route = pointer<int32_t>(p, 2)[position / vectors];
            uint4 value{};
            if (route >= 0) {
                const int64_t source = route / topk;
                if (source < source_rows)
                    value = pointer<uint4>(p, 0)[source * vectors + position % vectors];
                else atomicOr(pointer<int32_t>(p, 6), 2);
            }
            pointer<uint4>(p, 3)[position] = value;
        }
        // Row-major receive scales become MN-major local scales. Each warp
        // writes neighboring destination rows; every scale bit is retained.
        for (int64_t position = index; position < int64_t(active) * (width / 128);
             position += int64_t(gridDim.x) * blockDim.x) {
            const int64_t row = position % active, segment = position / active;
            const int route = pointer<int32_t>(p, 2)[row];
            float scale = 1e-4f / 448.f;
            if (route >= 0 && route / topk < source_rows)
                scale = pointer<float>(p, 1)[int64_t(route / topk) * (width / 128) + segment];
            pointer<float>(p, 4)[segment * capacity + row] = scale;
        }
        return;
    }
    if constexpr (Op == Operation::QuantizeRowsDeviceRows ||
                  Op == Operation::ActivationQuantizeDeviceRows ||
                  Op == Operation::ActivationBackwardQuantizeDeviceRows) {
        constexpr Operation Base = Op == Operation::QuantizeRowsDeviceRows ? Operation::QuantizeRows :
            Op == Operation::ActivationQuantizeDeviceRows ? Operation::ActivationQuantize :
            Operation::ActivationBackwardQuantize;
        constexpr int count_index = Op == Operation::QuantizeRowsDeviceRows ? 3 :
            Op == Operation::ActivationQuantizeDeviceRows ? 4 : 5;
        const int32_t rows = __ldg(pointer<int32_t>(p, count_index));
        if (rows < 0 || int64_t(rows) > p.dim[0]) {
            if (thread == 0 && blockIdx.x == 0)
                atomicOr(pointer<int32_t>(p, count_index + 1), 2);
            return;
        }
        Params active = p;
        active.dim[0] = rows;
        active.dim[2] = MnMajorScales ? p.dim[0] : 0;
        const int64_t blocks = (int64_t(rows) * (p.dim[1] / 128) + 7) / 8;
        // The reused arithmetic is warp-local. Its last partial warp returns
        // uniformly without touching inactive rows or requiring a CTA barrier.
        for (int64_t block = block_index; block < blocks; block += gridDim.x)
            run<Base, MnMajorScales>(active, block);
    } else if constexpr (Op == Operation::CountRoutes) {
        // ids, counts, route expert, within-expert row, status.
        // dims: route count, local experts, rank, number of ranks.
        if (index >= p.dim[0]) return;
        const int64_t id = pointer<int64_t>(p, 0)[index];
        const int experts = p.dim[1];
        int expert = -1, row = -1;
        if (id < -1 || id >= int64_t(experts) * p.dim[3]) {
            atomicOr(pointer<int32_t>(p, 4), 1);
        } else if (id >= 0 && id / experts == p.dim[2]) {
            expert = id % experts;
            row = atomicAdd(pointer<int32_t>(p, 1) + expert, 1);
        }
        pointer<int32_t>(p, 2)[index] = expert;
        pointer<int32_t>(p, 3)[index] = row;
    } else if constexpr (Op == Operation::PrefixRoutes) {
        // counts, offsets, aligned K lengths. dims: experts, alignment.
        if (index != 0) return;
        int offset = 0;
        pointer<int32_t>(p, 1)[0] = 0;
        for (int expert = 0; expert < p.dim[0]; ++expert) {
            const int count = pointer<int32_t>(p, 0)[expert];
            const int length = ((count + p.dim[1] - 1) / p.dim[1]) * p.dim[1];
            pointer<int32_t>(p, 2)[expert] = length;
            offset += length;
            pointer<int32_t>(p, 1)[expert + 1] = offset;
        }
    } else if constexpr (Op == Operation::InitializeLayout) {
        // offsets, per-row expert IDs, packed-row -> global route permutation.
        // dims: physical row capacity, experts.
        if (index >= p.dim[0]) return;
        const auto* offsets = pointer<int32_t>(p, 0);
        int expert = 0;
        if (index < offsets[p.dim[1]]) {
            int low = 0, high = p.dim[1];
            while (low < high) {
                const int middle = (low + high) / 2;
                if (offsets[middle + 1] <= index) low = middle + 1;
                else high = middle;
            }
            expert = low;
        }
        // Unused trailing rows are zero and use valid expert 0. This also
        // supports kernels that do not recognize a negative expert sentinel.
        pointer<int32_t>(p, 1)[index] = expert;
        pointer<int32_t>(p, 2)[index] = -1;
    } else if constexpr (Op == Operation::PackInput) {
        // gathered x, route experts, local rows, expert offsets, packed x,
        // permutation, status. dims: routes, H, top-k, physical capacity.
        const int64_t route = block_index;
        if (route >= p.dim[0]) return;
        const int expert = pointer<int32_t>(p, 1)[route];
        if (expert < 0) return;
        const int row = pointer<int32_t>(p, 3)[expert] + pointer<int32_t>(p, 2)[route];
        if (row >= p.dim[3]) {
            if (thread == 0) atomicOr(pointer<int32_t>(p, 6), 2);
            return;
        }
        if (thread == 0) pointer<int32_t>(p, 5)[row] = route;
        const auto* source = pointer<Bf16>(p, 0) + (route / p.dim[2]) * p.dim[1];
        auto* destination = pointer<Bf16>(p, 4) + int64_t(row) * p.dim[1];
        for (int64_t column = thread; column < p.dim[1]; column += blockDim.x)
            destination[column] = source[column];
    } else if constexpr (Op == Operation::UnpackRoutes) {
        // packed values, permutation, zero-initialized global route output.
        // dims: physical rows, H, global routes.
        if (index >= p.dim[0] * p.dim[1]) return;
        const int row = index / p.dim[1];
        const int route = pointer<int32_t>(p, 1)[row];
        if (route >= 0 && route < p.dim[2])
            pointer<Bf16>(p, 2)[int64_t(route) * p.dim[1] + index % p.dim[1]] = pointer<Bf16>(p, 0)[index];
    } else if constexpr (Op == Operation::PackRouteGrad) {
        // global route gradients, permutation, packed gradients.
        // dims: physical rows, H, global routes.
        if (index >= p.dim[0] * p.dim[1]) return;
        const int route = pointer<int32_t>(p, 1)[index / p.dim[1]];
        pointer<Bf16>(p, 2)[index] = route >= 0 && route < p.dim[2]
            ? pointer<Bf16>(p, 0)[int64_t(route) * p.dim[1] + index % p.dim[1]]
            : __float2bfloat16(0.f);
    } else if constexpr (Op == Operation::Combine) {
        // local route outputs, full-input scores, full-input y/dx.
        // dims: chunk tokens, H, top-k, input tokens, chunk start, apply scores.
        const bool vectorized = p.dim[1] % 8 == 0 &&
            (reinterpret_cast<uint64_t>(p.ptr[0]) & 15u) == 0 &&
            (reinterpret_cast<uint64_t>(p.ptr[2]) & 15u) == 0;
        if (vectorized) {
            const int64_t vectors = p.dim[1] / 8;
            const int64_t token = index / vectors, column = (index % vectors) * 8;
            const int64_t source_token = token + p.dim[4];
            if (token >= p.dim[0] || source_token >= p.dim[3]) return;
            float values[8] = {};
            // Keep slot accumulation order, sharing each score/address across
            // eight features and issuing aligned 16-byte memory operations.
            for (int slot = 0; slot < p.dim[2]; ++slot) {
                const float score = p.dim[5] ? pointer<float>(p, 1)[source_token * p.dim[2] + slot] : 1.f;
                const uint4 raw = *reinterpret_cast<const uint4*>(pointer<Bf16>(p, 0) +
                    (token * p.dim[2] + slot) * p.dim[1] + column);
                const auto* input = reinterpret_cast<const Bf16*>(&raw);
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    values[j] += __bfloat162float(input[j]) * score;
            }
            uint4 packed;
            auto* output = reinterpret_cast<Bf16*>(&packed);
            #pragma unroll
            for (int j = 0; j < 8; ++j) output[j] = __float2bfloat16(values[j]);
            *reinterpret_cast<uint4*>(pointer<Bf16>(p, 2) + source_token * p.dim[1] + column) = packed;
            return;
        }
        const int64_t token = index / p.dim[1], column = index % p.dim[1];
        const int64_t source_token = token + p.dim[4];
        if (token >= p.dim[0] || source_token >= p.dim[3]) return;
        float value = 0;
        for (int slot = 0; slot < p.dim[2]; ++slot) {
            const float score = p.dim[5] ? pointer<float>(p, 1)[source_token * p.dim[2] + slot] : 1.f;
            value += __bfloat162float(pointer<Bf16>(p, 0)[(token * p.dim[2] + slot) * p.dim[1] + column]) * score;
        }
        pointer<Bf16>(p, 2)[source_token * p.dim[1] + column] = __float2bfloat16(value);
    } else if constexpr (Op == Operation::Activation || Op == Operation::ActivationBackward) {
        // Forward: gate/up, hidden. Backward: gate/up, dHidden, dGateUp.
        // dims: physical rows, intermediate size.
        if (index >= p.dim[0] * p.dim[1]) return;
        const int64_t row = index / p.dim[1], column = index % p.dim[1];
        const int64_t offset = row * 2 * p.dim[1]
            + (p.dim[2] == 8 ? column / 8 * 16 + column % 8 : column);
        const int64_t up_step = p.dim[2] == 8 ? 8 : p.dim[1];
        const float gate = __bfloat162float(pointer<Bf16>(p, 0)[offset]);
        const float up = __bfloat162float(pointer<Bf16>(p, 0)[offset + up_step]);
        const float sigmoid = 1.f / (1.f + expf(-gate));
        const float silu = gate * sigmoid;
        if constexpr (Op == Operation::Activation) {
            pointer<Bf16>(p, 1)[index] = __float2bfloat16(silu * up);
        } else {
            const float dh = __bfloat162float(pointer<Bf16>(p, 1)[index]);
            pointer<Bf16>(p, 2)[offset] = __float2bfloat16(dh * up * sigmoid * (1.f + gate * (1.f - sigmoid)));
            pointer<Bf16>(p, 2)[offset + up_step] = __float2bfloat16(dh * silu);
        }
    } else if constexpr (Op == Operation::ActivationQuantize || Op == Operation::ActivationBackwardQuantize) {
        // Preserve the BF16 activation/gradient rounding boundary for wgrad,
        // then immediately quantize those same values for the next FP8 GEMM.
        constexpr bool backward = Op == Operation::ActivationBackwardQuantize;
        const int64_t group = block_index * (blockDim.x / 32) + warp;
        const int64_t segments = p.dim[1] / 128;
        if (group >= p.dim[0] * segments) return;
        const int64_t row = group / segments, segment = group % segments;
        const int64_t input_base = row * 2 * p.dim[1] + segment * 128;
        const int64_t output_base = row * (backward ? 2 : 1) * p.dim[1] + segment * 128;
        float first[4], second[4], maximum_first = 0.f, maximum_second = 0.f;
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            const int column = lane * 2 + (element / 2) * 64 + element % 2;
            const float gate = __bfloat162float(pointer<Bf16>(p, 0)[input_base + column]);
            const float up = __bfloat162float(pointer<Bf16>(p, 0)[input_base + column + p.dim[1]]);
            const float sigmoid = 1.f / (1.f + expf(-gate));
            const float silu = gate * sigmoid;
            if constexpr (backward) {
                const float dh = __bfloat162float(pointer<Bf16>(p, 1)[row * p.dim[1] + segment * 128 + column]);
                const Bf16 dg = __float2bfloat16(dh * up * sigmoid * (1.f + gate * (1.f - sigmoid)));
                const Bf16 du = __float2bfloat16(dh * silu);
                pointer<Bf16>(p, 2)[output_base + column] = dg;
                pointer<Bf16>(p, 2)[output_base + column + p.dim[1]] = du;
                first[element] = __bfloat162float(dg);
                second[element] = __bfloat162float(du);
                maximum_second = fmaxf(maximum_second, fabsf(second[element]));
            } else {
                const Bf16 hidden = __float2bfloat16(silu * up);
                pointer<Bf16>(p, 1)[output_base + column] = hidden;
                first[element] = __bfloat162float(hidden);
            }
            maximum_first = fmaxf(maximum_first, fabsf(first[element]));
        }
        const float scale_first = fmaxf(warp_max(maximum_first), 1e-4f) / 448.f;
        auto* quantized = pointer<uint16_t>(p, backward ? 3 : 2);
        auto* scales = pointer<float>(p, backward ? 4 : 3);
        if (lane == 0) {
            if constexpr (MnMajorScales) scales[segment * p.dim[2] + row] = scale_first;
            else scales[output_base / 128] = scale_first;
        }
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair)
            quantized[output_base / 2 + lane + pair * 32] = epilogue::quant::fp8_pair(
                first[2 * pair] / scale_first, first[2 * pair + 1] / scale_first);
        if constexpr (backward) {
            const float scale_second = fmaxf(warp_max(maximum_second), 1e-4f) / 448.f;
            if (lane == 0) {
                if constexpr (MnMajorScales) scales[(segment + segments) * p.dim[2] + row] = scale_second;
                else scales[(output_base + p.dim[1]) / 128] = scale_second;
            }
            #pragma unroll
            for (int pair = 0; pair < 2; ++pair)
                quantized[(output_base + p.dim[1]) / 2 + lane + pair * 32] = epilogue::quant::fp8_pair(
                    second[2 * pair] / scale_second, second[2 * pair + 1] / scale_second);
        }
    } else if constexpr (Op == Operation::QuantizeRows) {
        // BF16 matrix, E4M3 matrix, FP32 scales. dims: rows, K (128 aligned).
        const int64_t group = block_index * (blockDim.x / 32) + warp;
        const int64_t groups = p.dim[0] * (p.dim[1] / 128);
        if (group >= groups) return;  // Uniform within each participating warp.
        const int64_t base = group * 128;
        float values[4], maximum = 0.f;
        #pragma unroll
        for (int element = 0; element < 4; ++element) {
            values[element] = __bfloat162float(pointer<Bf16>(p, 0)[base + lane * 2 + (element / 2) * 64 + element % 2]);
            maximum = fmaxf(maximum, fabsf(values[element]));
        }
        maximum = warp_max(maximum);
        const float scale = fmaxf(maximum, 1e-4f) / 448.f;
        if (lane == 0) {
            if constexpr (MnMajorScales) {
                const int64_t segments = p.dim[1] / 128;
                pointer<float>(p, 2)[(group % segments) * p.dim[2] + group / segments] = scale;
            } else pointer<float>(p, 2)[group] = scale;
        }
        auto* output = pointer<uint16_t>(p, 1);
        #pragma unroll
        for (int pair = 0; pair < 2; ++pair)
            output[base / 2 + lane + pair * 32] = epilogue::quant::fp8_pair(
                values[pair * 2] / scale, values[pair * 2 + 1] / scale);
    } else if constexpr (Op == Operation::QuantizeWeightsCachedFloat ||
                         Op == Operation::QuantizeWeightsCachedInteger) {
        // One 128x128 block, with every BF16 input loaded once and retained
        // as packed registers until the block scale is known.
        const int n_blocks = p.dim[1] / 128, k_blocks = p.dim[2] / 128;
        const int group = block_index / (n_blocks * k_blocks);
        const int tile = block_index % (n_blocks * k_blocks);
        const int row_base = (tile / k_blocks) * 128, col_base = (tile % k_blocks) * 128;
        uint4 cached[8];
        float maximums[4] = {0.f, 0.f, 0.f, 0.f};
        unsigned integer_maximum = 0;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int vector = thread + slot * 256;
            const int64_t offset = (int64_t(group) * p.dim[1] + row_base + vector / 16) * p.dim[2]
                                   + col_base + (vector % 16) * 8;
            cached[slot] = *reinterpret_cast<const uint4*>(pointer<Bf16>(p, 0) + offset);
            const unsigned words[4] = {cached[slot].x, cached[slot].y, cached[slot].z, cached[slot].w};
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const unsigned word = words[pair];
                if constexpr (Op == Operation::QuantizeWeightsCachedInteger) {
                    const unsigned lo = word & 0x7fffu, hi = (word >> 16) & 0x7fffu;
                    // Match fmaxf: ignore NaNs, retain infinity and unsigned zero.
                    if (lo <= 0x7f80u && lo > integer_maximum) integer_maximum = lo;
                    if (hi <= 0x7f80u && hi > integer_maximum) integer_maximum = hi;
                } else {
                    maximums[pair] = fmaxf(maximums[pair], fabsf(__uint_as_float(word << 16)));
                    maximums[pair] = fmaxf(maximums[pair], fabsf(__uint_as_float(word & 0xffff0000u)));
                }
            }
        }
        float maximum;
        if constexpr (Op == Operation::QuantizeWeightsCachedInteger)
            maximum = __uint_as_float(integer_maximum << 16);
        else maximum = fmaxf(fmaxf(maximums[0], maximums[1]), fmaxf(maximums[2], maximums[3]));
        maximum = warp_max(maximum);
        __shared__ float maxima[8];
        if (lane == 0) maxima[warp] = maximum;
        __syncthreads();
        if (warp == 0) {
            maximum = warp_max(lane < 8 ? maxima[lane] : 0.f);
            if (lane == 0) maxima[0] = fmaxf(maximum, 1e-4f) / 448.f;
        }
        __syncthreads();
        const float scale = maxima[0];
        if (thread == 0) pointer<float>(p, 2)[block_index] = scale;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const unsigned words[4] = {cached[slot].x, cached[slot].y, cached[slot].z, cached[slot].w};
            unsigned converted[4];
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair)
                converted[pair] = epilogue::quant::fp8_pair(
                    __uint_as_float(words[pair] << 16) / scale,
                    __uint_as_float(words[pair] & 0xffff0000u) / scale);
            const uint2 packed = make_uint2(converted[0] | (converted[1] << 16),
                                            converted[2] | (converted[3] << 16));
            const int vector = thread + slot * 256;
            const int64_t offset = (int64_t(group) * p.dim[1] + row_base + vector / 16) * p.dim[2]
                                   + col_base + (vector % 16) * 8;
            *reinterpret_cast<uint2*>(pointer<uint8_t>(p, 1) + offset) = packed;
        }
    } else if constexpr (Op == Operation::QuantizeWeights) {
        // BF16 weights[G,N,K], E4M3 weights, FP32 block scales[G,N/128,K/128].
        // dims: G,N,K; N/K are 128 aligned.
        const int n_blocks = p.dim[1] / 128, k_blocks = p.dim[2] / 128;
        const int group = block_index / (n_blocks * k_blocks);
        const int tile = block_index % (n_blocks * k_blocks);
        const int row_base = (tile / k_blocks) * 128, col_base = (tile % k_blocks) * 128;
        float maximum = 0.f;
        for (int element = thread; element < 128 * 128; element += blockDim.x) {
            const int64_t offset = (int64_t(group) * p.dim[1] + row_base + element / 128) * p.dim[2] + col_base + element % 128;
            maximum = fmaxf(maximum, fabsf(__bfloat162float(pointer<Bf16>(p, 0)[offset])));
        }
        maximum = warp_max(maximum);
        __shared__ float maxima[8];
        if (lane == 0) maxima[warp] = maximum;
        __syncthreads();
        if (warp == 0) {
            maximum = warp_max(lane < blockDim.x / 32 ? maxima[lane] : 0.f);
            if (lane == 0) maxima[0] = fmaxf(maximum, 1e-4f) / 448.f;
        }
        __syncthreads();
        const float scale = maxima[0];
        if (thread == 0) pointer<float>(p, 2)[block_index] = scale;
        for (int pair = thread; pair < 128 * 64; pair += blockDim.x) {
            const int64_t offset = (int64_t(group) * p.dim[1] + row_base + pair / 64) * p.dim[2] + col_base + (pair % 64) * 2;
            pointer<uint16_t>(p, 1)[offset / 2] = epilogue::quant::fp8_pair(
                __bfloat162float(pointer<Bf16>(p, 0)[offset]) / scale,
                __bfloat162float(pointer<Bf16>(p, 0)[offset + 1]) / scale);
        }
    } else if constexpr (Op == Operation::WeightedGradient) {
        // full-input dy, full-input scores, local per-route dy.
        // dims: chunk tokens,H,top-k,input tokens,chunk start.
        if (index >= p.dim[0] * p.dim[1] * p.dim[2]) return;
        const int64_t route = index / p.dim[1], column = index % p.dim[1];
        const int64_t token = route / p.dim[2] + p.dim[4];
        float value = 0;
        if (token < p.dim[3])
            value = __bfloat162float(pointer<Bf16>(p, 0)[token * p.dim[1] + column])
                    * pointer<float>(p, 1)[token * p.dim[2] + route % p.dim[2]];
        pointer<Bf16>(p, 2)[index] = __float2bfloat16(value);
    } else if constexpr (Op == Operation::ScoreGradient) {
        // local unweighted expert outputs, full-input dy, full-input dscores.
        // dims: chunk tokens,H,top-k,input tokens,chunk start.
        const int64_t route = block_index;
        const int64_t token = route / p.dim[2] + p.dim[4];
        if (route >= p.dim[0] * p.dim[2] || token >= p.dim[3]) return;
        float value = 0;
        for (int64_t column = thread; column < p.dim[1]; column += blockDim.x)
            value += __bfloat162float(pointer<Bf16>(p, 0)[route * p.dim[1] + column])
                     * __bfloat162float(pointer<Bf16>(p, 1)[token * p.dim[1] + column]);
        value = warp_sum(value);
        __shared__ float sums[8];
        if (lane == 0) sums[warp] = value;
        __syncthreads();
        if (warp == 0) {
            value = warp_sum(lane < blockDim.x / 32 ? sums[lane] : 0.f);
            if (lane == 0) pointer<float>(p, 2)[token * p.dim[2] + route % p.dim[2]] = value;
        }
    } else if constexpr (Op == Operation::ActivationDeviceRows) {
        const int64_t rows = pointer<int32_t>(p, 2)[0];
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 3), 2);
        } else for (int64_t position = index; position < rows * p.dim[1];
                    position += int64_t(gridDim.x) * blockDim.x) {
            const int64_t column = position % p.dim[1];
            const int64_t offset = position / p.dim[1] * 2 * p.dim[1]
                + (p.dim[3] == 8 ? column / 8 * 16 + column % 8 : column);
            const int64_t up_step = p.dim[3] == 8 ? 8 : p.dim[1];
            const float gate = __bfloat162float(pointer<Bf16>(p, 0)[offset]);
            const float up = __bfloat162float(pointer<Bf16>(p, 0)[offset + up_step]);
            const float sigmoid = 1.f / (1.f + expf(-gate)), silu = gate * sigmoid;
            if (!p.dim[2]) pointer<Bf16>(p, 1)[position] = __float2bfloat16(silu * up);
            else {
                const float dh = __bfloat162float(pointer<Bf16>(p, 1)[position]);
                pointer<Bf16>(p, 0)[offset] = __float2bfloat16(dh * up * sigmoid * (1.f + gate * (1.f - sigmoid)));
                pointer<Bf16>(p, 0)[offset + up_step] = __float2bfloat16(dh * silu);
            }
        }
    } else if constexpr (Op == Operation::TransformDeviceRows || Op == Operation::TransformDeviceRowsVector) {
        const int64_t rows = pointer<int32_t>(p, 3)[0];
        constexpr int elements = Op == Operation::TransformDeviceRowsVector ? 8 : 1;
        using Value = std::conditional_t<elements == 8, uint4, Bf16>;
        const int64_t width = p.dim[1] / elements;
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 4), 2);
        } else for (int64_t position = index; position < rows * width;
                    position += int64_t(gridDim.x) * blockDim.x) {
            const int route = pointer<int32_t>(p, 1)[position / width];
            const bool valid = route >= 0 && route < p.dim[2];
            const int64_t offset = int64_t(route) * width + position % width;
            if (p.dim[3]) pointer<Value>(p, 2)[position] = valid ? pointer<Value>(p, 0)[offset] : Value{};
            else if (valid) pointer<Value>(p, 2)[offset] = pointer<Value>(p, 0)[position];
        }
    } else if constexpr (Op == Operation::ZeroDeviceRows) {
        const int64_t rows = pointer<int32_t>(p, 1)[0];
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 2), 2);
        } else for (int64_t position = index; position < rows * p.dim[1];
                    position += int64_t(gridDim.x) * blockDim.x)
            pointer<Bf16>(p, 0)[position] = __float2bfloat16(0.f);
    } else if constexpr (Op == Operation::BuildPermutationOnly) {
        if (index >= p.dim[0]) return;
        const int expert = pointer<int32_t>(p, 0)[index];
        if (expert < 0) return;
        const int row = pointer<int32_t>(p, 2)[expert] + pointer<int32_t>(p, 1)[index];
        if (row >= p.dim[1]) atomicOr(pointer<int32_t>(p, 4), 2);
        else pointer<int32_t>(p, 3)[row] = index;
    } else if constexpr (Op == Operation::PackInputDeviceRows) {
        const int64_t rows = pointer<int32_t>(p, 3)[0], width = p.dim[1] / 8;
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 4), 2);
        } else for (int64_t position = index; position < rows * width;
                    position += int64_t(gridDim.x) * blockDim.x) {
            const int route = pointer<int32_t>(p, 1)[position / width];
            uint4 value{};
            if (route >= 0) {
                const int64_t source_row = route / p.dim[3];
                if (source_row < p.dim[2]) value = pointer<uint4>(p, 0)[source_row * width + position % width];
                else atomicOr(pointer<int32_t>(p, 4), 2);
            }
            pointer<uint4>(p, 2)[position] = value;
        }
    } else if constexpr (Op == Operation::DeepEPReduceWindowRows) {
        const int64_t active_rows = pointer<int32_t>(p, 6)[0];
        if (active_rows < 0 || active_rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 7), 2);
            return;
        }
        if (active_rows == 0) return;
        const int64_t width = p.dim[1] / 8;
        for (int64_t token = block_index * 8 + warp; token < p.dim[2];
             token += int64_t(gridDim.x) * 8) {
            // Cache route positions/weights across the warp once per token.
            // The receive accumulator starts at zero, so absent/padded tokens
            // and inactive windows require no H-wide writes.
            int packed_row = -1;
            float weight = 1.f;
            if (lane < p.dim[3]) {
                const int64_t route = token * p.dim[3] + lane;
                const int expert = pointer<int32_t>(p, 1)[route];
                if (expert >= p.dim[4]) atomicOr(pointer<int32_t>(p, 7), 2);
                else if (expert >= 0) {
                    const int64_t global_row = int64_t(pointer<int32_t>(p, 3)[expert])
                                               + pointer<int32_t>(p, 2)[route];
                    if (global_row < 0 || global_row >= p.dim[7]) atomicOr(pointer<int32_t>(p, 7), 2);
                    else if (global_row >= p.dim[5] && global_row < p.dim[5] + active_rows) {
                        packed_row = global_row - p.dim[5];
                        if (p.dim[6]) weight = pointer<float>(p, 4)[route];
                    }
                }
            }
            const uint32_t mask = __ballot_sync(0xffffffffu, packed_row >= 0);
            if (!mask) continue;
            for (int64_t column = lane; column < width; column += 32) {
                float sum[8]{};
                for (uint32_t pending = mask; pending; pending &= pending - 1) {
                    const int slot = __ffs(pending) - 1;
                    const int row = __shfl_sync(0xffffffffu, packed_row, slot);
                    const float score = __shfl_sync(0xffffffffu, weight, slot);
                    const uint4 raw = pointer<uint4>(p, 0)[int64_t(row) * width + column];
                    const auto* values = reinterpret_cast<const Bf16*>(&raw);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) sum[j] += __bfloat162float(values[j]) * score;
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    pointer<float>(p, 5)[token * p.dim[1] + column * 8 + j] += sum[j];
            }
        }
    } else if constexpr (Op == Operation::DeepEPReduceRows) {
        // Invert the route layout from count/prefix metadata. This reduces
        // top-k in registers and writes BF16 directly, without FP32 atomics,
        // a zeroed FP32 receive matrix, or a separate conversion pass.
        const int64_t width = p.dim[1] / 8;
        for (int64_t position = index; position < p.dim[2] * width;
             position += int64_t(gridDim.x) * blockDim.x) {
            const int64_t token = position / width, column = position % width;
            float sum[8]{};
            for (int slot = 0; slot < p.dim[3]; ++slot) {
                const int64_t route = token * p.dim[3] + slot;
                const int expert = pointer<int32_t>(p, 1)[route];
                if (expert < 0) continue;
                if (expert >= p.dim[4]) {
                    atomicOr(pointer<int32_t>(p, 6), 2);
                    continue;
                }
                const int row = pointer<int32_t>(p, 3)[expert] + pointer<int32_t>(p, 2)[route];
                if (row < 0 || row >= p.dim[0]) {
                    atomicOr(pointer<int32_t>(p, 6), 2);
                    continue;
                }
                const uint4 raw = pointer<uint4>(p, 0)[int64_t(row) * width + column];
                const auto* values = reinterpret_cast<const Bf16*>(&raw);
                const float weight = p.dim[5] ? pointer<float>(p, 4)[route] : 1.f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) sum[j] += __bfloat162float(values[j]) * weight;
            }
            uint4 raw{};
            auto* values = reinterpret_cast<Bf16*>(&raw);
            #pragma unroll
            for (int j = 0; j < 8; ++j) values[j] = __float2bfloat16(sum[j]);
            pointer<uint4>(p, 5)[position] = raw;
        }
    } else if constexpr (Op == Operation::DeepEPBackwardRows) {
        // Gather DY, apply router weights, and reduce dScore without a
        // [receive_tokens, topk, hidden] expanded temporary.
        const int64_t rows = pointer<int32_t>(p, 6)[0];
        if (rows < 0 || rows > p.dim[0]) {
            if (index == 0) atomicOr(pointer<int32_t>(p, 7), 2);
            return;
        }
        for (int64_t row = block_index * 8 + warp; row < rows;
             row += int64_t(gridDim.x) * 8) {
            const int route = pointer<int32_t>(p, 2)[row];
            const bool valid = route >= 0 && route < p.dim[2] * p.dim[3];
            if (route >= p.dim[2] * p.dim[3] && lane == 0)
                atomicOr(pointer<int32_t>(p, 7), 2);
            const float weight = valid ? pointer<float>(p, 1)[route] : 0.f;
            float dot = 0.f;
            for (int64_t col = lane; col < p.dim[1]; col += 32) {
                const float dy = valid ? __bfloat162float(pointer<Bf16>(p, 0)[
                    int64_t(route / p.dim[3]) * p.dim[1] + col]) : 0.f;
                pointer<Bf16>(p, 4)[row * p.dim[1] + col] = __float2bfloat16(dy * weight);
                if (valid) dot += dy * __bfloat162float(pointer<Bf16>(p, 3)[row * p.dim[1] + col]);
            }
            dot = warp_sum(dot);
            if (valid && lane == 0) pointer<float>(p, 5)[route] = dot;
        }
    } else if constexpr (Op == Operation::TransposeWeights) {
        // Two independent expert-weight transposes share one launch. Transfer
        // BF16 bits in aligned vectors. XOR vector columns avoid a single
        // repeated TSM column for the transpose's strided lane accesses.
        constexpr int tile_size = 128;
        __shared__ __align__(16) uint4 tile[tile_size][tile_size / 8];
        const int64_t first_tiles = p.dim[0] * (2 * p.dim[2] / tile_size) * (p.dim[1] / tile_size);
        const bool first = block_index < first_tiles;
        const int64_t rows = first ? 2 * p.dim[2] : p.dim[1];
        const int64_t columns = first ? p.dim[1] : p.dim[2];
        const int64_t tile_index = first ? block_index : block_index - first_tiles;
        const int64_t column_tiles = columns / tile_size;
        const int64_t tiles_per_group = (rows / tile_size) * column_tiles;
        const int64_t group = tile_index / tiles_per_group;
        const int64_t local_tile = tile_index % tiles_per_group;
        const int64_t tile_row = local_tile / column_tiles, tile_column = local_tile % column_tiles;
        const auto* source = pointer<uint4>(p, first ? 0 : 1);
        auto* destination = pointer<uint4>(p, first ? 2 : 3);
        #pragma unroll
        for (int vector = thread; vector < tile_size * tile_size / 8; vector += 256) {
            const int row = vector / (tile_size / 8), column = vector % (tile_size / 8) * 8;
            const int64_t offset = ((group * rows + tile_row * tile_size + row) * columns
                                   + tile_column * tile_size + column) / 8;
            tile[row][(column / 8) ^ (row / 8)] = source[offset];
        }
        __syncthreads();
        #pragma unroll
        for (int vector = thread; vector < tile_size * tile_size / 8; vector += 256) {
            const int row = vector / (tile_size / 8), column = vector % (tile_size / 8) * 8;
            uint32_t words[4];
            const auto read_bits = [&](int source_row) {
                const uint4 value = tile[source_row][(row / 8) ^ (source_row / 8)];
                const int part = (row % 8) / 2;
                const uint32_t word = part == 0 ? value.x : part == 1 ? value.y : part == 2 ? value.z : value.w;
                return (word >> (16 * (row & 1))) & 0xffffu;
            };
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair)
                words[pair] = read_bits(column + 2 * pair) | (read_bits(column + 2 * pair + 1) << 16);
            const int64_t offset = ((group * columns + tile_column * tile_size + row) * rows
                                   + tile_row * tile_size + column) / 8;
            destination[offset] = make_uint4(words[0], words[1], words[2], words[3]);
        }
    } else if constexpr (Op == Operation::ReadChunk) {
        // full x, full IDs, chunk x, chunk IDs, valid token count, status.
        // dims: input capacity, chunk tokens,H,top-k,chunk start.
        const int64_t valid = pointer<int32_t>(p, 4)[0];
        if (index == 0 && (valid < 0 || valid > p.dim[0])) atomicOr(pointer<int32_t>(p, 5), 4);
        const int64_t available = valid < 0 ? 0 : valid > p.dim[0] ? p.dim[0] : valid;
        if (index < p.dim[1] * p.dim[2]) {
            const int64_t token = index / p.dim[2] + p.dim[4];
            pointer<Bf16>(p, 2)[index] = token < available
                ? pointer<Bf16>(p, 0)[token * p.dim[2] + index % p.dim[2]] : __float2bfloat16(0.f);
        }
        if (index < p.dim[1] * p.dim[3]) {
            const int64_t token = index / p.dim[3] + p.dim[4];
            pointer<int64_t>(p, 3)[index] = token < available
                ? pointer<int64_t>(p, 1)[token * p.dim[3] + index % p.dim[3]] : -1;
        }
    }
}

}  // namespace deep_gemm::training
