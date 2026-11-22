"""Isolated Tensor Cell measurement; observed throughput, not a hardware ceiling."""
from pathlib import Path
import hashlib,json,subprocess
root=Path.cwd()
source=r'''
#include <hggc_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <deep_gemm/common/fused_gemm_common.cuh>
#include <cute/ppu_tensor_mix.hpp>
#include <cutlass/gemm/config/gemm_operands.hpp>
#include <cute/atom/mma_traits_ppu0015.hpp>
using namespace cute;
#define CHECK(call) do {auto status=(call);if(status!=hggcSuccess){fprintf(stderr,"%s: %s\n",#call,hggcGetErrorString(status));exit(1);}}while(0)
template<int BM,int BN,int WM,int WN>
__launch_bounds__(32*(BM/WM)*(BN/WN),1)
__global__ void register_mma(float* output,int steps,int aval,int bval){
 using Src=cutlass::bfloat16_t;
 using Inst=typename cutlass::gemm::config::GetAiuMmaInst<cutlass::arch::PPU0015,Src,Src,float>::type;
 using MMA=TiledMMA<MMA_Atom<Inst>,Layout<Shape<Int<BM/WM>,Int<BN/WN>,_1>>>;
 extern __shared__ __align__(128) unsigned char scratch[];
 // The reservation is identical to the stage3 production WG. Only this
 // launch prologue accesses TSM; the timed MMA loop consumes registers.
 if(threadIdx.x==0)scratch[0]=static_cast<unsigned char>(aval);
 __syncthreads();
 auto shape_a=make_shape(Int<BM>{},Int<64>{});
 auto shape_b=make_shape(Int<BN>{},Int<64>{});
 auto sa=make_tensor(make_smem_ptr(reinterpret_cast<Src*>(scratch)),shape_a,make_stride(_1{},Int<BM>{}));
 auto sb=make_tensor(make_smem_ptr(reinterpret_cast<Src*>(scratch)),shape_b,make_stride(_1{},Int<BN>{}));
 MMA mma;
 auto thread_mma=mma.get_thread_slice(threadIdx.x);
 Tensor a=thread_mma.partition_fragment_A(sa);
 Tensor b=thread_mma.partition_fragment_B(sb);
 Tensor acc=partition_fragment_C(mma,make_shape(Int<BM>{},Int<BN>{}));
 fill(a,Src(float(aval)));fill(b,Src(float(bval)));clear(acc);
 CUTLASS_PRAGMA_NO_UNROLL
 for(int step=0;step<steps;++step){
  CUTLASS_PRAGMA_UNROLL
  for(int kb=0;kb<size<2>(a);++kb)
   cute::gemm(mma,acc,a(_,_,kb),b(_,_,kb),acc);
 }
 CUTLASS_PRAGMA_UNROLL
 for(int j=0;j<size(acc);++j)
  output[(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*size(acc)+j]=acc(j);
}
template<int BM,int BN,int WM,int WN>
void measure(int cu,const char* chip){
 constexpr int threads=32*(BM/WM)*(BN/WN),shared=196608,steps=4096;
 auto kernel=register_mma<BM,BN,WM,WN>;
 CHECK(hggcFuncSetAttribute(kernel,hggcFuncAttributeMaxDynamicSharedMemorySize,shared));
 hggcFuncAttributes attr;CHECK(hggcFuncGetAttributes(&attr,kernel));
 size_t elements=size_t(cu)*BM*BN;
 float* out;CHECK(hggcMalloc(reinterpret_cast<void**>(&out),elements*sizeof(float)));
 std::vector<float> host(elements);
 auto launch=[&](int a,int b){register_mma<BM,BN,WM,WN><<<cu,threads,shared>>>(out,steps,a,b);CHECK(hggcGetLastError());};
 for(auto seeds: {std::pair<int,int>{1,1},{2,3},{3,1}}){
  CHECK(hggcMemset(out,0xa5,elements*sizeof(float)));launch(seeds.first,seeds.second);
  CHECK(hggcMemcpy(host.data(),out,elements*sizeof(float),hggcMemcpyDeviceToHost));
  float expected=float(steps*64*seeds.first*seeds.second);
  for(size_t j=0;j<elements;++j)if(host[j]!=expected){fprintf(stderr,"MMA checksum mismatch at %zu: %g vs %g\n",j,host[j],expected);exit(2);}
 }
 for(int k=0;k<5;++k)launch(1,1);
 CHECK(hggcDeviceSynchronize());
 std::vector<float> samples;
 for(int repeat=0;repeat<5;++repeat){
  hggcEvent_t begin,end;CHECK(hggcEventCreate(&begin));CHECK(hggcEventCreate(&end));
  CHECK(hggcEventRecord(begin));
  for(int iteration=0;iteration<20;++iteration)launch(1,1);
  CHECK(hggcEventRecord(end));CHECK(hggcEventSynchronize(end));
  float ms;CHECK(hggcEventElapsedTime(&ms,begin,end));samples.push_back(ms/20);
  CHECK(hggcEventDestroy(begin));CHECK(hggcEventDestroy(end));
 }
 CHECK(hggcMemcpy(host.data(),out,elements*sizeof(float),hggcMemcpyDeviceToHost));
 for(float value:host)if(value!=float(steps*64)){fprintf(stderr,"Post-timing checksum failed\n");exit(3);}
 auto sorted=samples;std::sort(sorted.begin(),sorted.end());
 double flops=2.0*cu*BM*BN*64*steps,ms=sorted[2];
 printf("{\"case\":\"register_bf16_compute_diagnostic\",\"chip\":\"%s\",\"active_cus\":%d,\"block_m\":%d,\"block_n\":%d,\"warp_m\":%d,\"warp_n\":%d,\"threads\":%d,\"registers\":%d,\"local_bytes\":%zu,\"dynamic_shared_bytes\":%d,\"steps\":%d,\"every_output_exact\":true,\"fresh_seed_pairs\":3,\"median_event_ms\":%.9f,\"observed_tflops\":%.9f,\"samples_ms\":[%.9f,%.9f,%.9f,%.9f,%.9f],\"scope\":\"Register-resident BF16/FP32 MMA plus launch prologue and result store; no AIU mainloop or networking; measured throughput is not an upper bound\"}\n",
 chip,cu,BM,BN,WM,WN,threads,attr.numRegs,attr.localSizeBytes,shared,steps,ms,flops/(ms*1e9),samples[0],samples[1],samples[2],samples[3],samples[4]);fflush(stdout);
 CHECK(hggcFree(out));
}
int main(){
 hggcDeviceProp prop;CHECK(hggcGetDeviceProperties(&prop,0));
 measure<256,256,64,64>(prop.multiProcessorCount,prop.name);
 measure<128,256,64,64>(prop.multiProcessorCount,prop.name);
 measure<256,256,32,128>(prop.multiProcessorCount,prop.name);
}
'''
path=root/'build/campaign/bf16-register-throughput-r2.cpp';path.write_text(source)
binary=path.with_suffix('')
command=['/usr/local/PPU_SDK/bin/hgcc','-std=c++17','-O3','-arch=ppu_15','-x','cu',
 '-I'+str(root/'third-party/actlize_v1.0.0/include'),'-I'+str(root/'deep_gemm/include'),
 '-Xllvm','-ppu-patch-fence-ppu=false','-Xllvm','-ppu-simt-branch=false',
 str(path),'-o',str(binary)]
print(json.dumps(dict(case='register_bf16_diagnostic_build',source_sha256=hashlib.sha256(source.encode()).hexdigest(),command=command)),flush=True)
subprocess.run(command,check=True)
subprocess.run([str(binary)],check=True)
