"""Finite subgroup publication proof for proposed in-CTA input helpers."""
from pathlib import Path
import hashlib,json,subprocess
root=Path.cwd()
source=r'''
#include <hggc_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(call) do {auto s=(call);if(s!=hggcSuccess){fprintf(stderr,"%s: %s\n",#call,hggcGetErrorString(s));exit(1);}}while(0)
__host__ __device__ uint32_t expected(uint32_t tag,uint32_t block,uint32_t phase,uint32_t word){
 return tag ^ (word*2654435761u+phase*2246822519u+block*3266489917u);
}
template<int HelperThreads>
__global__ void subgroup_pipeline(uint32_t* payload,uint32_t* errors,int phases,int words,uint32_t tag){
 constexpr int Compute=512,Total=Compute+HelperThreads;
 __shared__ uint32_t scratch[Compute];
 if(threadIdx.x>=Compute){
  const int helper=threadIdx.x-Compute;
  for(int phase=0;phase<phases;++phase){
   auto* dst=payload+(size_t(blockIdx.x)*phases+phase)*words;
   for(int word=helper;word<words;word+=HelperThreads)
    dst[word]=expected(tag,blockIdx.x,phase,word);
   // Publish each helper's global stores before the whole team handoff.
   __threadfence();
   __ppu_barrier_sync(1,Total);
  }
  return;
 }
 for(int phase=0;phase<phases;++phase){
  __ppu_barrier_sync(1,Total);
  auto* src=payload+(size_t(blockIdx.x)*phases+phase)*words;
  uint32_t bad=0;
  for(int word=threadIdx.x;word<words;word+=Compute)
   bad+=src[word]!=expected(tag,blockIdx.x,phase,word);
  // Reuse consumer-only TSM independently of helpers prefetching next phase.
  for(int round=0;round<8;++round){
   scratch[threadIdx.x]=src[threadIdx.x]+round;
   __ppu_barrier_sync(0,Compute);
   const int neighbour=(threadIdx.x+17)%Compute;
   bad+=scratch[neighbour]!=expected(tag,blockIdx.x,phase,neighbour)+round;
   __ppu_barrier_sync(0,Compute);
  }
  if(bad)atomicAdd(errors+blockIdx.x,bad);
 }
}
template<int Helpers>
void run(int blocks,int phases,uint32_t tag){
 constexpr int Words=4096,shared=196608;
 const size_t count=size_t(blocks)*phases*Words;
 uint32_t *allocation,*errors;
 CHECK(hggcMalloc(reinterpret_cast<void**>(&allocation),(count+2)*sizeof(uint32_t)));
 CHECK(hggcMalloc(reinterpret_cast<void**>(&errors),blocks*sizeof(uint32_t)));
 CHECK(hggcMemset(allocation,0xa5,(count+2)*sizeof(uint32_t)));
 CHECK(hggcMemset(errors,0,blocks*sizeof(uint32_t)));
 CHECK(hggcFuncSetAttribute(subgroup_pipeline<Helpers>,hggcFuncAttributeMaxDynamicSharedMemorySize,shared));
 subgroup_pipeline<Helpers><<<blocks,512+Helpers,shared>>>(allocation+1,errors,phases,Words,tag);
 CHECK(hggcGetLastError());
 std::vector<uint32_t> values(count+2),status(blocks);
 CHECK(hggcMemcpy(values.data(),allocation,(count+2)*sizeof(uint32_t),hggcMemcpyDeviceToHost));
 CHECK(hggcMemcpy(status.data(),errors,blocks*sizeof(uint32_t),hggcMemcpyDeviceToHost));
 if(values.front()!=0xa5a5a5a5u||values.back()!=0xa5a5a5a5u){fprintf(stderr,"Canary changed\n");exit(2);}
 for(auto x:status)if(x){fprintf(stderr,"GPU publication mismatch\n");exit(3);}
 for(int block=0;block<blocks;++block)for(int phase=0;phase<phases;++phase)for(int word=0;word<Words;++word){
  size_t index=(size_t(block)*phases+phase)*Words+word;
  if(values[index+1]!=expected(tag,block,phase,word)){fprintf(stderr,"CPU reference mismatch\n");exit(4);}
 }
 hggcFuncAttributes attr;CHECK(hggcFuncGetAttributes(&attr,subgroup_pipeline<Helpers>));
 printf("{\"case\":\"ppu_subgroup_pipeline_reference\",\"blocks\":%d,\"compute_threads\":512,\"helper_threads\":%d,\"total_threads\":%d,\"phases\":%d,\"tag\":%u,\"payload_words_checked\":%zu,\"registers\":%d,\"local_bytes\":%zu,\"dynamic_shared_bytes\":%d,\"gpu_and_cpu_reference_exact\":true,\"canaries_exact\":true,\"passed\":true}\n",blocks,Helpers,512+Helpers,phases,tag,count,attr.numRegs,attr.localSizeBytes,shared);fflush(stdout);
 CHECK(hggcFree(errors));CHECK(hggcFree(allocation));
}
int main(){
 for(uint32_t tag:{551197u,551229u,551263u})
  for(int phases:{1,2,7,31})for(int blocks:{1,39}){
   run<32>(blocks,phases,tag);run<64>(blocks,phases,tag);
  }
}
'''
path=root/'build/campaign/ppu-subgroup-pipeline-r2.cpp';path.write_text(source)
binary=path.with_suffix('')
command=['/usr/local/PPU_SDK/bin/hgcc','-std=c++17','-O3','-arch=ppu_15','-x','cu',
 '-Xllvm','-ppu-patch-fence-ppu=false','-Xllvm','-ppu-simt-branch=false',str(path),'-o',str(binary)]
print(json.dumps(dict(case='ppu_subgroup_pipeline_build',source_sha256=hashlib.sha256(source.encode()).hexdigest(),command=command)),flush=True)
subprocess.run(command,check=True);subprocess.run([str(binary)],check=True,timeout=90)
