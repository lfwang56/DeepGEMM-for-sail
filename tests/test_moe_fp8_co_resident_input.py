"""Resource-guarded full-grid input overlap with exact serialized controls."""
import gc
import json
import torch
import deep_gemm as dg
from test_moe_peer_quantized_tile_publication import fixture as producer_fixture
from test_moe_peer_fp8_input import fixture as joint_fixture


def resources(elastic=False):
    g,r,h,n=16,65536,8192,4096
    dims=[r,h,4096,10,32,128,dg.get_num_sms(),1]
    producer=dg._C._moe_training_peer_input_occupancy(0,dims)
    assert producer>=2
    a=torch.empty(r,h,device='cuda',dtype=torch.float8_e4m3fn)
    sa=torch.empty(h//128,r,device='cuda')
    b=torch.empty(g,n,h,device='cuda',dtype=torch.float8_e4m3fn)
    sb=torch.empty(g,n//128,h//128,device='cuda')
    out=torch.empty(r,n,device='cuda',dtype=torch.bfloat16)
    lengths=torch.zeros(g,device='cuda',dtype=torch.int32)
    permutation=torch.full((r,),-1,device='cuda',dtype=torch.int32)
    destinations=torch.tensor([out.data_ptr()],device='cuda',dtype=torch.int64)
    ready=torch.zeros(r//128,device='cuda',dtype=torch.int32)
    status=torch.zeros(1,device='cuda',dtype=torch.int32)
    counter=torch.zeros_like(status)
    def compile(config):
        return dg._C._moe_training_peer_fp8_gemm((a,sa.t()),(b,sb),out,lengths,permutation,
            destinations,r,keep_local=True,compile_only=True,config=config,input_control=[ready,status]+([counter] if elastic else []),
            reserved_input_ctas=dg.get_num_sms(),co_resident_input=True)
    rejected=False
    try:compile([])
    except RuntimeError as error:
        if 'at least two resident CTAs' not in str(error):raise
        rejected=True
    assert rejected,'unsafe default geometry must be rejected before launch'
    grid=compile([128,128,32,64,3])
    assert grid==dg.get_num_sms()
    if elastic:
        lengths[0]=-128
        out.fill_(-7)
        dg._C._moe_training_peer_fp8_gemm((a,sa.t()),(b,sb),out,lengths,permutation,
            destinations,r,keep_local=True,config=[128,128,32,64,3],input_control=[ready,status,counter],
            reserved_input_ctas=dg.get_num_sms(),co_resident_input=True)
        torch.cuda.synchronize()
        assert status.item()==2 and torch.all(out==-7)
        lengths.zero_();status.zero_();counter.zero_()
        dg._C._moe_training_peer_fp8_gemm((a,sa.t()),(b,sb),out,lengths,permutation,
            destinations,r,keep_local=True,config=[128,128,32,64,3],input_control=[ready,status,counter],
            reserved_input_ctas=dg.get_num_sms(),co_resident_input=True)
        torch.cuda.synchronize()
        assert status.item()==0 and counter.item()==dg.get_num_sms() and torch.all(out==-7)
    print(json.dumps(dict(case='fp8_co_resident_resource_guard',producer_blocks_per_cu=producer,
        consumer_minimum_blocks_per_cu=2,unsafe_default_rejected_before_launch=rejected,
        producer_grid=dg.get_num_sms(),consumer_grid=grid,elastic_workers=elastic,
        invalid_layout_guard_and_retry=elastic,passed=True)),flush=True)


def main():
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(116116)
    resources()
    gc.collect();torch.cuda.synchronize()
    for width in (128,512,8192):producer_fixture(width,co_resident=True)
    config=[128,128,32,64,3]
    joint_fixture(4,512,512,2048,256,[[128,256,384,512],[0,128,512,128],[0,0,0,0],[512]*4],
                  ctas=dg.get_num_sms(),threads=256,bf16_source=True,co_resident=True,config=config)
    joint_fixture(16,4096,8192,65536,4096,[[0]*16,[2560]*16],benchmark=True,
                  ctas=dg.get_num_sms(),threads=256,bf16_source=True,co_resident=True,config=config)

if __name__=='__main__':main()
