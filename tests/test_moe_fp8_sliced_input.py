"""Hidden-dimension stripes publish one release increment per completed stripe."""
import gc
import json
import torch
import deep_gemm as dg
from test_moe_fp8_co_resident_input import resources
from test_moe_peer_quantized_tile_publication import fixture as producer_fixture
from test_moe_peer_fp8_input import fixture as joint_fixture


def schema_guards():
    r,h,n=128,256,128
    x=torch.zeros(r,h,device='cuda',dtype=torch.bfloat16)
    table=torch.tensor([x.data_ptr()],device='cuda',dtype=torch.int64)
    perm=torch.zeros(r,device='cuda',dtype=torch.int32)
    status=torch.zeros(1,device='cuda',dtype=torch.int32)
    blocks=torch.zeros_like(status);ready=torch.zeros_like(status)
    q=torch.empty(r,h,device='cuda',dtype=torch.float8_e4m3fn)
    sf=torch.empty(h//128,r,device='cuda')
    w=torch.empty(1,n,h,device='cuda',dtype=torch.float8_e4m3fn)
    sw=torch.empty(1,n//128,h//128,device='cuda')
    out=torch.empty(r,n,device='cuda',dtype=torch.bfloat16)
    lengths=torch.zeros(1,device='cuda',dtype=torch.int32)
    rejected=[]
    for slices in (0,3,4,32):
        try:
            dg._C._moe_training_peer_launch(25,[table,perm,x,status,blocks,q,sf,ready],
                [r,h,r,1,1,128,dg.get_num_sms(),2,slices],True)
        except RuntimeError:rejected.append(('producer',slices))
        else:raise AssertionError(('producer accepted invalid slices',slices))
        try:
            dg._C._moe_training_peer_fp8_gemm((q,sf.t()),(w,sw),out,lengths,perm,table,r,
                compile_only=True,input_control=[ready,status],reserved_input_ctas=dg.get_num_sms(),
                co_resident_input=True,config=[128,128,32,64,3],input_ready_count=slices)
        except RuntimeError:rejected.append(('consumer',slices))
        else:raise AssertionError(('consumer accepted invalid slices',slices))
    for slices in (1,2):
        occ=dg._C._moe_training_peer_input_occupancy(0,[r,h,r,1,1,128,dg.get_num_sms(),2,slices])
        assert occ>=2
    print(json.dumps(dict(case='fp8_sliced_input_schema_guard',rejected=rejected,
                         valid_slices=[1,2],passed=True)),flush=True)


def main():
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(118118)
    schema_guards();resources(elastic=True)
    gc.collect();torch.cuda.synchronize()
    for width,slices in ((128,1),(512,2),(512,4),(8192,8),(8192,16)):
        producer_fixture(width,co_resident=True,elastic_workers=True,slices=slices)
    config=[128,128,32,64,3]
    joint_fixture(4,512,512,2048,256,[[128,256,384,512],[0,128,512,128],[0,0,0,0],[512]*4],
        ctas=4*dg.get_num_sms(),threads=256,bf16_source=True,co_resident=True,config=config,
        elastic_workers=True,slices=2)
    for multiplier in (1,2,4):
        for slices in (1,2,4,8,16):
            joint_fixture(16,4096,8192,65536,4096,[[0]*16,[2560]*16],benchmark=True,
                ctas=multiplier*dg.get_num_sms(),threads=256,bf16_source=True,co_resident=True,
                config=config,elastic_workers=True,slices=slices)
            gc.collect();torch.cuda.synchronize()

if __name__=='__main__':main()
