"""Producer resources become additional GEMM workers after input preparation."""
import gc
import torch
import deep_gemm as dg
from test_moe_fp8_co_resident_input import resources
from test_moe_peer_quantized_tile_publication import fixture as producer_fixture
from test_moe_peer_fp8_input import fixture as joint_fixture

torch.cuda.set_device(0)
torch.set_num_threads(2)
torch.manual_seed(117117)
resources(elastic=True)
gc.collect();torch.cuda.synchronize()
for width in (128,512,8192):producer_fixture(width,co_resident=True,elastic_workers=True)
config=[128,128,32,64,3]
joint_fixture(4,512,512,2048,256,[[128,256,384,512],[0,128,512,128],[0,0,0,0],[512]*4],
    ctas=4*dg.get_num_sms(),threads=256,bf16_source=True,co_resident=True,config=config,elastic_workers=True)
for multiplier in (1,2,4):
    joint_fixture(16,4096,8192,65536,4096,[[0]*16,[2560]*16],benchmark=True,
        ctas=multiplier*dg.get_num_sms(),threads=256,bf16_source=True,co_resident=True,config=config,elastic_workers=True)
    gc.collect();torch.cuda.synchronize()
