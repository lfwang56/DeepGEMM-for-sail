"""Independent CPU FP32 and dynamic graph checks for full-width wgrad tiles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--library')
a = p.parse_args()
if a.library:
    sys.path.insert(0,str(Path(a.library).resolve()))
import torch
import deep_gemm as dg

torch.cuda.set_device(0)
torch.set_num_threads(2)
torch.manual_seed(192831)
torch.backends.cuda.matmul.allow_tf32 = False
capacity, experts = 65536,16
print(json.dumps(dict(case='wgrad_geometry_library',extension=dg._C.__file__,
    sha256=hashlib.sha256(Path(dg._C.__file__).read_bytes()).hexdigest())),flush=True)
for role,m,n in (('dw1',4096,8192),('dw2',8192,2048)):
    left = torch.zeros(capacity,m,device='cuda',dtype=torch.bfloat16)
    right = torch.zeros(capacity,n,device='cuda',dtype=torch.bfloat16)
    out = torch.zeros(experts,m,n,device='cuda',dtype=torch.float32)
    lengths = torch.zeros(experts,device='cuda',dtype=torch.int32)
    # Compile before capture; the GPU lengths remain mutable graph inputs.
    dg.k_grouped_bf16_gemm_tn_device(left,right,out,lengths,capacity)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dg.k_grouped_bf16_gemm_tn_device(left,right,out,lengths,capacity)
    patterns = ([128,0,256,128]+[0]*12, [0,2560]+[0]*14)
    for pattern,ks in enumerate(patterns):
        count = sum(ks)
        lc = (torch.randn(count,m)*.1).bfloat16()
        rc = (torch.randn(count,n)*.1).bfloat16()
        left[:count].copy_(lc)
        right[:count].copy_(rc)
        lengths.copy_(torch.tensor(ks,dtype=torch.int32))
        # Nonzero initial values exercise beta=1; two replays also verify that
        # accumulated FP32 outputs and dynamic empty groups remain correct.
        out.fill_(.03125)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        begin = 0
        errors = []
        for group,k in enumerate(ks):
            if not k:
                assert bool((out[group]==.03125).all()),(role,pattern,group)
                continue
            expected = (lc[begin:begin+k].float().t() @ rc[begin:begin+k].float())*2+.03125
            actual = out[group].cpu()
            difference = actual-expected
            error = float(difference.square().sum().sqrt()/expected.square().sum().sqrt())
            assert bool(torch.isfinite(actual).all()) and error<2e-4,(role,pattern,group,error)
            errors.append(error)
            begin += k
        print(json.dumps(dict(case='wgrad_geometry_reference',role=role,pattern=pattern,
            lengths=ks,capacity=capacity,relative_rms=errors,passed=True)),flush=True)
    del graph,left,right,out,lengths
    torch.cuda.empty_cache()
