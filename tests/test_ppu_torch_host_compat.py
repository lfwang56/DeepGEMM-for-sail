"""Exercise native BLAS layouts/dtypes across the Torch CUDA-compat boundary."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import deep_gemm as dg

p = argparse.ArgumentParser()
p.add_argument('--expect-managed-rejection', action='store_true')
a = p.parse_args()
torch.cuda.set_device(0)
torch.set_num_threads(2)
torch.manual_seed(61921)
torch.backends.cuda.matmul.allow_tf32 = False
m, n, k = 128, 96, 256
for dtype in (torch.bfloat16, torch.float16, torch.float32):
    left = (torch.randn(m,k,device='cuda')*.1).to(dtype)
    right = (torch.randn(k,n,device='cuda')*.1).to(dtype)
    for orientation in ('nn','nt','tn','tt'):
        lhs = left if orientation[0]=='n' else left.t().contiguous()
        rhs = right if orientation[1]=='n' else right.t().contiguous()
        for accumulate in (False,True):
            # The public DeepGEMM BLAS wrapper permits BF16 or FP32 output.
            output_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
            out = torch.empty(m,n,device='cuda',dtype=output_dtype)
            initial = (torch.randn_like(out)*.01) if accumulate else None
            getattr(dg._C,'acblaslt_gemm_'+orientation)(lhs,rhs,out,initial)
            expected = left.float() @ right.float()
            if initial is not None:
                expected.add_(initial.float())
            error = float((out.float()-expected).square().sum().sqrt()/expected.square().sum().sqrt())
            limit = .008 if dtype==torch.bfloat16 else .003
            assert torch.isfinite(out).all() and error<limit,(dtype,orientation,accumulate,error)
            print(json.dumps(dict(case='native_acblaslt_dtype_layout',dtype=str(dtype),
                output_dtype=str(output_dtype),
                orientation=orientation,accumulate=accumulate,relative_rms=error,passed=True)),flush=True)
if a.expect_managed_rejection:
    library = str(Path(dg.__file__).resolve().parent.parent)
    program = 'import sys; sys.path.insert(0,'+repr(library)+'); import deep_gemm; deep_gemm.get_num_sms()'
    result = subprocess.run([sys.executable,'-c',program],text=True,capture_output=True,
                            env=dict(os.environ,DG_USE_PYTORCH_ACBLASLT_HANDLE='1'))
    output = result.stdout+result.stderr
    assert result.returncode and 'owned native acBLASLt handle' in output,(result.returncode,output[-1200:])
    print(json.dumps(dict(case='incompatible_managed_handle_rejected',passed=True,
                         child_returncode=result.returncode)),flush=True)
