"""Bitwise quantized-weight transpose, mutable graphs, and complete prep timing."""
import json
import math
import torch
import deep_gemm as dg
from deep_gemm.training.native import _launch


def guarded(shape, dtype):
    size=math.prod(shape)
    raw=torch.full((size+32,),113,dtype=torch.uint8 if dtype==torch.float8_e4m3fn else torch.int32,device='cuda')
    view=raw[16:16+size].view(dtype).reshape(shape)
    return raw,view


def check_bits(e,h,i):
    shapes=((e,2*i,h),(e,h,i))
    src=[];dest=[];guards=[]
    for n,k in ((2*i,h),(h,i)):
        data=torch.randint(0,256,(e,n,k),device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn)
        # Include arbitrary SF bits, signed zero, infinities and NaN payloads.
        sf=torch.randint(-(2**31),2**31-1,(e,n//128,k//128),device='cuda',dtype=torch.int32).view(torch.float32)
        dq,q=guarded((e,k,n),torch.float8_e4m3fn);ds,s=guarded((e,k//128,n//128),torch.float32)
        src.extend((data,sf));dest.extend((q,s));guards.extend((dq,ds))
    def launch():_launch(35,[*src,*dest],[e,h,i])
    launch();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):launch()
    for change in range(3):
        for index,value in enumerate(src):
            if index%2==0:value.view(torch.uint8).random_(0,256)
            else:value.view(torch.int32).random_(-(2**31),2**31-1)
        graph.replay();torch.cuda.synchronize()
        for source,output in zip(src,dest):
            raw=torch.uint8 if source.dtype==torch.float8_e4m3fn else torch.int32
            assert torch.equal(output.view(raw),source.view(raw).transpose(1,2).contiguous())
        for raw in guards:
            assert bool(torch.all(raw[:16]==113)) and bool(torch.all(raw[-16:]==113))
    print(json.dumps(dict(case='fp8_weight_transpose_bits',experts=e,hidden=h,intermediate=i,graph=True,mutable_replays=3,every_byte=True,special_scale_bits=True,canaries=True,passed=True)),flush=True)


def check_quantization(operation):
    e,h,i=2,512,256
    weights=[(torch.randn(e,2*i,h,device='cuda')*.05).bfloat16(),(torch.randn(e,h,i,device='cuda')*.07).bfloat16()]
    weights[0][:,0,0]=0;weights[1][0].zero_()
    quant=[];result=[];reference=[]
    for w in weights:
        g,n,k=w.shape
        quant.extend((torch.empty_like(w,dtype=torch.float8_e4m3fn),torch.empty(g,n//128,k//128,device='cuda')))
        result.extend((torch.empty(g,k,n,device='cuda',dtype=torch.float8_e4m3fn),torch.empty(g,k//128,n//128,device='cuda')))
        reference.extend((torch.empty(g,k,n,device='cuda',dtype=torch.float8_e4m3fn),torch.empty(g,k//128,n//128,device='cuda')))
    for change in range(3):
        weights[0].mul_(1.01);weights[1].add_(.003)
        for j,w in enumerate(weights):
            _launch(operation,[w,*quant[2*j:2*j+2]],list(w.shape))
            wt=w.transpose(1,2).contiguous()
            _launch(operation,[wt,*reference[2*j:2*j+2]],list(wt.shape))
        _launch(35,[*quant,*result],[e,h,i]);torch.cuda.synchronize()
        for x,y in zip(result,reference):
            dtype=torch.uint8 if x.dtype==torch.float8_e4m3fn else torch.int32
            assert torch.equal(x.view(dtype),y.view(dtype))
    print(json.dumps(dict(case='fp8_weight_transpose_quantizer_equivalence',quantizer=operation,changed_weights=3,every_byte=True,passed=True)),flush=True)


def prep_timing():
    e,h,i=16,8192,2048
    weights=[(torch.randn(e,2*i,h,device='cuda')*.01).bfloat16(),(torch.randn(e,h,i,device='cuda')*.01).bfloat16()]
    transposed=[torch.empty_like(w.transpose(1,2),memory_format=torch.contiguous_format) for w in weights]
    quant=[];result=[]
    for w in weights:
        g,n,k=w.shape
        quant.extend((torch.empty_like(w,dtype=torch.float8_e4m3fn),torch.empty(g,n//128,k//128,device='cuda')))
        result.extend((torch.empty(g,k,n,device='cuda',dtype=torch.float8_e4m3fn),torch.empty(g,k//128,n//128,device='cuda')))
    def forward_quant():
        for j,w in enumerate(weights):_launch(30,[w,*quant[2*j:2*j+2]],list(w.shape))
    def candidate():forward_quant();_launch(35,[*quant,*result],[e,h,i])
    def control():forward_quant();_launch(32,[*quant,*result],[e,h,i])
    def measure(fn):
        for _ in range(2):fn()
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        values=[]
        for _ in range(7):
            a.record()
            for _ in range(5):graph.replay()
            b.record();b.synchronize();values.append(a.elapsed_time(b)/5)
        return sorted(values)[len(values)//2]
    before,after=measure(control),measure(candidate)
    baseline_transpose=measure(lambda:_launch(32,[*quant,*result],[e,h,i]))
    swizzled_transpose=measure(lambda:_launch(35,[*quant,*result],[e,h,i]))
    print(json.dumps(dict(case='fp8_full_weight_preparation',experts=e,hidden=h,intermediate=i,control_ms=before,quantized_transpose_and_forward_context_ms=after,transpose32_ms=baseline_transpose,transpose35_ms=swizzled_transpose,speedup=before/after,scope='FWD+matching BWD weight preparation only, all current-step quantization included; no GEMM or EP timing claim',passed=True)),flush=True)


torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(908212)
for shape in ((1,256,128),(2,512,256),(3,768,384)):
    check_bits(*shape)
for op in (10,29,30):check_quantization(op)
prep_timing()
