"""Eight independent device trials: meaningful DW gold checks and local timing."""
from pathlib import Path
import argparse,hashlib,json,os,sys,traceback

p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--lut-root',required=True);a=p.parse_args()
variant=int(os.environ.get('LOCAL_RANK','0'))
choices=[None,(256,256,64,64,64,3),(256,256,64,64,64,2),(128,256,64,32,64,3),(128,128,64,32,32,3),(128,128,64,32,32,2),(256,128,64,64,32,3),(128,128,128,32,32,2)]
folder=Path(a.lut_root)/f'v{variant}';folder.mkdir(parents=True,exist_ok=True)
tile=choices[variant]
lut=folder/'ZW-M890P_bf16_kgroupedcontiguous.lut'
if tile:
    lut.write_text('# groups M N maxK BM BN BK WM WN stages\n'+''.join(f"16 {m} {n} 65536 "+' '.join(map(str,tile))+'\n' for m,n in ((2048,4096),(4096,1024))))
os.environ['SAIL_DEEPGEMM_LUT_PATH']=str(folder)
sys.path.insert(0,a.library)
import torch
import deep_gemm as dg

torch.cuda.set_device(variant);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(391728)
print(json.dumps({'case':'verified_isolated_library','library':str(Path(a.library).resolve()),'torch':torch.__version__,'extension':str(dg._C.__file__),'extension_sha256':hashlib.sha256(Path(dg._C.__file__).read_bytes()).hexdigest(),'variant':variant,'tile':tile,'lut_sha256':hashlib.sha256(lut.read_bytes()).hexdigest() if lut.exists() else None}),flush=True)

try:
    for role,m,n in [('dw1',2048,4096),('dw2',4096,1024)]:
        cap=65536;e=16
        left=torch.empty(cap,m,device='cuda',dtype=torch.bfloat16)
        right=torch.empty(cap,n,device='cuda',dtype=torch.bfloat16)
        output=torch.empty(e,m,n,device='cuda',dtype=torch.float32)
        lengths=torch.zeros(e,device='cuda',dtype=torch.int32)
        dg.k_grouped_bf16_gemm_tn_device(left,right,output,lengths,cap)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):dg.k_grouped_bf16_gemm_tn_device(left,right,output,lengths,cap)
        for pattern,ks in enumerate(([128,0,256,128]+[0]*12,[0,2560]+[0]*14)):
            count=sum(ks);left[:count].normal_(std=.1);right[:count].normal_(std=.1)
            lengths.copy_(torch.tensor(ks,dtype=torch.int32,device='cuda'));output.fill_(.03125)
            graph.replay();graph.replay();torch.cuda.synchronize()
            start=0;errors=[]
            for group,k in enumerate(ks):
                if k:
                    expected=2*(left[start:start+k].float().t()@right[start:start+k].float())+.03125
                    error=float((output[group]-expected).square().sum().sqrt()/expected.square().sum().sqrt().clamp_min(1e-20))
                    assert error<2e-4 and bool(torch.isfinite(output[group]).all()),(role,pattern,group,error)
                    errors.append(error);start+=k
                else:assert bool((output[group]==.03125).all()),(role,pattern,group,'empty beta=1')
            print(json.dumps({'case':'mok_dw_independent_reference','variant':variant,'tile':tile,'role':role,'pattern':pattern,'lengths':ks,'relative_rms':errors,'passed':True}),flush=True)
        for tokens in (2048,4096):
            ks=[(1280 if tokens==2048 else 2560)+(256 if j%3==0 else 0) for j in range(e)]
            count=sum(ks);assert count<=cap
            left[:count].normal_(std=.1);right[:count].normal_(std=.1)
            lengths.copy_(torch.tensor(ks,dtype=torch.int32,device='cuda'));output.zero_()
            for _ in range(5):graph.replay()
            torch.cuda.synchronize();samples=[]
            for _ in range(5):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):graph.replay()
                end.record();end.synchronize();samples.append(begin.elapsed_time(end)/20)
            print(json.dumps({'case':'mok_dw_isolated_benchmark','variant':variant,'tile':tile,'role':role,'tokens_equivalent_per_rank':tokens,'groups':e,'m':m,'n':n,'lengths':ks,'capacity':cap,'samples_event_ms':samples,'median_event_ms':sorted(samples)[2],'device':torch.cuda.get_device_name(),'torch':torch.__version__,'scope':'One BF16-input FP32-output beta=1 DW; not a MoE step','passed':True}),flush=True)
        del graph,left,right,output,lengths
        torch.cuda.empty_cache()
    print(json.dumps({'case':'mok_dw_trial_finished','variant':variant,'passed':True}),flush=True)
except Exception as error:
    print(json.dumps({'case':'mok_dw_trial_finished','variant':variant,'passed':False,'error':repr(error)}),flush=True)
    traceback.print_exc()
