"""Output-tile WGrad chunks preserve beta=1 and every accumulation bit."""
import argparse
import json
import os
import torch
import deep_gemm as dg


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qwen',action='store_true');args=parser.parse_args()
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(491813)
    shapes=[(3,768,640,1024)]
    if args.qwen:shapes += [(16,4096,8192,65536),(16,8192,2048,65536)]
    for groups,m,n,capacity in shapes:
        a=(torch.randn(capacity,m,device='cuda')*.01).bfloat16()
        b=(torch.randn(capacity,n,device='cuda')*.01).bfloat16()
        initial=torch.randn(groups,m,n,device='cuda')*.001
        plain=initial.clone();chunked=initial.clone()
        lengths=torch.full((groups,),2560 if args.qwen and groups==16 else 256,device='cuda',dtype=torch.int32)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        for chunk in ((257,512,1024) if groups==16 else (1,5,17,100000)):
            lengths.fill_(2560 if args.qwen and groups==16 else 256)
            os.environ['DG_BF16_WGRAD_TILE_CHUNK']='0'
            with torch.cuda.stream(stream):
                plain.copy_(initial);dg.k_grouped_bf16_gemm_tn_device(a,b,plain,lengths,capacity)
            os.environ['DG_BF16_WGRAD_TILE_CHUNK']=str(chunk)
            def run():
                chunked.copy_(initial)
                dg.k_grouped_bf16_gemm_tn_device(a,b,chunked,lengths,capacity)
            with torch.cuda.stream(stream):run()
            stream.synchronize();torch.testing.assert_close(plain,chunked,rtol=0,atol=0)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):run()
            for pattern in ('mixed','empty'):
                a.mul_(.91);b.add_(.0001)
                lengths.zero_()
                if pattern=='mixed':lengths[groups//2]=256
                os.environ['DG_BF16_WGRAD_TILE_CHUNK']='0'
                with torch.cuda.stream(stream):
                    plain.copy_(initial);dg.k_grouped_bf16_gemm_tn_device(a,b,plain,lengths,capacity)
                stream.synchronize();graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(plain,chunked,rtol=0,atol=0)
            print(json.dumps(dict(case='wgrad_tile_chunks_reference',groups=groups,m=m,n=n,
                capacity=capacity,chunk=chunk,graph=True,nonzero_beta_one=True,
                changed_inputs_and_lengths=True,bitwise_equal=True,passed=True)),flush=True)
            del graph
        for invalid in ('-1',str(2**21)):
            os.environ['DG_BF16_WGRAD_TILE_CHUNK']=invalid
            before=chunked.clone()
            try:dg.k_grouped_bf16_gemm_tn_device(a,b,chunked,lengths,capacity)
            except RuntimeError:pass
            else:raise AssertionError('invalid output chunk accepted')
            torch.testing.assert_close(before,chunked,rtol=0,atol=0)
        os.environ['DG_BF16_WGRAD_TILE_CHUNK']='1'
        try:dg.k_grouped_bf16_gemm_tn_device(a,b,chunked,lengths,capacity,torch.empty(32768,device='cuda',dtype=torch.int64))
        except RuntimeError:pass
        else:raise AssertionError('ambiguous multi-launch probe accepted')
        os.environ['DG_BF16_WGRAD_TILE_CHUNK']='0'
        del a,b,initial,plain,chunked,lengths


if __name__=='__main__':main()
