"""Exact byte/scale packing with GPU row counts, changed routes and canaries."""
import json
import torch
from deep_gemm.training.native import _launch


def main():
 torch.cuda.set_device(0);torch.manual_seed(83179);torch.set_num_threads(2)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for width in (128,512,8192):
   n,capacity,k=83,512,10
   q=torch.randint(0,256,(n,width),device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn)
   sf=torch.randn(n,width//128,device='cuda')
   sf.view(torch.int32).flatten()[::19]=0x7fc00123
   padding_bf=torch.zeros(1,128,device='cuda',dtype=torch.bfloat16)
   padding_q=torch.empty(1,128,device='cuda',dtype=torch.float8_e4m3fn)
   padding_sf=torch.empty(1,1,device='cuda')
   _launch(9,[padding_bf,padding_q,padding_sf],[1,128])
   permutation=torch.full((capacity,),-1,device='cuda',dtype=torch.int32)
   out=torch.empty(capacity,width,device='cuda',dtype=torch.float8_e4m3fn)
   scales=torch.empty(width//128,capacity,device='cuda')
   count=torch.zeros(1,device='cuda',dtype=torch.int32);status=torch.zeros_like(count)
   def step():_launch(31,[q,sf,permutation,out,scales,count,status],[capacity,width,n,k])
   step();stream.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream):step()
   for active in (0,1,129,512):
    ids=torch.arange(capacity,device='cuda')*31%(n*k)
    ids[::7]=-1;permutation.copy_(ids.int());count.fill_(active)
    out.view(torch.uint8).fill_(117);scales.fill_(-13);status.zero_()
    graph.replay();stream.synchronize()
    safe=(ids[:active]//k).clamp(min=0)
    expected=q.view(torch.uint8)[safe].clone();expected[ids[:active]<0]=0
    expected_sf=sf[safe].clone();expected_sf[ids[:active]<0]=padding_sf[0,0]
    assert status.item()==0 and torch.equal(out[:active].view(torch.uint8),expected)
    assert torch.equal(scales[:,:active].t().contiguous().view(torch.int32),expected_sf.view(torch.int32))
    assert torch.all(out[active:].view(torch.uint8)==117) and torch.all(scales[:,active:]==-13)
    print(json.dumps(dict(case='pack_fp8_rows_reference',width=width,active=active,bytes=True,scale_bits=True,canaries=True,graph=True,passed=True)),flush=True)
   for bad in (-1,capacity+1):
    count.fill_(bad);status.zero_();out.view(torch.uint8).fill_(91);scales.fill_(-17)
    graph.replay();stream.synchronize()
    assert status.item()==2 and torch.all(out.view(torch.uint8)==91) and torch.all(scales==-17)
    print(json.dumps(dict(case='pack_fp8_rows_bad_count_reference',width=width,active=bad,passed=True)),flush=True)
   del graph
 stream.synchronize()


if __name__=='__main__':main()
