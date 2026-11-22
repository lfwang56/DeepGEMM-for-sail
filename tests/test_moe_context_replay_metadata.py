"""GPU-counted replay masks must retain metadata and reuse only live windows."""
import json
import torch
import deep_gemm as dg


def main():
 torch.cuda.set_device(0);torch.set_num_threads(2)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  e,capacity=2,128
  offsets=torch.zeros(e+1,device='cuda',dtype=torch.int32)
  lengths=torch.zeros(e,device='cuda',dtype=torch.int32)
  blocks=torch.zeros(1,device='cuda',dtype=torch.int32)
  global_offsets=torch.zeros_like(offsets)
  ro=torch.empty_like(offsets);rl=torch.empty_like(lengths);rb=torch.empty_like(blocks);cb=torch.empty_like(blocks)
  status=torch.zeros_like(blocks);windows=torch.zeros_like(blocks);rows=torch.zeros(1,device='cuda',dtype=torch.int64)
  for start in (0,128,256,512):
   def step():dg._C._moe_training_peer_launch(20,[offsets,lengths,blocks,global_offsets,ro,rl,rb,cb,status,windows,rows],[e,capacity,start],False)
   step();stream.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph,stream=stream):step()
   for cumulative in ([0,0,0],[0,128,128],[0,128,384],[0,256,640]):
    actual=torch.tensor(cumulative,device='cuda',dtype=torch.int32).sub(start).clamp(0,capacity)
    global_offsets.copy_(torch.tensor(cumulative,device='cuda',dtype=torch.int32));offsets.copy_(actual)
    lengths.copy_(actual.diff());blocks.fill_(int(actual[-1].item())//128)
    status.zero_();windows.zero_();rows.zero_();graph.replay();stream.synchronize()
    cached=cumulative[-1]>0 and (cumulative[-1]-1)//capacity*capacity==start
    assert status.item()==0 and torch.equal(ro,torch.zeros_like(actual) if cached else actual)
    assert torch.equal(rl,torch.zeros_like(lengths) if cached else lengths)
    assert rb.item()==(0 if cached else blocks.item()) and cb.item()==(blocks.item() if cached else 0)
    assert windows.item()==int(cached) and rows.item()==(actual[-1].item() if cached else 0)
    assert torch.equal(offsets,actual)
    print(json.dumps(dict(case='context_replay_metadata_reference',window=start,global_offsets=cumulative,cached=cached,graph=True,passed=True)),flush=True)
   del graph
 stream.synchronize()


if __name__=='__main__':main()
