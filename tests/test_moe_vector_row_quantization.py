"""Exact vectorized row quantization against the unchanged native arithmetic."""
import gc,json,torch
import deep_gemm as dg

def fixture(capacity,width,measure=False):
    x=(torch.randn(capacity,width,device='cuda')*.2).bfloat16()
    qraw=torch.full((capacity*width+256,),117,device='cuda',dtype=torch.uint8)
    q=qraw[128:-128].view(torch.float8_e4m3fn).reshape(capacity,width)
    sfraw=torch.full((capacity*(width//128)+64,),-13.,device='cuda');sf=sfraw[32:-32]
    ref=torch.empty_like(q);refsf=torch.empty_like(sf)
    rows=torch.zeros(1,device='cuda',dtype=torch.int32);status=torch.zeros_like(rows)
    for layout in (0,1):
        def launch(op):dg._C._moe_training_launch(op,[x,q if op==34 else ref,sf if op==34 else refsf,rows,status],[capacity,width,layout])
        launch(34)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):launch(34)
        active_cases=(0,1,129,capacity) if not measure else (min(40960,capacity),)
        patterns=('random','zero','special','updated') if not measure else ('random',)
        for pattern in patterns:
            if pattern=='zero':x.zero_()
            elif pattern in ('special','updated'):
                x.normal_(0.,2. if pattern=='updated' else .2)
                if pattern=='special':
                    bits=torch.tensor([0,-32768,1,-32767,0x7f7f,-129,0x7f80,-128,0x7fc1,-63],device='cuda',dtype=torch.int16)
                    for i in range(bits.numel()):x.view(torch.int16).flatten()[i::193]=bits[i]
            for active in active_cases:
                rows.fill_(active);qraw.fill_(117);sfraw.fill_(-13);ref.view(torch.uint8).fill_(117);refsf.fill_(-13);status.zero_()
                launch(26);graph.replay();torch.cuda.synchronize()
                gates=dict(q_bits=torch.equal(q.view(torch.uint8),ref.view(torch.uint8)),scale_bits=torch.equal(sf.view(torch.uint8),refsf.view(torch.uint8)),q_guard=bool(torch.all(qraw[:128]==117)&torch.all(qraw[-128:]==117)),sf_guard=bool(torch.all(sfraw[:32]==-13)&torch.all(sfraw[-32:]==-13)),status=status.item()==0)
                print(json.dumps(dict(case='vector_row_quantization_reference',capacity=capacity,width=width,layout=layout,active=active,pattern=pattern,gates=gates,graph=True,every_element=True,passed=all(gates.values()))),flush=True)
                assert all(gates.values()),gates
        for bad in (-1,capacity+1):
            rows.fill_(bad);qraw.fill_(117);sfraw.fill_(-13);status.zero_();graph.replay();torch.cuda.synchronize()
            assert status.item()==2 and torch.all(qraw==117) and torch.all(sfraw==-13)
        if measure:
            rows.fill_(min(40960,capacity));x.normal_(0.,.2);status.zero_()
            def timing(op):
                for _ in range(3):launch(op)
                torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):launch(op)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);samples=[]
                for _ in range(7):
                    begin.record()
                    for _ in range(10):g.replay()
                    end.record();end.synchronize();samples.append(begin.elapsed_time(end)/10)
                return sorted(samples)[3]
            before,after=timing(26),timing(34)
            print(json.dumps(dict(case='vector_row_quantization_performance',capacity=capacity,width=width,layout=layout,active=rows.item(),before_ms=before,after_ms=after,speedup=before/after,scope='Row quantization only, unchanged arithmetic',passed=True)),flush=True)
        del graph
    del x,q,sf,ref,refsf;gc.collect();torch.cuda.synchronize()

def main():
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(100519)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dg._C._moe_training_precompile(0,[26,34])
        for width in (128,256,8192):fixture(513,width)
        fixture(65536,8192,True)
    stream.synchronize()
if __name__=='__main__':main()
