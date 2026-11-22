"""Send completed source chunks while the next expert chunk executes."""
import torch
import torch.distributed as dist
from .hierarchical_pipeline import _join


class ChunkReductions:
    def __init__(self,plan,own):
        self.plan=plan
        self.own=[]
        self.outputs=[]
        self.work=[]
        self.partials=[]
        self.calls=0
        for value in own:
            shape=(plan.max_tokens,*value.shape[1:])
            if value.shape[0]==plan.max_tokens:
                padded=value
            else:
                padded=torch.zeros(shape,device=value.device,dtype=value.dtype)
                padded[:value.shape[0]].copy_(value)
            self.own.append(padded)
            self.outputs.append(torch.empty(shape,device=value.device,dtype=value.dtype))

    def _enqueue(self,remote,start,count):
        if count!=min(self.plan.chunk_tokens,self.plan.max_tokens-start):
            raise RuntimeError('remote output chunk disagrees with source capacity')
        main=torch.cuda.current_stream(self.plan.device)
        aux=self.plan._source_pair_stream
        partials=[]
        for own,other in zip(self.own,remote):
            left,right=own[start:start+count],other[start:start+count]
            pieces=(left,right) if self.plan.groups.domain_id==0 else (right,left)
            partials.append(torch.cat(pieces,dim=0))
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            for partial,output in zip(partials,self.outputs):
                target=output[start:start+count]
                partial.record_stream(aux);target.record_stream(aux)
                self.work.append(dist.reduce_scatter_tensor(target,partial,group=self.plan.groups.pair,async_op=True))
        self.partials.extend(partials)
        self.calls+=1
        if self.plan.hybrid_chunk_reduce_serialized:
            _join(self.plan,self.work)
            self.work.clear();self.partials.clear()

    def forward(self,output,start,count):
        self._enqueue((output,),start,count)

    def backward(self,dx,dscores,start,count):
        self._enqueue((dx,dscores),start,count)

    def finish(self,actual_tokens):
        if self.calls!=(self.plan.max_tokens+self.plan.chunk_tokens-1)//self.plan.chunk_tokens:
            raise RuntimeError('missing remote output chunk')
        if self.work:_join(self.plan,self.work)
        self.work.clear();self.partials.clear()
        return tuple(output[:actual_tokens] for output in self.outputs)
