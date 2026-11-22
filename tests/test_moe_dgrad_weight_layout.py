"""Identical-input BF16 GEMM checks for optional original [K,N] weights."""
import argparse
import json
import torch
import deep_gemm as dg


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qwen',action='store_true');args=parser.parse_args()
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.manual_seed(194913)
    shapes=[(3,128,256)] + ([(16,2048,8192),(16,8192,4096)] if args.qwen else [])
    for groups,n,k in shapes:
        rows=groups*256
        x=(torch.randn(rows,k,device='cuda')*.01).bfloat16()
        weight=(torch.randn(groups,k,n,device='cuda')*.02).bfloat16()
        transposed=weight.transpose(1,2).contiguous()
        old=torch.zeros(rows,n,device='cuda',dtype=torch.bfloat16);new=old.clone()
        identity=torch.arange(rows,device='cuda',dtype=torch.int32)
        work=torch.tensor([[g,256,g,g*256] for g in range(groups)],device='cuda',dtype=torch.int32)
        count=torch.tensor(groups,device='cuda',dtype=torch.int32)
        peers=torch.tensor([x.data_ptr()],device='cuda',dtype=torch.int64)
        def run():
            old.zero_();new.zero_();transposed.copy_(weight.transpose(1,2))
            for rhs,out,kn in ((transposed,old,False),(weight,new,True)):
                dg._C._moe_training_peer_gemm(x,rhs,out,peers,peers,identity,identity,work,count,
                    rows,rows,False,False,False,config=(256,64,64,64,3),block_m=256,weight_kn=kn)
        run();torch.cuda.synchronize();torch.testing.assert_close(old,new,rtol=0,atol=0)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
        for pattern in ('full','ragged','empty'):
            x.mul_(.97);weight.add_(.0001)
            work[:,1].fill_(127 if pattern=='ragged' else 256)
            count.fill_(0 if pattern=='empty' else groups)
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(old,new,rtol=0,atol=0)
        print(json.dumps(dict(case='dgrad_weight_layout_reference',groups=groups,n=n,k=k,
            graph=True,changed_inputs_weights_and_active_rows=True,bitwise_equal=True,passed=True)),flush=True)
        del x,weight,transposed,old,new,graph


if __name__=='__main__':main()
