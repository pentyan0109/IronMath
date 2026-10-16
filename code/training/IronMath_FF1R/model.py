import torch
from torch import nn
class FF(nn.Module):
    def __init__(self,H:int):
        super().__init__();self.H=H;self.projection=nn.Linear(4,16);self.core=nn.GRUCell(16,H);self.residual_gate=nn.Parameter(torch.zeros(()));self.head=nn.Linear(H,2)
    @torch.jit.export
    def step(self,x:torch.Tensor,h:torch.Tensor):
        h=self.core(self.projection(x),h)+torch.tanh(self.residual_gate)*h
        return h,self.head(h)
    @torch.jit.export
    def block(self,x:torch.Tensor,h:torch.Tensor):
        out=torch.jit.annotate(list[torch.Tensor],[])
        for i in range(x.size(1)):
            h,z=self.step(x[:,i],h);out.append(z)
        return torch.stack(out,1),h
    def forward(self,x:torch.Tensor):return self.block(x,torch.zeros(x.size(0),self.H,dtype=x.dtype,device=x.device))[0]
    @torch.jit.export
    def repeat(self,h:torch.Tensor,symbol:int,count:int):
        b=h.size(0);x=torch.zeros(b,4);x[:,symbol]=1;r=torch.zeros(b,4);r[:,3]=1
        movements=torch.empty(count,b);margins=torch.empty(count,b);labels=torch.arange(b)%2
        for i in range(count):
            old=h;h,z=self.step(x,h);movements[i]=torch.linalg.vector_norm(h-old,dim=1)
            if symbol==2:_,z=self.step(r,h)
            margins[i]=z.gather(1,labels.unsqueeze(1)).squeeze(1)-z.gather(1,(1-labels).unsqueeze(1)).squeeze(1)
        return h,movements,margins
