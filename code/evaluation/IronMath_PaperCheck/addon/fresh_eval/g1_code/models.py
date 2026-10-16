import torch
from torch import nn

class Base(nn.Module):
    def __init__(self):
        super().__init__();self.H=64;self.state_size=64;self.projection=nn.Linear(4,16)
    @torch.jit.export
    def advance(self,x:torch.Tensor,state:torch.Tensor):
        raise RuntimeError('abstract recurrence')
    @torch.jit.export
    def readout(self,state:torch.Tensor):
        return self.head(state[:,:64])
    @torch.jit.export
    def step(self,x:torch.Tensor,state:torch.Tensor):
        state=self.advance(x,state)
        return state,self.readout(state)
    def forward(self,x:torch.Tensor):
        state=torch.zeros(x.size(0),self.state_size,dtype=x.dtype,device=x.device)
        out=torch.jit.annotate(list[torch.Tensor],[])
        for i in range(x.size(1)):
            state=self.advance(x[:,i],state);out.append(self.readout(state))
        return torch.stack(out,1)
    @torch.jit.export
    def block(self,x:torch.Tensor,state:torch.Tensor):
        out=torch.jit.annotate(list[torch.Tensor],[])
        for i in range(x.size(1)):
            state,z=self.step(x[:,i],state);out.append(z)
        return torch.stack(out,1),state
    @torch.jit.export
    def repeat(self,state:torch.Tensor,symbol:int,count:int):
        b=state.size(0);x=torch.zeros(b,4);x[:,symbol]=1;r=torch.zeros(b,4);r[:,3]=1
        movements=torch.empty(count,b);margins=torch.empty(count,b);labels=torch.arange(b)%2
        state_finite=torch.ones(count,b,dtype=torch.bool);logit_finite=torch.ones(count,b,dtype=torch.bool)
        for i in range(count):
            old=state;state,z=self.step(x,state);movements[i]=torch.linalg.vector_norm(state-old,dim=1)
            state_finite[i]=torch.isfinite(state).all(-1)
            logit_finite[i]=torch.isfinite(z).all(-1)
            if symbol==2:_,z=self.step(r,state)
            logit_finite[i]=logit_finite[i]&torch.isfinite(z).all(-1)
            margins[i]=z.gather(1,labels[:,None])[:,0]-z.gather(1,(1-labels)[:,None])[:,0]
        return state,movements,margins,state_finite,logit_finite

class ResidualGRU(Base):
    def __init__(self):
        super().__init__();self.core=nn.GRUCell(16,64);self.residual_gate=nn.Parameter(torch.zeros(()));self.head=nn.Linear(64,2)
    @torch.jit.export
    def advance(self,x:torch.Tensor,state:torch.Tensor):
        return self.core(self.projection(x),state)+torch.tanh(self.residual_gate)*state

class GRU(Base):
    def __init__(self):
        super().__init__();self.core=nn.GRUCell(16,64);self.head=nn.Linear(64,2)
    @torch.jit.export
    def advance(self,x:torch.Tensor,state:torch.Tensor):
        return self.core(self.projection(x),state)

class LSTM(Base):
    def __init__(self):
        super().__init__();self.state_size=128;self.core=nn.LSTMCell(16,64);self.head=nn.Linear(64,2)
    @torch.jit.export
    def advance(self,x:torch.Tensor,state:torch.Tensor):
        h,c=self.core(self.projection(x),(state[:,:64],state[:,64:]))
        return torch.cat((h,c),dim=1)

class RNN(Base):
    def __init__(self):
        super().__init__();self.core=nn.RNNCell(16,64,nonlinearity='tanh');self.head=nn.Linear(64,2)
    @torch.jit.export
    def advance(self,x:torch.Tensor,state:torch.Tensor):
        return self.core(self.projection(x),state)

class Linear(Base):
    def __init__(self):
        super().__init__();self.theta=nn.Parameter(torch.empty(64));nn.init.uniform_(self.theta,0,3)
        self.core=nn.Linear(16,64);self.head=nn.Sequential(nn.Linear(64,64),nn.Tanh(),nn.Linear(64,2))
    @torch.jit.export
    def advance(self,x:torch.Tensor,state:torch.Tensor):
        return torch.sigmoid(self.theta)*state+self.core(self.projection(x))

ARCHS=['residual_gru','gru','lstm','rnn','linear']
FACTORY=dict(zip(ARCHS,[ResidualGRU,GRU,LSTM,RNN,Linear]))
def make(architecture):return FACTORY[architecture]()
