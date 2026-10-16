"""Untrained implementation checks; not experimental training/diagnostics."""
from g1_utils import *
import torch
from data import draw,tensor,labels,sequential_labels
def preflight():
    rows=[]
    expected=dict(residual_gru=15955,gru=15954,lstm=21202,rnn=5458,linear=5522)
    for arch in ARCHS:
        torch.manual_seed(123);m=make(arch);s=torch.jit.script(m);x=tensor(draw(123,3,8));h=torch.zeros(3,m.state_size)
        with torch.no_grad():
            direct=m(x);batch=s(x);part,hh=s.block(x[:,:4],h);rest,hh=s.block(x[:,4:],hh);zs=[]
            for i in range(8):h,z=m.step(x[:,i],h);zs.append(z)
            assert torch.equal(direct,batch) and torch.equal(direct,torch.stack(zs,1)) and torch.equal(direct,torch.cat([part,rest],1))
            assert torch.equal(m(x),m(x)), 'problem reset leakage'
            en,mv,mg,sf,lf=s.repeat(torch.zeros(8,m.state_size),2,10)
            assert en.shape==(8,m.state_size) and sf.shape==lf.shape==mv.shape==mg.shape==(10,8)
        count=sum(p.numel() for p in m.parameters());assert count==expected[arch]
        a=draw(12,4,8);z=m(tensor(a));mask=torch.from_numpy(a==3);y=torch.from_numpy(labels(a).astype(np.int64));loss=torch.nn.functional.cross_entropy(z[mask],y[mask]);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        path=R/f'quality_roundtrip_{arch}.pt';torch.save(m.state_dict(),path);n=make(arch);n.load_state_dict(torch.load(path,weights_only=True));assert torch.equal(m(x),n(x))
        assert np.array_equal(labels(a),sequential_labels(a))
        rows.append(dict(architecture=arch,parameters=count,state_dimension=m.state_size,forward_step_script_chunk_exact=True,backward_finite=True,save_load_exact=True))
    dump('QUALITY_PRETRAIN.json',dict(utc=now(),passed=True,untrained_implementation_tests=True,long_datasets_not_generated=True,models=rows))
    print('PRETRAIN QUALITY PASS',rows,flush=True)
if __name__=='__main__':setup();preflight()
