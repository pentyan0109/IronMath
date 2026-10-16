import unittest,time
import numpy as np,torch
from common import R
from data import draw,tensor,labels,sequential_labels,longset
from model import FF
class Tests(unittest.TestCase):
    def test_oracles(self):
        a=draw(42,32,16);self.assertTrue(np.array_equal(labels(a),sequential_labels(a)))
        b=np.array([[3,2,3,1,3,0,3]],np.uint8);self.assertEqual(labels(b).tolist(),[[0,0,0,1,1,0,0]])
    def test_structure(self):
        for h,n in ((32,4947),(64,15955)):
            m=FF(h);self.assertEqual(sum(p.numel() for p in m.parameters()),n);self.assertEqual(float(m.residual_gate),0)
    def test_step_script_reset_chunk(self):
        for h in (32,64):
            torch.manual_seed(99);m=FF(h).eval();s=torch.jit.script(m);a=tensor(draw(5,8,16));hh=torch.zeros(8,h);zs=[]
            with torch.inference_mode():
                eager=m(a);script=s(a)
                for i in range(16):hh,z=m.step(a[:,i],hh);zs.append(z)
                self.assertTrue(torch.equal(eager.argmax(-1),torch.stack(zs,1).argmax(-1)));self.assertTrue(torch.equal(eager.argmax(-1),script.argmax(-1)));self.assertTrue(torch.equal(s(a),s(a)))
                z0,h0=s.block(a[:,:8],torch.zeros(8,h));z1,h1=s.block(a[:,8:],h0);self.assertTrue(torch.equal(torch.cat((z0,z1),1),script))
    def test_backward_and_save(self):
        torch.manual_seed(9);m=FF(32);a=draw(9,4,8);mask=torch.from_numpy(a==3);y=torch.from_numpy(labels(a).astype(np.int64));z=m(tensor(a));torch.nn.functional.cross_entropy(z[mask],y[mask]).backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        path=R/'quality_roundtrip.pt';torch.save(m.state_dict(),path);n=FF(32);n.load_state_dict(torch.load(path,weights_only=True));self.assertTrue(torch.equal(m(tensor(a)),n(tensor(a))))
    def test_sampling_reproducible(self):self.assertTrue(np.array_equal(draw(12,32,8),draw(12,32,8)))
    def test_jacobian(self):
        m=FF(32);h=torch.zeros(32);x=torch.zeros(1,4);x[0,2]=1;j=torch.autograd.functional.jacobian(lambda hh:m.step(x,hh[None])[0][0],h);self.assertEqual(j.shape,(32,32));self.assertTrue(torch.isfinite(torch.linalg.eigvals(j.double())).all())
    def test_repeat_readout_and_nonmutation(self):
        m=FF(32).eval();s=torch.jit.script(m)
        for symbol in (2,3):
            h0=torch.zeros(8,32);x=torch.zeros(8,4);x[:,symbol]=1;r=torch.zeros(8,4);r[:,3]=1;hh=h0;labels=torch.arange(8)%2
            with torch.inference_mode():
                end,mv,mg=s.repeat(h0,symbol,10)
                for i in range(10):
                    old=hh;hh,z=m.step(x,hh)
                    self.assertTrue(torch.equal(mv[i],torch.linalg.vector_norm(hh-old,dim=1)))
                    if symbol==2:_,z=m.step(r,hh)
                    expected=z.gather(1,labels[:,None])[:,0]-z.gather(1,(1-labels)[:,None])[:,0]
                    self.assertTrue(torch.equal(mg[i],expected))
                self.assertTrue(torch.equal(end,hh))
    def test_long_generator_only_after_freezes(self):
        # Test only after A/B freezes; never generate a long dataset in pretraining QA.
        if not (R/'PREDICTIONS_B_FROZEN.json').exists():self.skipTest('long generation intentionally gated until both prediction freezes')
        for t in (32,64,128,512,2048,8192,65000):
            x,meta=longset(t);self.assertEqual(x.shape,(68,t));self.assertEqual(sum(r['multiplicity'] for r in meta),96)
            self.assertTrue(np.array_equal(labels(x),sequential_labels(x)))
            for j,r in enumerate(meta):
                if r['category']=='rewrite':
                    pp=[0,t//4,t//2,3*t//4];self.assertEqual(np.flatnonzero(x[j]<2).tolist(),pp)
                    for k,p in enumerate(pp):self.assertEqual(int(x[j,p]),r['value'] if k%2==0 else 1-r['value']);self.assertEqual(int(x[j,p+1]),3)
                if r['category'] in ('long_N','long_R'):
                    self.assertTrue(np.array_equal(x[j,:8],draw(70000+r['j'],1,8)[0]));self.assertEqual(x[j,8],r['value']);self.assertEqual(x[j,-1],3)
                    if r['category']=='long_N':self.assertTrue((x[j,9:-1]==2).all())
                    else:self.assertTrue((x[j,9:]==3).all())
