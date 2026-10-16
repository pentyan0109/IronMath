import io,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import *

class Quality(unittest.TestCase):
    def test_roundtrip(self):
        for n in range(-1000,1001):self.assertEqual(decode(encode(n,8,3),3),n)
        for w in (8,16,32,64,128,256,512,1024,2048):
            n=ternary_length(w)
            for value in (0,1,(1<<w)-1,2*((1<<w)-1)):
                self.assertEqual(decode(encode(value,n,3),3),value)
    def test_all_small_arithmetic_and_complement(self):
        for a in range(16):
            for b in range(16):
                for op in ('ADD','SUB'):
                    if op=='SUB' and a<b:continue
                    r=dict(a=a,b=b,op=op,width=4)
                    for k in ('B0','B1','T0','T1'):
                        dd,ev,ll=propagation(r,k);self.assertEqual(decode(dd,3 if k.startswith('T') else 2),a+b if op=='ADD' else a-b)
                    if op=='SUB':
                        aa,bb,c=inputs(r,'B1');self.assertEqual(decode(bb,2),31-b);self.assertEqual(c,[1,0,0,0,0])
                        self.assertEqual((a+decode(bb,2)+1)%32,a-b)
    def test_carry_table_independent(self):
        for a in (-1,0,1):
            for b in (-1,0,1):
                for c in (-1,0,1):
                    s=a+b+c;choices=[(r,z) for r in (-1,0,1) for z in (-1,0,1) if r+3*z==s]
                    self.assertEqual(len(choices),1);r,z=choices[0];self.assertEqual(r,(s+1)%3-1)
    def test_no_answer_inputs(self):
        r=dict(a=255,b=10,op='SUB',width=8)
        for k in ('B0','B1','T0','T1'):
            x,y,ll=batch([r],k);xx,yy,_=batch([dict(r,answer=-900)],k,labels=False)
            self.assertTrue(torch.equal(x,xx));self.assertTrue((yy==-100).all())
    def test_model_cpu_reset_causal_save_backward(self):
        torch.set_num_threads(2)
        for k in ('B0','B1','T0','T1'):
            torch.manual_seed(0);m=Residual(k);s=torch.jit.script(m)
            self.assertEqual(sum(p.numel() for p in m.parameters()),1000880 if k.startswith('T') else 1000334)
            self.assertTrue(all(p.device.type=='cpu' for p in m.parameters()))
            x=torch.rand(2,9,3);other=x.clone();other[:,5:]=1-other[:,5:]
            self.assertTrue(torch.equal(s(x)[:,:5],s(other)[:,:5]));self.assertTrue(torch.equal(s(x),s(x)))
            self.assertTrue(torch.allclose(s(x[:1]),s(x)[:1],atol=1e-5))
            stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0);mm=Residual(k);mm.load_state_dict(torch.load(stream,weights_only=True));self.assertTrue(torch.equal(m(x),mm(x)))
            s(x).square().mean().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in s.parameters()))
    def test_padding_and_unified_inputs(self):
        rows=[dict(a=255,b=255,op='ADD',width=8),dict(a=255,b=10,op='SUB',width=8)]
        for k in ('B0','B1','T0','T1'):
            x,y,ll=batch(rows,k)
            for i,r in enumerate(rows):self.assertEqual(decode((y[i,:ll[i]]-(1 if k.startswith('T') else 0)).tolist(),3 if k.startswith('T') else 2),r['a']+r['b'] if r['op']=='ADD' else r['a']-r['b'])
            if k=='B0':self.assertEqual(int(y[1,8]),-100)
            if k=='T1':self.assertTrue((x[:,:,2]==0).all())

if __name__=='__main__':unittest.main()
