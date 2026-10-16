import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import *

class Quality(unittest.TestCase):
    def test_widths_order_padding(self):
        rows=[dict(a=255,b=1,op='ADD',width=8),dict(a=10,b=3,op='SUB',width=8)]
        xr,yr,n=batch(rows,'R');xl,yl,_=batch(rows,'L')
        for i,l in enumerate(n):
            self.assertTrue(torch.equal(xr[i,:l].flip(0),xl[i,:l]));self.assertTrue(torch.equal(yr[i,:l].flip(0),yl[i,:l]))
        self.assertTrue(torch.equal(xl[1,8],torch.zeros(3)));self.assertEqual(int(yl[1,8]),-100)
    def test_bit_oracle(self):
        for a in range(16):
            for b in range(16):
                for op in ('ADD','SUB'):
                    if op=='SUB' and a<b:continue
                    r=annotate(dict(a=a,b=b,op=op,width=4));self.assertEqual(r['answer'],a+b if op=='ADD' else a-b)
    def test_native_equivalence(self):
        torch.set_num_threads(2);torch.manual_seed(0);m=Residual();n=NativeReference();n.load_state_dict(m.state_dict());s=torch.jit.script(m)
        x=torch.rand(7,9,3)
        self.assertLess(float((m(x)-n(x)).abs().max()),1e-4);self.assertTrue(torch.equal(m(x).argmax(-1),n(x).argmax(-1)))
        self.assertTrue(torch.allclose(m(x),s(x),atol=1e-5))
        self.assertEqual(sum(p.numel() for p in m.parameters()),1000334)
    def test_causal_output(self):
        m=torch.jit.script(Residual());x=torch.rand(3,9,3);y=x.clone();y[:,4:]=1-y[:,4:]
        self.assertTrue(torch.equal(m(x)[:,:4],m(y)[:,:4]))
    def test_MSB_impossibility_counterexample(self):
        rows=[dict(a=255,b=0,op='ADD',width=8),dict(a=255,b=1,op='ADD',width=8)]
        x,y,_=batch(rows,'L');self.assertTrue(torch.equal(x[0,0],x[1,0]));self.assertNotEqual(int(y[0,0]),int(y[1,0]))
        m=Residual();self.assertTrue(torch.equal(m(x)[0,0],m(x)[1,0]))
    def test_problem_reset_and_reload(self):
        torch.manual_seed(2);m=Residual();x=torch.rand(2,9,3);self.assertTrue(torch.equal(m(x),m(x)))
        n=Residual();n.load_state_dict(m.state_dict());self.assertTrue(torch.equal(m(x),n(x)))
    def test_missing_answers_are_not_consensus(self):
        from run import statistics_pair
        toy=dict(id='missing',correct=0,prediction=None,wrong_positions=[0])
        s,a,_=statistics_pair([toy],[toy]);self.assertEqual(a['agreement_count'],0)
        self.assertEqual(s['same_wrong'],0);self.assertEqual(s['different_wrong'],0);self.assertEqual(s['both_wrong_missing'],1)

if __name__=='__main__':unittest.main()
