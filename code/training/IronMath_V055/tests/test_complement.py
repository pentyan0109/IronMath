import io,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core import *

class Quality(unittest.TestCase):
    def test_entrypoint_after_function_definitions(self):
        import ast
        path=Path(__file__).resolve().parents[1]/'run.py';tree=ast.parse(path.read_text(encoding='utf-8'))
        entry=[n for n in tree.body if isinstance(n,ast.If) and '__name__' in ast.unparse(n.test)]
        self.assertEqual(len(entry),1);self.assertGreater(entry[0].lineno,max(n.end_lineno for n in tree.body if isinstance(n,ast.FunctionDef)))
    def test_exhaustive_complements(self):
        for b in range(256):
            neg=(-b)%256;flip=b^255
            self.assertEqual(neg,(flip+1)&255);self.assertEqual(integer(bits(neg,8)),neg)
            for a in range(b,256):
                self.assertEqual((a+neg)&255,a-b);self.assertEqual((a+flip+1)&255,a-b)
    def test_big_conversion_guard(self):
        for w in (8,16,32,64,128,256,512,1024,2048):
            for a,b in [((1<<w)-1,0),((1<<w)-1,1),((1<<w)-1,(1<<w)-1),(1<<(w-1),1)]:
                r=dict(a=a,b=b,op='SUB',width=w,longest_propagation=0);x,y,ll=encode([r],labels=True)
                self.assertEqual(ll,[w+1]);self.assertTrue(torch.equal(x[0,w],torch.zeros(3)));self.assertTrue((x[:,:,2]==0).all())
                raw=integer(y[0].tolist());self.assertEqual(raw,a+((-b)%(1<<w)));self.assertEqual(raw&((1<<w)-1),a-b)
                self.assertEqual(int(y[0,w]),int(b>0))
    def test_native_B1_guard_distinction(self):
        r=dict(a=10,b=3,op='SUB',width=8);x,y,_=encode([r],'B1_native',True)
        self.assertEqual(x[0,8].tolist(),[0,1,0]);self.assertEqual(float(x[0,0,2]),1);self.assertEqual(integer(y[0].tolist()),7)
        literal_raw=10+(3^255)+1;self.assertEqual(literal_raw>>8,1);self.assertEqual(literal_raw&255,7)
    def test_no_labels_in_inputs(self):
        r=dict(a=200,b=170,op='SUB',width=8);x,y,_=encode([r],labels=False)
        xx,yy,_=encode([dict(r,answer=-999)],labels=False);self.assertTrue(torch.equal(x,xx));self.assertTrue((y==-100).all())
    def test_cpu_forward_reset_reload_causality_backward(self):
        torch.set_num_threads(2);torch.manual_seed(0);m=Residual();s=torch.jit.script(m);x=torch.rand(3,9,3)
        self.assertEqual(sum(p.numel() for p in m.parameters()),1000334);self.assertTrue(all(p.device.type=='cpu' for p in m.parameters()))
        self.assertTrue(torch.equal(s(x),s(x)));xx=x.clone();xx[:,5:]=0;self.assertTrue(torch.equal(s(x)[:,:5],s(xx)[:,:5]))
        self.assertTrue(torch.allclose(s(x[:1]),s(x)[:1],atol=1e-5));z=io.BytesIO();torch.save(m.state_dict(),z);z.seek(0);n=Residual();n.load_state_dict(torch.load(z,weights_only=True));self.assertTrue(torch.equal(m(x),n(x)))
        s(x).square().mean().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in s.parameters()))
    def test_schedule(self):
        from data import phase
        self.assertEqual(phase('F2','C1',599,100),'ADD_only');self.assertEqual(phase('F2','C1',600,0),'mixed')
        self.assertEqual(phase('F2','C2',0,44.99),'ADD_only');self.assertEqual(phase('F2','C2',0,45),'mixed')
        self.assertEqual(phase('F0','C2',2000,90),'ADD_only');self.assertEqual(phase('F1','C1',0,0),'mixed')
if __name__=='__main__':unittest.main()
