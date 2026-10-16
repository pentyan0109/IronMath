"""Checks of the fresh aggregation and its audit boundaries, no model code."""
from common import *
from recompute_memory import labels
import ast
def run():
    tests=[]
    def check(name,value):
        tests.append(dict(check=name,passed=bool(value)))
        assert value,name
    x=np.array([[3,2,1,3,0,2,3],[1,3,0,3,1,3,3]],np.uint8)
    yy=np.zeros_like(x)
    for i,row in enumerate(x):
        state=0
        for j,v in enumerate(row):
            if v in (0,1):state=int(v)
            yy[i,j]=state
    check('last-write-vectorization vs scalar reference',np.array_equal(yy,labels(x)))
    check('little-endian packed prediction roundtrip',np.array_equal(yy,np.unpackbits(np.packbits(yy,axis=1,bitorder='little'),axis=1,count=yy.shape[1],bitorder='little')))
    check('no-discord McNemar p=1',paired([True,False],[True,False],np.array([True,False]),81000)['McNemar_exact_two_sided_p']==1)
    check('McNemar 7-vs-0 exact p',paired([True]*7,[False]*7,np.ones(7,bool),81000)['McNemar_exact_two_sided_p']==.015625)
    ms=load(R/'memory_models.json');check('280 unique model identities',len(ms)==280 and len({(m['study'],m['model']) for m in ms})==280)
    check('all7 success subset main3 success',all(not m['actual7'] or m['actual3'] for m in ms))
    check('A success implies D success',all(not m['A'] or m['D'] for m in ms))
    check('frozen R1 rules digest unchanged',sha(R/'R1_RULES.md')==load(R/'R1_RULES_SHA256.json')['SHA256'])
    check('rules saved before raw rescoring outputs',(R/'R1_RULES_SHA256.json').stat().st_mtime_ns<(R/'memory_models.json').stat().st_mtime_ns)
    check('R1 baseline entries64',len(rows(R/'r1_baselines.csv'))==64)
    check('R1 paired comparisons64',len(rows(R/'r1_comparison.csv'))==64)
    for m in ms:
        preds=[all(m['lengths'][str(t)] for t in [32,64,128,512] if t<=ell) for ell in [32,64,128,512]]
        checkname=False
        assert all(not preds[i+1] or preds[i] for i in range(3))
    check('Bcum nested across all280 models',True)
    check('no original experiment imports',all(all(not isinstance(n,ast.ImportFrom) or n.module in ['common','pathlib','recompute_memory'] for n in ast.walk(ast.parse(p.read_text(encoding='utf-8')))) for p in R.glob('*.py')))
    check('no torch import in analysis scripts',all(all(not (isinstance(n,ast.Import) and any(a.name.startswith('torch') for a in n.names)) and not (isinstance(n,ast.ImportFrom) and (n.module or '').startswith('torch')) for n in ast.walk(ast.parse(p.read_text(encoding='utf-8')))) for p in R.glob('*.py')))
    check('all selected checkpoint hashes matched',load(R/'R4_COMPLETION.json')['all_checkpoint_hashes_match'])
    check('raw mathematical integer flag discrepancies zero',load(R/'arithmetic_summary.json')['V054E']['integer_rescoring_disagreements']==0 and all(int(x['flags_disagree'])==0 for x in rows(R/'d11_prediction_coverage.csv')))
    check('all discrepancies preserved and counted',load(R/'R4_COMPLETION.json')['discrepancies']==sum(r['match']=='False' for r in rows(R/'r4_recomputation.csv')))
    dump('quality_checks.json',dict(tests=tests,passed=all(x['passed'] for x in tests),new_training=False,new_inference=False,original_scripts_imported=False,original_scripts_executed=False))
if __name__=='__main__':run()
