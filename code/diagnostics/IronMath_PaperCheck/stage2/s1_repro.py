"""Checks that the 20,000-iteration / 1e-6 condition reproduces the G1 frozen predictions and diagnostic details."""
import sys,json
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;G=S.parent.parent/'IronMath_G1'
sys.path.insert(0,str(S))
from s1_common import judge
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def run(write=True):
    frozen={m['model']:m for m in load(G/'main_PREDICTIONS_FROZEN.json')['models']};rows=[]
    for p in sorted((S/'s1_records').glob('*.json')):
        rec=load(p);mid=rec['model'];tr=rec['counts']['20000']['trajectories'];A,D=judge(tr,1e-6);old=load(G/'diagnostic_records'/f'{mid}.json')
        det={(d['neutral'],d['prefix'],d['value']):d for d in old['details']};bad=[]
        for r in tr:
            d=det[r['neutral'],r['prefix'],r['value']]
            if d['last100_max_movement']!=r['last100_max_movement']:bad.append(('movement',r['neutral'],r['prefix'],r['value']))
            if d['readout']!=r['readout']:bad.append(('readout',r['neutral'],r['prefix'],r['value']))
            if d['spectral_radius'] is not None and d['spectral_radius']!=r['spectral_radius']:bad.append(('rho',r['neutral'],r['prefix'],r['value']))
            if d['first_negative_margin_step']!=r['first_negative_margin_step']:bad.append(('margin',r['neutral'],r['prefix'],r['value']))
        with np.load(S/'s1_endpoints'/f'{mid}.npz') as z:
            for n in 'NR':
                with np.load(G/'diagnostic_traces'/f'{mid}_{n}.npz') as o:
                    if not np.array_equal(o['endpoints'],z[f'{n}_20000'],equal_nan=True):bad.append(('endpoints',n))
        f=frozen[mid];rows.append(dict(model=mid,A=A,D=D,frozen_A=f['A'],frozen_D=f['D'],A_match=A==f['A'],D_match=D==f['D'],G1_A_record=old['A'],detail_mismatches=bad))
    ok=all(r['A_match'] and r['D_match'] for r in rows);exact=all(not r['detail_mismatches'] for r in rows)
    res=dict(models=len(rows),predictions_reproduced=ok,details_bitwise_reproduced=exact,mismatching_models=[r for r in rows if not(r['A_match'] and r['D_match']) or r['detail_mismatches']])
    if write:(S/'s1_reproduction_check.json').write_text(json.dumps(dict(**res,rows=rows),ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(res,ensure_ascii=False)[:3000]);return res
if __name__=='__main__':run('--nowrite' not in sys.argv)
