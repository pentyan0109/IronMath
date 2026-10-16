"""Independent saved-data checks. Deliberately does not import audit_v072 or experiment code."""
import os
os.environ['OMP_NUM_THREADS']='1'
import csv,json,gzip,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np
O=Path(__file__).resolve().parent;X=O.parent;W=O.parents[2]
def rr(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    q=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(1048576),b''):q.update(a)
    return q.hexdigest()
def truth(v):return str(v).lower() in ['true','1']
def out(n,rs):
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with (O/n).open('w',encoding='utf-8',newline='') as f:q=csv.DictWriter(f,fieldnames=keys);q.writeheader();q.writerows(rs)
def main():
    manifest=json.loads((O/'INPUT_SHA256.json').read_text(encoding='utf-8'));checks=[]
    def record(name,test,n):
        checks.append(dict(check=name,passed=bool(test),n=n));assert test,name
    floor=rr(O/'E2_ALL_CONTROLS_TRIALS.csv');lookup={(r['seed'],r['op'],r['state'],r['pair'],r['at'],r['condition']):r for r in floor};p=[json.loads(s) for s in (W/'v054t_transition/data/probes.jsonl').read_text().splitlines()];p=[r for r in p if r['k']==4096 and r['prefix']=='min' and r['suffix']=='s0'];n=0
    with gzip.open(X/'audit/E2_all_curves.jsonl.gz','rt') as f:
        for line in f:
            a=json.loads(line);probe=p[a['index']];r=lookup[(str(a['seed']),probe['op'],str(probe['state']),probe['pair'],str(a['at']),a['condition'])];v=a['margins'][a['at']-1:];bad=[i for i,z in enumerate(v) if z<0];first=bad[0]+a['at'] if bad else None
            assert (float(r['post_first_error']) if r['post_first_error'] else None)==first and truth(r['left_censored_latent_onset'])==(v[0]<0) and truth(r['right_censored_first_error'])==(not bad)
            assert int(r['post_recoveries'])==sum(v[i]<0 and v[i+1]>=0 for i in range(len(v)-1));n+=1
    record('E2 all curves independently rescored',n==1350,n)
    # Hold predictions are rescored from integer bit strings, not E3 aggregate flags.
    hp=rr(X/'E3_independent_replication/E3_HOLD_RESULTS.csv');hlookup={(r['model'],r['pid']):r for r in hp};probes={str(r['pid']):r for r in [json.loads(s) for s in (W/'v054t_transition/data/probes.jsonl').read_text().splitlines()]};outcome=defaultdict(list);n=0
    with gzip.open(X/'E3_independent_replication/predictions.jsonl.gz','rt') as f:
        for line in f:
            a=json.loads(line)
            if a['kind']!='hold':continue
            assert a['prediction'] is not None,'missing finite mask/prediction: cannot infer'
            r=hlookup[(a['model'],a['id'])];p=probes[a['id']];delta=int(a['prediction'],16)^int(a['truth'],16);bits=(delta>>p['hold_start'])&((1<<p['k'])-1);first=(bits&-bits).bit_length() if bits else None
            assert (float(r['first_hold_error']) if r['first_hold_error'] else None)==first;assert truth(r['hold_censored'])==(first is None);outcome[(a['model'],a['op'],a['pair'],str(a['state']))].append(first);n+=1
    record('E3 hold first errors from saved integer bit predictions',n==15360,n)
    sides=rr(O/'E3_ENDPOINT_SIDE_OUTCOME.csv');diag={(r['model'],r['op'],r['pair'],r['state']):r for r in rr(X/'E3_independent_replication/E3_DIAGNOSTICS.csv')}
    for r in sides:
        key=tuple(r[k] for k in ['model','op','pair','state']);failed=any(t is not None for t in outcome[key]);assert (r['outcome']=='failure')==failed;assert truth(r['endpoint_correct'])==(int(diag[key]['endpoint_output'])==int(diag[key]['target']))
    record('E3 side readout and all-240 outcomes',len(sides)==240,len(sides))
    pairs=rr(O/'E3_ENDPOINT_OUTCOME.csv')
    for p in pairs:
        s=[r for r in sides if all(r[k]==p[k] for k in ['model','op','pair'])];assert len(s)==2 and (p['outcome']=='failure')==any(r['outcome']=='failure' for r in s)
    record('E3 pair outcome uses either carry side',len(pairs)==80,len(pairs))
    # Inspect saved checkpoint tensors only; never construct a model or run forward.
    import torch
    torch.set_num_threads(1);schemas=[];refpath=X/'E3_independent_replication/checkpoints/i100_c0/best_validation.pt';ref=torch.load(refpath,map_location='cpu',weights_only=False)['model'];refshape=[(k,list(v.shape),str(v.dtype)) for k,v in ref.items()]
    for directory in ['IronMath_V053','IronMath_V054','IronMath_V055']:
        for cp in sorted((W/directory/'checkpoints').glob('*/best_validation.pt')):
            manifest[str(cp)]={'bytes':cp.stat().st_size,'sha256':sha(cp)};z=torch.load(cp,map_location='cpu',weights_only=False);sd=z['model'];shape=[(k,list(v.shape),str(v.dtype)) for k,v in sd.items()];schemas.append(dict(checkpoint=str(cp.relative_to(W)),parameters=sum(v.numel() for v in sd.values()),keys_shapes_dtypes_order_match_E3=shape==refshape,initial_SHA256=z.get('initial_SHA256',z.get('initial_sha256','missing')),schema=json.dumps(shape)))
    models=rr(X/'E3_independent_replication/E3_MODEL_RESULTS.csv')
    for m in models:
        cp=X/f"E3_independent_replication/checkpoints/{m['model']}/best_validation.pt";manifest[str(cp)]={'bytes':cp.stat().st_size,'sha256':sha(cp)};z=torch.load(cp,map_location='cpu',weights_only=False);assert z['initial_SHA256']==m['initial_SHA256'];assert sha(cp)==m['selected_checkpoint_SHA256'];assert [(k,list(v.shape),str(v.dtype)) for k,v in z['model'].items()]==refshape
    out('E3_CHECKPOINT_SCHEMA_AUDIT.csv',schemas);record('E3 selected checkpoint metadata and SHA',True,len(models))
    # Prior saved audit evidence reconstructed initial states, no reconstruction rerun here.
    for p in [X/'audit/E3_TRAINING_CHECKS.csv',X/'audit/INITIALIZATION_REPRODUCTION.json']:
        manifest[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
    out('INDEPENDENT_CHECKS.csv',checks)
    original_changes=[p for p,v in manifest.items() if sha(p)!=v['sha256']];record('All consumed original sources SHA unchanged',not original_changes,len(manifest));out('INDEPENDENT_CHECKS.csv',checks)
    (O/'INPUT_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (O/'INDEPENDENT_AUDIT.json').write_text(json.dumps(dict(passed=True,checks=checks,new_model_calls=0,new_training=0,original_changes=original_changes),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Independent saved-data audit PASS; no model forward or new models',flush=True)
if __name__=='__main__':main()
