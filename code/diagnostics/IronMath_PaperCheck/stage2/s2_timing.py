"""S2: measured inference time of the length-<=ell tests on the stored G1 evaluation sets (T=32,64,128,512) for every
model, checking predictions bit-for-bit against IronMath_G1/saved_predictions; then 3 repeats of both the diagnostic
(all four iteration counts) and the tests for two models per structure. Writes only into stage2/."""
import sys,json,time,csv
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
import s1_run as K
import numpy as np,torch
G=K.G;LENS=[32,64,128,512]
def labels(a):
    writes=a<2;idx=np.maximum.accumulate(np.where(writes,np.arange(a.shape[1])[None,:],-1),axis=1)
    picked=np.take_along_axis(a,np.maximum(idx,0),axis=1);return np.where(idx<0,0,picked).astype(np.uint8)
def datasets():
    man=K.load(G/'main_LONG_DATA_FROZEN.json');out={}
    for f in man['files']:
        if f['T'] in LENS:
            assert K.sha(G/f['path'])==f['SHA256']
            with np.load(G/f['path']) as z:out[f['T']]=z['symbols'].copy()
    return out
def btest(m,data):
    """Returns per-length wall/CPU time, per-length success, predictions. Script time is measured separately."""
    t=time.perf_counter();c=time.process_time();s=torch.jit.script(m);script=(time.perf_counter()-t,time.process_time()-c);res={}
    for T in LENS:
        x=data[T];t=time.perf_counter();c=time.process_time()
        y=labels(x);state=torch.zeros(len(x),m.state_size);pred=np.zeros_like(x);invalid=np.zeros(x.shape,dtype=bool)
        with torch.inference_mode():
            for off in range(0,T,1024):
                z,state=s.block(K.tensor(x[:,off:off+1024]),state);pred[:,off:off+1024]=z.argmax(-1).numpy();invalid[:,off:off+1024]=~torch.isfinite(z).all(-1).numpy()
        mask=x==3;success=not bool((mask&((pred!=y)|invalid)).any())
        res[T]=dict(wall=time.perf_counter()-t,CPU=time.process_time()-c,success=success,pred=pred,invalid=invalid)
    return script,res
def saved_match(mid,T,pred,invalid):
    with np.load(G/'saved_predictions'/f'{mid}_T{T}.npz') as z:
        b=np.unpackbits(z['bits'],axis=1,bitorder='little',count=T);v=np.unpackbits(z['invalid'],axis=1,bitorder='little',count=T)
    return bool(np.array_equal(b,pred) and np.array_equal(v.astype(bool),invalid))
def main():
    data=datasets();assets=[K.load(p) for p in sorted((G/'assets').glob('*.json'))]
    scores={}
    with (S.parent/'memory_length_scores.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['study']=='G1':scores[r['model'],int(r['length'])]=r['success']=='True'
    others,_=K.processes();log=S/'s2_process_checks.jsonl'
    with log.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=K.utc(),stage='start',other_processes=others),ensure_ascii=False)+'\n')
    rows=[]
    for i,a in enumerate(assets):
        m=K.load_model(a);script,res=btest(m,data)
        for T in LENS:rows.append(dict(model=a['model'],architecture=a['architecture'],L=a['L'],seed=a['seed'],T=T,wall_seconds=res[T]['wall'],CPU_seconds=res[T]['CPU'],success=res[T]['success'],saved_predictions_bitwise_match=saved_match(a['model'],T,res[T]['pred'],res[T]['invalid']),stage1_success_match=res[T]['success']==scores[a['model'],T],script_wall_seconds=script[0]))
        if i%40==39:print('S2 btest',i+1,flush=True)
    with (S/'s2_btest_times.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    # repeats: two models per structure (L8_s20 and L16_s20), 3 times, diagnostic (4 counts) and tests
    rep=[]
    for arch in K_ARCH:
        for mid in [f'{arch}_L8_s20',f'{arch}_L16_s20']:
            a=[x for x in assets if x['model']==mid][0];m=K.load_model(a)
            for k in range(3):
                for count in K.COUNTS:
                    _,T,C,_=K.run_count(m,a,count);rep.append(dict(model=mid,architecture=arch,repeat=k+1,method='A',iterations=count,ell=None,wall_seconds=T['total_excluding_script'],prep=T['prep'],iterate=T['iterate'],readout=T['readout'],jacobian_eig=T['jacobian_eig']))
                _,res=btest(m,data);acc=0.
                for T in LENS:acc+=res[T]['wall'];rep.append(dict(model=mid,architecture=arch,repeat=k+1,method='B_cum',iterations=None,ell=T,wall_seconds=acc))
        print('S2 repeats',arch,flush=True)
    with (S/'s2_repeats.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in rep for k in r)));w.writeheader();w.writerows(rep)
    others,_=K.processes()
    with log.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=K.utc(),stage='end',other_processes=others),ensure_ascii=False)+'\n')
    print('S2 done',all(r['saved_predictions_bitwise_match'] and r['stage1_success_match'] for r in rows),flush=True)
K_ARCH=['residual_gru','gru','lstm','rnn','linear']
if __name__=='__main__':main()
