"""S1 diagnostic sensitivity run (post hoc). Re-runs the G1 diagnostic at 500/2000/5000/20000 iterations with
the byte-identical G1 model code (stage2/g1_code/models.py, SHA256-checked). Thresholds are applied afterwards.
Also records per-phase wall/CPU time, which S2 uses. Reads IronMath_G1 only; writes only into stage2/."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import sys,json,time,math,hashlib,datetime,subprocess,itertools
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;W=S.parent.parent;G=W/'IronMath_G1'
MODELS_SHA='abdc03640b3f56139aa59edba1fe40a77f8d4e183e9b73e6aed842878d1eee2b'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
assert sha(S/'g1_code'/'models.py')==MODELS_SHA==json.loads((G/'PREREGISTRATION.json').read_text(encoding='utf-8'))['code_SHA256']['models.py']
sys.path.insert(0,str(S/'g1_code'))
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
from models import make
COUNTS=[500,2000,5000,20000];P=[.1,.1,.5,.3]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def draw(seed,n,t):return np.random.default_rng(seed).choice(4,size=(n,t),p=P).astype(np.uint8)
def tensor(a):return torch.nn.functional.one_hot(torch.from_numpy(a.astype(np.int64)),4).float()
def clean(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x
def processes():
    q=subprocess.run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python|julia|wsl.exe'} | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3"],capture_output=True,text=True)
    p=json.loads(q.stdout) if q.stdout.strip() else [];p=[p] if isinstance(p,dict) else p
    own={os.getpid(),os.getppid()};return [x for x in p if x['ProcessId'] not in own and x['ParentProcessId'] not in own],p
def load_model(asset):
    p=G/asset['path'];assert sha(p)==asset['SHA256']
    cp=torch.load(p,map_location='cpu',weights_only=False);m=make(cp['architecture']).eval();m.load_state_dict(cp['model'])
    for q in m.parameters():q.requires_grad_(False)
    return m
def postwrite_states(model):
    states=[]
    with torch.inference_mode():
        for prefix in draw(30000,4,8):
            for value in [0,1]:
                a=np.append(prefix,value).astype(np.uint8)[None,:]
                _,state=model.block(tensor(a),torch.zeros(1,model.state_size));states.append(state[0])
    return torch.stack(states)
def decay(mv):
    """Descriptive summary of the movement series over the last <=1000 steps."""
    w=mv[-1000:];ok=np.isfinite(w)&(w>0)
    out=dict(window=int(len(w)),first=float(w[0]),last=float(w[-1]),ratio_last_over_first=float(w[-1]/w[0]) if w[0]>0 and np.isfinite(w[0]) and np.isfinite(w[-1]) else None)
    if ok.sum()>=10:
        x=np.arange(len(w))[ok];sl=np.polyfit(x,np.log(w[ok].astype(np.float64)),1)[0];out['per_step_geometric_factor']=float(np.exp(sl))
        lw=np.log(w[ok].astype(np.float64));pred=np.polyval(np.polyfit(x,lw,1),x);out['log_linear_R2']=float(1-((lw-pred)**2).sum()/max(((lw-lw.mean())**2).sum(),1e-300))
    else:out['per_step_geometric_factor']=None;out['log_linear_R2']=None
    return out
def run_count(m,asset,count):
    """One full diagnostic at a given iteration count; returns trajectories, timings, endpoint arrays."""
    T={};C={}
    t=time.perf_counter();c=time.process_time();s=torch.jit.script(m);T['script']=time.perf_counter()-t;C['script']=time.process_time()-c
    t=time.perf_counter();c=time.process_time();starts=postwrite_states(s);T['prep']=time.perf_counter()-t;C['prep']=time.process_time()-c
    t=time.perf_counter();c=time.process_time();runs={}
    for symbol in (2,3):
        with torch.inference_mode():runs[symbol]=s.repeat(starts.clone(),symbol,count)
    T['iterate']=time.perf_counter()-t;C['iterate']=time.process_time()-c
    t=time.perf_counter();c=time.process_time();reads={}
    xr=torch.zeros(1,4);xr[0,3]=1
    for symbol in (2,3):
        ends=runs[symbol][0]
        for j in range(8):
            with torch.inference_mode():z=m.step(xr,ends[j][None])[1] if symbol==2 else m.readout(ends[j][None])
            fin=bool(torch.isfinite(z).all());reads[symbol,j]=(int(z.argmax(-1)) if fin else None,fin)
    T['readout']=time.perf_counter()-t;C['readout']=time.process_time()-c
    t=time.perf_counter();c=time.process_time();rhos={}
    for symbol in (2,3):
        xx=torch.zeros(1,4);xx[0,symbol]=1;ends=runs[symbol][0]
        for j in range(8):
            terminal=ends[j].detach().clone()
            if bool(torch.isfinite(terminal).all()):
                jac=torch.autograd.functional.jacobian(lambda st:m.step(xx,st[None])[0][0],terminal)
                ev=torch.linalg.eigvals(jac.double());rhos[symbol,j]=float(ev.abs().max()) if bool(torch.isfinite(ev).all()) else None
            else:rhos[symbol,j]=None
    T['jacobian_eig']=time.perf_counter()-t;C['jacobian_eig']=time.process_time()-c
    T['total_excluding_script']=T['prep']+T['iterate']+T['readout']+T['jacobian_eig'];C['total_excluding_script']=C['prep']+C['iterate']+C['readout']+C['jacobian_eig']
    traj=[];arrays={}
    for symbol,neutral in ((2,'N'),(3,'R')):
        ends,mv,mg,sf,lf=[x.numpy() for x in runs[symbol]];arrays[neutral]=ends
        run_nonfinite=not all(np.isfinite(x).all() for x in (ends,mv,mg)) or not bool(sf.all() and lf.all())
        for j in range(8):
            prefix,value=divmod(j,2);last=mv[-100:,j];neg=np.flatnonzero(mg[:,j]<0)
            r=dict(neutral=neutral,prefix=prefix,value=value,endpoint_finite=bool(np.isfinite(ends[j]).all()),last100_finite=bool(np.isfinite(last).all()),last100_max_movement=float(np.max(last)),
                   spectral_radius=rhos[symbol,j],readout=reads[symbol,j][0],readout_finite=reads[symbol,j][1],first_negative_margin_step=int(neg[0])+1 if len(neg) else None,
                   run_nonfinite=run_nonfinite,movement_decay=decay(mv[:,j]))
            if asset['architecture']=='lstm' and r['endpoint_finite']:
                st=torch.from_numpy(ends[j].copy())[None];h,cc=st[:,:64],st[:,64:];xx=torch.zeros(1,4);xx[0,symbol]=1
                with torch.inference_mode():
                    g=m.core.weight_ih@m.projection(xx)[0]+m.core.bias_ih+m.core.weight_hh@h[0]+m.core.bias_hh;f=torch.sigmoid(g[64:128])
                r.update(c_L2=float(torch.linalg.vector_norm(cc)),c_max_abs=float(cc.abs().max()),forget_gate_mean=float(f.mean()),forget_gate_min=float(f.min()),forget_gate_max=float(f.max()))
            traj.append(r)
    return traj,T,C,arrays
def run(models=None,tag='main'):
    out=S/'s1_records';out.mkdir(exist_ok=True);ep=S/'s1_endpoints';ep.mkdir(exist_ok=True)
    assets=[load(p) for p in sorted((G/'assets').glob('*.json'))]
    if models:assets=[a for a in assets if a['model'] in models]
    others,_=processes();log=S/f's1_process_checks_{tag}.jsonl'
    with log.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=utc(),stage='start',other_processes=others),ensure_ascii=False)+'\n')
    for i,a in enumerate(assets):
        rp=out/f"{a['model']}.json"
        if rp.exists():continue
        t=time.perf_counter();m=load_model(a);load_s=time.perf_counter()-t;rec=dict(model=a['model'],architecture=a['architecture'],L=a['L'],seed=a['seed'],checkpoint_SHA256=a['SHA256'],load_wall_seconds=load_s,counts={})
        arr={}
        for count in COUNTS:
            traj,T,C,arrays=run_count(m,a,count);rec['counts'][str(count)]=dict(trajectories=traj,wall_seconds=T,CPU_seconds=C)
            for k,v in arrays.items():arr[f'{k}_{count}']=v
        rec['completed_utc']=utc();np.savez_compressed(ep/f"{a['model']}.npz",**arr)
        rp.write_text(json.dumps(clean(rec),ensure_ascii=False,indent=1,allow_nan=False),encoding='utf-8')
        if i%20==19:
            others,_=processes()
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=utc(),stage=f'after_{i+1}',other_processes=others),ensure_ascii=False)+'\n')
        print('S1',i+1,a['model'],round(sum(rec['counts'][str(c)]['wall_seconds']['total_excluding_script'] for c in COUNTS),2),flush=True)
    others,_=processes()
    with log.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(utc=utc(),stage='end',other_processes=others),ensure_ascii=False)+'\n')
if __name__=='__main__':
    run(sys.argv[2:] if len(sys.argv)>2 else None,sys.argv[1] if len(sys.argv)>1 else 'main')
