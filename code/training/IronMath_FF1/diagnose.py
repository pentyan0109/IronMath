import time,itertools
from common import *
def diagnose_all():
    import numpy as np,torch
    from data import draw,tensor
    registrations_ok();assert read('TRAINING_COMPLETE.json')['models']==40;assert not (R/'PREDICTIONS_A_FROZEN.json').exists()
    prefixes=draw(30000,4,8);dump('diagnostic_prefixes.json',dict(seed=30000,symbols=prefixes.tolist()));rows=[];pairs=[];pred=[];resources=[];(R/'diagnostic_traces').mkdir()
    for mid,h,l,seed in model_ids():
        t=time.perf_counter();cpu=time.process_time();m,asset=load_model(mid);s=torch.jit.script(m);start=[]
        with torch.inference_mode():
            xx=tensor(draw(31000,8,16));normal=m(xx);hh=torch.zeros(8,h);zs=[]
            for pos in range(16):hh,zz=m.step(xx[:,pos],hh);zs.append(zz)
            assert torch.equal(normal.argmax(-1),torch.stack(zs,1).argmax(-1)) and torch.equal(normal.argmax(-1),s(xx).argmax(-1)), 'trained checkpoint step/forward mismatch'
        with torch.inference_mode():
            for prefix in prefixes:
                for value in (0,1):
                    a=np.append(prefix,value).astype(np.uint8)[None,:];z,h0=s.block(tensor(a),torch.zeros(1,h));start.append(h0[0])
            starts=torch.stack(start)
        units=[];pairflags={}
        for symbol in (2,3):
            with torch.inference_mode():ends,moves,margins=s.repeat(starts.clone(),symbol,20000)
            en=ends.numpy();mv=moves.numpy();mg=margins.numpy();neutral='N' if symbol==2 else 'R';prefixrows=[]
            for p in range(4):
                rr=[]
                for v in (0,1):
                    j=p*2+v;terminal=ends[j].detach().clone();conv=bool(np.isfinite(en[j]).all() and np.isfinite(mv[-100:,j]).all() and np.max(mv[-100:,j])<1e-6)
                    xx=torch.zeros(1,4);xx[0,symbol]=1;xr=torch.zeros(1,4);xr[0,3]=1
                    with torch.inference_mode():_,z=m.step(xr,terminal[None]) if symbol==2 else (terminal[None],m.head(terminal[None]))
                    finite=bool(torch.isfinite(z).all());out=int(z.argmax(-1)) if finite else None;rho=None;eigs=[]
                    if conv:
                        jac=torch.autograd.functional.jacobian(lambda hh:m.step(xx,hh[None])[0][0],terminal)
                        ev=torch.linalg.eigvals(jac.double());rho=float(ev.abs().max());eigs=[[float(x.real),float(x.imag)] for x in ev]
                    good=conv and rho is not None and rho<1 and out==v
                    negative=np.flatnonzero(mg[:,j]<0);invalid=np.flatnonzero(~np.isfinite(mg[:,j]));first=int(negative[0])+1 if len(negative) else None
                    r=dict(model=mid,H=h,L=l,seed=seed,neutral=neutral,value=v,prefix=p,converged=conv,last100_max_movement=float(np.max(mv[-100:,j])),readout=out,readout_finite=finite,spectral_radius=rho,retention_prediction=good,first_negative_margin_step=first,first_nonfinite_margin_step=int(invalid[0])+1 if len(invalid) else None,eigenvalues=eigs)
                    rows.append(r);rr.append(r);prefixrows.append(r)
                distance=float(torch.linalg.vector_norm(ends[2*p]-ends[2*p+1]));two=bool(rr[0]['converged'] and rr[1]['converged'] and distance>=1e-3 and all(r['spectral_radius'] is not None and r['spectral_radius']<1 for r in rr) and rr[0]['readout']==0 and rr[1]['readout']==1)
                pairs.append(dict(model=mid,H=h,L=l,seed=seed,neutral=neutral,prefix=p,distance=distance,two_attractors=two));pairflags.setdefault(neutral,[]).append(two)
            same_value_distances={str(v):[float(np.linalg.norm(en[2*i+v]-en[2*j+v])) for i,j in itertools.combinations(range(4),2)] for v in (0,1)}
            np.savez_compressed(R/'diagnostic_traces'/f'{mid}_{neutral}.npz',endpoints=en,movements=mv,margins=mg)
            for v in (0,1):
                rr=[r for r in prefixrows if r['value']==v];holds=all(r['retention_prediction'] for r in rr);bad=[r for r in rr if not r['retention_prediction']];breaks=[r['first_negative_margin_step'] for r in bad if r['first_negative_margin_step'] is not None]
                units.append(dict(value=v,neutral=neutral,predict_holds=holds,predicted_break_step=min(breaks) if breaks else None,same_value_endpoint_distances=same_value_distances[str(v)],multiple_endpoints_at_1e_minus3=any(d>=1e-3 for d in same_value_distances[str(v)])))
        pred.append(dict(model=mid,H=h,L=l,seed=seed,predict_success=all(r['predict_holds'] for r in units),units=units,two_attractors={n:all(f) for n,f in pairflags.items()},checkpoint_SHA256=asset['SHA256']))
        resources.append(dict(model=mid,stage='diagnostics',wall_seconds=time.perf_counter()-t,CPU_seconds=time.process_time()-cpu,peak_RAM_bytes=memory()));print('DIAGNOSED',mid,flush=True)
    dump('diagnostic_details.json',rows);csvout('attractor_pairs.csv',pairs);csvout('diagnostic_resources.csv',resources)
    dump('PREDICTIONS_A_FROZEN.json',dict(frozen_utc=now(),models=pred,registration_SHA256=sha(R/'PREREGISTRATION.json'),checkpoints_SHA256=sha(R/'CHECKPOINTS_FROZEN.json')))
    dump('PREDICTIONS_A_SHA256.json',dict(frozen_utc=now(),SHA256=sha(R/'PREDICTIONS_A_FROZEN.json')))
def short_B():
    import torch,numpy as np
    from data import draw,tensor,labels
    registrations_ok();a=read('PREDICTIONS_A_SHA256.json');assert sha(R/'PREDICTIONS_A_FROZEN.json')==a['SHA256'];assert not (R/'long_data').exists();pred=[]
    for mid,h,l,seed in model_ids():
        x=draw(40000,256,2*l);y=labels(x);mask=x==3;m,_=load_model(mid);s=torch.jit.script(m)
        with torch.inference_mode():z=s(tensor(x));bits=z.argmax(-1).numpy();valid=torch.isfinite(z).all(-1).numpy()
        total=int(mask.sum());correct=int(((bits==y)&valid&mask).sum());pred.append(dict(model=mid,H=h,L=l,seed=seed,correct=correct,R_count=total,accuracy=correct/total,predict_success=correct==total,dataset_symbols_SHA256=__import__('hashlib').sha256(x.tobytes()).hexdigest()))
    dump('PREDICTIONS_B_FROZEN.json',dict(frozen_utc=now(),models=pred,A_SHA256=a['SHA256']));dump('PREDICTIONS_B_SHA256.json',dict(frozen_utc=now(),SHA256=sha(R/'PREDICTIONS_B_FROZEN.json')))
