from g1_utils import *
import torch,itertools
from data import draw,tensor,labels

def postwrite_states(model):
    states=[]
    with torch.inference_mode():
        for prefix in draw(30000,4,8):
            for value in [0,1]:
                a=np.append(prefix,value).astype(np.uint8)[None,:]
                _,state=model.block(tensor(a),torch.zeros(1,model.state_size));states.append(state[0])
    return torch.stack(states)

def architecture_audit(assets):
    registrations_ok();checks=[]
    for asset in assets:
        m,cp=load_checkpoint(asset);s=torch.jit.script(m);x=tensor(draw(31000,8,16));state=torch.zeros(8,m.state_size);zs=[]
        with torch.inference_mode():
            direct=m(x)
            for i in range(16):state,z=m.step(x[:,i],state);zs.append(z)
            exact=torch.equal(direct,torch.stack(zs,1)) and torch.equal(direct,s(x))
            first,h=s.block(x[:,:8],torch.zeros(8,m.state_size));last,h=s.block(x[:,8:],h)
            exact=exact and torch.equal(direct,torch.cat([first,last],1))
        states=postwrite_states(m)
        for symbol in [2,3]:
            xx=torch.zeros(1,4);xx[0,symbol]=1
            for j,terminal in enumerate(states):
                ad=torch.autograd.functional.jacobian(lambda st:m.step(xx,st[None])[0][0],terminal)
                # All columns use FP32 central finite differences of the very same state map.
                eye=torch.eye(m.state_size,dtype=torch.float32)*1e-4
                with torch.inference_mode():
                    plus=m.step(xx.expand(m.state_size,-1),terminal[None]+eye)[0]
                    minus=m.step(xx.expand(m.state_size,-1),terminal[None]-eye)[0]
                    fd=((plus-minus)/(2e-4)).T
                error=float(torch.linalg.vector_norm(ad-fd)/torch.linalg.vector_norm(ad).clamp_min(1e-12))
                passed=exact and math.isfinite(error) and error<=1e-2
                checks.append(dict(model=asset['model'],architecture=asset['architecture'],L=asset['L'],seed=asset['seed'],neutral='N' if symbol==2 else 'R',start=j,state_dimension=m.state_size,forward_step_FP32_exact=exact,jacobian_relative_Frobenius_error=error,passed=passed,epsilon=1e-4))
    result=dict(utc=now(),passed=all(c['passed'] for c in checks),models=len(assets),architecture_count=len(set(a['architecture'] for a in assets)),checks=checks)
    dump('g1_diagnostic_audit.json',result)
    if not result['passed']:raise RuntimeError('Architecture diagnostic audit failed; stop before any long evaluation, do not alter tolerances or silently exclude architecture')
    print('ARCHITECTURE DIAGNOSTIC AUDIT PASS',len(checks),flush=True)

def diagnose_one(asset):
    mid=asset['model']
    if (R/'diagnostic_records'/f'{mid}.json').exists():return read(f'diagnostic_records/{mid}.json')
    assert read('g1_diagnostic_audit.json')['passed']
    registrations_ok();t=time.perf_counter();cpu=time.process_time();m,cp=load_checkpoint(asset);s=torch.jit.script(m);starts=postwrite_states(s)
    details=[];pairs=[];units=[];flags={};D_nonfinite=False;D_converged=True;analytic=[]
    (R/'diagnostic_traces').mkdir(exist_ok=True)
    for symbol,neutral in [(2,'N'),(3,'R')]:
        with torch.inference_mode():ends,moves,margins,state_finite,logit_finite=s.repeat(starts.clone(),symbol,20000)
        en=ends.numpy();mv=moves.numpy();mg=margins.numpy();sf=state_finite.numpy();lf=logit_finite.numpy();rr=[]
        D_nonfinite=D_nonfinite or not all(np.isfinite(x).all() for x in [en,mv,mg]) or not bool(sf.all() and lf.all())
        xx=torch.zeros(1,4);xx[0,symbol]=1;xr=torch.zeros(1,4);xr[0,3]=1
        for prefix in range(4):
            pair=[]
            for value in [0,1]:
                j=2*prefix+value;terminal=ends[j].detach().clone();conv=bool(np.isfinite(en[j]).all() and np.isfinite(mv[-100:,j]).all() and np.max(mv[-100:,j])<1e-6);D_converged=D_converged and conv
                with torch.inference_mode():z=m.step(xr,terminal[None])[1] if symbol==2 else m.readout(terminal[None])
                finite=bool(torch.isfinite(z).all());readout=int(z.argmax(-1)) if finite else None;rho=None
                if conv:
                    jac=torch.autograd.functional.jacobian(lambda st:m.step(xx,st[None])[0][0],terminal)
                    rho=float(torch.linalg.eigvals(jac.double()).abs().max())
                good=conv and rho is not None and rho<1 and readout==value
                neg=np.flatnonzero(mg[:,j]<0);inv=np.flatnonzero(~np.isfinite(mg[:,j]))
                r=dict(model=mid,architecture=asset['architecture'],L=asset['L'],seed=asset['seed'],neutral=neutral,prefix=prefix,value=value,state_dimension=m.state_size,converged=conv,last100_max_movement=float(np.max(mv[-100:,j])),spectral_radius=rho,readout=readout,readout_finite=finite,retention_prediction=good,first_negative_margin_step=int(neg[0])+1 if len(neg) else None,first_nonfinite_margin_step=int(inv[0])+1 if len(inv) else None)
                details.append(r);rr.append(r);pair.append(r)
            distance=float(torch.linalg.vector_norm(ends[2*prefix]-ends[2*prefix+1]));two=all(r['converged'] and r['spectral_radius']<1 for r in pair) and distance>=1e-3 and pair[0]['readout']==0 and pair[1]['readout']==1
            pairs.append(dict(model=mid,neutral=neutral,prefix=prefix,distance=distance,two_attractors=two));flags.setdefault(neutral,[]).append(two)
        for value in [0,1]:
            vr=[r for r in rr if r['value']==value];breaks=[r['first_negative_margin_step'] for r in vr if not r['retention_prediction'] and r['first_negative_margin_step'] is not None]
            units.append(dict(value=value,neutral=neutral,predict_holds=all(r['retention_prediction'] for r in vr),predicted_break_step=min(breaks) if breaks else None,same_value_endpoint_distances=[float(np.linalg.norm(en[2*i+value]-en[2*j+value])) for i,j in itertools.combinations(range(4),2)]))
        np.savez_compressed(R/'diagnostic_traces'/f'{mid}_{neutral}.npz',endpoints=en,movements=mv,margins=mg,state_finite=sf,logit_finite=lf)
        if asset['architecture']=='linear':
            with torch.inference_mode():lam=torch.sigmoid(m.theta).double().numpy();q=m.core(m.projection(xx))[0].double().numpy()
            if not ((lam>0)&(lam<1)).all():raise RuntimeError('FP32 lambda outside strict (0,1); record and stop, no unregistered clipping')
            point=q/(1-lam);tau=-1/np.log(lam)
            differences=[]
            for j,r in enumerate(rr):
                diff=float(np.linalg.norm(en[j].astype(np.float64)-point)) if r['converged'] else None;differences.append(diff)
                analytic.append(dict(model=mid,neutral=neutral,prefix=r['prefix'],value=r['value'],converged=r['converged'],analytic_numerical_L2_difference=diff,lambda_max=float(lam.max()),lambda_median=float(np.median(lam)),tau_max=float(tau.max()),tau_median=float(np.median(tau))))
            (R/'linear_analytic').mkdir(exist_ok=True);np.savez_compressed(R/'linear_analytic'/f'{mid}_{neutral}.npz',lambdas=lam,taus=tau,q=q,analytic_point=point,numerical_endpoints=en)
    diag_wall=time.perf_counter()-t;diag_cpu=time.process_time()-cpu;t=time.perf_counter();cpu=time.process_time()
    x=draw(40000,256,2*asset['L']);y=labels(x);mask=x==3
    with torch.inference_mode():z=s(tensor(x));bits=z.argmax(-1).numpy();valid=torch.isfinite(z).all(-1).numpy()
    n=int(mask.sum());correct=int(((bits==y)&valid&mask).sum())
    result=dict(model=mid,architecture=asset['architecture'],L=asset['L'],seed=asset['seed'],A=all(u['predict_holds'] for u in units),B=correct==n,C_all_success=True,C_all_failure=False,D=D_converged and not D_nonfinite,all16_converged=D_converged,diagnostic_nonfinite=D_nonfinite,B_R_correct=correct,B_R_count=n,units=units,two_attractors={k:all(v) for k,v in flags.items()},details=details,pairs=pairs,linear_analytic=analytic,diagnostic_wall_seconds=diag_wall,diagnostic_CPU_seconds=diag_cpu,short_B_wall_seconds=time.perf_counter()-t,short_B_CPU_seconds=time.process_time()-cpu,peak_RAM_bytes=memory(),checkpoint_SHA256=asset['SHA256'],completed_utc=now())
    dump(f'diagnostic_records/{mid}.json',result);print('DIAGNOSED',mid,'A',result['A'],'D',result['D'],'wall',round(diag_wall,2),flush=True)
    return result
