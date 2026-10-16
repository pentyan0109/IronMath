import time
from common import *
def evaluate_all():
    import numpy as np,torch
    from data import tensor,labels,LENGTHS
    registrations_ok();ds=read('LONG_DATA_FROZEN.json');assert ds['A_SHA256']==sha(R/'PREDICTIONS_A_FROZEN.json') and ds['B_SHA256']==sha(R/'PREDICTIONS_B_FROZEN.json');results=[];resources=[];(R/'saved_predictions').mkdir()
    dump('LONG_EVALUATION_STARTED.json',dict(utc=now(),dataset_manifest_SHA256=sha(R/'LONG_DATA_FROZEN.json')))
    for mid,h,l,seed in model_ids():
        m,_=load_model(mid);s=torch.jit.script(m);t0=time.perf_counter();c0=time.process_time()
        for f in ds['files']:
            t=f['T'];p=R/f['path'];assert sha(p)==f['SHA256'];x=np.load(p)['symbols'];y=labels(x);meta=read(f'long_data/T{t}_metadata.json');hidden=torch.zeros(len(x),h);pred=np.zeros(x.shape,dtype=np.uint8);invalid=np.zeros(x.shape,dtype=bool)
            with torch.inference_mode():
                for offset in range(0,t,1024):
                    z,hidden=s.block(tensor(x[:,offset:offset+1024]),hidden);pred[:,offset:offset+1024]=z.argmax(-1).numpy();invalid[:,offset:offset+1024]=~torch.isfinite(z).all(-1).numpy()
            np.savez_compressed(R/'saved_predictions'/f'{mid}_T{t}.npz',bits=np.packbits(pred,axis=1,bitorder='little'),invalid=np.packbits(invalid,axis=1,bitorder='little'),length=t)
            for j,a in enumerate(meta):
                mask=x[j]==3;wrong=mask&((pred[j]!=y[j])|invalid[j]);where=np.flatnonzero(wrong);immediate=np.zeros(t,dtype=bool);immediate[1:]=(x[j,:-1]<2)&mask[1:];retention=np.arange(t)>=9
                holdbad=np.flatnonzero(wrong&retention);r=dict(model=mid,H=h,L=l,seed=seed,T=t,**a,R_count=int(mask.sum()),R_correct=int((mask&~wrong).sum()),sequence_exact=not bool(wrong.any()),first_error_position=int(where[0]) if len(where) else None,first_retention_R_error_position=int(holdbad[0]) if len(holdbad) else None,retention_R_count=int((mask&retention).sum()),retention_R_correct=int((mask&retention&~wrong).sum()),immediate_rewrite_R_count=int(immediate.sum()),immediate_rewrite_R_correct=int((immediate&~wrong).sum()),nonfinite_positions=int(invalid[j].sum()))
                results.append(r)
        resources.append(dict(model=mid,stage='long_evaluation',wall_seconds=time.perf_counter()-t0,CPU_seconds=time.process_time()-c0,peak_RAM_bytes=memory()));print('LONG EVALUATED',mid,round(time.perf_counter()-t0,2),flush=True)
    csvout('long_item_results.csv',results);csvout('evaluation_resources.csv',resources);dump('LONG_EVALUATION_COMPLETE.json',dict(utc=now(),models=40,items=len(results),new_training_after_freeze=0))
