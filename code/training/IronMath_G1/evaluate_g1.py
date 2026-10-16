from g1_utils import *
import torch
from data import longset,labels,sequential_labels,tensor,LENGTHS

def freeze_cohort(name,assets):
    key=f'{name}_PREDICTIONS_FROZEN.json'
    records=[read(f"diagnostic_records/{a['model']}.json") for a in assets]
    predictions=[{k:r[k] for k in ['model','architecture','L','seed','A','B','C_all_success','C_all_failure','D','checkpoint_SHA256']} for r in records]
    record_SHA={a['model']:sha(R/'diagnostic_records'/f"{a['model']}.json") for a in assets}
    dump(key,dict(frozen_utc=now(),cohort=name,models=predictions,diagnostic_record_SHA256=record_SHA,registration_SHA256=sha(R/'PREREGISTRATION.json')))
    dump(f'{name}_PREDICTIONS_SHA256.json',dict(utc=now(),SHA256=sha(R/key)))
    csvout(f'{name}_predictions.csv',predictions)
    print('FROZEN',name,len(assets),flush=True)

def generate_cohort(name):
    freeze=f'{name}_PREDICTIONS_FROZEN.json';h=sha(R/freeze);assert h==read(f'{name}_PREDICTIONS_SHA256.json')['SHA256']
    folder=R/f'{name}_long_data';assert not folder.exists();folder.mkdir();files=[]
    for T in LENGTHS:
        x,meta=longset(T);assert x.shape==(68,T) and sum(r['multiplicity'] for r in meta)==96
        assert np.array_equal(labels(x),sequential_labels(x)), 'long dataset oracle mismatch'
        p=folder/f'T{T}.npz';np.savez_compressed(p,symbols=x);dump(f'{name}_long_data/T{T}_metadata.json',meta)
        files.append(dict(T=T,path=p.relative_to(R).as_posix(),SHA256=sha(p),symbols_SHA256=__import__('hashlib').sha256(x.tobytes()).hexdigest(),effective_items=68,nominal_items=96))
    dump(f'{name}_LONG_DATA_FROZEN.json',dict(generated_utc=now(),cohort=name,prediction_SHA256=h,files=files))
    print('LONG DATA GENERATED',name,flush=True)

def evaluate_one(asset,cohort):
    registrations_ok();mid=asset['model'];out=f'evaluation_records/{mid}.json'
    if (R/out).exists():return read(out)
    ds=read(f'{cohort}_LONG_DATA_FROZEN.json');assert ds['prediction_SHA256']==sha(R/f'{cohort}_PREDICTIONS_FROZEN.json')
    assert any(r['model']==mid and r['checkpoint_SHA256']==asset['SHA256'] for r in read(f'{cohort}_PREDICTIONS_FROZEN.json')['models'])
    m,_=load_checkpoint(asset);s=torch.jit.script(m);results=[];t=time.perf_counter();cpu=time.process_time()
    (R/'saved_predictions').mkdir(exist_ok=True)
    for f in ds['files']:
        T=f['T'];assert sha(R/f['path'])==f['SHA256']
        with np.load(R/f['path']) as z:x=z['symbols']
        y=labels(x);meta=read(f'{cohort}_long_data/T{T}_metadata.json');state=torch.zeros(len(x),m.state_size);pred=np.zeros_like(x);invalid=np.zeros(x.shape,dtype=bool)
        with torch.inference_mode():
            for offset in range(0,T,1024):
                z,state=s.block(tensor(x[:,offset:offset+1024]),state);pred[:,offset:offset+1024]=z.argmax(-1).numpy();invalid[:,offset:offset+1024]=~torch.isfinite(z).all(-1).numpy()
        pp=R/'saved_predictions'/f'{mid}_T{T}.npz';assert not pp.exists();np.savez_compressed(pp,bits=np.packbits(pred,axis=1,bitorder='little'),invalid=np.packbits(invalid,axis=1,bitorder='little'),length=T)
        for j,md in enumerate(meta):
            mask=x[j]==3;wrong=mask&((pred[j]!=y[j])|invalid[j]);first=np.flatnonzero(wrong);nf=np.flatnonzero(invalid[j]);ret=np.arange(T)>=9;imm=np.zeros(T,bool);imm[1:]=(x[j,:-1]<2)&mask[1:]
            results.append(dict(model=mid,architecture=asset['architecture'],L=asset['L'],seed=asset['seed'],T=T,**md,R_correct=int((mask&~wrong).sum()),R_count=int(mask.sum()),sequence_exact=not bool(wrong.any()),first_error_position=int(first[0]) if len(first) else None,nonfinite_positions=int(invalid[j].sum()),first_nonfinite_position=int(nf[0]) if len(nf) else None,finite_wrong_R=int((mask&(pred[j]!=y[j])&~invalid[j]).sum()),nonfinite_R=int((mask&invalid[j]).sum()),retention_R_correct=int((mask&ret&~wrong).sum()),retention_R_count=int((mask&ret).sum()),immediate_rewrite_R_correct=int((mask&imm&~wrong).sum()),immediate_rewrite_R_count=int((mask&imm).sum()),prediction_path=pp.relative_to(R).as_posix(),prediction_SHA256=sha(pp)))
    record=dict(model=mid,architecture=asset['architecture'],L=asset['L'],seed=asset['seed'],cohort=cohort,started_after_prediction_freeze=True,completed_utc=now(),items=results,wall_seconds=time.perf_counter()-t,CPU_seconds=time.process_time()-cpu,peak_RAM_bytes=memory(),checkpoint_SHA256=asset['SHA256'],dataset_manifest_SHA256=sha(R/f'{cohort}_LONG_DATA_FROZEN.json'))
    dump(out,record);print('EVALUATED',mid,'wall',round(record['wall_seconds'],2),flush=True)
    return record
