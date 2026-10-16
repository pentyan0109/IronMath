import time,traceback,unittest,io
from common import *
def stage(name,fn):
    t=time.perf_counter();c=time.process_time();fn()
    dump(f'stage_resources/{name}.json',dict(stage=name,utc=now(),wall_seconds=time.perf_counter()-t,CPU_seconds=time.process_time()-c,peak_RAM_bytes=memory()))
    print('STAGE COMPLETE',name,now(),flush=True)
def diagnostic_D():
    import numpy as np
    result=[]
    for mid,h,l,seed in model_ids():
        nonfinite=False;conv=True
        for neutral in ['N','R']:
            with np.load(R/'diagnostic_traces'/f'{mid}_{neutral}.npz') as z:
                mv=z['movements'];mg=z['margins'];en=z['endpoints']
                assert mv.shape==mg.shape==(20000,8) and en.shape==(8,h)
                nonfinite=nonfinite or not all(np.isfinite(x).all() for x in [mv,mg,en])
                conv=conv and bool(np.isfinite(mv[-100:]).all() and np.max(mv[-100:])<1e-6 and np.isfinite(en).all())
        result.append(dict(model=mid,H=h,L=l,seed=seed,predict_success=conv and not nonfinite,all16_converged=conv,diagnostic_nonfinite=nonfinite))
    dump('PREDICTIONS_D_FROZEN.json',dict(frozen_utc=now(),models=result,definition=read('PREREGISTRATION.json')['D']))
    dump('PREDICTIONS_D_SHA256.json',dict(frozen_utc=now(),SHA256=sha(R/'PREDICTIONS_D_FROZEN.json')))
def freeze_all():
    for n in ['A','B','D']:
        assert sha(R/f'PREDICTIONS_{n}_FROZEN.json')==read(f'PREDICTIONS_{n}_SHA256.json')['SHA256']
    assert not (R/'long_data').exists()
    dump('PREDICTIONS_C_FROZEN.json',dict(frozen_utc=now(),models=[dict(model=m,all_success=True,all_failure=False) for m,h,l,s in model_ids()]))
    assets={n:read(f'PREDICTIONS_{n}_FROZEN.json') for n in ['A','B','C','D']}
    out=[]
    for m,h,l,s in model_ids():
        out.append(dict(model=m,H=h,L=l,seed=s,**{n:next(a for a in assets[n]['models'] if a['model']==m)['predict_success'] for n in ['A','B','D']},C_all_success=True,C_all_failure=False))
    csvout('ff1r_predictions.csv',out)
    dump('ALL_PREDICTIONS_FROZEN.json',dict(frozen_utc=now(),SHA256={n:sha(R/f'PREDICTIONS_{n}_FROZEN.json') for n in ['A','B','C','D']},prediction_csv_SHA256=sha(R/'ff1r_predictions.csv')))
def run():
    if (R/'FINAL_STATUS.json').exists() or (R/'STOPPED_ERROR.json').exists():raise RuntimeError('Already completed or stopped; no automatic rerun')
    setup();registrations_ok();process_gate()
    if not (R/'QUALITY_PRETRAIN.json').exists():
        import tests_ff1
        b=io.StringIO();q=unittest.TextTestRunner(stream=b,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests_ff1))
        (R/'QUALITY_PRETRAIN.txt').write_text(b.getvalue(),encoding='utf-8');dump('QUALITY_PRETRAIN.json',dict(passed=q.wasSuccessful(),tests=q.testsRun,skipped=len(q.skipped),utc=now()))
        assert q.wasSuccessful()
    from train import train_all
    from diagnose import diagnose_all,short_B
    from data import generate_long
    from evaluate import evaluate_all
    from analyze import analyze
    from manifest import snapshot
    from summarize_ff1r import summarize
    if not (R/'TRAINING_COMPLETE.json').exists():stage('training',train_all)
    if not (R/'PREDICTIONS_A_FROZEN.json').exists():stage('diagnostics_A',diagnose_all)
    if not (R/'PREDICTIONS_D_FROZEN.json').exists():stage('diagnostics_D',diagnostic_D)
    if not (R/'PREDICTIONS_B_FROZEN.json').exists():stage('short_B',short_B)
    if not (R/'ALL_PREDICTIONS_FROZEN.json').exists():stage('freeze_all',freeze_all)
    if not (R/'LONG_DATA_FROZEN.json').exists():
        f=read('ALL_PREDICTIONS_FROZEN.json')
        for n,h in f['SHA256'].items():assert sha(R/f'PREDICTIONS_{n}_FROZEN.json')==h
        stage('generate_long',generate_long)
    if not (R/'QUALITY_LONG.json').exists():
        import tests_ff1
        b=io.StringIO();q=unittest.TextTestRunner(stream=b,verbosity=2).run(unittest.TestSuite([tests_ff1.Tests('test_long_generator_only_after_freezes')]))
        (R/'QUALITY_LONG.txt').write_text(b.getvalue(),encoding='utf-8');assert q.wasSuccessful();dump('QUALITY_LONG.json',dict(passed=True,utc=now()))
    if not (R/'LONG_EVALUATION_COMPLETE.json').exists():stage('long_evaluation',evaluate_all)
    if not (R/'OUTPUT_AUDIT.json').exists():stage('independent_output_audit',analyze)
    if not (R/'WORKSPACE_AFTER.json').exists():stage('workspace_after',lambda:snapshot('after'))
    summarize()
if __name__=='__main__':
    try:run()
    except Exception as e:
        if not (R/'STOPPED_ERROR.json').exists():dump('STOPPED_ERROR.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),automatic_retraining=False))
        raise
