from g1_utils import *
import traceback

def measured(name,fn):
    t=time.perf_counter();cpu=time.process_time();result=fn();dump(f'stage_resources/{name}.json',dict(stage=name,utc=now(),wall_seconds=time.perf_counter()-t,CPU_seconds=time.process_time()-cpu,peak_RAM_bytes=memory()));print('STAGE COMPLETE',name,now(),flush=True);return result

def choose_budget(all_assets):
    # Resource fields ONLY; ALL200 model predictions were frozen before any long evaluation.
    estimates=[];actual_training=sum(a['train_wall_seconds'] for a in all_assets)
    actual_diagnostic=sum(read(f"diagnostic_records/{a['model']}.json")['diagnostic_wall_seconds']+read(f"diagnostic_records/{a['model']}.json")['short_B_wall_seconds'] for a in all_assets)
    for arch in ARCHS:
        models=[a for a in all_assets if a['architecture']==arch and a['seed']==20];times=[];eval_times=[]
        assert len(models)==2
        for a in models:
            d=read(f"diagnostic_records/{a['model']}.json");e=read(f"evaluation_records/{a['model']}.json")
            times.append(a['train_wall_seconds']+d['diagnostic_wall_seconds']+d['short_B_wall_seconds']+e['wall_seconds']);eval_times.append(e['wall_seconds'])
        mean=float(np.mean(times));em=float(np.mean(eval_times));estimates.append(dict(architecture=arch,timing_models=2,per_model_mean_seconds=mean,full40_seconds=40*mean,reduced20_seconds=20*mean,full_evaluation_seconds=40*em,reduced_evaluation_seconds=20*em))
    raw_full=sum(a['full40_seconds'] for a in estimates);actual_full=actual_training+actual_diagnostic+sum(a['full_evaluation_seconds'] for a in estimates);full=max(raw_full,actual_full);reduced=full>14400
    reduced_forecast=max(raw_full/2,actual_training+actual_diagnostic+sum(a['reduced_evaluation_seconds'] for a in estimates));estimate=reduced_forecast if reduced else full
    obj=dict(utc=now(),full_estimated_seconds=full,raw_two_model_forecast_seconds=raw_full,actual_all200_training_diagnostic_seconds=actual_training+actual_diagnostic,chosen_estimated_seconds=estimate,reduction_applied_once=reduced,seeds=list(range(20,30 if reduced else 40)),architectures=estimates,decision_uses_resource_values_only=True,within_budget=estimate<=14400,already_trained_models_retained=200,all_predictions_frozen_before_timing_evaluation=True)
    dump('BUDGET_DECISION.json',obj)
    if not obj['within_budget']:raise RuntimeError('Reduced estimate still exceeds4h; stop without further evaluation')
    print('BUDGET',obj,flush=True);return obj

def run():
    if (R/'FINAL_STATUS.json').exists() or (R/'STOPPED_ERROR.json').exists():raise RuntimeError('Prior completion/error: no automatic new runs')
    setup();registrations_ok();process_gate();assert read('QUALITY_PRETRAIN.json')['passed']
    from train_g1 import train_one
    from diagnostic_g1 import architecture_audit,diagnose_one
    from evaluate_g1 import freeze_cohort,generate_cohort,evaluate_one
    from independent_audit import audit_outputs
    from summarize_g1 import summarize
    from manifest import snapshot
    if not (R/'TRAINING_STARTED.json').exists():dump('TRAINING_STARTED.json',dict(utc=now(),registration_SHA256=sha(R/'PREREGISTRATION.json')))
    all_assets=[]
    for mid,a,l,s in conditions(range(20,40)):
        all_assets.append(read(f'assets/{mid}.json') if (R/f'assets/{mid}.json').exists() else train_one(mid,a,l,s))
    if not (R/'TRAINING_COMPLETE.json').exists():dump('TRAINING_COMPLETE.json',dict(utc=now(),models=200,seeds=list(range(20,40))))
    timing=[a for a in all_assets if a['seed']==20]
    if not (R/'g1_diagnostic_audit.json').exists():measured('architecture_diagnostic_audit',lambda:architecture_audit(timing))
    assert read('g1_diagnostic_audit.json')['passed']
    for asset in all_assets:diagnose_one(asset)
    if not (R/'main_PREDICTIONS_FROZEN.json').exists():freeze_cohort('main',all_assets)
    if not (R/'main_LONG_DATA_FROZEN.json').exists():measured('main_data_generation_and_QA',lambda:generate_cohort('main'))
    for asset in timing:evaluate_one(asset,'main')
    decision=read('BUDGET_DECISION.json') if (R/'BUDGET_DECISION.json').exists() else choose_budget(all_assets)
    assert decision['within_budget'];selected=[a for a in all_assets if a['seed'] in decision['seeds']]
    for asset in selected:evaluate_one(asset,'main')
    if not (R/'LONG_EVALUATION_COMPLETE.json').exists():dump('LONG_EVALUATION_COMPLETE.json',dict(utc=now(),models=len(selected),all_trained_models=200,timing_models_reused_once=10,new_training_after_freeze=False))
    if not (R/'SAVED_OUTPUT_AUDIT.json').exists():measured('independent_saved_output_audit',lambda:audit_outputs(selected))
    if not (R/'WORKSPACE_AFTER.json').exists():measured('workspace_after',lambda:snapshot('after'))
    summarize(selected,all_assets)

if __name__=='__main__':
    try:run()
    except Exception as e:
        if not (R/'STOPPED_ERROR.json').exists():
            dump('STOPPED_ERROR.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),no_automatic_retraining=True))
            if not (R/'REPORT_G1.md').exists():write_utf8('REPORT_G1.md','# G1 停止報告\n\n指定停止条件または実装・監査エラーにより停止。未完了のP1・P2を成功扱いしない。\n\n理由：'+repr(e)+'\n\nSTOPPED_ERROR.jsonとg1_diagnostic_audit.json（存在時）に詳細を保存。追加学習・自動修正・再学習を行わない。\n')
            preservation=None
            try:
                from manifest import snapshot
                if (R/'WORKSPACE_BEFORE.json').exists() and not (R/'WORKSPACE_AFTER.json').exists():snapshot('after')
                if (R/'WORKSPACE_COMPARISON.json').exists():preservation=read('WORKSPACE_COMPARISON.json')
            except Exception as audit_error:preservation=dict(audit_error=repr(audit_error))
            if not (R/'g1_integrity_audit.json').exists():dump('g1_integrity_audit.json',dict(utc=now(),passed=False,experiment_stopped=True,reason=repr(e),preservation=preservation,no_automatic_retraining=True))
        raise
