import sys,time,traceback,unittest,io
from common import *
def quality():
    import tests_ff1
    t=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromModule(tests_ff1);buf=io.StringIO();r=unittest.TextTestRunner(stream=buf,verbosity=2).run(suite);(R/'QUALITY_PRETRAIN.txt').write_text(buf.getvalue(),encoding='utf-8')
    dump('QUALITY_PRETRAIN.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,skipped=len(r.skipped),long_generation_intentionally_skipped=True,wall_seconds=time.perf_counter()-t))
    if not r.wasSuccessful():raise RuntimeError('pretraining quality failed; stop without modifying frozen protocol')
def run():
    if (R/'FINAL_STATUS.json').exists():raise RuntimeError('FF1 already completed; no rerun permitted')
    if (R/'STOPPED_ERROR.json').exists():raise RuntimeError('prior error recorded; inspect and ask before continuing')
    setup();registrations_ok();process_gate()
    if not (R/'QUALITY_PRETRAIN.json').exists():quality()
    from train import train_all
    from diagnose import diagnose_all,short_B
    from data import generate_long
    from evaluate import evaluate_all
    from analyze import analyze
    from manifest import snapshot
    from report import report
    def stage(name,fn):
        t=time.perf_counter();c=time.process_time();fn();dump(f'stage_resources/{name}.json',dict(stage=name,utc=now(),wall_seconds=time.perf_counter()-t,CPU_seconds=time.process_time()-c,peak_RAM_bytes=memory()))
    if not (R/'TRAINING_COMPLETE.json').exists():stage('training',train_all)
    if not (R/'PREDICTIONS_A_FROZEN.json').exists():stage('diagnostics_freeze_A',diagnose_all)
    if not (R/'PREDICTIONS_B_FROZEN.json').exists():stage('short_freeze_B',short_B)
    if not (R/'LONG_DATA_FROZEN.json').exists():stage('generate_long',generate_long)
    import tests_ff1
    buf=io.StringIO();q=unittest.TextTestRunner(stream=buf,verbosity=2).run(unittest.TestSuite([tests_ff1.Tests('test_long_generator_only_after_freezes')]))
    if not (R/'QUALITY_LONG_GENERATOR.txt').exists():(R/'QUALITY_LONG_GENERATOR.txt').write_text(buf.getvalue(),encoding='utf-8')
    if not q.wasSuccessful():raise RuntimeError('long-generator quality failed; stop, no automatic repairs')
    if not (R/'LONG_EVALUATION_COMPLETE.json').exists():stage('long_evaluation',evaluate_all)
    if not (R/'OUTPUT_AUDIT.json').exists():stage('output_audit',analyze)
    if not (R/'WORKSPACE_AFTER.json').exists():stage('workspace_after',lambda:snapshot('after'))
    report()
if __name__=='__main__':
    try:run()
    except Exception as e:
        if not (R/'STOPPED_ERROR.json').exists():dump('STOPPED_ERROR.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),automatic_retraining=False))
        raise
