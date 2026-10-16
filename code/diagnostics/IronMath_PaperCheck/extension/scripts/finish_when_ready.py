"""Serial handoff after E3; fixed resource stop, no extra experiments."""
from common import *
import signal
def e3_processes():
    q=subprocess.run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'extension[\\/]scripts[\\/]e3.py' } | Select-Object ProcessId,CommandLine | ConvertTo-Json"],capture_output=True,text=True)
    if q.returncode:raise RuntimeError('Cannot verify owned E3 processes for resource guard: '+q.stderr)
    v=json.loads(q.stdout) if q.stdout.strip() else [];return v if isinstance(v,list) else [v]
def main():
    owned=e3_processes();dump('audit/SERIAL_HANDOFF.json',dict(utc=utc(),waiting_for='E3 COMPLETE flag',only_owned_E3_processes=owned,stages=['aggregate','independent audit','reports + v07'],no_additional_experiments=True))
    while not (R/'E3_independent_replication/COMPLETE.json').exists():
        failures=R/'audit/failures.jsonl'
        if failures.exists() and any(json.loads(x)['stage']=='E3' for x in failures.read_text(encoding='utf-8').splitlines() if x.strip()):raise RuntimeError('E3 failure recorded; preserve incomplete work, do not continue')
        if time.time()-read(R/'audit/STARTED.json')['start_epoch']>=7200:
            current=e3_processes();allowed={x['ProcessId'] for x in owned}
            for p in current:
                if p['ProcessId'] in allowed:os.kill(p['ProcessId'],signal.SIGTERM)
            dump('audit/RESOURCE_STOP.json',dict(utc=utc(),fixed_limit_seconds=7200,only_task_owned_E3_terminated=True,completed_models=len(list((R/'E3_independent_replication/checkpoints').glob('*/evaluated.json'))),unfinished_not_success=True));raise RuntimeError('Fixed resource ceiling reached')
        if not e3_processes():raise RuntimeError('E3 process exited without COMPLETE; do not infer success')
        time.sleep(5)
    for stage in ['aggregate.py','audit_independent.py','report.py']:
        print('SERIAL STAGE',stage,flush=True)
        q=subprocess.run([sys.executable,'-B',str(R/'scripts'/stage)],cwd=W)
        if q.returncode:raise RuntimeError(f'{stage} failed; stop without additional experiments')
    print('EXTENSION FINISHED AND STOPPED',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:stage_failure('serial handoff',e);raise
