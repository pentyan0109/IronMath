"""Record read-only assets (path + SHA256) and concurrently running Python processes."""
import hashlib, json, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


groups = {
    'V054_C1_B0_checkpoints': sorted(p for s in range(3) for p in (ROOT / f'IronMath_V054/checkpoints/C1_B0_s{s}').rglob('*') if p.is_file()),
    'V054_model_code': [ROOT / 'IronMath_V054/model.py'],
    'V054_saved_predictions_all_9_widths': [ROOT / 'IronMath_V054/predictions.jsonl.gz', ROOT / 'IronMath_V054/failure_cases.jsonl.gz'],
    'V054T': [ROOT / 'v054t_transition/results_per_problem.csv', ROOT / 'v054t_transition/data/probes.jsonl'],
    'V051_A1_code_for_stage1_reuse': [ROOT / 'IronMath_V4/v051/src/model.py'],
}
out = {g: [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha256(p)) for p in ps] for g, ps in groups.items()}
out['V054E_endurance'] = 'not used: v054e_endurance/REPORT_V054E.md absent (endurance test not finished)'
ps = subprocess.run(['powershell', '-NoProfile', '-Command',
                     "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json"],
                    capture_output=True, text=True).stdout
out['concurrent_python_processes'] = dict(recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), processes=json.loads(ps) if ps.strip() else [])
json.dump(out, open(HERE / 'asset_paths.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)
print(json.dumps(out, indent=1, ensure_ascii=False, default=str)[:4000])
