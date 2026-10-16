"""Read schema previews only; imports no prior experiment scripts."""
from pathlib import Path
import json,csv,gzip,sys
sys.set_int_max_str_digits(0)
W=Path(__file__).resolve().parent.parent
files=[
'v054e_endurance/results_per_problem.csv','v054e_endurance/confirmation.csv','v054e_endurance/data/main_2048.json','v054e_endurance/data/manifest.json',
'v054r_replication/data/raw/R_V055_F0.json','v054r_replication/R_fixed_points.csv','v054r_replication/R_durability.csv','v054r_replication/asset_paths.json',
'decimal_d10/data/raw/fixed_seed0.jsonl','decimal_d10/data/raw/probes_seed0.jsonl','decimal_d10/data/raw/eval_seed0.jsonl','decimal_d10/analysis_summary.json',
'IronMath_V058/problems.jsonl.gz','IronMath_V058/PREDICTIONS_V058_FROZEN.json','IronMath_V058/checkpoint_assets.json','IronMath_V058b/PREDICTIONS_V058_FROZEN.json',
'IronMath_D11/data/raw/eval_control_s0.jsonl','IronMath_D11/data/raw/probes_control_s0.jsonl','IronMath_D11/data/raw/fixed_control_s0.jsonl','IronMath_D11/data/raw/pairs_control_s0.jsonl','IronMath_D11/analysis_summary.json']
def short(x):
    if isinstance(x,dict):return {k:(short(v) if isinstance(v,(list,dict)) else str(v)[:160]) for k,v in list(x.items())[:14]}
    if isinstance(x,list):return {'list_length':len(x),'first':short(x[0]) if x else None}
    return str(x)[:160]
for f in files:
    p=W/f
    if not p.exists():print(f,'MISSING');continue
    if f.endswith('.csv'):
        with p.open(encoding='utf-8-sig',newline='') as q:r=next(csv.DictReader(q))
    elif '.jsonl' in f:
        op=gzip.open if f.endswith('.gz') else open
        with op(p,'rt',encoding='utf-8') as q:r=json.loads(next(q))
    else:r=json.loads(p.read_text(encoding='utf-8-sig'))
    print(f,json.dumps(short(r),ensure_ascii=False))
print('V054E files:',[(p.relative_to(W).as_posix(),p.stat().st_size) for p in (W/'v054e_endurance').rglob('*') if p.is_file() and p.suffix not in ['.py','.md','.csv','.json']][:30])
