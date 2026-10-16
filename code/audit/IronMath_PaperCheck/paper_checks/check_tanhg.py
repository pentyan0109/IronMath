"""CHECK_TANHG: read residual_gate g from the selected checkpoints of the residual GRUs
in FF1, FF1-R and G1, and cross-tabulate the sign of tanh(g) against divergence.

Read-only. No model is built and no forward pass is run: the state_dict is loaded with
torch.load and only the scalar 'residual_gate' is read. Writes only into this folder.
Run from minimum-ai-v0:  .venv/Scripts/python.exe -I -B IronMath_PaperCheck/paper_checks/check_tanhg.py
"""
import csv, hashlib, json, math, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_csv(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def checkpoints():
    rows = []
    for exp, folder in [('FF1', 'IronMath_FF1'), ('FF1-R', 'IronMath_FF1R')]:
        frozen = json.loads((ROOT / folder / 'CHECKPOINTS_FROZEN.json').read_text(encoding='utf-8'))['models']
        for model, a in frozen.items():
            rows.append(dict(experiment=exp, model=model, path=ROOT / folder / a['path'], SHA256=a['SHA256'],
                             selected_update=a['selected_update']))
    for p in sorted((ROOT / 'IronMath_G1' / 'assets').glob('residual_gru_*.json')):
        a = json.loads(p.read_text(encoding='utf-8'))
        rows.append(dict(experiment='G1', model=a['model'], path=ROOT / 'IronMath_G1' / a['path'], SHA256=a['SHA256'],
                         selected_update=a['selected_update']))
    return rows


def classifications():
    c = {}
    for r in read_csv(ROOT / 'IronMath_FF1A' / 'ff1a_model_classification.csv'):
        c[('FF1', r['model'])] = (r['classification'], r['actual_success'])
    for r in read_csv(ROOT / 'IronMath_FF1R' / 'ff1r_model_results.csv'):
        c[('FF1-R', r['model'])] = (r['classification'], r['actual_success'])
    for r in read_csv(ROOT / 'IronMath_G1' / 'g1_results.csv'):
        if r['architecture'] == 'residual_gru':
            c[('G1', r['model'])] = (r['classification'], r['actual_success'])
    return c


def main():
    for name in ['CHECK_TANHG_models.csv', 'CHECK_TANHG_summary.json']:
        if (OUT / name).exists():
            sys.exit(f'{name} exists; not overwriting')
    cls = classifications()
    ff1a = {r['model']: r for r in read_csv(ROOT / 'IronMath_FF1A' / 'ff1a_residual_analysis.csv')}
    rows = []
    for a in checkpoints():
        digest = sha256(a['path'])
        assert digest == a['SHA256'], (a['model'], digest)
        cp = torch.load(a['path'], map_location='cpu', weights_only=False)
        sd = cp['model']
        g = sd['residual_gate']
        assert g.dim() == 0 and g.dtype == torch.float32
        tanh32 = float(torch.tanh(g))
        tanh64 = math.tanh(float(g))
        classification, success = cls[(a['experiment'], a['model'])]
        r = dict(experiment=a['experiment'], model=a['model'], checkpoint=str(a['path'].relative_to(ROOT)).replace('\\', '/'),
                 selected_update=a['selected_update'], SHA256_match=True, g=float(g), tanh_g_FP32=tanh32,
                 tanh_g_FP64=tanh64, tanh_g_positive=tanh32 > 0, classification=classification,
                 divergent=classification == '発散型', actual_success=success)
        if a['experiment'] == 'FF1':
            ref = ff1a[a['model']]
            r['FF1A_tanh_g_FP32'] = float(ref['tanh_g_FP32'])
            r['FF1A_match'] = float(ref['tanh_g_FP32']) == tanh32 and float(ref['g']) == float(g)
            r['FF1A_long_nonfinite'] = ref['long_nonfinite']
        rows.append(r)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / 'CHECK_TANHG_models.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for exp in ['FF1', 'FF1-R', 'G1', 'all']:
        sub = [r for r in rows if exp == 'all' or r['experiment'] == exp]
        t = {}
        for div in [True, False]:
            for pos in [True, False]:
                t[f"{'divergent' if div else 'non_divergent'}|{'tanh_g_gt_0' if pos else 'tanh_g_le_0'}"] = sum(
                    r['divergent'] == div and r['tanh_g_positive'] == pos for r in sub)
        div_g = [r['tanh_g_FP32'] for r in sub if r['divergent']]
        non_g = [r['tanh_g_FP32'] for r in sub if not r['divergent']]
        summary[exp] = dict(models=len(sub), table=t,
                            divergent_tanh_g_range=[min(div_g), max(div_g)] if div_g else None,
                            non_divergent_tanh_g_range=[min(non_g), max(non_g)] if non_g else None,
                            sign_FP32_equals_sign_FP64=all((r['tanh_g_FP32'] > 0) == (r['tanh_g_FP64'] > 0) for r in sub),
                            exact_zero_tanh_g=sum(r['tanh_g_FP32'] == 0 for r in sub))
    summary['FF1_vs_FF1A_all_match'] = all(r['FF1A_match'] for r in rows if r['experiment'] == 'FF1')
    summary['FF1A_long_nonfinite_equals_divergent'] = all(
        (r['FF1A_long_nonfinite'] == 'True') == r['divergent'] for r in rows if r['experiment'] == 'FF1')
    summary['torch'] = torch.__version__
    (OUT / 'CHECK_TANHG_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
