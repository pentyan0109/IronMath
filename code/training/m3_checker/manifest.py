"""Preservation audit: SHA256 manifest of minimum-ai-v0 (excluding m3_checker, IronMath_V055, v054h_dynamics, v054e_endurance).
usage: python -B manifest.py before|after|compare [tag]   (tag: suffix for a second audit round, e.g. resume)"""
import hashlib, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXCLUDE = {'m3_checker', 'IronMath_V055', 'v054h_dynamics', 'v054e_endurance'}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def scan():
    out = {}; errors = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = Path(dirpath).relative_to(ROOT)
        if rel == Path('.'):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE]
        for fn in filenames:
            p = Path(dirpath) / fn; key = (rel / fn).as_posix()
            try:
                out[key] = [sha256(p), p.stat().st_size]
            except OSError as e:
                errors[key] = str(e)
    return out, errors


if __name__ == '__main__':
    mode = sys.argv[1]; tag = ('_' + sys.argv[2]) if len(sys.argv) > 2 else ''
    if mode in ('before', 'after'):
        t = time.time(); files, errors = scan()
        json.dump(dict(mode=mode, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), root=str(ROOT), excluded=sorted(EXCLUDE),
                       n_files=len(files), errors=errors, files=files, seconds=round(time.time() - t, 1)),
                  open(HERE / 'data' / f'manifest_{mode}{tag}.json', 'w'), indent=0)
        print(mode, len(files), 'files', len(errors), 'errors', round(time.time() - t, 1), 's')
    else:
        b = json.load(open(HERE / 'data' / f'manifest_before{tag}.json')); a = json.load(open(HERE / 'data' / f'manifest_after{tag}.json'))
        fb, fa = b['files'], a['files']
        changed = sorted(k for k in fb.keys() & fa.keys() if fb[k][0] != fa[k][0])
        removed = sorted(fb.keys() - fa.keys()); added = sorted(fa.keys() - fb.keys())
        top = lambda ks: sorted({k.split('/')[0] for k in ks})
        res = dict(before_time=b['time'], after_time=a['time'], n_before=len(fb), n_after=len(fa),
                   unchanged=len(fb.keys() & fa.keys()) - len(changed), changed=changed, removed=removed, added=added,
                   top_level_dirs_with_differences=top(changed + removed + added),
                   read_errors_before=b['errors'], read_errors_after=a['errors'],
                   identical=not (changed or removed or added))
        json.dump(res, open(HERE / f'preservation_check{tag}.json', 'w'), indent=1, ensure_ascii=False)
        print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 20 else f'{len(v)} items') for k, v in res.items()}, ensure_ascii=False, indent=1))
