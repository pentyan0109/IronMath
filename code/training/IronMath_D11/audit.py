"""IronMath D11 preservation audit (method fixed in CLARIFICATION_ANSWERS_20261015.md, independent of FF1).
usage: python -B audit.py before | after
SHA256 manifest of every file under minimum-ai-v0 except IronMath_D11/, before and after the experiment.
Changed / deleted existing files are listed (never repaired or restored).
Top-level directories that did not exist before: name and creation time only; never entered; origin not asserted."""
import hashlib, json, os, sys, time
from pathlib import Path

D = Path(__file__).resolve().parent
WORK = D.parent
EXCLUDED_TOP = {D.name}
sys.dont_write_bytecode = True


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def tops():
    out = {}
    for e in sorted(os.scandir(WORK), key=lambda e: e.name):
        st = e.stat(follow_symlinks=False)
        out[e.name] = dict(is_dir=e.is_dir(follow_symlinks=False), created=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_ctime)))
    return out


def manifest(skip=()):
    out = {}; errors = []
    for e in sorted(os.scandir(WORK), key=lambda e: e.name):
        if e.name in EXCLUDED_TOP or e.name in skip:
            continue
        if e.is_file(follow_symlinks=False):
            out[e.name] = sha(e.path)
        elif e.is_dir(follow_symlinks=False):
            for root, dirs, files in os.walk(e.path):
                dirs.sort()
                for fn in sorted(files):
                    p = os.path.join(root, fn)
                    if os.path.islink(p):
                        continue
                    try:
                        out[os.path.relpath(p, WORK).replace('\\', '/')] = sha(p)
                    except OSError as ex:
                        errors.append(dict(path=os.path.relpath(p, WORK), error=repr(ex)))
    return out, errors


def before():
    p = D / 'audit' / 'manifest_before.json'; assert not p.exists(), 'before manifest already taken'
    p.parent.mkdir(exist_ok=True); t0 = time.strftime('%Y-%m-%d %H:%M:%S'); t = time.perf_counter()
    tp = tops(); m, err = manifest()
    p.write_text(json.dumps(dict(started=t0, finished=time.strftime('%Y-%m-%d %H:%M:%S'), seconds=time.perf_counter() - t, top_level=tp,
                                 n_files=len(m), read_errors=err, excluded=sorted(EXCLUDED_TOP), files=m), ensure_ascii=False), encoding='utf-8')
    (D / 'audit' / 'manifest_before.sha256').write_text(sha(p) + '  manifest_before.json\n', encoding='utf-8')
    print('before', len(m), 'files', round(time.perf_counter() - t), 's errors', len(err))


def after():
    b = json.loads((D / 'audit' / 'manifest_before.json').read_text(encoding='utf-8'))
    t0 = time.strftime('%Y-%m-%d %H:%M:%S'); t = time.perf_counter(); tp = tops()
    new_top = {k: v for k, v in tp.items() if k not in b['top_level'] and k not in EXCLUDED_TOP}
    m, err = manifest(skip=set(new_top))  # new top-level dirs are not entered
    bf = b['files']
    changed = sorted(k for k in bf if k in m and m[k] != bf[k]); deleted = sorted(set(bf) - set(m)); added = sorted(set(m) - set(bf))
    res = dict(before_started=b['started'], before_finished=b['finished'], after_started=t0, after_finished=time.strftime('%Y-%m-%d %H:%M:%S'),
               n_before=len(bf), n_after=len(m), n_changed=len(changed), n_deleted=len(deleted), n_added_in_existing_tops=len(added),
               changed=changed, deleted=deleted, added_in_existing_tops=added, read_errors_after=err,
               new_top_level_dirs_not_entered=new_top, existing_unchanged=not (changed or deleted), excluded=sorted(EXCLUDED_TOP),
               note='changes are recorded only; nothing was repaired or restored. Origin of new top-level entries is not asserted.')
    (D / 'audit' / 'preservation_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('changed', len(changed), 'deleted', len(deleted), 'added', len(added), 'new tops', list(new_top))


if __name__ == '__main__':
    {'before': before, 'after': after}[sys.argv[1]]()
