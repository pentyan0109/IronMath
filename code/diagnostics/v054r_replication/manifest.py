"""SHA256 manifest of minimum-ai-v0 excluding v054r_replication, m3_checker, IronMath_V057.
usage: python -B manifest.py before|after
'after' hashes only top-level entries that existed 'before'; new top-level entries are listed by name only
(contents not read) as additions by other work. Nothing is repaired."""
import ctypes, hashlib, json, os, sys, time
from pathlib import Path

T = Path(__file__).resolve().parent
WORK = T.parent
EXCLUDE = {'v054r_replication', 'm3_checker', 'IronMath_V057'}


def below_normal():
    try:
        k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p
        return bool(k.SetPriorityClass(ctypes.c_void_p(k.GetCurrentProcess()), 0x4000))
    except Exception as e:
        return repr(e)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def build(allowed_top=None):
    out = {}; errors = {}; top = sorted(os.listdir(WORK)); new_top = []
    for root, dirs, files in os.walk(WORK):
        rel = Path(root).relative_to(WORK)
        if rel == Path('.'):
            keep = [d for d in dirs if d not in EXCLUDE]
            if allowed_top is not None:
                new_top += [d + '/' for d in keep if d not in allowed_top]
                keep = [d for d in keep if d in allowed_top]
                new_top += [f for f in files if f not in allowed_top]
                files = [f for f in files if f in allowed_top]
            dirs[:] = keep
        dirs.sort()
        for name in sorted(files):
            p = Path(root) / name; key = str(p.relative_to(WORK)).replace('\\', '/')
            try:
                out[key] = sha(p)
            except OSError as e:
                errors[key] = repr(e)
    return out, errors, [t for t in top if t not in EXCLUDE], sorted(new_top)


def main(stage):
    pr = below_normal(); t = time.perf_counter()
    allowed = None
    if stage == 'after':
        b = json.loads((T / 'data' / 'manifest_before.json').read_text(encoding='utf-8')); allowed = set(b['top_level'])
    m, err, top, new_top = build(allowed); wall = time.perf_counter() - t
    (T / 'data').mkdir(exist_ok=True)
    (T / 'data' / f'manifest_{stage}.json').write_text(json.dumps(dict(root=str(WORK), excluded=sorted(EXCLUDE), below_normal_priority=pr, top_level=top,
                                                                       new_top_level_not_read=new_top, files=len(m), unreadable=err, wall_seconds=wall, sha256=m), indent=1), encoding='utf-8')
    print(stage, len(m), 'files', round(wall, 1), 's', len(err), 'unreadable', 'priority', pr, 'new_top', new_top)
    if stage == 'after':
        before = b['sha256']
        added = sorted(set(m) - set(before)); removed = sorted(set(before) - set(m))
        changed = sorted(k for k in set(m) & set(before) if m[k] != before[k])
        res = dict(scope=f'{WORK} 配下の全ファイル（除外: {sorted(EXCLUDE)}）',
                   files_before=len(before), files_after=len(m), added=added, removed=removed, changed=changed,
                   new_top_level_by_other_work_not_read=new_top,
                   unreadable_before=b['unreadable'], unreadable_after=err,
                   identical_within_preexisting_top_level=not (added or removed or changed), repaired=False)
        (T / 'preservation_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print('identical', res['identical_within_preexisting_top_level'], 'added', len(added), 'removed', len(removed), 'changed', len(changed))


if __name__ == '__main__':
    main(sys.argv[1])
