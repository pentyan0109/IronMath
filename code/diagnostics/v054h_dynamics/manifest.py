"""SHA256 manifest of minimum-ai-v0 (excluding v054h_dynamics, v054e_endurance, IronMath_V055).
usage: python -B manifest.py before|after
'after' compares with before and writes preservation_check.json. Nothing is repaired."""
import hashlib, json, os, sys, time
from pathlib import Path

T = Path(__file__).resolve().parent
WORK = T.parent
EXCLUDE = {'v054h_dynamics', 'v054e_endurance', 'IronMath_V055'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def build():
    out = {}; errors = {}
    for root, dirs, files in os.walk(WORK):
        rel = Path(root).relative_to(WORK)
        if rel == Path('.'):
            dirs[:] = [d for d in dirs if d not in EXCLUDE]
        dirs.sort()
        for name in sorted(files):
            p = Path(root) / name; key = str(p.relative_to(WORK)).replace('\\', '/')
            try:
                out[key] = sha(p)
            except OSError as e:
                errors[key] = repr(e)
    return out, errors


def main(stage):
    t = time.perf_counter(); m, err = build(); wall = time.perf_counter() - t
    (T / 'data').mkdir(exist_ok=True)
    (T / 'data' / f'manifest_{stage}.json').write_text(json.dumps(dict(root=str(WORK), excluded=sorted(EXCLUDE), files=len(m), unreadable=err, wall_seconds=wall, sha256=m), indent=1), encoding='utf-8')
    print(stage, len(m), 'files', round(wall, 1), 's', len(err), 'unreadable')
    if stage == 'after':
        b = json.loads((T / 'data' / 'manifest_before.json').read_text(encoding='utf-8'))
        before = b['sha256']
        added = sorted(set(m) - set(before)); removed = sorted(set(before) - set(m))
        changed = sorted(k for k in set(m) & set(before) if m[k] != before[k])
        res = dict(scope=f'{WORK} 配下の全ファイル（除外: {sorted(EXCLUDE)}）',
                   scope_note='V054Tと同じ範囲（minimum-ai-v0配下）。v054t_transition は対象に含む（読み取りのみのため変化しないはず）。',
                   files_before=len(before), files_after=len(m), added=added, removed=removed, changed=changed,
                   unreadable_before=b['unreadable'], unreadable_after=err,
                   identical=not (added or removed or changed), repaired=False)
        (T / 'preservation_check.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print('identical', res['identical'], 'added', len(added), 'removed', len(removed), 'changed', len(changed))


if __name__ == '__main__':
    main(sys.argv[1])
