"""Preservation audit. usage: python -B preserve.py before|after
before/after: SHA256 of every file under the read-from experiment folders and of the external checkpoints/model files,
plus (size, mtime_ns) of every file in the workspace except .venv and this reverify folder. 'after' compares with 'before'."""
from rv_common import *

TARGETS = [W / 'IronMath_D11', W / 'decimal_d10', W / 'v054r_replication', W / 'v054t_transition' / 'data']
EXTRA = [Path(v['path']) for v in load(W / 'v054r_replication' / 'asset_paths.json')['files'].values()]


def snapshot():
    hashes = {}
    for t in TARGETS:
        for p in sorted(t.rglob('*')):
            if p.is_file():
                hashes[str(p.relative_to(W))] = sha(p)
    for p in EXTRA:
        hashes[str(p.relative_to(W))] = sha(p)
    stat = {}
    for root, dirs, files in os.walk(W):
        rp = Path(root)
        dirs[:] = [d for d in dirs if not (rp / d == W / '.venv' or rp / d == RV)]
        for f in files:
            p = rp / f
            try:
                st = p.stat(); stat[str(p.relative_to(W))] = [st.st_size, st.st_mtime_ns]
            except OSError as e:
                stat[str(p.relative_to(W))] = repr(e)
    return dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), hashes=hashes, stat=stat)


if __name__ == '__main__':
    mode = sys.argv[1]; snap = snapshot()
    if mode == 'before':
        dump(RV / 'preservation_before.json', snap); print('files hashed', len(snap['hashes']), 'stat', len(snap['stat']))
    else:
        b = load(RV / 'preservation_before.json'); dump(RV / 'preservation_after.json', snap)
        hchg = sorted(k for k in b['hashes'] if snap['hashes'].get(k) != b['hashes'][k])
        added = sorted(set(snap['stat']) - set(b['stat'])); removed = sorted(set(b['stat']) - set(snap['stat']))
        changed = sorted(k for k in set(b['stat']) & set(snap['stat']) if b['stat'][k] != snap['stat'][k])
        def when(k, s):
            v = s['stat'].get(k)
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(v[1] / 1e9)) if isinstance(v, list) else None
        res = dict(before=b['time'], after=snap['time'], hashed_files=len(b['hashes']), hash_changed=hchg,
                   workspace_files_before=len(b['stat']),
                   workspace_added=[dict(path=k, mtime=when(k, snap)) for k in added],
                   workspace_removed=[dict(path=k, mtime_before=when(k, b)) for k in removed],
                   workspace_changed=[dict(path=k, mtime_before=when(k, b), mtime_after=when(k, snap)) for k in changed],
                   note='Changes are listed with name and modification time only; the process that made them is not inferred.')
        dump(RV / 'preservation_comparison.json', res); print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()}))
