"""Shared helpers for the inference-only re-verification (run with python -B).
Writes only under IronMath_PaperCheck/reverify/. Original code is never imported; only the model class files are loaded by path."""
import gzip, hashlib, importlib.util, json, os, platform, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
RV = Path(__file__).resolve().parent
W = RV.parents[1]  # minimum-ai-v0
OUT = RV / 'out'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def jl_read(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8-sig') as f:
        return [json.loads(l) for l in f if l.strip()]


def jl_write(p, rows):
    p = Path(p); assert RV in p.resolve().parents, p
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, 'wt', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def dump(p, x):
    p = Path(p); assert RV in p.resolve().parents, p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=1), encoding='utf-8')


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load_class(path, name):
    """load a model file by path without writing bytecode; the file's own directory is put on sys.path only during the load"""
    path = Path(path); sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f'rv_{path.stem}_{abs(hash(str(path)))}', path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(path.parent))
    return getattr(mod, name)


def check_sha(p, expected, log):
    got = sha(p); ok = got.lower() == expected.lower()
    log.append(dict(path=str(Path(p).relative_to(W)), expected=expected, actual=got, match=ok))
    if not ok:
        raise SystemExit(f'STOP: SHA256 mismatch {p}')
    return got


def modelhash(m):
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def env(torch, threads):
    import numpy
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__, numpy=numpy.__version__,
                threads_set=threads, threads_effective=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                mkl=torch.backends.mkl.is_available(), pid=os.getpid(), time=time.strftime('%Y-%m-%d %H:%M:%S'), executable=sys.executable)
