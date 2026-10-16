"""Stage 3: generate a fresh G1-style evaluation set (seeds only changed) and evaluate all 200 G1 models.

Rules: RULES_FRESH_EVAL.md (SHA256 frozen). No training. Checkpoints / frozen predictions / original data read-only.
Writes only into addon/fresh_eval/. Resumable per model.
Run from workspace root: .venv/Scripts/python.exe -I -B IronMath_PaperCheck/addon/fresh_eval/fresh_eval.py generate|evaluate
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "2"
import datetime, hashlib, json, sys, time, types
from pathlib import Path
import numpy as np

S = Path(__file__).resolve().parent
W = S.parent.parent.parent
G = W / "IronMath_G1"
MODELS_SHA = "abdc03640b3f56139aa59edba1fe40a77f8d4e183e9b73e6aed842878d1eee2b"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4194304), b""):
            h.update(b)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dumpx(p, obj):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, allow_nan=False)


frozen = json.loads((S / "RULES_SHA256.json").read_text(encoding="utf-8"))["SHA256"]
assert all(sha(S / f) == h for f, h in frozen.items()), "rules changed"
assert sha(S / "g1_code" / "models.py") == MODELS_SHA
# data_fresh.py must differ from G1 data.py only in the two seed lines
orig = (G / "data.py").read_text(encoding="utf-8").splitlines()
new = (S / "data_fresh.py").read_text(encoding="utf-8").splitlines()
diff = [(i, a, b) for i, (a, b) in enumerate(zip(orig, new)) if a != b]
assert len(orig) == len(new) and [d[0] for d in diff] == [24, 28], diff
assert "86000+t" in diff[0][2] and "86200+j" in diff[1][2]

stub = types.ModuleType("common")  # data_fresh imports G1 common; the generator functions do not use it
for name in ("R", "read", "sha", "dump", "now"):
    setattr(stub, name, None)
sys.modules["common"] = stub
sys.path.insert(0, str(S)); sys.path.insert(1, str(S / "g1_code"))
import torch
torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
import data_fresh as D
from models import make

LENGTHS = D.LENGTHS
assert LENGTHS == [32, 64, 128, 512, 2048, 8192, 65000]
NEW_SEEDS = {86000 + t for t in LENGTHS} | {86200 + j for j in range(8)}
USED_SEEDS = ({10000 + s for s in range(20, 40)} | {20000 + s for s in range(20, 40)} | {30000, 31000, 40000, 81000, 12, 123}
              | {50000 + t for t in LENGTHS} | {70000 + j for j in range(8)} | {84000, 85000})
DATA = S / "fresh_long_data"


def generate():
    assert not (NEW_SEEDS & USED_SEEDS), NEW_SEEDS & USED_SEEDS
    DATA.mkdir()
    files, overlap = [], {}
    for T in LENGTHS:
        x, meta = D.longset(T)
        assert x.shape == (68, T) and sum(r["multiplicity"] for r in meta) == 96
        assert np.array_equal(D.labels(x), D.sequential_labels(x)), "oracle mismatch"
        with np.load(G / "main_long_data" / f"T{T}.npz") as z:
            xo = z["symbols"]
        om = json.loads((G / "main_long_data" / f"T{T}_metadata.json").read_text(encoding="utf-8"))
        assert [(m["category"], m["value"], m["j"], m["multiplicity"]) for m in om] == \
               [(m["category"], m["value"], m["j"], m["multiplicity"]) for m in meta]
        same = {}
        for i, m in enumerate(meta):
            same.setdefault(m["category"], 0)
            same[m["category"]] += bool(np.array_equal(x[i], xo[i]))
        overlap[T] = same
        p = DATA / f"T{T}.npz"
        np.savez_compressed(p, symbols=x)
        dumpx(DATA / f"T{T}_metadata.json", meta)
        files.append(dict(T=T, path=p.relative_to(S).as_posix(), SHA256=sha(p),
                          symbols_SHA256=hashlib.sha256(x.tobytes()).hexdigest(),
                          metadata_SHA256=sha(DATA / f"T{T}_metadata.json"), effective_items=68, nominal_items=96))
    dumpx(S / "FRESH_DATA_FROZEN.json", dict(generated_utc=utc(), new_seeds=sorted(NEW_SEEDS), checked_against=sorted(USED_SEEDS),
                                              seed_overlap=[], rows_identical_to_original_by_category=overlap, files=files,
                                              rules_SHA256=frozen, data_fresh_SHA256=sha(S / "data_fresh.py")))
    print("GENERATED", json.dumps(overlap), flush=True)


def evaluate():
    fz = json.loads((S / "FRESH_DATA_FROZEN.json").read_text(encoding="utf-8"))
    pred_frozen = G / "main_PREDICTIONS_FROZEN.json"
    assert sha(pred_frozen) == json.loads((G / "main_PREDICTIONS_SHA256.json").read_text(encoding="utf-8"))["SHA256"]
    data = []
    for f in fz["files"]:
        assert sha(S / f["path"]) == f["SHA256"]
        with np.load(S / f["path"]) as z:
            x = z["symbols"]
        meta = json.loads((DATA / f"T{f['T']}_metadata.json").read_text(encoding="utf-8"))
        assert sha(DATA / f"T{f['T']}_metadata.json") == f["metadata_SHA256"]
        data.append((f["T"], x, D.labels(x), meta))
    (S / "evaluation_records").mkdir(exist_ok=True); (S / "saved_predictions").mkdir(exist_ok=True)
    assets = [json.loads(p.read_text(encoding="utf-8-sig")) for p in sorted((G / "assets").glob("*.json"))]
    assert len(assets) == 200
    t0 = time.perf_counter()
    for n, a in enumerate(assets):
        mid = a["model"]; out = S / "evaluation_records" / f"{mid}.json"
        if out.exists():
            continue
        assert sha(G / a["path"]) == a["SHA256"]
        cp = torch.load(G / a["path"], map_location="cpu", weights_only=False)
        m = make(cp["architecture"]).eval(); m.load_state_dict(cp["model"])
        for q in m.parameters():
            q.requires_grad_(False)
        s = torch.jit.script(m); results = []; t = time.perf_counter(); cpu = time.process_time()
        for T, x, y, meta in data:
            state = torch.zeros(len(x), m.state_size); pred = np.zeros_like(x); invalid = np.zeros(x.shape, dtype=bool)
            with torch.inference_mode():
                for off in range(0, T, 1024):
                    z, state = s.block(D.tensor(x[:, off:off + 1024]), state)
                    pred[:, off:off + 1024] = z.argmax(-1).numpy(); invalid[:, off:off + 1024] = ~torch.isfinite(z).all(-1).numpy()
            pp = S / "saved_predictions" / f"{mid}_T{T}.npz"
            assert not pp.exists()
            np.savez_compressed(pp, bits=np.packbits(pred, axis=1, bitorder="little"),
                                invalid=np.packbits(invalid, axis=1, bitorder="little"), length=T)
            for j, md in enumerate(meta):
                mask = x[j] == 3; wrong = mask & ((pred[j] != y[j]) | invalid[j]); first = np.flatnonzero(wrong)
                ret = np.arange(T) >= 9
                results.append(dict(model=mid, T=T, **md, R_correct=int((mask & ~wrong).sum()), R_count=int(mask.sum()),
                                    sequence_exact=not bool(wrong.any()), first_error_position=int(first[0]) if len(first) else None,
                                    nonfinite_positions=int(invalid[j].sum()), finite_wrong_R=int((mask & (pred[j] != y[j]) & ~invalid[j]).sum()),
                                    retention_R_correct=int((mask & ret & ~wrong).sum()), retention_R_count=int((mask & ret).sum()),
                                    prediction_SHA256=sha(pp)))
        rec = dict(model=mid, architecture=a["architecture"], L=a["L"], seed=a["seed"], checkpoint_SHA256=a["SHA256"],
                   items=results, wall_seconds=time.perf_counter() - t, CPU_seconds=time.process_time() - cpu,
                   completed_utc=utc(), fresh_data_frozen_SHA256=sha(S / "FRESH_DATA_FROZEN.json"))
        dumpx(out, rec)
        print("EVAL", n + 1, mid, round(rec["wall_seconds"], 1), "elapsed", round(time.perf_counter() - t0), flush=True)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    {"generate": generate, "evaluate": evaluate}[sys.argv[1]]()
