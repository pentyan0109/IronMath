"""Break-timing re-aggregation: workspace SHA256 manifest (standard library only).

Excludes IronMath_PaperCheck/; does not follow links/junctions; writes only into
break_timing/ and never overwrites. Usage: python preservation_bt.py before|after
"""
import datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path

B = Path(__file__).resolve().parent
X = B.parent
W = X.parent


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 * 2**20), b""):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    p = B / name
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def processes(stage):
    q = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python|julia|wsl.exe'} | "
                        "Select-Object Name,ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3"],
                       capture_output=True, text=True)
    p = json.loads(q.stdout) if q.stdout.strip() else []
    save(f"processes_{stage}.json", {"utc": utc(), "returncode": q.returncode, "processes": p})


def snapshot(stage):
    t = time.perf_counter()
    files, links, errors = {}, [], []
    for folder, subs, names in os.walk(W, followlinks=False, onerror=lambda e: errors.append(repr(e))):
        parent = Path(folder)
        keep = []
        for n in subs:
            p = parent / n
            if p == X:
                continue
            if p.is_symlink() or p.is_junction():
                links.append({"path": p.relative_to(W).as_posix(), "target": os.readlink(p)})
            else:
                keep.append(n)
        subs[:] = keep
        for n in names:
            p = parent / n
            try:
                a = p.stat(); h = sha(p); b = p.stat()
                files[p.relative_to(W).as_posix()] = {"SHA256": h, "bytes": b.st_size, "mtime_ns": b.st_mtime_ns,
                                                      "stable_while_hashing": (a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns)}
            except Exception as e:
                errors.append({"path": str(p), "error": repr(e)})
    save(f"WORKSPACE_{stage.upper()}.json", {"utc": utc(), "excluded": "IronMath_PaperCheck/", "files": files,
                                              "links_not_followed": links, "errors": errors,
                                              "total_bytes": sum(v["bytes"] for v in files.values()),
                                              "wall_seconds": time.perf_counter() - t})
    if stage == "after":
        old = json.loads((B / "WORKSPACE_BEFORE.json").read_text(encoding="utf-8"))["files"]
        changed = sorted(p for p, v in old.items() if p in files and v["SHA256"] != files[p]["SHA256"])
        missing = sorted(p for p in old if p not in files)
        added = sorted(set(files) - set(old))
        tm = lambda p: datetime.datetime.fromtimestamp(files[p]["mtime_ns"] / 1e9).isoformat(timespec="seconds")
        save("WORKSPACE_COMPARISON.json", {
            "utc": utc(), "original_files_unchanged": not changed and not missing, "original_count": len(old),
            "changed": [{"path": p, "mtime": tm(p)} for p in changed], "missing": missing,
            "added": [{"path": p, "mtime": tm(p)} for p in added],
            "note": "Other activity is recorded by name and modification time only; its origin is not inferred."})
    print("SNAPSHOT", stage, len(files), "files", len(errors), "errors", flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    processes(st)
    snapshot(st)
