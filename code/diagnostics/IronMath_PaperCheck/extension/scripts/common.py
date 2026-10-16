from pathlib import Path
import json,csv,hashlib,datetime,sys,time,os,subprocess,ctypes,platform,shutil
import numpy as np
R=Path(__file__).resolve().parents[1];W=R.parents[1]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(path,obj):
    p=R/path;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def md(path,text):
    p=R/path;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(text,encoding='utf-8');assert p.read_text(encoding='utf-8')==text and '\ufffd' not in text
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(path,rr):
    if not rr:return
    p=R/path;p.parent.mkdir(exist_ok=True,parents=True);keys=list(dict.fromkeys(k for r in rr for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:q=csv.DictWriter(f,keys);q.writeheader();q.writerows(rr)
def process_snapshot(stage):
    q=subprocess.run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^(python|julia)'} | Select-Object Name,ProcessId,ParentProcessId,CommandLine | ConvertTo-Json"],capture_output=True,text=True)
    dd=json.loads(q.stdout) if q.stdout.strip() else [];dump(f'audit/processes_{stage}.json',dict(utc=utc(),processes=dd,error=q.stderr.strip()))
    return dd
def memory():
    from ctypes import wintypes
    class PMC(ctypes.Structure):_fields_=[('cb',wintypes.DWORD),('fault',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ('peak','working','qp','q','qn','n','page','peakpage')]
    p=PMC();p.cb=ctypes.sizeof(p);k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=wintypes.HANDLE
    fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD];assert fn(k.GetCurrentProcess(),ctypes.byref(p),p.cb)
    if p.peak>2*1024**3:raise MemoryError('Registered 2 GiB process-peak ceiling exceeded')
    return p.peak
def first_persistent(margins):
    """Correct class uses margin>0; zero is handled separately by real argmax elsewhere."""
    bad=np.asarray(margins)<0;idx=np.flatnonzero(bad);first=int(idx[0]) if len(idx) else None
    persistent=(int(np.flatnonzero(~bad)[-1])+1 if (~bad).any() else 0) if bad[-1] else None
    recovery=int(np.sum(bad[:-1]&~bad[1:]));return first,persistent,recovery
def cluster_ci(values,seed=98000):
    v=np.asarray(values,float);ix=np.random.default_rng(seed).integers(0,len(v),(10000,len(v)));return np.percentile(v[ix].mean(1),[2.5,97.5]).tolist()
def stage_failure(stage,e):
    import traceback
    p=R/'audit/failures.jsonl';p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(stage=stage,utc=utc(),error=repr(e),traceback=traceback.format_exc(),results_not_imputed=True),ensure_ascii=False)+'\n')
