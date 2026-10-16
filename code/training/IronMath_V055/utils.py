import csv,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent;WORK=R.parent;C=json.loads((R/'config.json').read_text())
def dump(name,obj):
    (R/name).parent.mkdir(parents=True,exist_ok=True);(R/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:'NA' if v is None else v for k,v in row.items()} for row in rows)
def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def protected():
    return {str(p.relative_to(WORK)):sha(p) for folder in ('IronMath_V4/v050','IronMath_V4/v051','IronMath_V052','IronMath_V053','IronMath_V054','IronMath_Abacus') for p in (WORK/folder).rglob('*') if p.is_file()}
def setup(seed):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
def modelhash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()
def memory():
    import ctypes
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('fault',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ('peak','working','qp','q','qn','n','page','peakpage')]
    p=PMC();p.cb=ctypes.sizeof(p);kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=wintypes.HANDLE
    fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD];assert fn(kernel.GetCurrentProcess(),ctypes.byref(p),p.cb)
    if p.peak>C['max_peak_RAM_MiB']*2**20:raise MemoryError('registered RAM ceiling')
    return p.peak
