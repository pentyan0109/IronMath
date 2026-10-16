import csv,hashlib,json,os,sys,time,ctypes,subprocess,math
from pathlib import Path
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent; W=R.parent
for name in ('TMP','TEMP','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
    p=R/'runtime_cache'/name;p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(n):return json.loads((R/n).read_text(encoding='utf-8-sig'))
def dump(n,x):
    p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    def finite(a):
        if isinstance(a,float) and not math.isfinite(a):return None
        if isinstance(a,dict):return {k:finite(v) for k,v in a.items()}
        if isinstance(a,(list,tuple)):return [finite(v) for v in a]
        return a
    p.write_text(json.dumps(finite(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(n,rows):
    p=R/n;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    keys=list(dict.fromkeys(k for r in rows for k in r)) if rows else ['no_rows']
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def setup():
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def memory():
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('fault',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in ('peak','working','qp','q','qn','n','page','peakpage')]
    p=PMC();p.cb=ctypes.sizeof(p);k=ctypes.windll.kernel32;k.GetCurrentProcess.restype=wintypes.HANDLE
    fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD];assert fn(k.GetCurrentProcess(),ctypes.byref(p),p.cb)
    return p.peak
def process_gate():
    code="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python|julia|wsl.exe' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
    q=subprocess.run(['powershell','-NoProfile','-Command',code],capture_output=True,text=True,check=True)
    procs=json.loads(q.stdout) if q.stdout.strip() else [];procs=procs if isinstance(procs,list) else [procs]
    own={os.getpid()};parents={int(p['ProcessId']):int(p['ParentProcessId']) for p in procs}
    pid=os.getpid()
    while pid in parents and parents[pid] not in own:
        pid=parents[pid];own.add(pid)
    other=[p for p in procs if int(p['ProcessId']) not in own]
    record={'utc':now(),'own_and_ancestors':sorted(own),'processes':procs,'other_processes':other}
    with (R/'process_checks.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    if other:raise RuntimeError('OTHER EXPERIMENT PROCESS: stop and ask; no process killed')
def registrations_ok():
    reg=read('PREREGISTRATION.json')
    assert sha(R/'PREREGISTRATION.json')==read('PREREGISTRATION_SHA256.json')['SHA256']
    for p,h in reg['code_SHA256'].items():assert sha(R/p)==h,p
    return reg
def model_ids():return [(f'H{h}_L{l}_s{s:02}',h,l,s) for h in (32,64) for l in (8,16) for s in range(10,20)]
def load_model(mid):
    import torch
    from model import FF
    a=read('CHECKPOINTS_FROZEN.json')['models'][mid];assert sha(R/a['path'])==a['SHA256']
    cp=torch.load(R/a['path'],map_location='cpu',weights_only=False);m=FF(cp['H']).eval();m.load_state_dict(cp['model'])
    for p in m.parameters():p.requires_grad_(False)
    return m,a
