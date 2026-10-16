"""New independent workspace manifest implementation, standard library only."""
import os,json,time,hashlib,datetime,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parent
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,obj):
    p=R/name
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def processes(stage):
    q=subprocess.run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python|julia|wsl.exe'} | Select-Object Name,ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3"],capture_output=True,text=True)
    p=json.loads(q.stdout) if q.stdout.strip() else [];save('processes_'+stage+'.json',dict(utc=utc(),returncode=q.returncode,processes=p,parallel_readonly_analysis_permitted=True))
def snapshot(stage):
    t=time.perf_counter();dirs={p.name:dict(creation_time_utc_epoch=p.stat().st_ctime) for p in W.iterdir() if p.is_dir() and p!=R}
    original=dirs if stage=='before' else json.loads((R/'WORKSPACE_BEFORE.json').read_text(encoding='utf-8'))['top_level_directories']
    new={n:v for n,v in dirs.items() if n not in original};files={};links=[];errors=[]
    for folder,subs,names in os.walk(W,followlinks=False,onerror=lambda e:errors.append(repr(e))):
        parent=Path(folder);keep=[]
        for name in subs:
            p=parent/name
            if p==R or (parent==W and name in new):continue
            if p.is_symlink() or p.is_junction():links.append(dict(path=p.relative_to(W).as_posix(),target=os.readlink(p)))
            else:keep.append(name)
        subs[:]=keep
        for name in names:
            p=parent/name
            try:
                a=p.stat();h=sha(p);b=p.stat()
                if (a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns):raise RuntimeError('file changed while hashing')
                files[p.relative_to(W).as_posix()]=dict(SHA256=h,bytes=b.st_size,mtime_ns=b.st_mtime_ns)
            except Exception as e:errors.append(dict(path=str(p),error=repr(e)))
    save('WORKSPACE_'+stage.upper()+'.json',dict(utc=utc(),files=files,top_level_directories=dirs,new_top_level_directories_name_and_creation_only=new,links_not_followed=links,errors=errors,total_bytes=sum(v['bytes'] for v in files.values()),wall_seconds=time.perf_counter()-t))
    if stage=='after':
        old=json.loads((R/'WORKSPACE_BEFORE.json').read_text(encoding='utf-8'))['files'];changed=[p for p,v in old.items() if p not in files or v['SHA256']!=files[p]['SHA256']]
        save('WORKSPACE_COMPARISON.json',dict(utc=utc(),original_files_unchanged=not changed,original_count=len(old),changed_or_missing=changed,added_files_in_existing_locations=sorted(set(files)-set(old)),new_top_level_directories_name_and_creation_only=new,origin_not_inferred=True))
    print('SNAPSHOT',stage,len(files),'files',len(errors),'errors',flush=True)
    if errors:raise RuntimeError('Incomplete preservation manifest')
if __name__=='__main__':
    import sys
    stage=sys.argv[1];processes(stage)
    if stage=='before':
        s=(R/'R1_RULES.md').read_text(encoding='utf-8');assert '\ufffd' not in s and '???' not in s
        save('R1_RULES_SHA256.json',dict(utc=utc(),SHA256=sha(R/'R1_RULES.md'),UTF8_readback_verified=True))
    snapshot(stage)
