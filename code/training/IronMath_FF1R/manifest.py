"""Hash all workspace files except FF1; never writes outside FF1."""
import os,json,time
from common import R,W,sha,dump,read,now
def topdirs():return {p.name:{'creation_time_utc_epoch':p.stat().st_ctime} for p in W.iterdir() if p.is_dir() and p!=R}
def snapshot(stage):
    start=time.perf_counter();dirs=topdirs();initial=dirs if stage=='before' else read('WORKSPACE_BEFORE.json')['top_level_directories'];new={n:v for n,v in dirs.items() if n not in initial}
    files={};links=[];errors=[];total=0
    def walkerror(e):errors.append(repr(e))
    for folder,sub,names in os.walk(W,followlinks=False,onerror=walkerror):
        p=type(W)(folder)
        sub[:]=[s for s in sub if p/s!=R and not (p==W and s in new)]
        for s in list(sub):
            q=p/s
            if q.is_symlink() or q.is_junction():links.append({'path':str(q.relative_to(W)),'target':os.readlink(q)});sub.remove(s)
        for name in names:
            q=p/name
            try:
                before=q.stat();h=sha(q);after=q.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('changed while hashing')
                files[q.relative_to(W).as_posix()]={'SHA256':h,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns};total+=after.st_size
            except Exception as e:errors.append({'path':str(q),'error':repr(e)})
    obj=dict(utc=now(),files=files,top_level_directories=dirs,new_top_level_directories_name_and_creation_only=new,directory_links_not_followed=links,errors=errors,total_bytes=total,wall_seconds=time.perf_counter()-start)
    dump(f'WORKSPACE_{stage.upper()}.json',obj)
    print('WORKSPACE',stage,len(files),total,'bytes',len(errors),'errors',flush=True)
    if errors:raise RuntimeError('incomplete workspace manifest; stop and ask')
    if stage=='after':
        old=read('WORKSPACE_BEFORE.json')['files'];changed=[p for p,v in old.items() if p not in files or files[p]['SHA256']!=v['SHA256']]
        added=sorted(set(files)-set(old));result=dict(original_files_unchanged=not changed,original_count=len(old),changed_or_missing=changed,added_files_in_existing_locations=added,new_top_level_directories_name_and_creation_only=new,origin_not_inferred=True)
        dump('WORKSPACE_COMPARISON.json',result)
        if changed:raise RuntimeError('workspace preservation failed')
    return obj
