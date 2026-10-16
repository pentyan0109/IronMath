"""Read-only numerical check helpers. Placement only as explicitly requested."""
from pathlib import Path
import hashlib,json,difflib,datetime,csv
W=Path(__file__).resolve().parent.parent
OUT=W/'IronMath_ALL/v02_audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def loadcsv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def prepare():
    OUT.mkdir(exist_ok=True)
    record=OUT/'placement.json'
    if record.exists():return json.loads(record.read_text(encoding='utf-8'))
    source=Path('inputs/main (4).tex');previous=Path('inputs/main (3).tex')
    target=W/'paper/main.tex';old=(W/'paper/old').resolve()
    assert old.is_relative_to((W/'paper').resolve())
    srcsha=sha(source);prevsha=sha(previous);originalsha=sha(target)
    assert originalsha==prevsha,'Current main differs from reviewed previous edition; preserve for clarification'
    assert srcsha!=prevsha
    # Preserve bytes, with collision checks, within the explicitly named paper folder.
    archived=old/f'main_v02_pre_main3_{originalsha[:12]}.tex'
    assert not archived.exists()
    old.mkdir(exist_ok=True);target.rename(archived)
    with target.open('xb') as f:f.write(source.read_bytes())
    assert sha(target)==srcsha and sha(archived)==originalsha
    with (W/'paper/SOURCES.txt').open('a',encoding='utf-8',newline='') as f:
        f.write(f'\nmain.tex — v0.2 差分数値照合\n  取得元: {source}\n  配置日時: {now()}\n  SHA256: {srcsha}\n  前回照合版: {previous}\n  前回版SHA256: {prevsha}\n  既存原稿の退避先: {archived.relative_to(W).as_posix()}\n  退避版SHA256: {originalsha}\n  内容はコピー元とバイト単位で同一。本文の修正なし。\n')
    a=previous.read_text(encoding='utf-8-sig').splitlines();b=target.read_text(encoding='utf-8-sig').splitlines()
    changes=[]
    for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if tag=='equal':continue
        changes.append(dict(type=tag,previous_lines=[i+1,j],current_lines=[k+1,l],removed=a[i:j],added=b[k:l]))
    obj=dict(placed_at=now(),source=str(source),source_SHA256=srcsha,previous=str(previous),previous_SHA256=prevsha,archived=str(archived),main_SHA256=sha(target),changes=changes)
    record.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'main3_to_main4.diff').write_text('\n'.join(difflib.unified_diff(a,b,fromfile='main (3).tex',tofile='paper/main.tex',n=0))+'\n',encoding='utf-8')
    (OUT/'added_changed_lines.txt').write_text('\n'.join(f'{c["current_lines"][0]+n}: {s}' for c in changes for n,s in enumerate(c['added']))+'\n',encoding='utf-8')
    print('Placed',srcsha,'changes',len(changes),'added/changed',sum(len(c['added']) for c in changes))
    return obj
if __name__=='__main__':prepare()
