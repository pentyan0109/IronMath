"""Copy FF1 implementation without importing it; only model_ids seeds change."""
from pathlib import Path
import hashlib,json,datetime
R=Path(__file__).resolve().parent; S=R.parent/'IronMath_FF1'
copies=['common.py','manifest.py','data.py','model.py','train.py','diagnose.py','evaluate.py','analyze.py','tests_ff1.py']
records=[]
for name in copies:
    source=(S/name).read_bytes();out=source
    if name=='common.py':
        assert source.count(b'for s in range(10)]')==1
        out=source.replace(b'for s in range(10)]',b'for s in range(10,20)]')
    p=R/name
    if p.exists():raise FileExistsError(p)
    p.write_bytes(out)
    records.append({'file':name,'source_SHA256':hashlib.sha256(source).hexdigest(),'copy_SHA256':hashlib.sha256(out).hexdigest(),'identical':source==out,'change':'model_ids seed range 0..9 -> 10..19 only' if name=='common.py' else 'none; runtime path is derived from __file__'})
(R/'SOURCE_COPY_AUDIT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records},indent=2),encoding='utf-8')
print('Copied',len(records),'files; diagnose.py and model.py byte-identical.')
