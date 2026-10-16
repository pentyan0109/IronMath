"""Copy audited utilities and FF1 data generator into the new output only."""
from pathlib import Path
import hashlib,json,datetime
R=Path(__file__).resolve().parent;S=R.parent/'IronMath_FF1'
audit=[]
for name in ['common.py','manifest.py','data.py']:
    original=(S/name).read_bytes();out=original
    if name=='common.py':out=original.split(b'\r\ndef model_ids():')[0] if b'\r\ndef model_ids():' in original else original.split(b'\ndef model_ids():')[0]
    if name=='data.py':out=original.split(b'\r\ndef generate_long():')[0] if b'\r\ndef generate_long():' in original else original.split(b'\ndef generate_long():')[0]
    p=R/name
    if p.exists():raise FileExistsError(p)
    p.write_bytes(out)
    audit.append(dict(file=name,original_SHA256=hashlib.sha256(original).hexdigest(),copied_SHA256=hashlib.sha256(out).hexdigest(),change='none' if out==original else 'retain prefix only; remove experiment-specific trailing functions; all retained bytes identical'))
(R/'SOURCE_COPY_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
theory=R/'THEORY_G1.md';s=theory.read_text(encoding='utf-8');assert '定理と証明' in s and '\ufffd' not in s and '???' not in s
(R/'THEORY_G1_SHA256.json').write_text(json.dumps(dict(SHA256=hashlib.sha256(theory.read_bytes()).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),UTF8_readback_verified=True),indent=2),encoding='utf-8')
print('Prepared utility copies; THEORY UTF-8 and SHA verified.')
