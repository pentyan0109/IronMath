import hashlib,json
from pathlib import Path
from core import transformed
R=Path(__file__).resolve().parent;WORK=R.parent
def identity(r):
    a,b,n,cin,flag=transformed(r,'P');return (n,*sorted((a,b)))
def rawkey(r):return r['width'],*sorted((r['a'],r['b']))
def prepare():
    source={p.stem:json.loads(p.read_text()) for p in sorted((WORK/'IronMath_V053/data').glob('*.json'))}
    test={identity(r) for name,rows in source.items() if '_Test' in name for r in rows}
    data={k:v for k,v in source.items() if '_Test' in k};excluded=[]
    for op in ('ADD','SUB'):
        keep=[]
        for r in source[op+'_Validation']:
            if identity(r) in test:excluded.append(dict(id=r['id'],op=op,split='Validation',reason='converted input alias of fixed Test'))
            else:keep.append(r)
        data[op+'_Validation']=keep
    validation={identity(r) for op in ('ADD','SUB') for r in data[op+'_Validation']}
    for op in ('ADD','SUB'):
        keep=[]
        for r in source[op+'_Train']:
            reason='converted input alias of fixed Test' if identity(r) in test else 'converted input alias of retained Validation' if identity(r) in validation else None
            if reason:excluded.append(dict(id=r['id'],op=op,split='Train',reason=reason))
            else:keep.append(r)
        data[op+'_Train']=keep
    train={identity(r) for op in ('ADD','SUB') for r in data[op+'_Train']}
    assert not(train&validation or train&test or validation&test)
    oldADD={rawkey(r) for r in source['ADD_Train']}
    for name,rows in data.items():
        if '_Test' not in name:continue
        for r in rows:
            a,b,n,_,_=transformed(r,'P')
            r['old_ADD_train_input_familiar']=int(r['width']==8 and (8,*sorted((a,b))) in oldADD)
    return source,data,excluded

def candidates(data,rng):
    # Draw all streams even if not used; the same seed gives common prefixes.
    streams=[]
    for op in ('ADD','ADD','SUB'):
        rows=[];pool=data[op+'_Train']
        for _ in range(8):
            r=dict(pool[rng.randrange(len(pool))])
            if op=='ADD' and rng.randrange(2):r['a'],r['b']=r['b'],r['a']
            rows.append(r)
        streams.append(rows)
    return streams
def phase(condition,budget,step,cpu):
    return 'mixed' if condition=='F1' or condition=='F2' and (step>=600 if budget=='C1' else cpu>=45) else 'ADD_only'
