"""Read-only audit of exact input identity after deterministic representation conversion.
Integer tuples are unique digit-stream signatures because fixed-length radix encodings
are bijective (verified separately). Training ADD augmentation is included.
"""
import json
from pathlib import Path
R=Path(__file__).resolve().parent;source=R.parent/'IronMath_V053/data'
def tn(w):
    n=1;capacity=1
    while capacity<2*((1<<w)-1):n+=1;capacity=3*capacity+1
    return n
data={p.stem:json.loads(p.read_text()) for p in source.glob('*.json')};out=[]
for k in ('B0','B1','T0','T1'):
    splitsets={s:set() for s in ('Train','Validation','Test')}
    for name,rows in data.items():
        sp='Train' if name.endswith('_Train') else 'Validation' if name.endswith('_Validation') else 'Test'
        for r in rows:
            n=tn(r['width']) if k[0]=='T' else r['width']+int(k=='B1' or r['op']=='ADD')
            pairs=[(r['a'],r['b'])]
            if sp=='Train' and r['op']=='ADD':pairs.append((r['b'],r['a']))
            for a,b in pairs:
                flag=int(r['op']=='SUB')
                if k=='B1' and flag:b=(1<<n)-1-b
                if k=='T1':b=-b if flag else b;flag=0
                splitsets[sp].add((n,a,b,flag))
    for a,b in (('Train','Validation'),('Train','Test'),('Validation','Test')):
        shared=splitsets[a]&splitsets[b];out.append(dict(model=k,split1=a,split2=b,exact_transformed_input_overlap=len(shared),ADD_training_augmentation_included=True))
(R/'transformed_input_split_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out));assert all(r['exact_transformed_input_overlap']==0 for r in out)
