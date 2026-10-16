import json,random
from pathlib import Path
from model import annotate

def key(r):
    a,b=(sorted((r['a'],r['b'])) if r['op']=='ADD' else (r['a'],r['b']))
    return (r['width'],r['op'],a,b)

def generate(work,config):
    old=work/'IronMath_V4/v051/data';rng=random.Random(config['data_seed']);data={};used=set();metadata=[]
    for op in ('ADD','SUB'):
        for split in ('Train','Validation'):
            original=json.loads((old/f'{op}_{split}.json').read_text());rows=[]
            for r in original:
                assert key(r) not in used;used.add(key(r));rows.append(annotate(dict(r,origin='V051_'+split)))
            data[f'{op}_{split}']=rows
    for w in config['widths']:
        for op in ('ADD','SUB'):
            source=json.loads((old/f'{op}_Test{w}.json').read_text());rows=[]
            if w==8:
                pool=[annotate(dict(r,origin='V051_Test8')) for r in source];rng.shuffle(pool);taken=set()
                for category in ('no_propagation','short_propagation','long_propagation','zero_operand','near_maximum','bit_boundary'):
                    count=0
                    for r in pool:
                        if count>=64:break
                        if category not in r['attributes'] or key(r) in taken:continue
                        taken.add(key(r));rows.append(r);count+=1
                for r in pool:
                    if len(rows)>=512:break
                    if key(r) in taken:continue
                    taken.add(key(r));rows.append(r)
                assert len(rows)==512
                for r in rows:assert key(r) not in used;used.add(key(r))
            else:
                for r in source:
                    if key(r) in used:continue
                    used.add(key(r));rows.append(annotate(dict(r,origin='V051_Test_existing')))
                inherited=len(rows);counter=0
                while len(rows)<512:
                    kind=('random','no_propagation','long_propagation','zero_operand','extrema','boundary')[counter%6];counter+=1;limit=(1<<w)-1
                    if kind=='random':a,b=rng.getrandbits(w),rng.getrandbits(w)
                    elif kind=='no_propagation':
                        a=rng.getrandbits(w);b=rng.getrandbits(w)&(~a if op=='ADD' else a)
                    elif kind=='long_propagation':
                        k=rng.randrange(max(1,w//2),w);upper=rng.getrandbits(w-k)<<k
                        a=upper|((1<<k)-1) if op=='ADD' else upper|(1<<k)
                        a&=limit;b=rng.randrange(1,4)
                    elif kind=='zero_operand':a,b=rng.getrandbits(w),0
                    elif kind=='extrema':a,b=limit-rng.randrange(9),rng.getrandbits(w)
                    else:a,b=(1<<rng.randrange(w)),rng.getrandbits(w)
                    a,b=sorted((a,b),reverse=op=='SUB')
                    r=dict(id=f'V053_{op}_{w}_{a}_{b}',a=a,b=b,op=op,width=w,category=kind,origin='V053_'+kind)
                    if key(r) in used:continue
                    used.add(key(r));rows.append(annotate(r))
                metadata.append(dict(width=w,op=op,original_rows=len(source),inherited_unique=inherited,new_rows=512-inherited))
            rng.shuffle(rows);data[f'{op}_Test{w}']=rows
    return data,metadata
