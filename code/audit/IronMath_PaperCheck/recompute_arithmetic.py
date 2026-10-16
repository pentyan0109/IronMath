"""Independent saved-record arithmetic aggregation. No original script imports."""
from common import *
import sys,gzip,collections
sys.set_int_max_str_digits(0)
def stream(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def bootstrap(d):
    d=np.asarray(d,float);ix=np.random.default_rng(66000).integers(0,len(d),(10000,len(d)))
    return float(d.mean()),np.percentile(d[ix].mean(axis=1),[2.5,97.5]).tolist()
def run():
    c=Comparisons();audit=[];extra={};cells=[]
    root=W/'v054e_endurance';datasets={}
    for p in (root/'data').glob('*.json'):
        if p.stem.startswith(('main_','confirmation_')):datasets[p.stem]={x['id']:x for x in load(p)}
    counts=collections.defaultdict(lambda:[0,0]);disagree=0
    for r in rows(root/'results_per_problem.csv'):
        if r['set_type'] not in ['main','confirmation']:continue
        x=datasets[r['set_name']][r['problem_id']];truth=int(x['a'],16)+int(x['b'],16)
        assert truth==int(x['answer'],16)
        correct=int(r['prediction_hex'],16)==truth;disagree+=int(correct!=bool(int(r['exact'])))
        a=counts[(r['set_type'],r['width'],r['seed'])];a[0]+=correct;a[1]+=1
    for r in rows(root/'summary.csv'):
        k=(r['set_type'],r['width'],r['seed'])
        if r['category']=='ALL' and k in counts:
            hit,n=counts[k]
            c.add('V054E','/'.join(k)+'/exact',int(r['exact']),hit,'summary.csv','prediction_hex vs independent a+b')
            c.add('V054E','/'.join(k)+'/rate',float(r['exact_rate']),hit/n,'summary.csv','prediction_hex vs independent a+b')
    extra['V054E']=dict(integer_rescoring_disagreements=disagree,counts=[dict(set=k[0],width=int(k[1]),seed=int(k[2]),correct=v[0],N=v[1]) for k,v in counts.items()])
    for width in sorted({k[1] for k in counts},key=int):
        for kind in ['main','confirmation']:
            vals=[v[0]/v[1] for k,v in counts.items() if k[:2]==(kind,width)]
            if vals:extra['V054E'].setdefault('means',[]).append(dict(set=kind,width=int(width),mean=float(np.mean(vals)),sample_SD=float(np.std(vals,ddof=1))))
    root=W/'v054r_replication';allcells=[]
    for p in sorted((root/'data/raw').glob('R_*.json')):
        raw=load(p)
        for f in raw['fixed_points']:
            if not f['converged']:continue
            rr=[x for x in raw['durability'] if all(x[k]==f[k] for k in ['seed','op','state','pair']) and x['k']==4096]
            assert len(rr)==16,(p,f,len(rr))
            read=f['h_star_output']==f['correct_output'];hold=all(x['exact']==1 for x in rr)
            allcells.append(dict(model=f['model'],seed=f['seed'],state=f['state'],pair=f['pair'],read_correct=read,hold_correct=hold,match=read==hold))
    subset=[x for x in allcells if x['model'].startswith('V055')]
    c.add('V054R','R1/matched',90,sum(x['match'] for x in allcells),'REPORT_V054R.md','raw fixed_points + k4096 probes','Probe outputs absent; aggregate saved per-probe exact flags, no independent bit regrading')
    c.add('V054R','R1/total',90,len(allcells),'REPORT_V054R.md','raw records')
    c.add('V054R','V055/matched',54,sum(x['match'] for x in subset),'REPORT_V054R.md','independently trained V055 raw subset')
    c.add('V054R','V055/total',54,len(subset),'REPORT_V054R.md','raw records');table('v054r_cells.csv',allcells)
    root=W/'decimal_d10';cc=[]
    for seed in range(3):
        fixed=list(stream(root/f'data/raw/fixed_seed{seed}.jsonl'));probe=list(stream(root/f'data/raw/probes_seed{seed}.jsonl'))
        for f in fixed:
            if not f['converged']:continue
            rr=[x for x in probe if x['k']==1024 and x['state']==f['state'] and x['pair']==f['pair']];assert len(rr)==16
            read=f['h_star_output']==f['correct_output'];hold=all(x['exact']==1 for x in rr)
            cc.append(dict(seed=seed,state=f['state'],pair=f['pair'],read_correct=read,hold_correct=hold,match=read==hold))
    c.add('D10','H4/matched',77,sum(x['match'] for x in cc),'analysis_summary.json / REPORT_D10.md','raw fixed and probes','Saved exact flags only; no probe output digits')
    c.add('D10','H4/total',78,len(cc),'analysis_summary.json','raw fixed/probes');table('d10_cells.csv',cc)
    for study in ['V058','V058b']:
        root=W/f'IronMath_{study}';fp=load(root/'PREDICTIONS_V058_FROZEN.json')['predictions'];pred={x['checkpoint']:x['prediction'] for x in fp};cnt=collections.defaultdict(lambda:[0,0]);badflags=0
        for x in stream(root/'problems.jsonl.gz'):
            key=(x['n'],f"{x['model']}-S{x['seed']}",x['order']);correct=x['prediction']==int(x['a'])*int(x['b'])
            badflags+=int(correct!=x['exact']);cnt[key][0]+=correct;cnt[key][1]+=1
            assert int(x['truth'])==int(x['a'])*int(x['b'])
        out=[]
        for ref in rows(root/'order_prediction_comparison.csv'):
            n=int(ref['n']);cp=ref['checkpoint'];a,na=cnt[n,cp,'O0'];b,nb=cnt[n,cp,'O1'];assert na==nb==512
            actual='O0' if a>b else 'O1' if b>a else 'tie';predict=pred[cp];matched=predict==actual
            out.append(dict(width=n,checkpoint=cp,prediction=predict,O0_correct=a,O1_correct=b,actual_best=actual,matched=matched,diagnostic_correct=a if predict=='O0' else b,diagnostic_minus_fixedO1=(a-b if predict=='O0' else 0)))
            for k,v in [('O0_correct',a),('O1_correct',b),('actual_best',actual),('matched',matched)]:
                ov=int(ref[k]) if k.endswith('correct') else ref[k]=='True' if k=='matched' else ref[k]
                c.add(study,f'{n}/{cp}/{k}',ov,v,'order_prediction_comparison.csv','problems.jsonl.gz predictions vs a*b')
        table(study.lower()+'_independent_judgment_table.csv',out)
        if study=='V058':
            extra[study]={str(n):dict(matched=sum(x['matched'] for x in out if x['width']==n),N=6,judgment='支持' if sum(x['matched'] for x in out if x['width']==n)>=5 else '不支持') for n in [64,128]}
            for n in [64,128]:c.add(study,f'{n}/H1','支持',extra[study][str(n)]['judgment'],'REPORT_V058.md','independent judgment table')
        else:
            for n in [64,128]:
                oo=[x for x in out if x['width']==n and x['prediction']=='O0'];rev=sum(x['O0_correct']<x['O1_correct'] for x in oo);wins=sum(x['O0_correct']>x['O1_correct'] for x in oo)
                judge='unsupported' if rev else 'supported' if oo and wins==len(oo) else 'indeterminate'
                ref=next(r for r in rows(root/'hypothesis_judgments.csv') if int(r['n'])==n)
                c.add(study,f'{n}/limited_hypothesis',ref['limited_hypothesis'],judge,'hypothesis_judgments.csv','raw integer outputs')
            extra[study]=dict(raw_flags_disagree=badflags,diagnostic_minus_O1={str(n):sum(x['diagnostic_minus_fixedO1'] for x in out if x['width']==n) for n in [64,128]})
        for key,v in load(root/'checkpoint_assets.json').items():
            p=W/v['path'];audit.append(dict(study=study,asset=key,path=v['path'],expected=v['SHA256'],actual=sha(p)))
    root=W/'IronMath_D11';reported=load(root/'analysis_summary.json');d={h:{g:[] for g in ['control','intervention']} for h in ['H1','H2','H3']};coverage=[]
    truthdata={x['id']:x for x in load(root/'data/test/Test2048_part1.json')}
    for g in ['control','intervention']:
        for seed in range(10):
            pr=list(stream(root/f'data/raw/probes_{g}_s{seed}.jsonl'));fx=list(stream(root/f'data/raw/fixed_{g}_s{seed}.jsonl'));pa=list(stream(root/f'data/raw/pairs_{g}_s{seed}.jsonl'))
            selected=[x for x in pr if x['state']==1 and sum(map(int,x['pair']))==9 and x['k']==1024]
            grouped=collections.defaultdict(list)
            for x in selected:grouped[x['pair']].append(x)
            assert len(grouped)==10 and all(len(v)==16 for v in grouped.values())
            d['H1'][g].append(sum(all(x['exact']==1 for x in v) for v in grouped.values()))
            validpairs=0
            for p in pa:
                if sum(map(int,p['pair']))!=9:continue
                ff=[x for x in fx if x['pair']==p['pair']];assert len(ff)==2
                validpairs+=bool(all(x['converged'] and x['h_star_output']==x['correct_output'] for x in ff) and p['distance_c0_vs_c1']>=1e-3 and p['rho_c0']<1 and p['rho_c1']<1)
            d['H2'][g].append(validpairs)
            ev=[x for x in stream(root/f'data/raw/eval_{g}_s{seed}.jsonl') if x['width']==2048 and x['part']==1];assert len(ev)==512
            hits=0;missing=0;mismatch=0
            for x in ev:
                item=truthdata[x['id']];truth=int(item['a'])+int(item['b']);assert truth==int(item['answer'])
                if x['prediction'] is None:
                    # Raw logger omits predictions on successes; do not invent output digits.
                    correct=bool(x['exact']);missing+=1
                else:
                    correct=int(x['prediction'])==truth;mismatch+=int(correct!=bool(x['exact']))
                hits+=correct
            d['H3'][g].append(hits/512);coverage.append(dict(group=g,seed=seed,N=512,prediction_missing=missing,flags_disagree=mismatch))
    for h in d:
        dif=np.array(d[h]['intervention'])-d[h]['control'];mean,ci=bootstrap(dif)
        old=reported[h]
        for g in ['control','intervention']:c.add('D11',h+'/'+g,[old[g][str(s)] for s in range(10)],d[h][g],'analysis_summary.json','raw probes/fixed/pairs/eval')
        c.add('D11',h+'/mean_difference',old.get('mean_diff','NOT_REPORTED'),mean,'analysis_summary.json','paired seed differences')
        c.add('D11',h+'/CI95',old.get('ci95','NOT_REPORTED'),ci,'analysis_summary.json','bootstrap seed66000','H1/H2 intervals newly calculated descriptive; H3 preregistered')
        if h=='H3':judge='支持' if mean>=.05 and ci[0]>0 else '不支持' if mean<=0 or ci[0]<=0<=ci[1] else '判定不能'
        else:
            n=int((dif>0).sum());judge='支持' if n>=8 else '不支持' if n<=5 else '判定不能'
        c.add('D11',h+'/judgment',old['verdict'],judge,'analysis_summary.json','registered rules')
        extra.setdefault('D11',{})[h]=dict(control=d[h]['control'],intervention=d[h]['intervention'],differences=dif.tolist(),mean_difference=mean,CI95=ci,judgment=judge)
    table('d11_prediction_coverage.csv',coverage)
    table('arithmetic_checkpoint_hash_checks.csv',audit)
    assert all(x['expected'].lower()==x['actual'].lower() for x in audit)
    table('r4_arithmetic_recomputation.csv',c.data);dump('arithmetic_summary.json',extra)
if __name__=='__main__':run()
