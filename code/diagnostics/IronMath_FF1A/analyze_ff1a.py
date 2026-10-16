import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,hashlib,time,datetime,ctypes,subprocess,math,statistics
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'IronMath_FF1'
start=time.perf_counter(); cpu=time.process_time()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def processes():
    p=subprocess.run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python'} | Select-Object Name,ProcessId | ConvertTo-Json"],capture_output=True,text=True)
    return {'returncode':p.returncode,'processes':json.loads(p.stdout) if p.stdout.strip() else [],'stderr':p.stderr}
kernel=ctypes.windll.kernel32
kernel.GetCurrentProcess.restype=ctypes.c_void_p
kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
kernel.SetPriorityClass.restype=ctypes.c_int
kernel.GetPriorityClass.argtypes=[ctypes.c_void_p]
kernel.GetPriorityClass.restype=ctypes.c_ulong
priority_ok=bool(kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000))
runtime={'start_utc':utc(),'python_start':processes(),'priority_set_ok':priority_ok,'priority_class':kernel.GetPriorityClass(kernel.GetCurrentProcess()),'torch_threads':torch.get_num_threads(),'timing_reference_only':True,'pid':os.getpid()}
def dump(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def writecsv(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
manifest={}
for line in (SRC/'SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines():
    h,p=line.split('  ',1);manifest[p.replace('\\','/')]=h
used={}; checks=[]
def readpath(rel):
    p=SRC/rel
    if rel not in used:
        actual=sha(p); expected=manifest.get(rel)
        checks.append({'file':rel,'expected':expected,'before':actual,'matches_manifest':actual==expected})
        if expected is None or actual!=expected: raise RuntimeError('Input SHA mismatch: '+rel)
        used[rel]=actual
    return p
def load(rel): return json.loads(readpath(rel).read_text(encoding='utf-8-sig'))
def rows(rel):
    with readpath(rel).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def truth(x): return x is True or x=='True'
details=load('diagnostic_details.json'); frozen=load('CHECKPOINTS_FROZEN.json')
aa={r['model']:r for r in load('PREDICTIONS_A_FROZEN.json')['models']}
bb={r['model']:r for r in load('PREDICTIONS_B_FROZEN.json')['models']}
diag={}; ds=[]
for mid in sorted(aa):
    rr=[r for r in details if r['model']==mid]; assert len(rr)==16
    complete=True; nonfinite=False; converged=True; traces=[]
    for neutral in ['N','R']:
        rel=f'diagnostic_traces/{mid}_{neutral}.npz'
        if not (SRC/rel).exists():complete=False;continue
        with np.load(readpath(rel)) as z:
            movement=z['movements'];margin=z['margins'];endpoint=z['endpoints']
            assert movement.shape==(20000,8) and margin.shape==(20000,8)
            nonfinite=nonfinite or not all(np.isfinite(v).all() for v in [movement,margin,endpoint])
            for r in [a for a in rr if a['neutral']==neutral]:
                j=2*r['prefix']+r['value'];c=bool(np.isfinite(movement[-100:,j]).all() and movement[-100:,j].max()<1e-6)
                assert c==r['converged'];converged=converged and c
    pred=False if nonfinite or not converged else (True if complete else None)
    diag[mid]={'D':pred,'all16_converged':converged if complete else None,'diagnostic_nonfinite':nonfinite,'complete':complete}
    ds.append({'model':mid,**diag[mid]})
dump('PREDICTIONS_D_LOG_ONLY.json',{'utc':utc(),'definition':'all 16 converge and no nonfinite recorded movement/margin/endpoint; missing if incomplete and no established failure','models':ds,'no_A_or_long_result_input':True})
# D above is calculated only from convergence and saved traces, independently of A and long outcomes.
subset_violations=[m for m in aa if aa[m]['predict_success'] and diag[m]['D'] is not True]
dump('STRUCTURAL_CHECK.json',{'utc':utc(),'A_success_subset_D_success':not subset_violations,'violations':subset_violations,'all_converged':sum(x['all16_converged'] is True for x in diag.values())})
items=rows('long_item_results.csv'); outcome={m:True for m in aa}; aggregate={m:{'nonfinite_positions':0,'finite_wrong_R':0,'nonfinite_R':0,'first_nonfinite':None} for m in aa}
# Independent rescoring of stored bit predictions and all-position invalid masks; no model invocation.
item_index={(r['model'],int(r['T']),r['id']):r for r in items}
for T in [32,64,128,512,2048,8192,65000]:
    with np.load(readpath(f'long_data/T{T}.npz')) as z: symbols=z['symbols']
    meta=load(f'long_data/T{T}_metadata.json')
    labels=np.zeros(symbols.shape,dtype=np.uint8);state=np.zeros(len(symbols),dtype=np.uint8)
    for pos in range(T):
        x=symbols[:,pos];state=np.where(x<2,x,state);labels[:,pos]=state
    mask=symbols==3
    for mid in sorted(aa):
        with np.load(readpath(f'saved_predictions/{mid}_T{T}.npz')) as z:
            pred=np.unpackbits(z['bits'],axis=1,bitorder='little')[:,:T]
            invalid=np.unpackbits(z['invalid'],axis=1,bitorder='little')[:,:T].astype(bool)
        wrong=mask&((pred!=labels)|invalid)
        agg=aggregate[mid];agg['nonfinite_positions']+=int(invalid.sum());agg['finite_wrong_R']+=int((mask&(pred!=labels)&~invalid).sum());agg['nonfinite_R']+=int((mask&invalid).sum())
        for j,md in enumerate(meta):
            row=item_index[mid,T,md['id']]
            assert int(row['R_count'])==int(mask[j].sum()) and int(row['R_correct'])==int((mask[j]&~wrong[j]).sum())
            assert int(row['nonfinite_positions'])==int(invalid[j].sum())
            if T>=2048 and wrong[j].any():outcome[mid]=False
            ix=np.flatnonzero(invalid[j])
            if len(ix):
                event={'T':T,'item':md['id'],'position_0based':int(ix[0])}
                # First under deterministic order T ascending, metadata order, position ascending.
                if agg['first_nonfinite'] is None:agg['first_nonfinite']=event
models=[];residual=[]
for mid in sorted(aa):
    cp=frozen['models'][mid];p=readpath(cp['path']);assert sha(p)==cp['SHA256']
    checkpoint=torch.load(p,map_location='cpu',weights_only=False)
    g=float(checkpoint['model']['residual_gate']);tg=float(torch.tanh(checkpoint['model']['residual_gate']))
    rr=[r for r in details if r['model']==mid];rho=[r['spectral_radius'] for r in rr if r['converged'] and r['spectral_radius'] is not None]
    ag=aggregate[mid];nf=ag['nonfinite_positions']>0;actual=outcome[mid]
    kind='発散型' if nf else ('有限誤答型' if not actual and ag['finite_wrong_R'] else ('成功' if actual else 'その他'))
    r={'model':mid,'H':cp['H'],'L':cp['L'],'seed':cp['seed'],'actual_success':actual,'A_success':aa[mid]['predict_success'],'B_success':bb[mid]['predict_success'],'C_success':False,'D_success':diag[mid]['D'],'classification':kind,**ag,**{k:v for k,v in diag[mid].items() if k!='D'},'two_attractors_N':aa[mid]['two_attractors']['N'],'two_attractors_R':aa[mid]['two_attractors']['R']}
    r['first_nonfinite']=json.dumps(r['first_nonfinite'],ensure_ascii=False)
    models.append(r)
    residual.append({'model':mid,'g':g,'tanh_g_FP32':tg,'one_plus_tanh_g':1+tg,'rho_converged_count':len(rho),'rho_median':statistics.median(rho) if rho else None,'rho_max':max(rho) if rho else None,'all16_converged':diag[mid]['all16_converged'],'long_nonfinite':nf})
comparison=[]
for group,mm in [('all40',models),('no_long_nonfinite',[r for r in models if not r['nonfinite_positions']]),('all16_converged',[r for r in models if r['all16_converged']])]:
    for method in 'ABCD':
        valid=[r for r in mm if r[method+'_success'] is not None]
        tp=sum(r[method+'_success'] and r['actual_success'] for r in valid);fp=sum(r[method+'_success'] and not r['actual_success'] for r in valid)
        tn=sum(not r[method+'_success'] and not r['actual_success'] for r in valid);fn=sum(not r[method+'_success'] and r['actual_success'] for r in valid)
        comparison.append({'group':group,'method':method,'N':len(mm),'missing':len(mm)-len(valid),'hits':tp+tn,'TP':tp,'FP':fp,'TN':tn,'FN':fn,'success_precision':tp/(tp+fp) if tp+fp else None,'success_recall':tp/(tp+fn) if tp+fn else None})
disagreements=[]
for r in models:
    if r['A_success']!=r['D_success']:
        rr=[x for x in details if x['model']==r['model']]
        reasons=sorted(set('readout_wrong_or_nonfinite' if not x['readout_finite'] or x['readout']!=x['value'] else 'rho_not_below_1' for x in rr if not x['retention_prediction']))
        disagreements.append({'model':r['model'],'A_success':r['A_success'],'D_success':r['D_success'],'actual_success':r['actual_success'],'A_only_correct':r['A_success']==r['actual_success'] and r['D_success']!=r['actual_success'],'D_only_correct':r['D_success']==r['actual_success'] and r['A_success']!=r['actual_success'],'reasons':';'.join(reasons)})
writecsv('ff1a_model_classification.csv',models);writecsv('ff1a_residual_analysis.csv',residual);writecsv('ff1a_predictor_comparison.csv',comparison)
if disagreements:writecsv('ff1a_disagreements.csv',disagreements)
else:(ROOT/'ff1a_disagreements.csv').write_text('model,A_success,D_success,actual_success,A_only_correct,D_only_correct,reasons\n',encoding='utf-8-sig')
groups=[]
for positive in [True,False]:
    rr=[r for r in residual if (r['tanh_g_FP32']>0)==positive]
    groups.append({'group':'tanh(g)>0' if positive else 'tanh(g)<=0','models':len(rr),'divergent':sum(r['long_nonfinite'] for r in rr),'proportion':sum(r['long_nonfinite'] for r in rr)/len(rr) if rr else None})
writecsv('ff1a_residual_groups.csv',groups)
for c in checks:c['after']=sha(SRC/c['file']);c['unchanged']=c['after']==c['before']
assert all(c['unchanged'] and c['matches_manifest'] for c in checks)
assert sum(r['actual_success']==r['A_success'] for r in models)==38
runtime.update(end_utc=utc(),python_end=processes(),wall_seconds=time.perf_counter()-start,CPU_seconds=time.process_time()-cpu)
dump('runtime.json',runtime)
audit={'status':'completed','existing_inputs':checks,'all_input_SHA_unchanged':True,'independent_rescored_items':len(items),'A_subset_D':not subset_violations,'no_model_forward_or_training':True,'D11_directory_not_accessed':True,'FF1_registered_A_hits_unchanged':38,'structural_note_SHA256':sha(ROOT/'STRUCTURAL_NOTE.md'),'script_SHA256':sha(Path(__file__)),'missing_D':sum(r['D_success'] is None for r in models)}
dump('ff1a_audit.json',audit)
main={r['method']:r for r in comparison if r['group']=='all40'}
finite={r['method']:r for r in comparison if r['group']=='no_long_nonfinite'}
case='ケースA' if main['D']['hits']>=main['A']['hits'] else 'ケースB（探索的。小差の解釈はケースCの留保も必要）'
text=(ROOT/'STRUCTURAL_NOTE.md').read_text(encoding='utf-8')+'\n## 構造の確認と全収束集合\n\n'
text+=f"A成功 ⊂ D成功 の確認：{not subset_violations}。違反：{subset_violations}。全16本収束：{sum(r['all16_converged'] for r in models)}モデル。\n\n"
text+='|モデル|A|D|実際|\n|---|---|---|---|\n'
for r in models:
    if r['all16_converged']:text+=f"|{r['model']}|{r['A_success']}|{r['D_success']}|{r['actual_success']}|\n"
text+='\n## 方式比較\n\nTP=成功を成功と予測、FP=失敗を成功と予測。適合率の分母0は欠測。\n\n|方式|的中|TP|FP|TN|FN|適合率|再現率|\n|---|---|---|---|---|---|---|---|\n'
for m,r in main.items():text+=f"|{m}|{r['hits']}/40|{r['TP']}|{r['FP']}|{r['TN']}|{r['FN']}|{r['success_precision']}|{r['success_recall']}|\n"
text+=f"\n分類：発散型{sum(r['classification']=='発散型' for r in models)}、有限誤答型{sum(r['classification']=='有限誤答型' for r in models)}、成功{sum(r['actual_success'] for r in models)}。成功モデルの非有限値は{sum(r['actual_success'] and r['nonfinite_positions']>0 for r in models)}件。全7長さ・非R位置も含む。有限誤答と非有限値の併存は分類CSVの別列に保存。\n\n"
text+=f"非発散{finite['A']['N']}モデル：A的中{finite['A']['hits']}、D的中{finite['D']['hits']}。有限誤答型で失敗を予測：A {sum(r['classification']=='有限誤答型' and not r['A_success'] for r in models)}、D {sum(r['classification']=='有限誤答型' and r['D_success'] is False for r in models)}。Aのみ正解{sum(r['A_only_correct'] for r in disagreements)}、Dのみ正解{sum(r['D_only_correct'] for r in disagreements)}。\n\n"
text+=f"判定：{case}。主任は『差が小さい』の数値境界を指定していないため、同等以上ならケースAという明示規則を優先し、A優位の小差を統計的な確証と扱わない。FF1の事前登録38/40は変更しない。\n\n"
text+='## 残差係数・安定性\n\n'+json.dumps(groups,ensure_ascii=False)+'\n\n収束した終点の保存済みスペクトル半径のみを集計。非収束終点は欠測。1+tanh(g)はスペクトル半径ではない。符号と発散の関連は記述的で、因果関係ではない。1点／2点は保存されたN/R別のtwo_attractorsフラグを掲載し、フラグが偽であるだけで「1点」と断定しない（非収束・不安定・読み出し失敗もあり得る）。\n\n'
text+='## 方法・制約・監査\n\nDは16本の保存トレースの最終100移動量、全移動量・読み出しマージン・終点の有限性から算出。完全な中間活性は保存されていない。非有限の移動量にはノルム計算のオーバーフローも含み、これだけでhidden自体の非有限とは断定しない。長系列の発散型は保存済みlogit非有限マスクで判定。初回位置は長さ昇順、保存項目順、系列内位置順の最初であり、別系列間の物理時刻ではない。有限誤答数はR位置のイベント数で、独立試行数ではない。rewrite/mixedは保存済み2種類のみを数え、8倍しない。\n\nDの今回のログ集計費用はruntimeに記録するが、診断生成費用とは異なる。既存診断は反復・読み出し・ヤコビアンを一括計測しておりD単独の生成費用は分離不能。AとDの診断生成費用が等しいとは主張しない。\n\n既存入力全件を元のSHA256SUMSと照合し、読み取り前後で不変を確認。保存予測19,040項目を独立再採点。新しいモデル実行・学習・診断推論なし。D11ディレクトリにアクセスせず、Pythonの名前とPIDだけを開始・終了時に取得。CPU1thread、優先度と資源はruntime.jsonを参照。時間は並行D11のため参考値。\n\n再現：`.venv/Scripts/python.exe -B IronMath_FF1A/analyze_ff1a.py`（同名の今回成果物を再生成するため、必要なら別コピーで実行）。追加実験は行わず停止。\n'
(ROOT/'REPORT_FF1A.md').write_text(text,encoding='utf-8')
print(json.dumps({'comparison':main,'classification':{k:sum(r['classification']==k for r in models) for k in ['発散型','有限誤答型','成功','その他']},'groups':groups,'case':case,'audited_files':len(checks),'runtime':runtime},ensure_ascii=False))
