"""Redraw from archived CSVs and fill explicitly sourced CHECKs; no model calls."""
import os
from pathlib import Path
O=Path(__file__).resolve().parent;X=O.parent;W=O.parents[2];A=X/'v072_audit'
os.environ['MPLCONFIGDIR']=str(O/'mpl_cache');os.environ['OMP_NUM_THREADS']='1'
import csv,json,hashlib,re,shutil,datetime,subprocess,statistics
from collections import defaultdict
import numpy as np
INPUTS={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for s in iter(lambda:f.read(1048576),b''):h.update(s)
    return h.hexdigest()
def src(p):
    p=Path(p);INPUTS.setdefault(str(p),dict(SHA256=sha(p),bytes=p.stat().st_size));return p
def rows(p):
    with src(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(n,s):
    p=O/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(s,encoding='utf-8');assert p.read_text(encoding='utf-8')==s and '\ufffd' not in s
def dump(n,v):write(n,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(n,rs):
    fields=list(dict.fromkeys(k for r in rs for k in r));p=O/n;p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('w',encoding='utf-8',newline='') as f:q=csv.DictWriter(f,fieldnames=fields);q.writeheader();q.writerows(rs)
def truth(v):return str(v).lower() in ('true','1')
def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    dest=O/'figures';dest.mkdir(exist_ok=True)
    rr=rows(A/'FIGURE1_SOURCE.csv');table('FIGURE1_SOURCE.csv',rr)
    fig,axes=plt.subplots(2,3,figsize=(10.1,5.8),sharex=True,sharey=True);fig.subplots_adjust(left=.075,right=.975,top=.94,bottom=.25,hspace=.19,wspace=.12)
    colors={'00':'#2864a0','01':'#dc7c24','10':'#397d4c','11':'#9e4d80'}
    for oi,op in enumerate(['ADD','SUB']):
        for seed in range(3):
            ax=axes[oi,seed];groups=defaultdict(list)
            for r in rr:
                if r['op']==op and int(r['seed'])==seed:groups[(r['state'],r['pair'])].append(r)
            assert len(groups)==6
            for (s,p),rs in sorted(groups.items()):
                rs.sort(key=lambda r:int(r['t']));ax.plot([int(r['t']) for r in rs],[float(r['margin']) for r in rs],color=colors[p],ls='-' if s=='0' else '--',lw=1.3)
            ax.axhline(0,color='black',lw=.6);ax.set_xscale('symlog',base=2,linthresh=8);ax.set_xticks([0,8,32,128,512],['0','8','32','128','512']);ax.set_xlim(0,512);ax.grid(alpha=.15);ax.set_title(f'{op}, seed {seed}',fontsize=11)
            if seed==0:ax.set_ylabel('Correct minus wrong logit',fontsize=9)
    handles=[Line2D([0],[0],color=colors[p],ls='-' if s==0 else '--',lw=1.4,label=f'(a,b)=({p[0]},{p[1]}), carry {s}') for s in [0,1] for p in ['00','01','10','11']]
    legend=fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.53,.004),frameon=False,fontsize=8.4,columnspacing=1.2,handlelength=2.2)
    fig.supxlabel('Iteration step t (symlog scale)',y=.155,fontsize=10)
    assert len(fig.legends)==1 and all(ax.get_legend() is None for ax in axes.flat)
    fig.savefig(dest/'figure1_arithmetic_margins.pdf');fig.savefig(dest/'figure1_arithmetic_margins.png',dpi=180);plt.close(fig)
    # The left panel uses the same 160 nonlinear G1 models. B_cum values are post-hoc tests.
    old=rows(A/'FIGURE2_SOURCE.csv');base=rows(W/'IronMath_PaperCheck/r1_baselines.csv');new=[]
    for label,method in [('A','A'),('D','D'),('Constant','C_all_failure')]:
        r=next(r for r in old if r['study']=='G1' and r['method']==method);new.append(dict(panel='G1',method=label,N=160,hits=int(r['hits']),source='v072_audit/FIGURE2_SOURCE.csv',source_method=method))
    for ell in [128,512]:
        r=next(r for r in base if r['experiment']=='G1' and r['structure']=='nonlinear' and r['method']=='B_cum' and int(r['ell'])==ell);new.append(dict(panel='G1',method=f'B_cum({ell})',N=160,hits=int(r['hits']),source='IronMath_PaperCheck/r1_baselines.csv',source_method='B_cum',ell=ell))
    assert [r['hits'] for r in new]==[140,91,84,119,143] and new[-1]['hits']>new[0]['hits']
    for r in old:
        if r['study']=='ABL1':new.append(dict(panel='ABL1',method=r['method'],N=160,hits=int(r['hits']),source='v072_audit/FIGURE2_SOURCE.csv',source_method=r['method']))
    table('FIGURE3_SOURCE.csv',new);fig,axes=plt.subplots(1,2,figsize=(10.1,3.8),sharey=True,layout='constrained')
    for ax,panel in zip(axes,['G1','ABL1']):
        rs=[r for r in new if r['panel']==panel];labels=[r['method'].replace('B_cum(128)',r'$B_{\mathrm{cum}}(128)$').replace('B_cum(512)',r'$B_{\mathrm{cum}}(512)$') for r in rs];bars=ax.bar(labels,[r['hits']/160*100 for r in rs],color=['#397d4c' if panel=='ABL1' and r['method'] in ['A1','A2'] else '#7696ae' for r in rs],width=.65)
        for bar,r in zip(bars,rs):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{r['hits']}/160",ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,104);ax.set_yticks([0,25,50,75,100]);ax.set_title(panel+' (same 160 nonlinear models)',fontsize=11);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.tick_params(axis='x',labelsize=9)
    assert len({tuple(b.get_facecolor()) for b in axes[0].patches})==1
    assert axes[0].patches[-1].get_height()>axes[0].patches[0].get_height()
    axes[0].set_ylabel('Prediction accuracy (%)');fig.savefig(dest/'figure3_g1_abl1.pdf');fig.savefig(dest/'figure3_g1_abl1.png',dpi=180);plt.close(fig)
    cc=rows(A/'E3_ENDPOINT_COUNTS_BY_MODEL.csv');table('FIGURE2_SOURCE.csv',cc);assert all(int(r['unclassified'])==0 for r in cc)
    fig,ax=plt.subplots(figsize=(10.1,3.9),layout='constrained');x=np.arange(10)
    for c,offset in [('0',-.19),('1',.19)]:
        a=[next(r for r in cc if r['initialization']==str(s) and r['data_condition']==c) for s in range(100,110)];bottom=np.zeros(10)
        for field,color,label in [('shared','#dc7c24','Shared candidate'),('distinct','#2864a0','Distinct')]:
            v=np.array([int(r[field]) for r in a]);ax.bar(x+offset,v,.34,bottom=bottom,color=color,label=label if c=='0' else None,hatch='' if c=='0' else '///',edgecolor='white',lw=.6);bottom+=v
        for xx in x+offset:ax.text(xx,4.08,'c'+c,ha='center',fontsize=8)
    ax.set_xticks(x,[str(s) for s in range(100,110)]);ax.set_xlabel('Initialization seed (paired streams c0 / c1)');ax.set_ylabel('Input pairs per trained model');ax.set_yticks([0,1,2,3,4]);ax.set_ylim(0,4.7);ax.legend(ncol=2,loc='upper center',fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    assert ax.get_legend_handles_labels()[1]==['Shared candidate','Distinct']
    fig.savefig(dest/'figure2_e3_initializations.pdf');fig.savefig(dest/'figure2_e3_initializations.png',dpi=180);plt.close(fig)
    print('Figures complete: left G1 counts 140,91,84,119,143; Bcum512 above A',flush=True)

def manuscript():
    main=src('inputs/main_v0.7.2.tex');bib=src('inputs/refs_v0.7.2.bib');backup=O/'source_originals';backup.mkdir(exist_ok=True)
    for p in [main,bib]:shutil.copyfile(p,backup/p.name)
    for p in ['E3_INITIALIZATION_AUDIT.md','E3_ENDPOINT_DISTANCE_AUDIT.md','TIMING_HISTORY_INDEPENDENCE.md']:src(A/p)
    s=main.read_text(encoding='utf-8');edits=[];logs=[]
    def replace_check(old,new,note):
        nonlocal s
        token='\\CHECK{'+old+'}';assert s.count(token)==1,(old,s.count(token));line=s[:s.index(token)].count('\n')+1;edits.append(dict(kind='CHECK',old=token,new=new,line=line));s=s.replace(token,new,1);logs.append(dict(line=line,old=old,replacement=new,evidence=note,status='partial' if '\\CHECK{' in new else 'filled'))
    ir=rows(A/'E3_INITIALIZATION_AUDIT.csv');assert len(ir)==10 and all(truth(r['paired_equal']) and int(r['other_initializations_duplicate'])==0 for r in ir);assert len({r['c0_initial_SHA256'] for r in ir})==10
    replace_check('confirm wording against E3\\_INITIALIZATION\\_AUDIT.md, including whether a direct comparison with the V053/V054/V055 initial hashes was possible',r'(direct tensor-hash comparison is possible with V053, V054 B0/B1 and V055; V054 T0/T1 have different three-output heads and are excluded)', 'E3_INITIALIZATION_AUDIT.md / .csv; E3_CHECKPOINT_SCHEMA_AUDIT.csv: 10 unique initial hashes, paired equality, binary schemas match; ternary heads differ')
    replace_check('norm used and typical state norm, from E3\\_ENDPOINT\\_DISTANCE\\_AUDIT.md',r'(Euclidean distance in the 545-dimensional hidden state; \CHECK{typical state norm is not tabulated in the saved endpoint-distance audit})','E3_ENDPOINT_DISTANCE_AUDIT.md: Euclidean norm,545 dimensions. Typical endpoint state norm is absent; no saved distance is substituted for a state norm and no new metric is computed.')
    dist=rows(A/'E3_ENDPOINT_DISTANCE_AUDIT.csv');assert len(dist)==80 and not any(truth(r['near_threshold_factor2']) or truth(r['near_threshold_10percent']) for r in dist)
    replace_check("state the audit's definition of ``near''",r'(zero pairs in $[0.9,1.1]\times10^{-3}$ or in the wider factor-two band $[0.5,2]\times10^{-3}$)', 'E3_ENDPOINT_DISTANCE_AUDIT.md / .csv: both frozen audit-neighborhood counts0/80; the actual classification threshold remains1e-3')
    hist=rows(A/'TIMING_HISTORY_SUMMARY.csv');assert float(next(r['range_max'] for r in hist if r['study']=='E3' and r['axis']=='across_prefixes_at_fixed_k'))==15;assert float(next(r['range_max'] for r in hist if r['study']=='V054T/H' and r['axis']=='across_prefixes_at_fixed_k'))==19
    replace_check('confirm k values and maxima against TIMING\\_HISTORY\\_INDEPENDENCE.md',r'(E3: $k\in\{16,64,256,1024\}$; original models: $k\in\{8,16,64,256,1024,4096\}$)', 'TIMING_HISTORY_INDEPENDENCE.md, SUMMARY.csv and OBSERVATIONS.csv: maxima15/19; fixed-history across-k ranges0; k values transcribed from archived rows')
    floor=rows(A/'E2_FLOOR_AUDIT.csv');conditions=[['transplant_correct'],['normal_1_correct'],['normal_1_failed'],[f'orthogonal{i}_correct' for i in range(3)],[f'orthogonal{i}_failed' for i in range(3)],[f'random{i}_correct' for i in range(3)],[f'random{i}_failed' for i in range(3)]]
    assert s.count('\\CHECK{}')==7
    for names in conditions:
        values=[]
        for name in names:
            a=[r for r in floor if int(r['at'])==8 and r['condition']==name];assert len(a)==10;values.append(sum(truth(r['left_censored_latent_onset']) for r in a))
        assert values==[0]*len(names);token='\\CHECK{}';line=s[:s.index(token)].count('\n')+1;s=s.replace(token,'0/10',1);edits.append(dict(kind='CHECK',old=token,new='0/10',line=line));logs.append(dict(line=line,old='empty immediate-error table cell',replacement='0/10',evidence='E2_FLOOR_AUDIT.csv,at=8,conditions='+','.join(names)+'; each direction separately0/10, directions are not treated as30 independent contexts',status='filled'))
    for before,after in [('figures/figure3_e3_initializations.pdf','figures/figure2_e3_initializations.pdf'),('figures/figure2_g1_abl1.pdf','figures/figure3_g1_abl1.pdf')]:
        assert s.count(before)==1;s=s.replace(before,after,1);edits.append(dict(kind='graphics_path',old=before,new=after))
    assert '\\bibliography{refs}' in s;s=s.replace('\\bibliography{refs}','\\bibliography{refs_v0.7.2}',1);edits.append(dict(kind='bibliography_path',old='\\bibliography{refs}',new='\\bibliography{refs_v0.7.2}'))
    write('main_v0.7.2.tex',s);shutil.copyfile(bib,O/bib.name)
    # Verify that no text outside allowed replacements has changed: replay edit list in order.
    check=main.read_text(encoding='utf-8')
    for e in edits:check=check.replace(e['old'],e['new'],1)
    assert check==s
    table('CHECK_FILL_LOG.csv',logs);dump('ALLOWED_EDITS.json',edits)
    detail='\n'.join(f"### 元の{r['line']}行\n- 状態：{r['status']}\n- 転記：`{r['replacement']}`\n- 出典：{r['evidence']}\n" for r in logs)
    write('CHECK_FILL_LOG.md','# CHECK転記記録\n\n本文のCHECK11箇所中10箇所を完了、1箇所は距離ノルム・次元のみ確認し典型状態ノルムを保留。典型状態ノルムは指定の既存監査文書・CSVに未掲載。新規計測は禁止されているため算出せず、距離中央値を状態ノルムと取り違えない。\n\n'+detail+'\n## 整合性と変更範囲\n\n今回転記した数値と対応本文の矛盾は0件。原稿の15/19step、初期SHA、0近傍、全Immediate0/10は既存CSVと一致。未解決は典型状態ノルムの出典欠落1件。本文の解釈やその他の科学的主張は修正・再監査していない。CHECK以外の変更はincludegraphics2箇所とbibliographyファイル名1箇所だけ。ALLOWED_EDITS.jsonの逐次適用で元原稿から全文が再現できることを検査した。\n\n図1のcarry表記は指定どおり共通凡例に統一し、SUBパネルでは同じラベルがborrowを指す。元のt=0規約を変更していない。図の順序は算術→E3→G1/ABL1。右パネルのABL1は事後分析、左のB_cumも事後ベースラインであり、G1の事前登録結果は変更しない。\n')
    for name in ['natbib.sty','booktabs.sty','plainnat.bst']:shutil.copyfile(src(A/'jmlr_build'/name),O/name)
    style=src(O/'jmlr2e.sty');assert style.read_text(encoding='utf-8').startswith('%');dump('STYLE_PROVENANCE.json',dict(official_guide='https://www.jmlr.org/format/format.html',official_style='https://raw.githubusercontent.com/JmlrOrg/jmlr-style-file/master/jmlr2e.sty',SHA256=sha(style),matches_prior_real_style=sha(style)==sha(src(W/'paper/jmlr2e.sty')),stub_style_used=False,failed_legacy_URL='https://www.jmlr.org/format/jmlr2e.sty returned404; official guide provided current GitHub URL'))
    dump('INPUT_SHA256.json',INPUTS);dump('PREPARED.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CHECKs_fully_filled=10,CHECKs_partially_filled=1,new_training=0,new_inference=0,new_Test=0,body_changes_limited_to_CHECK=True,figures=3))
    print('Manuscript prepared:10 CHECKs complete,1 partial (missing state norm); source files unchanged',flush=True)
if __name__=='__main__':figures();manuscript()
