"""Reports, measured figures and byte-identical JMLR compilation input; no manuscript edits."""
from audit_v072 import *
def tracked(p):
    manifest=json.loads((O/'INPUT_SHA256.json').read_text(encoding='utf-8'));p=Path(p);manifest.setdefault(str(p),{'bytes':p.stat().st_size,'sha256':sha(p)});(O/'INPUT_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');return p
def readrows(p):return rows(tracked(p))
def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    dest=O/'figures';dest.mkdir(exist_ok=True);t=readrows(W/'v054h_dynamics/trajectory.csv');t=[r for r in t if int(r['t'])<=512]
    table('FIGURE1_SOURCE.csv',t);fig,axes=plt.subplots(2,3,figsize=(10.1,7.6),sharex=True,sharey=True,layout='constrained');colors={'00':'#2864a0','01':'#dc7c24','10':'#397d4c','11':'#9e4d80'}
    for oi,op in enumerate(['ADD','SUB']):
        for seed in range(3):
            ax=axes[oi,seed];groups=defaultdict(list)
            for r in t:
                if r['op']==op and int(r['seed'])==seed:groups[(r['state'],r['pair'])].append(r)
            for (s,p),rs in sorted(groups.items()):
                rs.sort(key=lambda r:int(r['t']));ax.plot([int(r['t']) for r in rs],[float(r['margin']) for r in rs],color=colors[p],ls='-' if s=='0' else '--',lw=1.25,label=f'{p}, state {s}')
            ax.axhline(0,color='black',lw=.6);ax.set_xscale('symlog',base=2,linthresh=8);ax.set_xticks([0,8,32,128,512],['0','8','32','128','512']);ax.grid(alpha=.15);ax.set_title(f'{op}, seed {seed}');ax.legend(fontsize=6.6,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.24),frameon=False);ax.set_xlim(0,512)
            if oi==1:ax.set_xlabel('Stored trajectory index t (original convention)')
            if seed==0:ax.set_ylabel('Correct minus wrong logit')
    fig.savefig(dest/'figure1_arithmetic_margins.pdf');fig.savefig(dest/'figure1_arithmetic_margins.png',dpi=180);plt.close(fig)
    # Recompute accuracy from model-level predictions/outcomes, same 160 nonlinear G1 models.
    gr=readrows(W/'IronMath_G1/g1_results.csv');ar=readrows(W/'IronMath_PaperCheck/addon/abl1/abl1_model_results.csv');gr=[r for r in gr if r['architecture']!='linear'];ar=[r for r in ar if r['architecture']!='linear'];assert len(gr)==len(ar)==160 and {r['model'] for r in gr}=={r['model'] for r in ar}
    gtruth={r['model']:b(r['actual_success']) for r in gr};assert all(b(r['actual_success'])==gtruth[r['model']] for r in ar)
    tracked(W/'IronMath_PaperCheck/addon/abl1/RULES_ABL1.md')
    data=[]
    for study,rs,methods in [('G1',gr,['A','B','C_all_success','C_all_failure','D']),('ABL1',ar,['A0','A1','A2','A3','A4','A5'])]:
        for method in methods:
            hits=sum(b(r[method])==gtruth[r['model']] for r in rs);data.append(dict(study=study,method=method,N=160,hits=hits,accuracy=hits/160))
    table('FIGURE2_SOURCE.csv',data);fig,axes=plt.subplots(1,2,figsize=(10.1,3.8),sharey=True,layout='constrained')
    for ax,study in zip(axes,['G1','ABL1']):
        rs=[r for r in data if r['study']==study];labels=[r['method'].replace('C_all_success','All pass').replace('C_all_failure','All fail') for r in rs];val=[r['hits'] for r in rs];bars=ax.bar(labels,np.array(val)/160*100,color=['#2864a0' if r['method'] in ['A','A0'] else '#397d4c' if r['method'] in ['A1','A2'] else '#a4b3c1' for r in rs],width=.65)
        for bar,n in zip(bars,val):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{n}/160',ha='center',va='bottom',fontsize=8)
        ax.set_ylim(0,104);ax.set_yticks([0,25,50,75,100]);ax.set_title(study+' (same 160 nonlinear models)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Prediction accuracy (%)');fig.savefig(dest/'figure2_g1_abl1.pdf');fig.savefig(dest/'figure2_g1_abl1.png',dpi=180);plt.close(fig)
    cc=readrows(O/'E3_ENDPOINT_COUNTS_BY_MODEL.csv');table('FIGURE3_SOURCE.csv',cc);fig,ax=plt.subplots(figsize=(10.1,3.9),layout='constrained');x=np.arange(10);width=.34
    for c,offset in [('0',-.19),('1',.19)]:
        a=[next(r for r in cc if r['initialization']==str(s) and r['data_condition']==c) for s in range(100,110)];base=np.zeros(10)
        for field,color,label in [('shared','#dc7c24','Shared candidate'),('distinct','#2864a0','Distinct'),('unclassified','#9a9a9a','Unclassified (0)')]:
            y=np.array([int(r[field]) for r in a]);ax.bar(x+offset,y,width,bottom=base,color=color,label=label if c=='0' else None,hatch='' if c=='0' else '///',edgecolor='white',lw=.6);base+=y
        for xx,r in zip(x+offset,a):ax.text(xx,4.08,'c'+c,ha='center',fontsize=7)
    ax.set_xticks(x,[str(s) for s in range(100,110)]);ax.set_xlabel('Initialization seed (paired streams c0 / c1)');ax.set_ylabel('Input pairs per trained model');ax.set_yticks([0,1,2,3,4]);ax.set_ylim(0,4.7);ax.legend(ncol=3,loc='upper center',fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.savefig(dest/'figure3_e3_initializations.pdf');fig.savefig(dest/'figure3_e3_initializations.png',dpi=180);plt.close(fig)
    write('FIGURES.md','''# 実測図と転載用説明

Figure 1: Original V054H arithmetic margin trajectories, all 36 contexts (3 checkpoints × 12 contexts), displayed through original stored index t=512. Solid/dashed lines distinguish incoming state0/1; color denotes input bit pair. Horizontal zero is the readout boundary. Original trajectory t=0 already follows the first repeated update, unlike E2's 1-based step labels. This figure is descriptive, not a new prediction test. CSV provides every displayed point.

Figure 2: G1 frozen predictors and exploratory ABL1 component predictors on the same 160 nonlinear G1 models and same main outcome definition. The 40 linear models are excluded from both panels. ABL1 is post hoc, not an independent replication or a preregistered comparison. A0=140/160, A1=145/160; no significance claim is derived from the plotted counts alone. A2 requires no wrong or invalid readout anywhere in all diagnostic trajectories; A3 requires convergence and correct endpoint readout; A4 requires convergence and local stability; A5 is frozen D (convergence and trajectory finiteness). Exact definitions remain in RULES_ABL1.md. Labels and every count are available in FIGURE2_SOURCE.csv.

Figure 3: All 80 paired-endpoint cases from 20 E3 models. Each initialization has two presentation streams, each with four input pairs. 46 shared candidates, 34 distinct, zero unclassified. Shared means numerical convergence and distance<1e−3 in 545 dimensions; it is not proof of identical mathematical fixed points. Pairs and streams are not independent initialization units. CSV identifies every model. The zero-height unclassified category is retained in the legend.

PDF and 180dpi PNG versions are provided; the PDF is vector output. All inputs are saved measurements. No model inference, generated test data, or schematic curves are used.
''')
    print('Three measured figures saved',flush=True)

def plan():
    main=tracked(Path('inputs/main (8).tex'));v061=tracked(Path('inputs/main_v0.6.1.tex'));s=main.read_text(encoding='utf-8');t=v061.read_text(encoding='utf-8');assert 'v0.7.1' in s[:150] and 'Exact timing rules' in t
    exact=re.search(r'\\paragraph\{Exact timing rules\.\}.*?(?=\n\s*\n)',t,re.S).group(0);unpred=re.search(r'\\paragraph\{Unpredicted failures\.\}.*?(?=\n\s*\n)',t,re.S).group(0)
    write('V061_CONFIRMED_TIMING_TEXT.txt',exact+'\n\n'+unpred+'\n');c=readrows(O/'E3_ENDPOINT_CROSSTAB.csv');ds=readrows(O/'E3_ENDPOINT_DISTANCE_AUDIT.csv');init=readrows(O/'E3_ENDPOINT_COUNTS_BY_MODEL.csv');history=readrows(O/'TIMING_HISTORY_SUMMARY.csv')
    write('V072_REVISION_PLAN.md',f'''# v0.7.2 原稿変更計画（Claude担当、Codexは本文を変更しない）

数値監査を完了し、別スクリプトでE2全1350マージンとE3全15360hold整数予測を検算。原稿v0.7.1はDownloads/main (8).tex（SHA {sha(main)}）、確認済みv0.6.1はmain_v0.6.1.tex（SHA {sha(v061)}）。この計画は査読後の表示・解釈修正であり、凍結した実験判定は変更しない。新学習・新推論・新Testは0。本文改稿は追加指定DによりClaude担当。

## 修正項目

1. v0.7.1のDefinition audit note（現在276行）を削除し、V061_CONFIRMED_TIMING_TEXT.txtのExact timing rulesに置換する。算術D1はfirst strictly negative、持続条件なし、観測範囲±3との比較。記憶課題は非保持判定のprefixだけから最初の負マージンの最小を取り、全prefix保持なら時刻予測なし。argmax tie=class0とstrict marginの非対称性も維持。今回は元protocolを改訂しない。
2. 線形37件は「保持されたprefixで一度負マージンになったから予測あり」と書かない。全prefixが終点で保持判定されたため原規則では予測時刻なし。代替first-negative分析は事後感度分析として明確に分離。原v0.6.1のUnpredicted failuresを提供済み。
3. 貢献1・第5節タイトル・172行見出し『Fixed points explain which pairs break and when』を『Fixed-input trajectories and observed arithmetic failures』等の観測対応表現へ弱化。179行の『The network therefore does not implement a one-bit carry register』『This explains both』は内部計算全体の否定・因果的説明とせず『These measurements are consistent with input-dependent transient retention in the examined contexts』等へ限定する。該当行番号はv0.7.1に対する位置で改稿後は変動する。
4. 算術事前登録D1/D2、算術posthoc共有終点解析、FF1/G1の登録済みモデル成否、記憶課題の事後timing再解析、ABL1事後分解、E1/E2/E3拡張を分ける。旧G1 P1/P2やABL1 140対145を改変しない。
5. AbstractにE2の下限・対照とE3の全80pair内訳を追加。例として『At step8 all ten failing contexts reached the first observable error under three negative/transplant conditions; their identical mean shift of−34.3 is a floor, not equal latent effect. E3 yielded46 shared candidate pairs with failure and34 distinct pairs without observed failure through1024 steps.』を素材として提供。因果・独立コーパス・無限長成功とは書かない。
6. E3本文・表に全80pairクロス表と側単位D2形式を追加。共有46組は全て窓内破綻、別34組は全て最大1024で破綻なし。全240側では終点正194側は窓内誤答なし、誤46側は破綻。元36/36の分母は側であってpairではない。旧最大k4096対E3最大k1024という観測窓差を明示する。
7. 共有候補は『両側の20000反復後の最大移動量<1e−6、545次元終点距離<1e−3』の操作的判定。厳密機構の追加条件は局所rho<1・少なくとも片側誤読出し・同じ側で観測破綻。数値近接を同一の数学的固定点と同一視しない。±10%・factor2の閾値近傍はいずれも0件（閾値変更なし）。
8. E2は全方向0/1/2・両donorノルム・全時刻・全36文脈を付表で提供し、本文は10失敗文脈at8と明記。負法線2条件と失敗donorは全て10/10で最初の観測時点に誤答。負法線correct/failedの回復は6/10・1/10、failed transplant0/10で、同じfloorでも持続性は同じではない。左打切りと右打切りを明記。3方向内SDと文脈間SDを混同しない。
9. 旧算術の観測・時期予測を先に、E1有限時刻理論→E2介入→E3独立初期化の拡張を連続した節に整理。記憶課題とABL1は別の予測価値検証として整理。負の結果と制約を残す。
10. AI利用開示は『Claude Codeのみが全実験を実装実行』を修正。今回のE1〜E3実装・実行・独立検算、今回の追加集計・図・組版はOpenAI Codexが担当。Claudeは査読・v0.7.1/次版改稿、GPTは指示書・研究計画等に関与。古い全系列の実装担当は保存ログで確認できる範囲のみ断定し、人間著者が最終責任を負う。モデル自己申告だけで古い担当を推測しない。
11. 提供する実測Figure1〜3と各CSV・説明文を利用。Figure2は同じ160非線形モデルのみ、ABL1は事後分析と強調。Figure3は10初期化×2streamで独立単位10を維持。Figure1は元保存軌道のt規約を明示。
12. Codexは本文を改稿せず、byte-identical v0.7.1コピーをJMLRでコンパイル確認する。現版のエラー・警告は隠さずBUILD_REPORT.mdへ。将来Claudeのv0.7.2完成後の組版保証は今回できない。

## 時期予測について必ず追加する制限

第8節に『The timing prediction is computed by iterating the same holding input and is therefore close to the observation procedure. Its nontrivial empirical content concerns the degree to which break times transfer across preceding histories, not a computational shortcut or a general history-independent clock.』に相当する文を追加する。今回の保存データでは、固定prefix/suffixでkを変えた観測破綻時刻の範囲は全て0。しかし同じkでprefix/suffixを変えると旧V054T/Hは最大19step、E3は最大15step、範囲中央値8/5step。したがって『厳密な履歴非依存性が確認された』とは書かない。観測打切り・prefixの先行失敗・suffixも混ざる集計という制約を併記する。

結論では、E2はnorm一致の全方向を含めても法線方向への直後効果が強いが、負方向の−34.3は観測下限、他方向に対する潜在強度の比較ではない、と記載する。正法線の平均変化+7.1/+10.8、正解donor+7.0、random3方向平均−0.167/−3.767、orthogonal3方向平均−0.2/−3.5（correct/failedノルム、at8・10文脈）という全対照を提供する。幾何固有の因果性や永久修復は未証明。

## Q1〜Q5への回答

- Q1：はい。3条件は全10文脈でat8の観測下限に到達。平均−34.3が等しくても介入直後marginや回復は異なる。
- Q2：全方向を含めても法線の符号付き直後効果と観測時刻効果は際立つ。ただし負方向はfloorで飽和、共有アトラクタ幾何固有の因果性は示さない。
- Q3：別終点34組の破綻0、最大1024stepまで破綻なし34、評価不能0。共有46は破綻46。
- Q4：元D2と同じ側単位の対応はE3全240/240で再現した。共有候補は46/80、両提示条件で厳密機構を少なくとも1組持った初期化は9/10。ただし旧36/36・共有pair率・初期化単位再現率は別指標。独立初期化は10、同一Train母集団、同一構造、最大観測長は旧と異なる。
- Q5：全ケース・全対照を開示した条件付きの終点読み出し対応、状態方向への感度、同構造での再現を強く記述できる。潜在効果の連続的優越、共有構造の因果同定、厳密な履歴非依存、数学的に同一の固定点、無限長汎化は弱める。ABL1の単純readout優位を維持する。

追加実験・学習・本文改稿は実施せず、この計画・数値・図・組版診断をClaudeへ渡す成果物として保存して停止する。
''')
    build=O/'jmlr_build';build.mkdir(exist_ok=True)
    for src,dst in [(main,'main.tex'),(X/'paper/v07/refs.bib','refs.bib'),(X/'paper/v07/jmlr2e.sty','jmlr2e.sty')]:shutil.copyfile(tracked(src),build/dst)
    assert sha(build/'main.tex')==sha(main)
    dump('MANUSCRIPT_PRESERVATION.json',dict(original_v071=str(main),original_SHA256=sha(main),build_copy_SHA256=sha(build/'main.tex'),body_changed=False,revision_plan_only=True,confirmed_v061=str(v061),confirmed_v061_SHA256=sha(v061)))
    schema=readrows(O/'E3_CHECKPOINT_SCHEMA_AUDIT.csv');same=Counter(Path(r['checkpoint']).parts[0] for r in schema if b(r['keys_shapes_dtypes_order_match_E3']));diff=[r['checkpoint'] for r in schema if not b(r['keys_shapes_dtypes_order_match_E3'])]
    p=O/'E3_INITIALIZATION_AUDIT.md';s=p.read_text(encoding='utf-8');write(p.name,s+'\n## 保存checkpointスキーマの直接確認\n\n同じキー順・shape・dtype：'+str(dict(same))+'。非一致：'+str(len(diff))+'件（V054の3値head、詳細CSV）。旧モデルを生成せず保存済みtensorを読み取り、E3選択checkpoint20個のinitial SHAメタデータ・ファイルSHAも再確認。V053全条件、V054 B0/B1、V055は同じテンソル構造で初期SHAを直接比較できるが、V054 T0/T1の3値headは比較不可。V055の学習データ・変換方式は違うので、同構造でも同一学習課題とは呼ばない。hash手順はキー名+tensor bytes、ファイルcontainer hashと混同しない。\n')

if __name__=='__main__':
    assert json.loads((O/'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))['passed'];plan();figures()
