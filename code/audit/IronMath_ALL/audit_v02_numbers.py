"""Numerical source check only: standard library, saved text/CSV/JSON."""
from check_v02_diff import W,OUT,sha,now,loadcsv,prepare
from pathlib import Path
import json,csv,math,statistics,collections
J=lambda p:json.loads((W/p).read_text(encoding='utf-8-sig'))
TB='IronMath_PaperCheck/addon/timing_baseline/'
FP='IronMath_PaperCheck/addon/fp64/'
FR='IronMath_PaperCheck/addon/fresh_eval/'
claims=[];inputs={}
def cite(path,needle=None):
    p=W/path
    if not p.exists():raise FileNotFoundError(p)
    inputs[path]=sha(p)
    lines=p.read_text(encoding='utf-8-sig').splitlines()
    if needle is None:return f'`{path}`'
    found=[i+1 for i,s in enumerate(lines) if needle in s]
    assert found,(path,needle)
    return f'`{path}:{found[0]}`'
def add(id,line,claim,status,verified,source,note=''):
    assert status in ['一致','不一致','出典が見つからない']
    claims.append(dict(ID=id,line=str(line),claim=claim,status=status,verified=verified,source=source,note=note))
def ok(id,line,claim,verified,source,note=''):add(id,line,claim,'一致',verified,source,note)
def run():
    placement=prepare();main=W/'paper/main.tex';start=sha(main);assert start==placement['main_SHA256']
    summary={r['section']:r for r in loadcsv(W/(TB+'timing_baseline_summary.csv'))}
    units=loadcsv(W/(TB+'timing_baseline_units.csv'))
    spearman=loadcsv(W/(TB+'timing_baseline_spearman.csv'))
    # The seven new table rows: compare the stated population, rates and intervals.
    groups=[('Arithmetic','算術（V054T＋V054H）',261,10,70,0,[-19.25,-21.,-17.5]),('R non-divergent','記憶 R保持・非発散',262,75,35,17,[-6.,-9.75,-1.]),('G1 residual GRU','G1 residual_gru R',263,7,71,0,[-13.25,-82.25,-2.]),('G1 GRU','G1 gru R',264,8,75,0,[-19.5,-48.5,-8.5]),('G1 LSTM','G1 lstm R',265,17,59,0,[-18.75,-25.5,-13.75]),('G1 linear','G1 linear R',266,40,0,28,[1.,-1.,4.]),('R divergent','記憶 R保持・発散型',267,17,18,0,[-639.75,-1432.75,-88.25])]
    paired_details=[]
    for i,(name,key,line,n,dp,lp,d) in enumerate(groups,1):
        r=summary[key];assert int(r['cmp_L_n_units'])==n
        actual=[float(r[k]) for k in ['cmp_L_median_d','cmp_L_ci_low','cmp_L_ci_high']];assert actual==d
        source=cite(TB+'timing_baseline_summary.csv',key+',')+'; '+cite(TB+'REPORT_TIMING_BASELINE.md',key if key.startswith('G1') else '算術（V054T＋V054H）' if name=='Arithmetic' else '記憶 R保持・非発散' if name=='R non-divergent' else '記憶 R保持・発散型')
        if name=='R non-divergent':
            rr=[u for u in units if u['kind']=='R' and u['divergent']=='False' and u['baseline_L']]
            assert len(rr)==75
            diag=sum(abs(float(u['pred'])-float(u['actual_point']))<=3 for u in rr)
            base=sum(abs(float(u['baseline_L'])-float(u['actual_point']))<=3 for u in rr)
            assert (diag,base)==(25,13)
            add('T'+str(i),line,f'{name}: Units=75、Diag±3=35%、L±3=17%、d=−6.0、95%区間[−9.75,−1.0]','不一致','同じ比較可能75単位なら診断25/75=33.333…%（整数表示33%）、Lは13/75=17.333…%（17%）。35%は全79単位の28/79=35.443…%を丸めた値。dと区間は75単位で一致。',source+'; '+cite(TB+'timing_baseline_units.csv'),'分母混在。75を維持するならDiagを33%にするか、診断79単位・基準線75単位と分母を別記する。main.texは変更していない。')
            paired_details.append(dict(group=key,comparable_units=75,diag_within3=diag,baseline_within3=base,diag_rate=diag/75,baseline_rate=base/75,summary_diag_denominator=79,summary_diag_within3=28))
        else:
            assert round(float(r['pred_within3'])*100)==dp and round(float(r['L_within3'])*100)==lp
            ok('T'+str(i),line,f'{name}: Units={n}、Diag±3={dp}%、L±3={lp}%、dと95%区間',f'{n}; {float(r["pred_within3"])*100:.9g}%; {float(r["L_within3"])*100:.9g}%; d={actual[0]}, CI={actual[1:]}',source)
    # Within-architecture correlations and quantitative body text.
    for i,arch in enumerate(['residual_gru','gru','lstm','linear'],1):
        r=summary[f'G1 {arch} R'];expected=[.96,1.,.96,.97][i-1];assert round(float(r['pred_spearman']),2)==expected
        ok(f'S{i}',249,f'{arch}: Spearman={expected:.2f}、単位数={[7,8,17,40][i-1]}',f"rho={r['pred_spearman']}, n={r['n_units']}",cite(TB+'timing_baseline_spearman.csv','構造='+arch+','))
    ok('S5',249,'線形の予測は約8ステップ早い','符号つき誤差中央値−8.25',cite(TB+'timing_baseline_summary.csv','G1 linear R,'),'約8は中央値の要約。各単位が厳密に同じ誤差ではない。')
    ok('S6',251,'Mは記憶課題で多くがLと同じ','代替120単位（R59・N61）。同じ保持種の他単位がない場合はLへ代替。',cite(TB+'REPORT_TIMING_BASELINE.md','120単位'))
    for i,key in enumerate(['算術（V054T＋V054H）','G1 residual_gru R','G1 gru R','G1 lstm R'],7):
        r=summary[key];assert float(r['cmp_L_ci_high'])<0 and float(r['cmp_M_ci_high'])<0
        ok(f'S{i}',272,key+'で両基準線より時期予測が正確','対L・対Mとも誤差差の95%区間が負',cite(TB+'timing_baseline_summary.csv',key+','))
    rr=summary['G1 rnn R'];assert int(rr['n_units'])==1 and rr['cmp_L_verdict']=='対象なし'
    ok('S11',272,'Elman RNNはR保持1単位で比較不能','n=1、L/Mとも対象なし。相関も未定義。',cite(TB+'timing_baseline_summary.csv','G1 rnn R,'))
    rr=summary['G1 linear R'];assert float(rr['cmp_L_ci_low'])<=0<=float(rr['cmp_L_ci_high'])
    ok('S12',272,'線形モデルはLと明確な差なし','d=+1、95%区間[−1,+4]',cite(TB+'timing_baseline_summary.csv','G1 linear R,'),'対Mではd=−4、区間[−6.5,−3]。線形が基準線より良くないという要約はLに限定。')
    rr=summary['記憶 N保持・非発散'];assert round(float(rr['pred_frac_items_inside'])*100,1)==63.7 and round(float(rr['L_frac_items_inside'])*100,1)==20.4
    ok('S13',272,'N保持の区間内63.7%対L20.4%','452項目、診断288/452、L92/452。単位別差の区間[0,0.75]。',cite(TB+'timing_baseline_summary.csv','記憶 N保持・非発散,'),'長さ自体を独立単位にはしていない。単位別の割合差をbootstrap。')
    for i,arch in enumerate(['residual_gru','gru'],14):
        rr=summary[f'G1 {arch} N'];assert int(rr['n_units'])==3 and float(rr['cmp_L_ci_low'])>0
        ok(f'S{i}',272,f'N保持で差ありは{arch}、3単位',f"n=3, CI=[{rr['cmp_L_ci_low']},{rr['cmp_L_ci_high']}]",cite(TB+'timing_baseline_summary.csv',f'G1 {arch} N,'))
    ok('S16',272,'どの区分でも診断が基準線より不正確という明確な差なし','保存summaryの判定に「基準線より不正確」はなく、線形対Lは非有意。',cite(TB+'REPORT_TIMING_BASELINE.md','診断が基準線より劣った区分'))
    # Section 8 unchanged quantitative paragraph, explicitly requested body coverage.
    for id,line,key,claim,verify in [
        ('U1',247,'算術（V054T＋V054H）','算術10件、7/10が±3、符号誤差+2、rho0.96','n10、±3=70%、中央値+2、rho0.963636…'),
        ('U2',247,'記憶 R保持・非発散','非発散R79件、中央値−6、範囲内97.5%、rho0.97','n79、−6、77/79=97.468…%、rho0.96515478…'),
        ('U3',247,'G1 linear R','うち線形40件、中央値−8.25、±3は0','n40、−8.25、0/40'),
        ('U4',247,'記憶 R保持・発散型','発散R17件、中央値−7.5、rho0.75','n17、−7.5、rho0.752306…'),
        ('U5',247,'G1 rnn N','N保持G1は63.7%、RNN25.8%、外れは早すぎる','非発散N全体63.7168…%、RNN17/66=25.7576…%。早期性はbreak_timing報告で確認。')]:ok(id,line,claim,verify,cite(TB+'timing_baseline_summary.csv',key+',')+'; '+cite(TB+'REPORT_TIMING_BASELINE.md','N 保持'))
    ok('U6',274,'線形37単位は予測なし・誤答、別定義R32件中1件が範囲内','37=R32+N5。別定義Rの範囲内1/32=3.1%。事後感度分析。',cite(TB+'REPORT_TIMING_BASELINE.md','37単位'),'原稿では今回未変更だが第8節指定範囲として確認。')
    # Abstract: include unchanged numbers on the changed paragraph, as requested.
    g1=J('IronMath_G1/g1_metrics.json');r1=loadcsv(W/'IronMath_PaperCheck/r1_baselines.csv')
    ok('A1',34,'約100万パラメータ、8bit加減算で学習','1,000,334、8bit',cite('IronMath_V054/model.py','GRUCell')+'; '+cite('IronMath_V054/PREREGISTRATION.md','8'))
    ok('A2',34,'3seed平均95%超を65000bitまで維持','65000主評価平均96.0286458…%、全幅95%以上',cite('v054e_endurance/summary.csv','main,main_65000,65000,mean,ALL'))
    ok('A3',34,'伝播入力12中10で単一引き込み点','10/12（入力対×演算条件を跨ぐ）',cite('v054h_dynamics/REPORT_V054H.md','12件中10件'))
    ok('A4',34,'破綻する入力carry対の予測36/36','D2 36/36',cite('IronMath_ALL/ALL_PREDICTIONS.csv','D2,'))
    ok('A5',34,'失敗10件すべて観測範囲の±3以内、中央値±3は7/10','観測範囲内10/10（従って範囲±3以内も全件）。中央値±3は7/10。',cite(TB+'timing_baseline_summary.csv','算術（V054T＋V054H）,'))
    ok('A6',34,'非線形160モデル・4構造・長さ2048〜65000','4構造×40、主判定は2048/8192/65000',cite('IronMath_G1/REPORT_G1.md','160'))
    ok('A7',34,'A140、D91、p約5.9e−12',f"{g1['pooled_nonlinear']['methods'][0]['hits']}; D91; p={g1['pooled_nonlinear']['McNemar_exact_two_sided_p']}",cite('IronMath_G1/g1_metrics.json','McNemar_exact_two_sided_p'))
    ok('A8',34,'128までのテスト119、512まで143、512との差は不明確','119/160と143/160。512とのMcNemar p=0.6900379…',cite('IronMath_PaperCheck/r1_baselines.csv','G1,nonlinear,128,B_cum')+'; '+cite('IronMath_PaperCheck/r1_comparison.csv','G1,nonlinear,512,B_cum'))
    ok('A9',34,'診断は約7〜108倍の計算','0.770/0.114≈6.75、12.310/0.114≈107.98',cite('IronMath_PaperCheck/stage2/REPORT_STAGE2.md','108'),'実装・CPU依存の時間比較。')
    ok('A10',34,'対角線形40モデルすべて失敗','0/40成功、Aの失敗予測40/40',cite('IronMath_G1/REPORT_G1.md','|linear|'))
    ok('A11',34,'時期予測は算術・残差GRU・GRU・LSTMで基準線より良いが線形では良くない','S7〜S12の各区間で一致。線形の非優位は対L。記憶課題の誤差は早期側。',cite(TB+'REPORT_TIMING_BASELINE.md','区分のまとめ'),'相関と絶対的な時期精度を区別する。')
    ok('C1',366,'結論：160非線形・4構造・128までの直接テストより正確','G1 A140対B128119、p0.001065…',cite('IronMath_PaperCheck/r1_comparison.csv','G1,nonlinear,128,B_cum'))
    ok('C2',366,'結論：算術・3非線形構造で時期予測が良く、記憶では数step早い','residual_gru/gru/lstmのR保持の誤差差区間はいずれも負。符号誤差中央値−3/−2.75/−3。',cite(TB+'timing_baseline_summary.csv','G1 residual_gru R,')+'; '+cite(TB+'REPORT_TIMING_BASELINE.md','G1 gru R'))
    ok('C3',366,'結論：構造内でほぼ完全な順位関係、線形は基準線より良くない','3非線形のrho0.9643/1/0.9619。線形rho0.9709でも対Lの誤差差区間は0を含む。',cite(TB+'timing_baseline_spearman.csv','構造=linear,')+'; '+cite(TB+'timing_baseline_summary.csv','G1 linear R,'),'within each architectureは直前の3構造の範囲なら裏付けられる。RNNは1単位で順位相関未定義。線形の非優位はLとの比較。')
    # Explicitly requested Section 9, even though unchanged in this revision.
    fp=J(FP+'fp64_summary.json');fn=loadcsv(W/(FP+'fp64_false_negative_models.csv'));changes=loadcsv(W/(FP+'fp64_A_changes.csv'))
    ok('F1',306,'FP64、全200、20000反復、閾値1e−6、実行前登録','RULESと登録SHAあり、全200×16軌道、20000反復、閾値1e−6',cite(FP+'REPORT_FP64.md','全200モデル')+'; '+cite(FP+'RULES_SHA256.json'))
    assert len(fn)==4 and all(r['all16_converged_FP64']=='True' for r in fn)
    ok('F2',306,'LSTM4つすべてで全軌道収束、移動量約1e−15','4/4、最大移動量1.5511e−15〜3.6321e−15、Aは全件成功予測',cite(FP+'fp64_false_negative_models.csv'))
    assert fp['F2']['hits_FP64']=={'nonlinear':145,'linear':40}
    ok('F3',306,'全200でA的中180→185、非線形145/160','FP32 140+40=180、FP64 145+40=185',cite(FP+'fp64_summary.json','hits_FP64'))
    changed=[r for r in changes if r['A_FP32']!=r['A_FP64']];assert len(changed)==5 and all(r['A_FP64']==r['actual_success'] and r['A_FP32']!=r['actual_success'] for r in changed)
    ok('F4',306,'変わった5予測はすべて外れ→的中','5モデル。CSV6行のうちLSTM s35はDのみ変化でA変化には含めない。',cite(FP+'fp64_A_changes.csv'))
    assert fp['F3']['trajectories_converged_FP32']==3040 and fp['F3']['readout_correctness_changed']==0
    ok('F5',306,'FP32収束3040本の読み出し正誤はすべて不変','3040本、変化0',cite(FP+'fp64_summary.json','readout_correctness_changed')+'; '+cite(FP+'fp64_trajectories.csv'))
    ok('F6',306,'偽陰性はFP32の丸めによるという記述','同一FP32重みのFP64化で4件が収束、rhoと読み出しは維持。出典報告の解釈と一致。',cite(FP+'REPORT_FP64.md','丸めによる'),'数字が裏付けるのは精度変更時の観測差。一般的な因果証明の照合ではない。')
    fresh=J(FR+'fresh_summary.json');ch=loadcsv(W/(FR+'fresh_changed_models.csv'))
    ok('E1',309,'新しいseedで全200モデルを再評価、登録予測すべて成立','E1〜E5支持。予測Aが全200で的中した意味ではない。',cite(FR+'REPORT_FRESH_EVAL.md','全200モデル')+'; '+cite(FR+'fresh_summary.json'))
    assert fresh['E1']['success_fresh_nonlinear']==76
    ok('E2',309,'非線形成功76で不変','新76、元76、線形0',cite(FR+'fresh_summary.json','success_fresh_nonlinear'))
    assert len(ch)==2 and all(r['A']!=r['actual_success_fresh'] and r['A']==r['actual_success_original'] for r in ch)
    ok('E3',309,'成否変化2モデル、両方でAが外れに','residual_gru_L16_s29とrnn_L16_s36',cite(FR+'fresh_changed_models.csv'))
    ok('E4',309,'Aは138/160、元140',f"新{fresh['E2']['A_hits_nonlinear']}、元140",cite(FR+'fresh_summary.json','A_hits_nonlinear')+'; '+cite(FR+'REPORT_FRESH_EVAL.md','138'))
    ok('E5',309,'A対D p=6.4e−11',f"52対5、p={fresh['E3']['p']}",cite(FR+'fresh_summary.json','only_first_correct'))
    ok('E6',309,'B_cum512との差p0.69、Bの的中が3多い',f"A138、B141、p={fresh['E4']['p']}",cite(FR+'fresh_summary.json','E4')+'; '+cite(FR+'REPORT_FRESH_EVAL.md','141'))
    ok('E7',309,'全線形が失敗','線形成功0/40',cite(FR+'fresh_summary.json','linear_success'))
    ok('E8',309,'rewrite/mixedは同一、random/longの前置きだけ新規','7長さすべてでrandom/longの同一行0。rewrite/mixed各2行は決定的で同一。',cite(FR+'REPORT_FRESH_EVAL.md','同じ行の数'),'新集合全体が独立ではないという注意も原稿にある。')
    # Other modified numerical lines.
    cfg=J('IronMath_V054/config.json')
    assert [cfg[k] for k in ['learning_rate','weight_decay','gradient_clip','per_op','C1_updates']]==[.002,.01,1.,8,1200]
    ok('O1',140,'lr0.002、wd0.01、clip1、ADD8/SUB8、1200更新、3seed、8bit','設定ファイルと事前登録に一致。',cite('IronMath_V054/config.json','learning_rate')+'; '+cite('IronMath_V054/PREREGISTRATION.md','8'))
    ok('O2',140,'V051/V053の同じ初期化は記録の範囲で、独立モデルと数えない','V053/V054初期SHA一致。V051の初期SHAは欠測、最終重みが近似。',cite('IronMath_ALL/ALL_LINEAGE.md','初期重みのハッシュが一致')+'; '+cite('IronMath_ALL/ALL_LINEAGE.md','V051 A1 自身'))
    ok('O3',143,'主評価65000で95%超、確認95.51%、1seed94.73%','主96.0286…%、確認平均95.5078…%、seed2は485/512=94.7265625%',cite('v054e_endurance/confirmation.csv','65000')+'; '+cite('v054e_endurance/summary.csv','main,main_65000,65000,mean,ALL'))
    ok('O4',143,'randomは全幅正解、no_propagation86.8%、alternating約70%','random全幅100%。65000の86.8%、70.8%。',cite('v054e_endurance/REPORT_V054E.md','no_propagation (43)')+'; '+cite('v054e_endurance/REPORT_V054E.md','alternating (16)'))
    ok('O5',143,'4096〜65000の全幅、初回誤りはseed順67/20/27','0+(2^L−1)の位置はseedごとに不変。2048は範囲に含めない。',cite('v054e_endurance/REPORT_V054E.md','長さに依存しない固定失敗'))
    signrows=loadcsv(W/'IronMath_PaperCheck/paper_checks/CHECK_TANHG_models.csv');div=[r for r in signrows if r['divergent']=='True'];assert len(div)==70 and all(float(r['tanh_g_FP32'])>0 for r in div)
    ok('O6',235,'発散70件=FF1 22+FF1-R29+G1 19、全てtanh(g)>0','CSVを再集計して70件・全て正を確認。',cite('IronMath_PaperCheck/paper_checks/CHECK_TANHG_models.csv'))
    counts={s:collections.Counter(r['divergent'] for r in signrows if r['experiment']==s) for s in ['FF1','FF1-R','G1']}
    assert [counts[s]['True'] for s in counts]==[22,29,19]
    assert all(float(r['tanh_g_FP32'])>0 for r in signrows if r['experiment'] in ['FF1-R','G1'])
    neg=[r for r in signrows if r['experiment']=='FF1' and float(r['tanh_g_FP32'])<=0];assert len(neg)==2 and all(r['divergent']=='False' for r in neg)
    ok('O7',235,'FF1の非正係数2件は非発散、FF1-R/G1は全残差GRUで係数が正','FF1 0/2発散、FF1-R/G1各40件とも正。正でも非発散あり。',cite('IronMath_PaperCheck/paper_checks/CHECK_TANHG_models.csv'),'符号だけで発散・非発散を分離できないという限定は妥当。')
    ok('O8',312,'2threads、約7/108倍、500反復、反復はN/R別に8軌道バッチ','s1_run.pyのrepeatを各symbolごとに1回、8終点をfor j in range(8)で読み出し・Jacobian処理。',cite('IronMath_PaperCheck/stage2/s1_run.py','runs[symbol]=s.repeat')+'; '+cite('IronMath_PaperCheck/stage2/REPORT_STAGE2.md','108'))
    ok('O9',356,'seedごとの3初期SHAはV053/V054/V055で同一、V051は記録なし','初期SHAをCPU記録から確認。V051の初期SHA不明を明記する修正。',cite('IronMath_ALL/ALL_LINEAGE.md','初期重みのハッシュが一致')+'; '+cite('IronMath_ALL/ALL_LINEAGE.md','初期重みは V053/V054'))
    ok('O10',356,'V051/V054最終重みの最大差2.1e−4','seed別最大差1.2e−5、2.1e−4、9.6e−6の最大',cite('v054r_replication/data/V051_vs_V054_weights.json')+'; '+cite('IronMath_ALL/ALL_LINEAGE.md','2.1e-4'))
    ok('O11',393,'B1対B0、2048bit SUB全seed勝ち、ADD平均低下≤5point、両予算不支持','判定規則と結果に一致。',cite('IronMath_ALL/ALL_PREDICTIONS.csv','Operation unification supported separately for each budget'))
    ok('O12',394,'平衡三進T1対T0も同条件、両予算支持','同じall3seed/SUB・ADD低下≤5条件を継承。',cite('IronMath_ALL/ALL_PREDICTIONS.csv','Operation unification supported separately for each budget'))
    ok('O13',396,'V055 Z1全seed2048bit≥90%、結果84.2〜88.3%、両予算不支持','C1 84.18/85.35/88.09、C2 84.18/85.35/88.28。全体84.18〜88.28→84.2〜88.3。',cite('IronMath_V055/REPORT_V055.md','|C1|Z1|')+'; '+cite('IronMath_V055/REPORT_V055.md','|C2|Z1|'))
    ok('O14',397,'F1/F2の支持条件SUB全3seed厳密増加・ADD平均低下≤5、F1支持/F2不支持','原文条件に一致。',cite('IronMath_ALL/ALL_PREDICTIONS.csv','matching archived B0 native SUB'))
    ok('O15',405,'3seed平均≥95%の最大幅65000、評価上限','主評価で65000、より長い未評価幅には外挿しない。',cite('v054e_endurance/REPORT_V054E.md','平均95%耐久幅'))
    ok('O16',406,'全seed≥95%の最大幅65000、確認ではseed2 94.73%で不成立','主評価の最小95.5078…%、確認seed2 94.7266…%。',cite('v054e_endurance/REPORT_V054E.md','全seed95%耐久幅'))
    ok('O17',409,'M3≥99% at65000の支持、全seed2048で<99%の不支持条件、31.6〜43.4%','2048seed別31.6/32.0/43.4%、全て99%未満。',cite('IronMath_ALL/ALL_PREDICTIONS.csv','all 3 seeds: 65000-bit random final-residue accuracy'))
    # Bind citations, raw check population and immutable input hashes to the report.
    for p in [TB+'REPORT_TIMING_BASELINE.md',FP+'REPORT_FP64.md',FR+'REPORT_FRESH_EVAL.md',TB+'timing_baseline_units.csv',TB+'timing_baseline_summary.csv',TB+'timing_baseline_spearman.csv',FP+'fp64_trajectories.csv',FR+'fresh_models.csv']:
        cite(p)
    assert sha(main)==start and all(sha(W/p)==h for p,h in inputs.items())
    counts=collections.Counter(r['status'] for r in claims)
    with (OUT/'claims.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,list(claims[0]));writer.writeheader();writer.writerows(claims)
    changed={i for c in placement['changes'] for i in range(c['current_lines'][0],c['current_lines'][1]+1)}
    audit=dict(completed_at=now(),counts=dict(counts),total=len(claims),main_SHA256_before=start,main_SHA256_after=sha(main),main_unmodified=True,source_hashes_unchanged=True,sources_SHA256=inputs,paired_population_checks=paired_details,changed_lines=sorted(changed),explicit_unchanged_coverage=[245,247,274,306,309],new_learning=False,new_inference=False,experimental_scripts_imported=False)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    text=f'''# v0.2 差分の数値照合

作成：{now()}。保存ファイルの読み取り・集計のみ。新しい学習・推論なし。

## 1. 原稿・差分・範囲

- 最新受領版：`inputs/main (4).tex`。`paper/main.tex` にバイト単位で同一の内容を配置した。
- 最新版SHA256：`{start}`。
- 前回照合版：`main (3).tex`、SHA256 `{placement['previous_SHA256']}`。
- 既存の `paper/main.tex` は前回版とSHA一致を確認し、`{Path(placement['archived']).relative_to(W).as_posix()}` に退避した。取得元・SHAは `paper/SOURCES.txt` に追記した。
- 差分は{len(placement['changes'])}ブロック、追加・変更{sum(len(c['added']) for c in placement['changes'])}行。`v02_audit/added_changed_lines.txt` は新しい行だけ、`v02_audit/main3_to_main4.diff` は削除・変更前も含む差分。
- 行番号は配置後の `paper/main.tex`。主張の数値のまとまりを1件、表は1行を1件と数える。LaTeXの列指定・位置指定、版名・日付、式の一般形、モデル名・節番号、数値のない削除行は件数に含めない。
- 数字が変わっていなくても、変更された行に含まれる数値は照合した。指定により第8節本文の未変更行と第9節FP64・新評価集合（306・309行）も追加で照合した。その他の未変更箇所を全件再照合したものではない。

## 2. 件数と要点

|区分|件数|
|---|---:|
|一致（原稿の桁数への丸めを含む）|{counts['一致']}|
|不一致|{counts['不一致']}|
|出典が見つからない|{counts['出典が見つからない']}|
|合計|{len(claims)}|

**第8節の表 `tab:timingbase` の非発散R行に分母混在がある。** Units=75ならDiag.±3は25/75=33.3…%（整数表示33%）。原稿の35%は全79単位の28/79=35.44…%を丸めたもの。Lは13/75=17.3…%、差の中央値−6.0と95%区間[−9.75,−1.0]は75単位で一致する。診断割合を33%へ揃えるか、診断79・基準線75と分母を別記する必要がある。元のaddon報告書とsummaryにも79と75の異なる母集団が並記されているため、summaryの35.4%との単純一致だけでは表の整合性を確認できない。原稿自体は変更していない。

要旨・結論の時期の記述は、算術と3非線形構造（residual GRU/GRU/LSTM）および基準線Lとの比較という範囲で出典と一致する。RNNはR保持1単位で、順位相関・L/M比較は定義できない。線形の対Lは有意差なしだが、対Mでは診断が良い。したがって線形に関する「基準線より良くない」は対Lに限定して読む必要がある。

第9節FP64の4 LSTM、180→185（非線形145）、変化5件、3040軌道の読み出し不変と、新評価集合の成功76・A138・成否変化2件・各p値はいずれも一致した。FP64のCSVには6モデルが載るが、1件はDだけの変化で、Aの変化は5件である。

## 3. 項目別の照合

|ID|原稿行|原稿の数値・主張|判定|照合値／正しい値と範囲|出典|注記|
|---|---|---|---|---|---|---|
'''
    for r in claims:
        vals=[r[k] for k in ['ID','line','claim','status','verified','source','note']]
        text+='| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in vals)+' |\n'
    text+='''
## 4. 読み取り監査とUTF-8

`v02_audit/audit.json` に原稿・出典のSHA256、全項目の件数、75単位の再集計値を保存した。照合前後で `paper/main.tex` のSHA256が一致し、引用した出典ファイルのSHA256も全て一致した。元のcheckpoint・評価データ・報告書は書き換えていない。配置のための旧原稿退避とSOURCES追記だけは指示どおり実施した。

報告書はUTF-8で保存し、その直後にUTF-8で読み直して全文一致・置換文字なしを確認した。確認結果は `v02_audit/utf8_check.json`。照合後は停止し、原稿修正・実験再実行は行わない。
'''
    p=W/'IronMath_ALL/ALL_NUMBERS_CHECK_v02.md';assert not p.exists(),'Do not overwrite an existing v02 report'
    p.write_text(text,encoding='utf-8');back=p.read_text(encoding='utf-8');assert back==text and '\ufffd' not in back and '???' not in back
    (OUT/'utf8_check.json').write_text(json.dumps(dict(SHA256=sha(p),readback_exact=True,replacement_character_absent=True,counts=dict(counts),total=len(claims)),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(counts=dict(counts),total=len(claims),main_unchanged=True),ensure_ascii=False))
if __name__=='__main__':run()
