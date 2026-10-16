"""UTF-8 stage-2 report (rules freeze, R2, R3, S1, S2). Reads stage2 outputs only."""
import csv,json,hashlib,datetime
from pathlib import Path
S=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(x,n=5):
    x=float(x)
    if x!=0 and abs(x)<1e-3:return f'{x:.3g}'
    return str(int(x)) if x==int(x) else f'{x:.{n}f}'.rstrip('0').rstrip('.')
def sec(x):x=float(x);return f'{x:.3f}' if x>=.1 else f'{x:.4f}'
def mdtable(h,data):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in data)+'\n'
JA={'residual_gru':'残差GRU','gru':'通常のGRU','lstm':'LSTM','rnn':'Elman型RNN','linear':'線形','nonlinear':'非線形合算'}
ARCH=['residual_gru','gru','lstm','rnn'];CONDS=[f'{c}/{t}' for c in (500,2000,5000,20000) for t in ('1e-04','1e-05','1e-06')]
def say(r):
    p=float(r['McNemar_exact_two_sided_p']);a=int(r['A_hits']);b=int(r['other_hits'])
    if p>=.05:return '明確な差を検出できなかった'
    return 'A が多く当てた' if a>b else '相手が多く当てた'
def run():
    rules=load(S/'S1S2_RULES_SHA256.json');assert sha(S/'S1S2_RULES.md')==rules['SHA256']
    r2=rows(S/'r2_holm.csv');r3=rows(S/'r3_seed_cluster.csv');d0=load(S/'d0b_status.json');d1=load(S/'d0b_status_recheck.json')
    sm=load(S/'s1s2_summary.json');s1=rows(S/'s1_conditions.csv');cmp=rows(S/'s1_comparisons.csv');mis=rows(S/'s1_misclassification_counts.csv');det=rows(S/'s1_misclassification_detail.csv')
    lstm=rows(S/'s1_lstm_false_negative_dynamics.csv');tb=rows(S/'s2_time_breakdown.csv');var=rows(S/'s2_repeat_variability.csv');pts=rows(S/'s2_compute_matched_points.csv')
    before=load(S/'WORKSPACE_BEFORE.json');interim=load(S/'WORKSPACE_COMPARISON_AFTER_INTERIM.json')
    fc=S/'WORKSPACE_COMPARISON_AFTER.json';final=load(fc) if fc.exists() else None
    g=lambda grp,c:[r for r in s1 if r['group']==grp and r['condition']==c][0]
    t='''# IronMath 論文準備 第2段階：S1・S2・R2・R3

## 1. 位置づけ

本報告の分析はすべて事後分析である。G1 の事前登録結果（P1・P2 支持、20,000回・1e-6 の主結果）は変更しない。新しい学習は行っていない。S1 では G1 と同じ診断を、反復回数を変えて実際に実行した。S2 では、保存済みの評価集合で推論を実行し、時間を計った。元の checkpoint・凍結予測・評価データは読み取りだけで、変更していない。

'''
    t+=f"規則：指示書第3節・第4節をそのまま S1S2_RULES.md に書き写し、R2・R3・S1・S2 のどの集計よりも前に凍結した（SHA256 `{rules['SHA256']}`、{rules['utc'][:19]} UTC）。集計スクリプトは、毎回この SHA256 を照合してから実行する。\n\n"
    t+=f"D0b の完了確認：1回目（{d0['checked_local']}）は未完了だった（run_d0b.py が実行中で、metrics.json と報告書は未作成）。そのため、規則の凍結と R2・R3 のあと、いったん S1・S2 に入らずに中間の保存監査を行った。S1 を始める直前の2回目の確認（{d1['checked_local']}）では、RUN_COMPLETE.json が complete=true、報告書 D0b_報告.md が作成済みで、この確認自身を除く Python・Julia のプロセスはなかった。§5 の条件を満たしたので S1・S2 に進んだ。記録は d0b_status.json と d0b_status_recheck.json。\n\n"
    # R2
    t+='## 2. R2：Holm 法による多重比較の補正\n\n各群で、4構造（各40モデル）の McNemar 正確検定（両側）の p 値に Holm 法をかけた。「Aのみ正解／相手のみ正解」は、一方だけが主判定（2048・8192・65000）を当てたモデルの数。\n\n'
    t+=mdtable(['群','構造','A的中','相手的中','Aのみ正解','相手のみ正解','補正前p','Holm補正後p','補正後 p<0.05'],[[r['group'],JA[r['structure']],r['A_hits'],r['other_hits'],r['A_only_correct'],r['other_only_correct'],f(r['p_unadjusted'],6),f(r['p_holm'],6),'はい' if r['significant_holm_005']=='True' else 'いいえ'] for r in r2])
    q=lambda grp,a:[r for r in r2 if r['group'].startswith(grp) and r['structure']==a][0];l=q('3','lstm')
    t+=f"- 群1（A 対 D）：4構造すべてで、Holm 補正後も p < 0.05 で、A が D より多く当てた。\n- 群2（A 対 B_cum(128)）：補正後も p < 0.05 なのは残差GRU だけだった（補正後 p={f(q('2','residual_gru')['p_holm'],6)}、A が多く当てた）。通常のGRU・LSTM・Elman型RNN では、明確な差を検出できなかった。LSTM では、B_cum(128) が A より1件多く当てた（34 対 33）。\n- 群3（A 対 B_cum(512)）：補正後に p < 0.05 となる構造はなく、4構造とも明確な差を検出できなかった。ただし LSTM では、B_cum(512) だけが正解したモデルが {l['other_only_correct']}、A だけが正解したモデルが {l['A_only_correct']} で、補正前は p={f(l['p_unadjusted'],6)} と 0.05 未満だった（B_cum(512) のほうが正確な方向）。Holm 補正後は p={f(l['p_holm'],6)} だった。これは診断A に不利な結果である。\n\n群2・群3 の補正前 p 値は、第1段階 R1 の構造別 p 値と一致した（r2r3_record.json）。\n\n"
    # R3
    t+='## 3. R3：seed を単位にした区間\n\nG1 の非線形160モデル（seed 20〜39、各 seed に 4構造 × 2学習長 = 8モデル）で、seed を復元抽出するクラスターのブートストラップを行った（10,000回、numpy.random.default_rng(83000)）。比較のためのモデル単位の区間は、R1・R4 と同じ乱数 seed（A−D は 81000、A−B_cum は 82000）で計算し直したもので、第1段階の保存値と一致した。\n\n'
    t+=mdtable(['比較','的中率の差','seed 単位 95%区間','モデル単位 95%区間'],[[r['comparison'],f(r['accuracy_difference']),f"[{f(r['seed_cluster_CI_low'])}, {f(r['seed_cluster_CI_high'])}]",f"[{f(r['model_CI_low'])}, {f(r['model_CI_high'])}]"] for r in r3])
    t+='3つの比較のいずれでも、seed 単位の区間はモデル単位の区間より狭かった（seed 内の相関がない、または負のときに起こりうる。理由は検証していない）。区間が0を含むかどうかは、モデル単位と同じだった。A−D と A−B_cum(128) は0を含まず、A−B_cum(512) は0を含む（明確な差を検出できなかった）。seed ごとの的中数は r3_per_seed.csv。\n\n'
    # S1
    rp=sm['reproduction']
    t+=f"## 4. S1：診断の感度（事後の感度分析）\n\n### 4.1 実行と再現の確認\n\nG1 のモデル定義 models.py をそのまま複製し、SHA256 が事前登録の値（`abdc0364…`）と一致することを確認してから使った。200モデル（非線形160、線形40）のそれぞれについて、反復回数 500・2,000・5,000・20,000 ごとに診断（4前置き × 2値 × 2中立入力 = 16本、FP32、2スレッド）を実行した。各回は G1 と同じ順序（書き込みで出発点を作る → N 反復 → R 反復）で行った。収束の閾値（1e-4・1e-5・1e-6）は、同じ実行の結果に後から当てはめた（窓は最後の100ステップ）。読み出しとヤコビアンのスペクトル半径は、各反復回数の終点で評価した。\n\n**再現の確認：** 20,000回・1e-6 で、{rp['models']}モデルすべての A・D が凍結予測と一致した。終点の状態ベクトル、最後の100ステップの移動量、読み出し、スペクトル半径、最初に余裕が負になったステップも、G1 の保存記録とビット単位で一致した（s1_reproduction_check.json）。\n\n"
    t+='### 4.2 全12条件（G1 非線形160モデル）\n\n実際の成功は76モデル。「変化」は 20,000回・1e-6 からの A の的中数の変化。規則により、±5件以内を「頑健」と記述する。\n\n'
    t+=mdtable(['反復回数','閾値','A的中','A: TP/FP/TN/FN','D的中','変化','頑健'],[[r['iterations'],r['condition'].split('/')[1],r['A_hits'],f"{r['A_TP']}/{r['A_FP']}/{r['A_TN']}/{r['A_FN']}",r['D_hits'],('+' if int(r['A_hits_change_from_20000_1e6'])>0 else '')+r['A_hits_change_from_20000_1e6'],'頑健' if r['robust_within_5']=='True' else '頑健でない']+([] ) for r in s1 if r['group']=='nonlinear'])
    nr=[r['condition'] for r in s1 if r['group']=='nonlinear' and r['robust_within_5']!='True']
    t+=f"12条件のうち11条件が「頑健」だった。頑健でなかったのは {', '.join(c.replace('/1e-06','回・1e-6') for c in nr)}（−8件、140→132）である。閾値を 1e-4 か 1e-5 にゆるめると、2,000回以上ではどの条件でも 145（+5、頑健の上限ちょうど）になった。500回では、閾値を厳しくするほど的中数が下がった（142→138→132）。\n\n"
    t+='**構造別の A の的中数**（各40モデル。括弧内は D の的中数）：\n\n'
    t+=mdtable(['条件']+[JA[a] for a in ARCH]+['線形（記述）'],[[c]+[f"{g(a,c)['A_hits']}（{g(a,c)['D_hits']}）" for a in ARCH+['linear']] for c in CONDS])
    t+='Elman型RNN は12条件すべてで同じだった（37）。通常のGRU は 500回で下がった（1e-6 で 27）。LSTM は 20,000回・1e-6（33）が12条件の中で最も低く、2,000回以上で閾値を 1e-4・1e-5 にゆるめると 37 になった。線形40モデルは、A が全モデルで「失敗」と予測し、全モデルが実際に失敗したため、全条件で 40/40 だった（記述のみ）。\n\n'
    t+='### 4.3 各条件での比較（非線形160モデル）\n\nMcNemar 正確検定（両側）と、モデル単位のブートストラップ 95%区間（10,000回、numpy.random.default_rng(84000)）。差は A の的中率 − 相手の的中率。\n\n'
    t+=mdtable(['条件','比較','A的中','相手的中','Aのみ/相手のみ','p','差 [95%区間]','読み'],[[r['condition'],r['comparison'].replace('A vs ','A 対 '),r['A_hits'],r['other_hits'],f"{r['A_only_correct']}/{r['other_only_correct']}",f(r['McNemar_exact_two_sided_p'],4),f"{f(r['accuracy_difference'])} [{f(r['CI_low'])}, {f(r['CI_high'])}]",say(r)] for r in cmp if r['group']=='nonlinear'])
    t+='- A 対 D：12条件すべてで p < 0.05 で、A が多く当てた。\n- A 対 B_cum(128)：500回・1e-6（p=0.105、明確な差を検出できなかった）を除く11条件で、p < 0.05 で A が多く当てた。\n- A 対 B_cum(512)：12条件すべてで、明確な差を検出できなかった。点推定では、閾値 1e-6 の条件と 500回・1e-4 / 1e-5 の条件で B_cum(512) のほうが多く当て、2,000回以上・閾値 1e-4 / 1e-5 の条件では A が2件多かった。\n\n構造別の比較は s1_comparisons.csv に保存した。\n\n'
    b=sm['best_condition_exploratory']
    t+=f"**最良条件を選んだ比較（探索的）：** 12条件の中で A の的中数が最も多かったのは {b['A_hits']} 件で、2,000回以上・閾値 1e-4 / 1e-5 の6条件が並んだ。反復回数が最少の {b['condition'].replace('/1e-04','回・1e-4')} を代表として B_cum(512)（{b['B_cum512_hits']}）と比べると、Aのみ正解 {b['A_only_correct']}、B_cum(512)のみ正解 {b['other_only_correct']}、p={f(b['McNemar_exact_two_sided_p'],4)}、差 {f(b['accuracy_difference'])} [{f(b['CI_low'])}, {f(b['CI_high'])}] で、明確な差を検出できなかった。**この比較は、結果を見てから12条件の中で最良のものを選んでいるため、選択による楽観的な偏りがある。探索的結果であり、主結果には置き換えない。**\n\n"
    # 3.1
    t+='### 4.4 誤判定の種類\n\n分類規則：偽陰性（A は「失敗」、実際は成功）のうち、16本のうち1本でも非収束があれば「非収束」、すべて収束していれば「読み出しまたは安定性」（読み出しの誤り、またはスペクトル半径 ≥ 1）。偽陽性は、A が「成功」で実際は失敗。\n\n'
    t+=mdtable(['条件']+[f'{JA[a]} 偽陰性(非収束/読出・安定)/偽陽性' for a in ARCH],[[c]+[(lambda r:f"{r['FN_nonconvergence']}/{r['FN_readout_or_stability']}/{r['FP']}")([m for m in mis if m['condition']==c and m['structure']==a][0]) for a in ARCH] for c in CONDS])
    t+='線形は全条件で誤判定0件。読み出しまたは安定性による偽陰性は、すべて読み出しの誤りによるものだった。スペクトル半径 ≥ 1 による偽陰性はなかった（s1_misclassification_detail.csv）。Elman型RNN の偽陰性3件は全条件で同じモデルで、反復の長さや閾値に左右されない読み出しの誤りだった。\n\n'
    fp=[d for d in det if d['condition']=='20000/1e-06' and d['type']=='FP']
    t+='**20,000回・1e-6 での偽陽性**（失敗した評価カテゴリと、最も短い失敗長での最初の誤りの位置）：\n\n'
    t+=mdtable(['モデル','失敗した長さ','失敗カテゴリ','最初の失敗','最初の誤りの位置'],[[d['model'],', '.join(map(str,json.loads(d['failed_lengths']))),', '.join(json.loads(d['failed_categories'])),f"{d['first_failure_id']}",d['first_error_position']] for d in fp])
    t+='全条件の偽陽性は s1_misclassification_detail.csv に、同じ形式で保存した。\n\n'
    ref=sm['lstm_true_positive_reference'];main=[x for x in lstm if x['condition']=='20000/1e-06']
    t+='### 4.5 LSTM の偽陰性（記述のみ。原因は断定しない）\n\n20,000回・1e-6 での LSTM の偽陰性4モデルは、いずれも「非収束」に分類された。条件を満たさなかった7本の軌道の値：\n\n'
    t+=mdtable(['モデル','中立','前置き','値','最後100の最大移動量','ρ','読み出し','最後/最初の比','1ステップの減衰率','対数線形R²','‖c‖','max|c|','忘却ゲート平均'],[[x['model'],x['neutral'],x['prefix'],x['value'],f"{float(x['last100_max_movement']):.3e}",f(x['spectral_radius'],4),'正' if x['readout_correct']=='True' else '誤',f(x['ratio_last_over_first'],3),f"{float(x['per_step_geometric_factor']):.6f}",f(x['log_linear_R2'],2),f(x['c_L2'],2),f(x['c_max_abs'],2),f(x['forget_gate_mean'],3)] for x in main])
    t+=f"「最後/最初の比」と減衰率は、反復の最後の1,000ステップの移動量から求めた。比は最後の値と最初の値の比、減衰率は対数の直線当てはめの傾き。\n\n記述：7本とも、最後の100ステップの最大移動量は 1.01e-6〜1.11e-6 で、閾値 1e-6 をわずかに超えていた。スペクトル半径はすべて1未満（0.86〜0.97）、読み出しはすべて正しかった。移動量は、一定の割合で減ってはいなかった（1ステップの減衰率は約 1.0000、対数線形の当てはまり R² は約 0）。ほぼ一定の大きさのまま動き続けていた。‖c‖ は 6.84〜19.30 で、A が正しく「成功」と予測した LSTM の軌道（{ref['n_trajectories']}本）の中央値 {ref['c_L2_median']:.2f} と同程度か、それより大きかった。忘却ゲートの平均（0.45〜0.54）は、正しく予測した軌道の中央値 {f(ref['forget_mean_median'],3)} と大きな違いはなかった。正しく予測した軌道でも、移動量の最後/最初の比の中央値は {f(ref['ratio_median'],3)}（最後の100ステップの最大移動量の中央値 {ref['last100_median']:.2e}）で、減り続けてはいなかった。偽陰性の軌道は、この「動き続ける大きさ」が閾値を少し上回っていた点で違っていた。閾値を 1e-5 にゆるめると、2,000回以上ではこれら4モデルは正しく「成功」と予測された。全条件の値は s1_lstm_false_negative_dynamics.csv。\n\n"
    # S2
    t+='## 5. S2：計算時間の実測（事後の比較）\n\n### 5.1 計測の条件\n\n'
    t+='診断A の時間は、S1 の実行（反復回数ごとに1回、全200モデル）で計測した。長さ ℓ までのテストは、保存済みの評価集合（長さ 32・64・128・512、各68系列）で全200モデルの推論を実行して計測した。どちらも、計測中にほかの Python・Julia のプロセスがないことを、開始時・途中・終了時に確認した（s1_process_checks_main.jsonl、s2_process_checks.jsonl）。FP32、2スレッド。'
    t+=f"テストの推論の出力（予測bit列・非有限マスク）は、800件（200モデル × 4長さ）すべてで、G1 の保存予測とビット単位で一致した。成否も、第1段階の採点と一致した（{'一致' if sm['s2_btest_reproduces_saved'] else '不一致あり'}）。\n\n"
    t+='比較から除いた時間：checkpoint の読み込みと torch.jit.script の変換（両方式に共通。script はおおむね 0.02秒/回）、評価集合のディスクからの読み込み。テストの時間には、正解の計算、推論、R 位置の採点を含む。診断A の時間には、準備・反復・読み出し・ヤコビアンと固有値を含む。ヤコビアンは、各反復回数の終点のうち値が有限の16本すべてについて計算した（G1 の手順では収束した軌道だけで計算するため、この時間は上限の見積もりになる）。\n\n'
    t+='### 5.2 時間の内訳（1モデルあたりの平均、秒）\n\n'
    t+=mdtable(['群','反復回数','準備','反復','読み出し','ヤコビアン・固有値','合計','合計の最小〜最大'],[[JA[r['group']],r['iterations'],sec(r['prep_mean']),sec(r['iterate_mean']),sec(r['readout_mean']),sec(r['jacobian_eig_mean']),sec(r['total_excluding_script_mean']),f"{sec(r['total_min'])}〜{sec(r['total_max'])}"] for r in tb if r['method']=='A'])
    t+=mdtable(['群','テスト','処理した系列ステップ/モデル','合計（平均）','最小〜最大'],[[JA[r['group']],f"B_cum({r['ell']})",r['sequence_steps_per_model'],sec(r['total_excluding_script_mean']),f"{sec(r['total_min'])}〜{sec(r['total_max'])}"] for r in tb if r['method']=='B_cum'])
    nlA=[r for r in tb if r['group']=='nonlinear' and r['method']=='A'];nlB=[r for r in tb if r['group']=='nonlinear' and r['method']=='B_cum']
    t+=f"非線形合算で、最も安い診断A（500回、{sec(nlA[0]['total_excluding_script_mean'])}秒）は、B_cum(512)（{sec(nlB[3]['total_excluding_script_mean'])}秒）の約{float(nlA[0]['total_excluding_script_mean'])/float(nlB[3]['total_excluding_script_mean']):.0f}倍だった。事前登録の20,000回（{sec(nlA[3]['total_excluding_script_mean'])}秒）は、B_cum(512) の約{float(nlA[3]['total_excluding_script_mean'])/float(nlB[3]['total_excluding_script_mean']):.0f}倍だった。診断A の時間の大部分は反復で、500回ではヤコビアンと固有値の計算（LSTM は128次元のため約0.78秒）も大きな割合を占めた。B_cum の最大値のうち、長さ32の 0.118秒は最初に処理したモデル（gru_L16_s20）のもので、プロセスで最初の推論の立ち上がり時間を含む。\n\n"
    t+='### 5.3 時間のばらつき（各構造2モデル × 3回）\n\n各構造の L8_s20 と L16_s20 について、全計測を3回繰り返した。表は「（最大−最小）/中央値」の、その方式でのモデル間の最大値。\n\n'
    vv=[]
    for meth,key in [('A','iterations'),('B_cum','ell')]:
        for k in sorted({r[key] for r in var if r['method']==meth},key=int):
            sel=[r for r in var if r['method']==meth and r[key]==k];vv.append([meth if meth=='A' else 'B_cum',f'{k}回' if meth=='A' else f'ℓ={k}',sec(min(float(r['median']) for r in sel))+'〜'+sec(max(float(r['median']) for r in sel)),f"{max(float(r['range_over_median']) for r in sel):.2f}"])
    t+=mdtable(['方式','設定','中央値（10モデルの範囲、秒）','（最大−最小）/中央値の最大'],vv)
    amax=max(float(r['range_over_median']) for r in var if r['method']=='A');bmax=max(float(r['range_over_median']) for r in var if r['method']=='B_cum')
    amed=sorted(float(r['range_over_median']) for r in var if r['method']=='A')[len([r for r in var if r['method']=='A'])//2]
    t+=f"（最大−最小）/中央値 は、診断A で最大 {amax:.2f}（モデル・設定の中央値 {amed:.2f}）、B_cum で最大 {bmax:.2f} だった。幅が大きい例の多くは、3回のうち1回だけが遅かったものである。B_cum はもともとの時間が数ミリ秒〜0.1秒と短く、相対的なばらつきが大きくなりやすい。方式間の時間差（非線形合算の平均で約{float(nlA[0]['total_excluding_script_mean'])/float(nlB[3]['total_excluding_script_mean']):.0f}〜{float(nlA[3]['total_excluding_script_mean'])/float(nlB[3]['total_excluding_script_mean']):.0f}倍）は、このばらつきより十分に大きかった。モデルごとの値は s2_repeat_variability.csv。\n\n"
    t+='### 5.4 計算量をそろえた比較\n\n![非線形160モデル](s2_compute_matched_nonlinear.png)\n\n![構造別](s2_compute_matched_by_structure.png)\n\n'
    P=[p for p in pts if p['group']=='nonlinear']
    t+=mdtable(['方式・条件','1モデルあたりの平均時間（秒）','的中数（/160）'],[[p['condition'] if p['method']=='B_cum' else '診断A '+p['condition'],sec(p['mean_seconds_per_model']),p['hits']] for p in sorted(P,key=lambda p:float(p['mean_seconds_per_model']))])
    t+='構造別の値は s2_compute_matched_points.csv。\n\n**固定した表現による結論：**\n\n'
    v=sm['verdict']
    for grp in ['nonlinear']+ARCH:
        w=v[grp]['A_wins_exploratory']
        if w:
            names=', '.join(x['condition'] for x in w);bb=w[0]
            t+=f"- {JA[grp]}：ある計算時間以下で、診断A のいずれかの条件が、同じ計算時間以下の B_cum(ℓ) すべてより多く当てた。該当した条件は {names}（A {max(x['A_hits'] for x in w)}件。同じ時間以下で最も多く当てたテストは {bb['best_B_within']} の {bb['best_B_hits']}件）。**探索的結果である。**\n"
        else:t+=f"- {JA[grp]}：同じ計算時間の範囲では、診断A がテストより正確であることは示されなかった。\n"
    t+=f"\n補足（必ず併せて読むこと）：診断A のどの条件も、B_cum(512) を含むすべてのテストより時間がかかった。そのため上の比較は、すべてのテストとの比較になっている。非線形合算で該当した条件は、いずれも B_cum(512) より2件多いだけである。McNemar の p は {f(b['McNemar_exact_two_sided_p'],3)} で、明確な差を検出できなかった。一方、計算時間は B_cum(512) の約{float(nlA[1]['total_excluding_script_mean'])/float(nlB[3]['total_excluding_script_mean']):.0f}〜{float(nlA[3]['total_excluding_script_mean'])/float(nlB[3]['total_excluding_script_mean']):.0f}倍だった。該当した条件は、12条件の中から事後に見つけたものである。事前登録の条件（20,000回・1e-6、140件）は B_cum(512)（143件）より少なかった。LSTM と通常のGRU では、該当する条件はなかった。この比較は、評価集合（長さ32〜512）がすでに作られていることを前提にした実測であり、ほかの評価集合やハードウェアでの時間を示すものではない。\n\n"
    # audit
    t+=f"## 6. 保存監査・再現\n\n開始側の SHA256 manifest（IronMath_PaperCheck/ を除く）は {len(before['files']):,} ファイル、{before['total_bytes']:,} バイト、読取不能等は {len(before['errors'])} 件。\n\n"
    def cmpline(c,label):
        locs=sorted({p.rsplit('/',1)[0] for p in c['added_files_in_existing_locations']})
        return f"{label}：元ファイル不変={c['original_files_unchanged']}、変更・欠落 {len(c['changed_or_missing'])} 件"+(f"（{', '.join(c['changed_or_missing'])}）" if c['changed_or_missing'] else '')+f"、既存の場所に追加されたファイル {len(c['added_files_in_existing_locations'])} 件"+(f"（置き場所：{'、'.join(locs)}）" if locs else '')+f"、新しい最上位ディレクトリ {len(c['new_top_level_directories_name_and_creation_only'])} 件。\n\n"
    t+=cmpline(interim,'中間（R2・R3 のあと、D0b の実行中）')
    t+='中間で変更・追加があったのは、すべて IronMath_Discovery/D0b の中だった。並行して動いていた D0b の実行が、自分の途中ログに追記したものと、完了時に書き出したものである。本分析は D0b に書き込んでいない。\n\n'
    t+=cmpline(final,'最終（S1・S2 のあと）') if final else '最終の終了側 manifest は未確定。\n\n'
    if final:
        ai=load(S/'WORKSPACE_AFTER_INTERIM.json')['files'];af=load(S/'WORKSPACE_AFTER.json')['files']
        ch=[p for p in ai if p not in af or ai[p]['SHA256']!=af[p]['SHA256']];ad=[p for p in af if p not in ai]
        t+=f"中間と最終の manifest を直接比べると、変更・欠落は {len(ch)} 件、追加は {len(ad)} 件（{'、'.join(ad)}）だった。追加された D0b_報告.md の更新時刻は 11:31 で、S1 の開始（11:38）より前である。中間の監査（11:13〜11:34）が D0b のフォルダを読み終えたあとに作られたため、中間の manifest に入っていなかった。S1・S2 の実行中に変わった元ファイルはなかった。\n\n"
    t+='本分析の書き込みは、すべて IronMath_PaperCheck/stage2/ の中で行った。manifest はファイルを順に読み取って作るもので、ある瞬間の状態を一括で写し取ったもの（原子的なスナップショット）ではない。追加・変更の作成元は推測で断定しない。プロセスの記録は processes_before.json、processes_after_interim.json、processes_after.json にある。\n\n'
    t+='再現コマンド（ワークスペースのルートで実行。生成物は stage2/ に作られる）：\n\n```powershell\n.venv/Scripts/python.exe -B IronMath_PaperCheck/stage2/r2r3.py\n.venv/Scripts/python.exe -B IronMath_PaperCheck/stage2/s1_run.py main      # 約80分。既存の s1_records は飛ばす\n.venv/Scripts/python.exe -B IronMath_PaperCheck/stage2/s1_repro.py\n.venv/Scripts/python.exe -B IronMath_PaperCheck/stage2/s2_timing.py        # 計測中は他の高負荷処理を止める\n.venv/Scripts/python.exe -B IronMath_PaperCheck/stage2/s1s2_aggregate.py\n.venv/Scripts/python.exe -B IronMath_PaperCheck/stage2/write_report2.py\n```\n\n'
    t+='S1 の数値（終点・移動量・ρ・読み出し）は決定的で、G1 とビット単位で一致した。時間の値は実行ごとに変わる（§5.3）。保存監査は preservation2.py（before／after_interim／after）で行った。どの段階の記録も上書きしない作りなので、再実行するときは監査記録を別の場所に退避すること。S1 の最初の2モデル（gru_L16_s21、lstm_L8_s20）は、本番と同じコードによる試行の実行で記録したもので、そのまま使った（s1_process_checks_pilot.jsonl）。\n\n報告書は UTF-8 で書き、読み直して同じ文字列であること、置換文字がないことを確認した。報告の作成をもって停止する。推論のみの再検証は、別の指示書で行う。\n'
    p=S/'REPORT_STAGE2.md';p.write_text(t,encoding='utf-8');back=p.read_text(encoding='utf-8');assert back==t and '�' not in back and '???' not in back
    (S/'REPORT_UTF8_CHECK.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),SHA256=sha(p),readback_exact=True,replacement_character_absent=True,final_preservation_audit=final is not None),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':run()
