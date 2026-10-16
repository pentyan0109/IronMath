"""UTF-8 report from new independent computations only."""
from common import *
import datetime
def mdtable(headers,data):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in data)+'\n'
def run():
    gs=load(R/'memory_group_statistics.json');ar=load(R/'arithmetic_summary.json');completion=load(R/'R4_COMPLETION.json');br=rows(R/'r1_baselines.csv');cr=rows(R/'r1_comparison.csv')
    text='''# IronMath 論文準備 第1段階：R1・R4

## 1. 目的と分析の位置づけ

保存済みファイルのみを用い、R4の独立再集計とR1の長めの短系列ベースラインを実施した。新しい学習・モデル推論はなく、元の集計スクリプトはimportも実行もしていない。新規実装はこのディレクトリ内に分離した。

R1は同じ評価で得た長さ32〜512の結果を使う事後の頑健性分析である。独立した新規評価集合による追試ではない。G1の事前登録結果と混同せず、P1・P2の判定は変更しない。R2・R3・S1・S2は実施していない。

実際の成功は2048・8192・65000の全カテゴリの全R位置正解で統一した。FF1-Rについては元の全7長さ定義も併記した。モデルを反復単位とし、成功を陽性とする混同行列を保存した。

## 2. R4：独立再集計

'''
    text+=f"報告書に対応する保存集計との照合は{completion['comparisons']}項目、不一致{completion['discrepancies']}項目。元集計に明示されていない追加の記述値が{completion['unreported_descriptive_values']}項目ある。精密値はr4_recomputation.csv、制約はr4_discrepancies.mdに記録した。\n\n"
    text+='V054Eの報告書の標本SD表示に4件の差がある（百分率単位、報告→再計算を2桁に丸めた値）：4096bit 1.63→1.64、12000bit 1.32→1.33、16384bit 0.89→0.90、32768bit 0.40→0.41。保存集計の精密SDは再計算と一致した。表示時の丸め・切り捨て処理の差が候補だが生成経路は未確定のため、元報告書を修正せず、両方を保存した。正答数・平均正答率・仮説判定には差がなかった。\n\n'
    text+='### G1・FF1・FF1-R\n\n'
    data=[]
    for key,s in gs.items():
        study,g,definition=key.split('/');data.append([study,g,definition,s['A']['N'],s['actual_success'],s['A']['hits'],s['B']['hits'],s['C_all_success']['hits'],s['C_all_failure']['hits'],s['D']['hits']])
    text+=mdtable(['実験','構造','真値','N','成功数','A','B','C全成功','C全失敗','D'],data)
    p=gs['G1/nonlinear/actual3']['paired_AD']
    text+=f"G1非線形合算のAのみ正解は{p['A_only_correct']}、Dのみ正解は{p['other_only_correct']}。McNemar両側正確p={p['McNemar_exact_two_sided_p']:.12g}、的中率差A−D={p['accuracy_difference']:.5f}、モデル単位bootstrap 95%区間[{p['CI_low']:.5f}, {p['CI_high']:.5f}]（10,000回、seed81000）。構造別の混同行列・対応比較も再現した。G1 P1・P2はともに支持のままである。\n\n"
    tau=load(R/'linear_tau_recomputed.json');text+=f"線形再帰のλから−1/log(λ)を再計算した。N/Rで同じλを重複計数せず、40モデル×64成分の{tau['components']}成分について、時定数の中央値{tau['median']:.15g}、最大{tau['maximum']:.15g}ステップを再現した。\n\n"
    text+='FF1の元の主仮説は支持。FF1には有限誤答型に関する事前登録H2はなく、FF1-Rと同じ規則を遡及適用した参考値は、有限誤答11件中Aの失敗予測9件、Dは0件である。FF1-Rは3長さ・7長さとも成功5件、H1・H2支持で変わらなかった（有限誤答6件中Aは3件、Dは0件）。\n\n'
    text+='### 算術\n\nV054Eの幅別完全一致率は次のとおり。3seed平均と標本SDを百分率で表示する。\n\n'
    text+=mdtable(['集合','幅','seed0','seed1','seed2','平均±標本SD (%)'],[[r['set'],r['width']]+[next(f"{x['correct']}/{x['N']}" for x in ar['V054E']['counts'] if x['set']==r['set'] and x['width']==r['width'] and x['seed']==s) for s in range(3)]+[f"{r['mean']*100:.5f} ± {r['sample_SD']*100:.5f}"] for r in ar['V054E']['means']])
    text+='V054RのR1は90/90、独立学習V055部分集合は54/54、D10のH4は77/78を再現した。D10の不一致セル（seed1、carry1、入力45）も保持した。ここでの不一致は固定点診断と耐久試験の不一致であり、今回の再集計と報告値の不一致ではない。\n\n'
    text+='V058は64bitで5/6、128bitで6/6の予測一致。V058bは両幅とも唯一のO0予測F2-S2が正しく、限定仮説支持を再現した。固定O1に対する診断選択の改善は64bitで11問、128bitで58問（6checkpoint×512問の和）。同率予測はO1を選ぶ元の規則で扱った。全判定行・同率の外れもv058_independent_judgment_table.csvとv058b_independent_judgment_table.csvに保存した。一般的な演算選択能力や因果関係を示す結果とは扱わない。\n\n'
    text+=mdtable(['D11仮説','差の平均','95%区間','判定','区間の位置づけ'],[[h,f"{s['mean_difference']:.12g}",f"[{s['CI95'][0]:.12g}, {s['CI95'][1]:.12g}]",s['judgment'],'事前登録' if h=='H3' else '今回追加の記述値'] for h,s in ar['D11'].items()])
    text+='D11の差は介入−対照。H1/H2は10セル中の個数差、H3は完全一致率の差（−10.5273ポイント）であり、単位を混同しない。H1/H2の件数と判定、H3の差と区間は元記録と一致した。\n\n'
    cov=rows(R/'d11_prediction_coverage.csv');missing=sum(int(x['prediction_missing']) for x in cov)
    text+=f"再検証の限界：V054R・D10・D11の耐久Probeは生の出力桁列がなく、保存された試行別exactフラグを再集計した。D11主評価10,240試行のうち{missing:,}試行はprediction=nullで、保存されている予測整数のみ独立採点した。したがって、算術の全項目について元の正誤フラグの正しさまで検証できた、とは主張しない。欠測は新しい推論で補っていない。\n\n"
    text+='## 3. R1：長めの短系列ベースライン\n\nB_cumは指定長以下の全評価長・全カテゴリに誤答がない場合だけ成功を予測する。B_onlyは指定長だけを使用する。次の値は的中数。全混同行列、Aのみ／Bのみ正解、両側McNemar、10,000回の対応bootstrap区間（seed82000）はCSVに保存した。\n\n'
    data=[]
    for study,g in [('FF1','all'),('FF1R','all')]+[('G1',g) for g in ['nonlinear','residual_gru','gru','lstm','rnn','linear']]:
        s=gs[f'{study}/{g}/actual3']
        for method in ['B_cum','B_only']:
            rr=[next(r for r in br if r['experiment']==study and r['structure']==g and r['method']==method and int(r['ell'])==ell) for ell in [32,64,128,512]]
            data.append([study,g,s['A']['N'],s['A']['hits'],method]+[r['hits'] for r in rr])
    text+=mdtable(['実験','構造','N','A','方式','32','64','128','512'],data)
    text+='B_cum(512)の主判定との一致は、FF1 36/40、FF1-R 34/40、G1非線形143/160、線形40/40であった。512は学習長8の64倍、学習長16の32倍である。短いという呼称だけで低コストな診断とみなさない。\n\n'
    text+='## 4. 固定規則に基づく128・512の結論\n\n'
    for r in load(R/'r1_conclusions.json'):
        text+=f"**{r['sentence']}。** A={r['A_hits']}/160、B_cum={r['baseline_hits']}/160、Aのみ正解={r['A_only_correct']}、Bのみ正解={r['other_only_correct']}、p={r['McNemar_exact_two_sided_p']:.12g}。的中率差A−B={r['accuracy_difference']:.5f}、95%区間[{r['CI_low']:.5f}, {r['CI_high']:.5f}]。\n\n"
    text+='「同程度」はR1_RULES.mdで指定された非有意差の表現であり、統計的同等性を証明した意味ではない。512でAの優位性は確認されず、点推定ではB_cumが3件多く的中した。この否定的な結果も保存した。G1のP1・P2支持は変更しない。\n\n'
    text+='## 5. 計算量の目安\n\n各長さはrandom32、long_N16、long_R16、rewrite2、mixed2の計68種類を実処理している。rewrite/mixedの各値8重複は独立試行に数えず、保存配列の行数で計算した。\n\n'
    text+=mdtable(['ℓ','B_onlyのstep/model','B_cumのstep/model','40モデルB_cum','160モデルB_cum'],[[ell,68*ell,68*sum(t for t in [32,64,128,512] if t<=ell),40*68*sum(t for t in [32,64,128,512] if t<=ell),160*68*sum(t for t in [32,64,128,512] if t<=ell)] for ell in [32,64,128,512]])
    text+='これは系列ステップ数の目安であり、実測CPU時間・診断Aの費用・構造間の1step当たり費用を表さない。今回S2は実施していない。\n\n'
    text+='## 6. 保存監査・再現\n\n'
    before=load(R/'WORKSPACE_BEFORE.json') if (R/'WORKSPACE_BEFORE.json').exists() else None
    after=load(R/'WORKSPACE_COMPARISON.json') if (R/'WORKSPACE_COMPARISON.json').exists() else None
    if before:text+=f"開始側SHA256 manifestは{len(before['files']):,}ファイル、{before['total_bytes']:,}バイト。読取不能等は{len(before['errors'])}件。\n\n"
    else:text+='開始側の全ワークスペースSHA256監査は処理中。最終確定前の文書である。\n\n'
    if after:text+=f"終了側との照合：元ファイル不変={after['original_files_unchanged']}、変更・欠落{len(after['changed_or_missing'])}件、既存場所に追加されたファイル{len(after['added_files_in_existing_locations'])}件（置き場所：{'、'.join(sorted({f.rsplit('/',1)[0] for f in after['added_files_in_existing_locations']}))}。いずれもIronMath_PaperCheck外で、本分析のスクリプトは書き込んでいない）。新規最上位ディレクトリは名前と作成時刻のみWORKSPACE_COMPARISON.jsonに記録し、作成元を断定していない。\n\n"
    else:text+='終了側manifestと比較結果は未確定。\n\n'
    text+=f"選択checkpointのSHA256は{completion['checkpoint_hashes']}件すべて保存SHAと一致した。学習済み重みはロードせず、ファイルのハッシュだけを計算した。\n\n"
    text+='開始時に別のPython監査プロセスを確認した。プロセス名・PID・コマンドはprocesses_before.json、終了時はprocesses_after.jsonに記録する。並行処理を停止・変更していない。manifestは逐次読取で、原子的なファイルシステムsnapshotではない。並行処理による変化があれば隠さず照合結果へ記録する。\n\n'
    text+=f"R1規則SHA256：`{load(R/'R1_RULES_SHA256.json')['SHA256']}`。規則の凍結が生出力再採点ファイル作成より前であることを品質検査で確認した。統計・復元・入出力境界の品質検査はquality_checks.jsonに保存した。\n\n"
    text+='''既存コードは定義・保存形式を確認するためにテキストとして読んだだけで、import/実行していない。再現コマンド（ワークスペースルート、生成先は本ディレクトリ内）：

```powershell
.venv/Scripts/python.exe -B IronMath_PaperCheck/recompute_memory.py
.venv/Scripts/python.exe -B IronMath_PaperCheck/recompute_arithmetic.py
.venv/Scripts/python.exe -B IronMath_PaperCheck/complete_r4.py
.venv/Scripts/python.exe -B IronMath_PaperCheck/recompute_r1.py
.venv/Scripts/python.exe -B IronMath_PaperCheck/quality_checks.py
.venv/Scripts/python.exe -B IronMath_PaperCheck/write_report.py
```

保存監査は`preservation.py before`と`preservation.py after`で行った。これらのmanifestは既存ファイルを上書きしない設計であり、再実行で今回の監査記録を置き換えないこと。再集計コマンドはPaperCheck内の生成物だけを再作成する。

報告書はUTF-8で書き、直後に同じ文字列へ復号できることと置換文字がないことを確認した。報告完了後に停止する。
'''
    p=R/'REPORT_R1_R4.md';p.write_text(text,encoding='utf-8');back=p.read_text(encoding='utf-8');assert back==text and '\ufffd' not in back and '???' not in back
    dump('REPORT_UTF8_CHECK.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),SHA256=sha(p),readback_exact=True,replacement_character_absent=True,complete_preservation_audit=after is not None))
if __name__=='__main__':run()
