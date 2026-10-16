"""Finalize read-only audit handoff and hashes. Does not run any model."""
from audit_v072 import *
def main():
    manifest=json.loads((O/'INPUT_SHA256.json').read_text(encoding='utf-8'));manifest={p:v for p,v in manifest.items() if not Path(p).is_relative_to(O)}
    # Save original missing-donor rows, not just measured controls.
    old=rows(X/'E2_causal_intervention/E2_RESULTS.csv');na=[r for r in old if r['status']!='measured'];assert len(na)==78;table('E2_UNAVAILABLE_DONORS.csv',na)
    # Annotate actual fixed threshold on the old post-hoc pairs, without calling it old preregistration.
    previous=rows('v054h_dynamics/posthoc_fixed_point_pairs.csv');table('V054H_OLD_ENDPOINT_PAIRS.csv',[dict(**r,shared_under_E3_distance_threshold=float(r['distance_hstar_state0_vs_state1'])<.001,comparison_only_not_old_preregistered_shared_rule=True) for r in previous])
    manifest.update({p:v for p,v in SOURCES.items() if not Path(p).is_relative_to(O)})
    oldd2=rows(O/'V054H_D2_INDEPENDENT.csv');assert len(oldd2)==36 and all(b(r['matching_D2']) for r in oldd2)
    # Build warnings and inspection are handed off, not silently corrected in the manuscript.
    p=O/'V072_REVISION_PLAN.md';s=p.read_text(encoding='utf-8');s+='''
## 組版検査で判明した追加事項

本文不変のJMLR検査コピーで27ページを生成。元refs.bibにはentry内の全行%コメントがあり、classic BibTeXで15件の構文エラー。原本は不変のまま、別検査コピーでコメント行だけを除去するとBibTeXを含め全工程が終了0（フィールド値は変更なし）。既存本文のoverfull10件とhyperref3件が残るため、投稿用レイアウト合格とはしない。BUILD_REPORT.mdとJMLR_WARNINGS.csvを参照し、Claudeが次版で参考文献コメントと幅を修正する。Codexは本文を改稿していない。

元V054H D2の36/36も保存された整数予測から独立に再採点して確認（V054H_D2_INDEPENDENT.csv）。旧posthoc12pairの距離とE3閾値による参考分類はV054H_OLD_ENDPOINT_PAIRS.csv。旧D2の事前登録と旧共有距離の事後解釈は混同しない。
''';write(p.name,s)
    write('AUDIT_SUMMARY.md','''# IronMath v0.7.2 追加監査・Claude向け引継ぎ

新モデル学習0・新モデル推論0・新Test生成0。保存マージン・整数予測・終点tensor・checkpointメタデータから再集計し、旧集計コードのimport/実行なしで検算した。元結果、凍結規則、checkpoint、v0.7.1本文は上書きしていない。本文改稿は指定Dに従いClaudeへ残す。

## Q1〜Q5

1. **−34.3は観測下限。** at8の失敗状態移植と負法線2条件は各10/10で最初の評価時点に誤答。無介入の平均初回誤答42.3に対して8−42.3=−34.3で、潜在誤答開始の連続的な強度比較はできない。回復は順に0/10、6/10（correctノルム）、1/10（failedノルム）と異なる。
2. **全方向を含めても法線効果は記述的に際立つ。** at8・10失敗文脈、正法線+7.1/+10.8に対し、ランダム3方向の平均は−0.167/−3.767、直交は−0.2/−3.5（correct/failedノルム）。全方向の範囲・文脈内と文脈間の変動をCSVに提供した。負法線は下限で飽和しており、幾何固有の因果性や統計的な一般優越を主張しない。
3. **別終点34組は破綻0、最大1024まで破綻なし34、評価不能0。** 共有候補46/80=57.5%、その46組はすべて片側で窓内破綻。別終点の68側はすべて終点正読出し。無限長で成功したとは呼ばない。
4. **元D2の側単位対応は36/36から240/240へ再現。** E3の正読出し194側は窓内誤答なし、誤読出し46側は破綻。両streamで少なくとも1厳密共有過渡pairを持つのは9/10初期化。36側・80pair・10初期化は異なる単位。旧k最大4096対E3最大1024、同じ構造・Train母集団という範囲に限定する。
5. **強められるのは、全対照を含む状態方向への感度と、同構造の条件付き終点読出し対応の再現。** 共有アトラクタの数学的同一性・幾何固有の因果性・永久修復・厳密な履歴非依存性・他構造への一般性は主張しない。固定履歴でkを変えた観測時刻は一致したが、前置き間の範囲は旧最大19、E3最大15step。ABL1のA0 140/160対A1 145/160という否定的結果を保持する。

## 提出物

- E2_FLOOR_AUDIT.md / .csv：10失敗文脈の全条件・時刻、左/右打切り、直後margin、回復。
- E2_ALL_CONTROLS.md / .csv：方向別・全36/失敗10の集計。TRIALS、DIRECTION_WITHIN_CONTEXT、DIRECTION_GROUPED、UNAVAILABLE_DONORSのCSVで全行追跡。
- E3_ENDPOINT_OUTCOME_AUDIT.md、E3_ENDPOINT_OUTCOME.csv、E3_ENDPOINT_SIDE_OUTCOME.csv、E3_READOUT_D2_CROSSTAB.csv：全80pair・240側、旧D2互換の単位。
- E3_INITIALIZATION_AUDIT.md / .csv、E3_CHECKPOINT_SCHEMA_AUDIT.csv：初期SHAと保存checkpointのキー・shape・dtype。
- E3_ENDPOINT_DISTANCE_AUDIT.md / .csv：全距離と閾値近傍。閾値変更なし、近傍0件。
- TIMING_HISTORY_INDEPENDENCE.md / .csv、OBSERVATIONSとSUMMARY：既存履歴とk間の時刻変動、打切り。
- V072_REVISION_PLAN.md、V061_CONFIRMED_TIMING_TEXT.txt：Claude用の修正計画・確認済み旧規則。本文変更なし。
- figures/：実測PDF3枚、PNG、PDF再レンダリング。FIGURES.mdとFIGURE1/2/3_SOURCE.csvで元値を追跡。
- BUILD_REPORT.md、BUILD_CHECK.json、JMLR_WARNINGS.csv：JMLR検査。元BibTeXは15エラー。コメント除去だけの別コピーでは27ページ生成・全終了コード0、overfull10件が残り投稿用合格ではない。
- INDEPENDENT_AUDIT.json、INDEPENDENT_CHECKS.csv、INPUT_SHA256.json、FINAL_QA.json、SHA256_MANIFEST.json。

## 再現と順序

PowerShell、workspace rootで `.venv/Scripts/python.exe -B IronMath_PaperCheck/extension/v072_audit/audit_v072.py` → `verify_v072.py`。モデルコードはimportしない。checkpoint読取のTorchは1thread、モデル構築もforwardもしない。prepare_delivery.pyは監査通過を要求してから修正計画・図を作る（最終版には組版結果を追記済み）。build_check.pyは既存MiKTeXで検査コピーのみを組版。render_check.pyは生成済みPDFを描画する。初回の再現は別の新規出力先へ各スクリプトをコピーし、元の本監査成果物も保存すること。

数値監査完了→独立検算→修正計画/図→JMLR組版の順。時刻は各ステータスJSONとファイルmtimeを保存。今回の修正は監査用の表示・注釈・図レイアウト・BibTeX互換性だけで、実験条件や分類閾値を変えていない。新しい引用調査はしない。TeX依存の公式CTAN取得は研究データ取得とは別で、ソースとSHAを保存した。

入力SHAは実際に参照したファイルを対象とし、全ワークスペースのSHA監査ではない。コンパイル互換性コピー以外の本文・既存集計は読み取り専用。報告書はUTF-8保存後に読み直す。図PDFの全3ページと検査原稿27ページをレンダリングし、図3枚と原稿全ページのコンタクトシートを目視した。本文のはみ出し・書誌不備は引継ぎ、図には曲線を覆う凡例や切れたラベルを残さない。

指定の成果物を保存して停止。追加実験、学習、本文改稿は行わない。
''')
    # Mark visual findings explicitly instead of certifying the unmodified paper's layout.
    dump('VISUAL_QA.json',dict(figure_PDFs_reviewed=3,figure_layout_passed=True,manuscript_pages_contact_sheet_reviewed=27,manuscript_layout_passed=False,original_manuscript_unchanged=True,remaining_findings='10 overfull hboxes, 3 hyperref PDF-string warnings; original BibTeX comment syntax errors',new_models=0))
    deps=[]
    for name,base in [('natbib.dtx','natbib'),('natbib.ins','natbib'),('plainnat.bst','natbib'),('booktabs.dtx','booktabs'),('booktabs.ins','booktabs')]:
        p=O/'jmlr_build'/name;deps.append(dict(file=str(p.relative_to(O)),official_source=f'https://mirrors.ctan.org/macros/latex/contrib/{base}/{name}',bytes=p.stat().st_size,SHA256=sha(p),purpose='compile dependency only, no research data'))
    dump('TEX_DEPENDENCY_SOURCES.json',deps)
    # Recheck figure2 recorded numbers against archived summary without executing it.
    source_fig=rows(O/'FIGURE2_SOURCE.csv');old_a=rows('IronMath_PaperCheck/addon/abl1/abl1_architecture_summary.csv');old_g=rows('IronMath_G1/g1_architecture_comparison.csv')
    for r in source_fig:
        oldrow=next(z for z in old_a if z['group']=='nonlinear160' and z['diagnostic']==r['method']) if r['study']=='ABL1' else next(z for z in old_g if z['group']=='nonlinear_pooled' and z['method']==r['method']);assert r['hits']==oldrow['hits']
    manifest.update({p:v for p,v in SOURCES.items() if not Path(p).is_relative_to(O)});changed=[p for p,v in manifest.items() if sha(Path(p))!=v['sha256']];assert not changed
    dump('INPUT_SHA256.json',manifest);utf=[]
    for p in O.rglob('*.md'):
        text=p.read_text(encoding='utf-8');assert '\ufffd' not in text;utf.append(dict(path=str(p.relative_to(O)),SHA256=sha(p),UTF8_readback=True))
    dump('FINAL_QA.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),numerical_audit_passed=js(O/'INDEPENDENT_AUDIT.json')['passed'],original_inputs_checked=len(manifest),original_input_changes=changed,original_v071_body_unchanged=sha(Path('inputs/main (8).tex'))==sha(O/'jmlr_build/main.tex'),new_training=0,new_inference=0,new_Test_generation=0,UTF8_reports=utf,figures=3,body_revision_performed=False,stopped=True))
    dump('SHA256_MANIFEST.json',dict(self_excluded=True,files={str(p.relative_to(O)):{'bytes':p.stat().st_size,'SHA256':sha(p)} for p in O.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json'}))
    print('V072 AUDIT COMPLETE AND STOPPED; original input SHA unchanged; no manuscript body revision',flush=True)
if __name__=='__main__':main()
