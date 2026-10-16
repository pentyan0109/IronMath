"""Final saved-file QA and UTF-8 handoff, no scientific computation."""
from update_v072 import *
def main():
    manifest=json.loads((O/'INPUT_SHA256.json').read_text(encoding='utf-8'));changes=[p for p,v in manifest.items() if not Path(p).is_relative_to(O) and sha(Path(p))!=v['SHA256']];assert not changes
    prior=json.loads((A/'SHA256_MANIFEST.json').read_text(encoding='utf-8'));prior_changes=[p for p,v in prior['files'].items() if sha(A/p)!=v['SHA256']];assert not prior_changes
    s=(O/'main_v0.7.2.tex').read_text(encoding='utf-8');original=Path('inputs/main_v0.7.2.tex').read_text(encoding='utf-8');edits=json.loads((O/'ALLOWED_EDITS.json').read_text(encoding='utf-8'))
    for e in edits:original=original.replace(e['old'],e['new'],1)
    assert original==s
    references=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s);expected=['figures/figure1_arithmetic_margins.pdf','figures/figure2_e3_initializations.pdf','figures/figure3_g1_abl1.pdf'];assert references==expected and all((O/p).is_file() for p in references)
    actual='\n'.join(l for l in s.splitlines() if not l.lstrip().startswith('%'));remaining=re.findall(r'\\CHECK\{([^{}]*)\}',actual);assert len(remaining)==1 and 'typical state norm' in remaining[0]
    tabledata=rows(O/'FIGURE3_SOURCE.csv');assert [int(r['hits']) for r in tabledata if r['panel']=='G1']==[140,91,84,119,143]
    build=json.loads((O/'BUILD_SUMMARY.json').read_text(encoding='utf-8'));assert build['all_commands_exit0'] and build['error']==0 and build['bibtex_warning']==0
    write('README_DELIVERY.md','''# v0.7.2 図・CHECK修正版

新規学習・モデル推論・Test生成/評価は0。Downloads原本と過去の監査成果物を保存し、新規出力先だけに変更を適用した。

## 更新

- 図1：x軸を指定文言へ。共通凡例1個・8組（ADD/SUBの合法文脈の和集合）、carry表記。SUBではborrowに対応する。上下段の間隔を短縮、右端512の切れも解消。元CSVのt規約と全36曲線は不変。
- 図2：E3初期化内訳。Unclassified(0)凡例を除去。実数値は不変。
- 図3：左はA140・D91・Constant84・B_cum128=119・B_cum512=143。全て同じ色、B_cum512の棒はAより高い。右はA1/A2だけ色を変え、元の160モデルの値を維持。
- main_v0.7.2.texのincludegraphicsはこの図番号順へ修正。refs_v0.7.2.bibを指定して無変更で使用。CHECK以外の本文・図キャプションは変更していない。
- CHECK11箇所のうち10箇所を完了。1箇所はEuclidean/545次元だけを確認し、典型状態ノルムが指定の保存文書・CSVにないため部分保留。距離の中央値を状態ノルムとして使っていない。CHECK_FILL_LOG.md/.csvに転記値と出典。

## 組版

JMLR公式案内が指定する実物jmlr2e.sty（STYLE_PROVENANCE.json）を使用。スタブなし。提供されたbibをそのまま使い、LaTeX/BibTeX/LaTeX/LaTeXが全て終了0。32ページPDF。エラー0・BibTeX警告0、一般警告3・Overfull10・Underfull8はBUILD_REPORT.mdとBUILD_WARNINGS.csvに全件記録し、本文修正で隠していない。典型状態ノルムのCHECKが残り、レイアウト警告もあるため投稿完成稿とはしない。

全32ページのコンタクトシート、図3枚のPDF再レンダリングを目視確認。図番号はPDF中でも1:算術、2:E3、3:G1/ABL1。図を作るための新しい実験結果は一切生成していない。公式スタイル取得のみネット接続し、新しい研究データ・モデルは取得していない。

## 再現

workspace rootから `.venv/Scripts/python.exe -B IronMath_PaperCheck/extension/v072_figures_check/update_v072.py`、次に `compile_check.py`。入力CSVと提供原稿・bibはINPUT_SHA256.json、図の転記CSVはFIGURE1/2/3_SOURCE.csv、許可されたTeX差分はALLOWED_EDITS.json。既存版を保持するため、再現する場合は別の新規出力先へコードをコピーすること。

FINAL_QA.jsonとSHA256_MANIFEST.jsonを保存して停止。未解決の典型状態ノルムを埋めるには、既存の集計済み出典が必要。今回の指示にない新規ノルム計測は行わない。
''')
    utf=[]
    for p in O.glob('*.md'):
        text=p.read_text(encoding='utf-8');assert '\ufffd' not in text;utf.append(dict(path=p.name,UTF8_readback=True,SHA256=sha(p)))
    dump('FINAL_QA.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_training=0,new_inference=0,new_Test_evaluation=0,original_input_changes=changes,prior_audit_files_checked=len(prior['files']),prior_audit_changes=prior_changes,allowed_TeX_edits_verified=True,figure_reference_order=references,common_legend_reviewed=True,G1_left_hits=[140,91,84,119,143],Bcum512_above_A=True,remaining_CHECKs=remaining,fully_filled_CHECKs=10,partially_filled_CHECKs=1,transferred_values_conflicting_with_corresponding_text=0,figure_PDFs_visually_reviewed=3,manuscript_pages_contact_sheet_reviewed=32,UTF8= utf,build=build,stopped=True))
    dump('SHA256_MANIFEST.json',dict(self_excluded=True,files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,SHA256=sha(p)) for p in O.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json'}))
    print('Final QA passed; original inputs and all previous141 audit files unchanged; stopped',flush=True)
if __name__=='__main__':main()
