"""Compile a diagnostic copy. Manuscript body remains byte-identical throughout."""
from audit_v072 import *
EXE=Path('tex_bin')
def main():
    original=O/'jmlr_build';dest=O/'jmlr_build_comments_removed';dest.mkdir(exist_ok=True)
    for name in ['main.tex','refs.bib','jmlr2e.sty','natbib.sty','booktabs.sty','plainnat.bst']:shutil.copyfile(original/name,dest/name)
    before=(dest/'refs.bib').read_bytes();lines=before.splitlines(keepends=True);removed=[i+1 for i,s in enumerate(lines) if s.lstrip().startswith(b'%')];after=b''.join(s for s in lines if not s.lstrip().startswith(b'%'));(dest/'refs.bib').write_bytes(after)
    assert sha(original/'main.tex')==sha(dest/'main.tex');assert b''.join(s for s in before.splitlines(keepends=True) if not s.lstrip().startswith(b'%'))==after
    dump('BIBTEX_COMPATIBILITY_CHANGE.json',dict(original_bib=sha(original/'refs.bib'),compile_only_bib=sha(dest/'refs.bib'),removed_full_comment_lines=removed,all_other_bytes_identical=True,manuscript_body_bytes_identical=True,not_a_v072_manuscript=True,reason='Classic BibTeX rejects full % comment lines inside entries; original run reports 15 missing-field-name errors'))
    commands=[('pdflatex.exe',['--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex']),('bibtex.exe',['--disable-installer','main']),('pdflatex.exe',['--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex']),('pdflatex.exe',['--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error','main.tex'])];runs=[]
    for i,(exe,args) in enumerate(commands):
        q=subprocess.run([str(EXE/exe),*args],cwd=dest,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90);(dest/f'pass{i+1}.txt').write_bytes(q.stdout);runs.append(dict(command=[exe,*args],returncode=q.returncode));assert q.returncode==0,(exe,q.stdout[-1000:])
    log=(dest/'main.log').read_text(encoding='utf-8',errors='replace');bib=(dest/'main.blg').read_text(encoding='utf-8',errors='replace');warn=[]
    for label,pattern in [('overfull',r'Overfull \\hbox.*?(?=\n\n|\n\[|$)'),('undefined',r'[^\n]*(?:undefined|multiply defined)[^\n]*'),('hyperref',r'Package hyperref Warning:.*?(?=\n\n|$)')]:
        for m in re.finditer(pattern,log,re.S if label!='undefined' else 0):warn.append(dict(category=label,text=m.group(0)))
    table('JMLR_WARNINGS.csv',warn);bst_warnings=[s for s in bib.splitlines() if s.startswith('Warning--')];pages=re.search(r'Output written on main.pdf \((\d+) pages',log)
    dump('BUILD_CHECK.json',dict(original_body_SHA256=sha(original/'main.tex'),compile_body_SHA256=sha(dest/'main.tex'),same_body=True,original_bibtex_exit=1,original_bibtex_errors=15,compatibility_only_remove_full_comments=True,runs=runs,pages=int(pages[1]) if pages else None,warnings=Counter(r['category'] for r in warn),bibtex_warnings=bst_warnings,clean_publication_layout=False))
    write('BUILD_REPORT.md',f'''# JMLRスタイル組版検査

原稿はv0.7.1（Downloads/main (8).tex）のbyte-identicalコピー。v0.7.2本文は作成・改稿していない。元原稿・元refs・既存jmlr2eは全て不変。

- MiKTeX-pdfTeX 4.18 / MiKTeX24.1を既存インストールの絶対パスで実行。前回PATH未検出だったTeXが見つかった。
- 不足したnatbib/booktabs/plainnatの公式CTANソースを今回のjmlr_build内だけに取得。システムのTeXパッケージは変更しない。--disable-installerで自動導入を止め、参照解決を実施。
- 原本と同一のrefs.bibではBibTeXが15件のmissing field nameを報告。原因はentry内部の%コメント行。原本は修正せず、jmlr_build/main.pdfはこの失敗を含む診断出力であり完成稿として使わない。
- 別のjmlr_build_comments_removed/に限り、refs.bibの全行コメントだけを除去した検査用コピーを作成。フィールド・値・順序・本文は同一、削除行番号とSHAはBIBTEX_COMPATIBILITY_CHANGE.json。この機械的な互換性回避を原本へ反映していない。
- 検査コピーのLaTeX/BibTeX/LaTeX/LaTeX終了コードは全て0。{int(pages[1]) if pages else '不明'}ページPDFを生成。BibTeX警告は{len(bst_warnings)}件（詳細JSON）。新しい文献調査や書誌内容の正しさの認定はしていない。
- 最終LaTeXログのoverfullは{sum(r['category']=='overfull' for r in warn)}件、undefined/multiply-definedのログ行は{sum(r['category']=='undefined' for r in warn)}件。JMLR_WARNINGS.csvに位置と幅を保存。はみ出しをCodexが本文修正で隠すことはしない。**コンパイル成功と、投稿用レイアウト合格は別。現版はレイアウト修正が必要。**

次版の改稿後にClaudeが本文・表・長いパス/URLの幅と参考文献を整え、同じJMLRで再組版する必要がある。今回提供するPDFはv0.7.1の検査用であり、v0.7.2完成稿ではない。
''')
    print('JMLR compatibility-copy compilation PASS; original BibTeX defect preserved; layout warnings retained',flush=True)
if __name__=='__main__':main()
