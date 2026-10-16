"""Compile amended copy with provided bibliography and official JMLR style; retain warnings."""
from update_v072 import *
EXE=Path('tex_bin')
def main():
    commands=[('pdflatex.exe',['--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error','main_v0.7.2.tex']),('bibtex.exe',['--disable-installer','main_v0.7.2.aux']),('pdflatex.exe',['--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error','main_v0.7.2.tex']),('pdflatex.exe',['--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error','main_v0.7.2.tex'])]
    logs=O/'build_logs';logs.mkdir(exist_ok=True);runs=[]
    for i,(exe,args) in enumerate(commands):
        q=subprocess.run([str(EXE/exe),*args],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90);(logs/f'pass{i+1}.txt').write_bytes(q.stdout);runs.append(dict(pass_number=i+1,command=[exe,*args],returncode=q.returncode));print('Build pass',i+1,'exit',q.returncode,flush=True)
        if q.returncode:break
    dump('BUILD_RUNS.json',runs)
    if any(r['returncode'] for r in runs):raise RuntimeError('Compilation error recorded; do not alter manuscript claims to force compilation')
    p=O/'main_v0.7.2.log';s=p.read_text(encoding='utf-8',errors='replace');bib=(O/'main_v0.7.2.blg').read_text(encoding='utf-8',errors='replace');lines=s.splitlines();warning=[];patterns=[('overfull',r'Overfull \\[hv]box'),('underfull',r'Underfull \\[hv]box'),('warning',r'(?:LaTeX|Package [^\n]*?) Warning:'),('error',r'^!|LaTeX Error:|Emergency stop|Fatal error')]
    for i,line in enumerate(lines):
        for kind,pat in patterns:
            if re.search(pat,line):
                j=i+1
                while j<len(lines) and lines[j].strip() and j-i<10:j+=1
                message='\n'.join(lines[i:j]);location=re.search(r'(?:input line|at lines?|at line)\s+(\d+)(?:--(\d+))?',message);page=re.search(r'\bon page (\d+)',message);width=re.search(r'\(([\d.]+)pt too (?:wide|high)\)',message);warning.append(dict(category=kind,log_line=i+1,tex_line_start=location[1] if location else '',tex_line_end=location[2] if location and location[2] else '',page=page[1] if page else '',excess_points=width[1] if width else '',message=message));break
    for i,line in enumerate(bib.splitlines()):
        if line.startswith('Warning--'):warning.append(dict(category='bibtex_warning',log_line=i+1,message=line))
    table('BUILD_WARNINGS.csv',warning);counts={key:sum(r['category']==key for r in warning) for key in ['error','warning','overfull','underfull','bibtex_warning']};pages=int(re.search(r'Output written on main_v0.7.2.pdf \((\d+) pages',s)[1]);dump('BUILD_SUMMARY.json',dict(pages=pages,all_commands_exit0=True,**counts,unresolved_CHECKs=1,bibliography_exact_provided_file=sha(O/'refs_v0.7.2.bib')==sha(Path('inputs/refs_v0.7.2.bib')),official_style_SHA256=sha(O/'jmlr2e.sty'),stub_style=False))
    text='\n'.join(f"### {i+1}. {r['category']}\n```text\n{r['message']}\n```\n" for i,r in enumerate(warning))
    write('BUILD_REPORT.md',f'''# v0.7.2 組版結果

提供されたrefs_v0.7.2.bibを無変更で利用。JMLR公式案内がリンクする実物のjmlr2e.styでLaTeX→BibTeX→LaTeX→LaTeXを実施し、全終了コード0。スタブは使っていない。PDFは{pages}ページ。公式スタイルのSHAはSTYLE_PROVENANCE.json。元Downloadsの原稿・bibと過去成果物は不変。

- エラー：{counts['error']}
- LaTeX/Package警告：{counts['warning']}
- Overfull：{counts['overfull']}
- Underfull：{counts['underfull']}
- BibTeX警告：{counts['bibtex_warning']}

残るCHECKは典型状態ノルム1箇所（出典未掲載）であり、未確認値を埋めずに明示した。参考文献の%コメント移動・DOI underscore修正版をそのまま使用し、こちらで書誌内容を修正していない。本文の矛盾を修正していない。組版警告を消すための追加本文・表の修正も行っていない。全警告はBUILD_WARNINGS.csvと下記、全実行ログはbuild_logs/。終了コード0は投稿品質の合格とは別。

## 最終パスの警告一覧

{text}
''')
    preview=O/'previews';preview.mkdir(exist_ok=True)
    q=subprocess.run([str(EXE/'pdftoppm.exe'),'-r','65','-png',str(O/'main_v0.7.2.pdf'),str(preview/'page')],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120);(logs/'render.txt').write_bytes(q.stdout);assert q.returncode==0
    for name in ['figure1_arithmetic_margins','figure2_e3_initializations','figure3_g1_abl1']:
        q=subprocess.run([str(EXE/'pdftoppm.exe'),'-r','120','-png','-singlefile',str(O/f'figures/{name}.pdf'),str(O/f'figures/{name}_pdfproof')],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30);assert q.returncode==0
    from PIL import Image,ImageDraw
    pp=sorted(preview.glob('page-*.png'));assert len(pp)==pages
    for group in range(math.ceil(pages/12)):
        canvas=Image.new('RGB',(960,1020),'#dddddd');d=ImageDraw.Draw(canvas)
        for i,p in enumerate(pp[group*12:(group+1)*12]):
            with Image.open(p) as im:
                im.thumbnail((230,310));x=i%4*240+(240-im.width)//2;y=i//4*340+20;canvas.paste(im,(x,y));d.text((i%4*240+8,i//4*340+3),str(group*12+i+1),fill='black')
        canvas.save(preview/f'contact_{group+1}.png')
    print('Build and rendering complete:',counts,'pages',pages,flush=True)
if __name__=='__main__':
    import math
    main()
