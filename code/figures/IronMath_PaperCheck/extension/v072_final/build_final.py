"""JMLR compilation and saved-data-only final QA."""
from final_revision import *
import subprocess,math
EXE=Path('tex_bin')
def main():
    # Verify exact saved prefix bits, not merely the names r1/r2/r3.
    p=src(W/'v054t_transition/data/probes.jsonl');probes=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()];groups=defaultdict(list)
    for r in probes:
        n=r['hold_start'];groups[(r['op'],r['state'],r['pair'],r['prefix'],r['suffix'])].append((r['k'],n,int(r['a'],16)&((1<<n)-1),int(r['b'],16)&((1<<n)-1)))
    mismatches=[str(k) for k,v in groups.items() if len({tuple(a[1:]) for a in v})>1];assert not mismatches
    dump('SAVED_PREFIX_IDENTITY.json',dict(groups_checked=len(groups),mismatching_groups=mismatches,same_actual_pre_hold_bits_across_k=True,new_series_generated=0))
    report=O/'TIMING_CENSOR_RECHECK.md';write(report.name,report.read_text(encoding='utf-8')+'\n## 保存prefixビット列の追加確認\n\n元probes.jsonlの192個の(op,carry,pair,prefix名,suffix名)群について、hold_startおよびa/bのhold前ビット列をk間で照合。全群一致、異なる群0。従って今回の固定履歴比較は、prefix名だけでなく実際の保存入力prefixも同じである。系列生成・モデル実行はない。\n')
    logs=O/'build_logs';logs.mkdir(exist_ok=True);runs=[]
    for i,(exe,args) in enumerate([('pdflatex.exe',['--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error','main_v0.7.2.tex']),('bibtex.exe',['--disable-installer','main_v0.7.2.aux']),('pdflatex.exe',['--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error','main_v0.7.2.tex']),('pdflatex.exe',['--disable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error','main_v0.7.2.tex'])]):
        q=subprocess.run([str(EXE/exe),*args],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90);(logs/f'pass{i+1}.txt').write_bytes(q.stdout);runs.append(dict(command=[exe,*args],returncode=q.returncode));dump('BUILD_RUNS.json',runs);assert q.returncode==0,(exe,q.stdout[-1000:]);print('Build pass',i+1,'exit0',flush=True)
    log=(O/'main_v0.7.2.log').read_text(encoding='utf-8',errors='replace');bib=(O/'main_v0.7.2.blg').read_text(encoding='utf-8',errors='replace');lines=log.splitlines();warnings=[]
    for i,line in enumerate(lines):
        kind='overfull' if line.startswith('Overfull ') else 'underfull' if line.startswith('Underfull ') else 'warning' if re.search(r'(?:LaTeX|Package .*?) Warning:',line) else 'error' if re.search(r'^!|LaTeX Error:|Emergency stop|Fatal error',line) else None
        if kind:
            j=i+1
            while j<len(lines) and lines[j].strip() and j-i<10:j+=1
            message='\n'.join(lines[i:j]);loc=re.search(r'(?:input line|at lines?)\s+(\d+)(?:--(\d+))?',message);warnings.append(dict(category=kind,log_line=i+1,tex_line=loc[1] if loc else '',message=message))
    warnings.extend(dict(category='bibtex_warning',message=s) for s in bib.splitlines() if s.startswith('Warning--'));table('BUILD_WARNINGS.csv',warnings);counts={k:sum(r['category']==k for r in warnings) for k in ['error','warning','overfull','underfull','bibtex_warning']};undef=[r for r in warnings if re.search(r'undefined|multiply defined',r['message'],re.I)];s=(O/'main_v0.7.2.tex').read_text(encoding='utf-8');checks=s.count('\\CHECK');assert checks==0 and not undef and counts['error']==0;pages=int(re.search(r'Output written on main_v0.7.2.pdf \((\d+) pages',log)[1])
    assert sha(O/'jmlr2e.sty')==sha(P/'jmlr2e.sty') and sha(O/'refs_v0.7.2.bib')==sha(P/'refs_v0.7.2.bib')
    dump('BUILD_SUMMARY.json',dict(pages=pages,CHECK_remaining=checks,undefined_references=len(undef),**counts,all_commands_exit0=True,official_JMLR_style=True,stub=False))
    details='\n'.join(f"### {i+1}. {r['category']}\n```text\n{r['message']}\n```\n" for i,r in enumerate(warnings))
    write('BUILD_REPORT.md',f'''# v0.7.2 最終組版確認

CHECK残存0、未定義参照0、エラー0。LaTeX→BibTeX→LaTeX→LaTeXは全終了コード0、{pages}ページPDF生成。前段で公式配布とSHA一致を確認した本物のJMLRスタイルをそのまま使用。refs_v0.7.2.bibも提供版から無変更。旧原稿・旧集計を上書きせず、新しいv072_final/に保存。

一般警告{counts['warning']}、Overfull{counts['overfull']}、Underfull{counts['underfull']}、BibTeX警告{counts['bibtex_warning']}。警告を消すために指定外の本文・表・書誌の修正をしていない。詳細は下記とBUILD_WARNINGS.csv。コンパイルエラー0は、残るはみ出しを含めた投稿品質合格を意味しない。

## 最終パスの警告一覧

{details}
''')
    # Render all manuscript pages and revised Figure2; Figure1/3 are byte-identical copies.
    preview=O/'previews';preview.mkdir(exist_ok=True)
    q=subprocess.run([str(EXE/'pdftoppm.exe'),'-r','65','-png',str(O/'main_v0.7.2.pdf'),str(preview/'page')],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120);(logs/'render.txt').write_bytes(q.stdout);assert q.returncode==0
    q=subprocess.run([str(EXE/'pdftoppm.exe'),'-r','130','-png','-singlefile',str(O/'figures/figure2_e3_initializations.pdf'),str(O/'figures/figure2_pdfproof')],cwd=O,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30);assert q.returncode==0
    from PIL import Image,ImageDraw
    pp=sorted(preview.glob('page-*.png'));assert len(pp)==pages
    for group in range(math.ceil(pages/12)):
        im=Image.new('RGB',(960,1020),'#dddddd');d=ImageDraw.Draw(im)
        for i,p in enumerate(pp[group*12:(group+1)*12]):
            with Image.open(p) as a:
                a.thumbnail((230,310));x=i%4*240+(240-a.width)//2;y=i//4*340+20;im.paste(a,(x,y));d.text((i%4*240+8,i//4*340+3),str(group*12+i+1),fill='black')
        im.save(preview/f'contact_{group+1}.png')
    manifest=json.loads((O/'INPUT_SHA256.json').read_text(encoding='utf-8'));manifest.update(INPUTS);dump('INPUT_SHA256.json',manifest);print('Build complete',counts,'pages',pages,'CHECK0,undefined0,error0',flush=True)
if __name__=='__main__':main()
