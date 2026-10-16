"""Render existing PDFs only, for visual QA."""
from audit_v072 import *
def main():
    exe=Path('tex_bin/pdftoppm.exe');dest=O/'jmlr_build_comments_removed/previews';dest.mkdir(exist_ok=True)
    q=subprocess.run([str(exe),'-r','65','-png',str(dest.parent/'main.pdf'),str(dest/'page')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120);(dest/'render.log').write_bytes(q.stdout);assert q.returncode==0
    for name in ['figure1_arithmetic_margins','figure2_g1_abl1','figure3_e3_initializations']:
        q=subprocess.run([str(exe),'-r','110','-png','-singlefile',str(O/f'figures/{name}.pdf'),str(O/f'figures/{name}_pdfproof')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30);assert q.returncode==0
    from PIL import Image,ImageOps,ImageDraw
    pp=sorted(dest.glob('page-*.png'));assert len(pp)==27
    for group in range(3):
        canvas=Image.new('RGB',(4*240,3*340),'#dddddd');draw=ImageDraw.Draw(canvas)
        for i,p in enumerate(pp[group*12:(group+1)*12]):
            with Image.open(p) as im:
                im.thumbnail((230,310));x=(i%4)*240+(240-im.width)//2;y=(i//4)*340+20;canvas.paste(im,(x,y));draw.text(((i%4)*240+8,(i//4)*340+3),str(group*12+i+1),fill='black')
        canvas.save(dest/f'contact_{group+1}.png')
    dump('PDF_RENDER_CHECK.json',dict(rendered_pages=27,figure_PDFs_rendered=3,contact_sheets=3,pages_not_revised=True))
    print('27 manuscript pages and all3 figure PDFs rendered',flush=True)
if __name__=='__main__':main()
