"""Compare re-rendered figure PDFs with the PDFs used in the paper by rasterizing both with pdftoppm.

Usage: python compare_figures.py <reference figures dir> <re-rendered dir> <output dir> [pdftoppm executable]
Font hinting/antialiasing differs between matplotlib versions, so the comparison reports the share of pixels whose
grey value differs by more than 64 (of 255) and the bounding box of differences, and writes a difference image.
"""
import subprocess, sys, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops

ref, new, out = (Path(a).resolve() for a in sys.argv[1:4]); exe = sys.argv[4] if len(sys.argv) > 4 else 'pdftoppm'
out.mkdir(parents=True, exist_ok=True); res = {}
for name in ['figure1_arithmetic_margins', 'figure2_e3_initializations', 'figure3_g1_abl1']:
    imgs = []
    for tag, d in (('reference', ref), ('rerendered', new)):
        stem = out / f'{name}_{tag}'
        subprocess.run([exe, '-r', '100', '-png', '-singlefile', str(d / f'{name}.pdf'), str(stem)], check=True, capture_output=True)
        imgs.append(Image.open(f'{stem}.png').convert('L'))
    a, b = imgs
    same_size = a.size == b.size
    if not same_size: b = b.resize(a.size)
    d = np.abs(np.asarray(a, int) - np.asarray(b, int))
    ImageChops.difference(a, b).save(out / f'{name}_difference.png')
    res[name] = dict(size_reference=a.size, size_rerendered=imgs[1].size, same_size=same_size, max_abs_difference=int(d.max()),
                     share_pixels_diff_gt64=float((d > 64).mean()), diff_bbox=ImageChops.difference(a, b).getbbox())
json.dump(res, open(out / 'figure_comparison.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
