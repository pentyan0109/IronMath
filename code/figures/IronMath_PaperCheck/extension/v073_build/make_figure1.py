"""Figure 1 redraw from archived v072_audit/FIGURE1_SOURCE.csv (no model calls).
Identical to v072_figures_check/update_v072.py figures() Figure 1 except legend fontsize (8.4 -> LEGEND_FS) and the bottom margin it needs."""
import os, sys, csv, hashlib
from pathlib import Path
from collections import defaultdict
O = Path(__file__).resolve().parent; X = O.parent
os.environ['MPLCONFIGDIR'] = str(O / 'mpl_cache')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
LEGEND_FS = float(sys.argv[1]); OUT = Path(sys.argv[2]); BOTTOM = float(sys.argv[3]); LEG_Y = float(sys.argv[4]); XLAB_Y = float(sys.argv[5])
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.hashsalt': 'v073'})
src = X / 'v072_audit/FIGURE1_SOURCE.csv'
with src.open(encoding='utf-8-sig', newline='') as f: rr = list(csv.DictReader(f))
fig, axes = plt.subplots(2, 3, figsize=(10.1, 5.8), sharex=True, sharey=True); fig.subplots_adjust(left=.075, right=.975, top=.94, bottom=BOTTOM, hspace=.19, wspace=.12)
colors = {'00': '#2864a0', '01': '#dc7c24', '10': '#397d4c', '11': '#9e4d80'}
for oi, op in enumerate(['ADD', 'SUB']):
    for seed in range(3):
        ax = axes[oi, seed]; groups = defaultdict(list)
        for r in rr:
            if r['op'] == op and int(r['seed']) == seed: groups[(r['state'], r['pair'])].append(r)
        assert len(groups) == 6
        for (s, p), rs in sorted(groups.items()):
            rs.sort(key=lambda r: int(r['t'])); ax.plot([int(r['t']) for r in rs], [float(r['margin']) for r in rs], color=colors[p], ls='-' if s == '0' else '--', lw=1.3)
        ax.axhline(0, color='black', lw=.6); ax.set_xscale('symlog', base=2, linthresh=8); ax.set_xticks([0, 8, 32, 128, 512], ['0', '8', '32', '128', '512']); ax.set_xlim(0, 512); ax.grid(alpha=.15); ax.set_title(f'{op}, seed {seed}', fontsize=11)
        if seed == 0: ax.set_ylabel('Correct minus wrong logit', fontsize=9)
handles = [Line2D([0], [0], color=colors[p], ls='-' if s == 0 else '--', lw=1.4, label=f'(a,b)=({p[0]},{p[1]}), carry {s}') for s in [0, 1] for p in ['00', '01', '10', '11']]
leg = fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(.53, LEG_Y), frameon=False, fontsize=LEGEND_FS, columnspacing=1.2, handlelength=2.2)
fig.supxlabel('Iteration step t (symlog scale)', y=XLAB_Y, fontsize=10)
fig.canvas.draw(); bb = leg.get_window_extent(); W = fig.bbox.width
print('legend bbox px', round(bb.x0, 1), round(bb.x1, 1), 'fig width', W, 'inside', bb.x0 >= 0 and bb.x1 <= W, 'legend top(frac)', round(bb.y1 / fig.bbox.height, 4))
fig.savefig(OUT.with_suffix('.pdf')); fig.savefig(OUT.with_suffix('.png'), dpi=180); plt.close(fig)
print('source SHA256', hashlib.sha256(src.read_bytes()).hexdigest())
