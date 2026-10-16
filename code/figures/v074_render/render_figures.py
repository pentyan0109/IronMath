"""Re-render Figures 1-3 of the paper from the archived source CSVs (no model calls).

Usage: python render_figures.py <archive root> <output dir>
Figure code is copied from the original scripts (Figure 1: code/figures/IronMath_PaperCheck/extension/v073_build/make_figure1.py,
legend 10.92 pt; Figure 2: v072_final/final_revision.py figure2(); Figure 3: v072_figures_check/update_v072.py figures()).
Only input paths differ.
"""
import csv, os, sys
from collections import defaultdict
from pathlib import Path

A = Path(sys.argv[1]).resolve(); O = Path(sys.argv[2]).resolve(); O.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(O / 'mpl_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = A / 'results/IronMath_PaperCheck/extension'
def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})

# Figure 1 (legend fontsize 8.4 * 1.3)
rr = rows(R / 'v072_audit/FIGURE1_SOURCE.csv')
fig, axes = plt.subplots(2, 3, figsize=(10.1, 5.8), sharex=True, sharey=True); fig.subplots_adjust(left=.075, right=.975, top=.94, bottom=.25, hspace=.19, wspace=.12)
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
fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(.53, .004), frameon=False, fontsize=10.92, columnspacing=1.2, handlelength=2.2)
fig.supxlabel('Iteration step t (symlog scale)', y=.155, fontsize=10)
fig.savefig(O / 'figure1_arithmetic_margins.pdf'); plt.close(fig)

# Figure 2
cc = rows(R / 'v072_figures_check/FIGURE2_SOURCE.csv')
fig, ax = plt.subplots(figsize=(10.1, 3.9), layout='constrained'); x = np.arange(10)
for c, offset in [('0', -.19), ('1', .19)]:
    a = [next(r for r in cc if r['initialization'] == str(s) and r['data_condition'] == c) for s in range(100, 110)]; bottom = np.zeros(10)
    for field, color, label in [('shared', '#dc7c24', 'Shared endpoint'), ('distinct', '#2864a0', 'Distinct')]:
        v = np.array([int(r[field]) for r in a]); ax.bar(x + offset, v, .34, bottom=bottom, color=color, label=label if c == '0' else None, hatch='' if c == '0' else '///', edgecolor='white', lw=.6); bottom += v
    for xx in x + offset: ax.text(xx, 4.08, 'c' + c, ha='center', fontsize=8)
ax.set_xticks(x, [str(s) for s in range(100, 110)]); ax.set_xlabel('Initialization seed (paired streams c0 / c1)'); ax.set_ylabel('Input pairs per trained model'); ax.set_yticks([0, 1, 2, 3, 4]); ax.set_ylim(0, 4.7); ax.legend(ncol=2, loc='upper center', fontsize=9); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
fig.savefig(O / 'figure2_e3_initializations.pdf'); plt.close(fig)

# Figure 3
new = rows(R / 'v072_figures_check/FIGURE3_SOURCE.csv')
for r in new: r['hits'] = int(r['hits'])
fig, axes = plt.subplots(1, 2, figsize=(10.1, 3.8), sharey=True, layout='constrained')
for ax, panel in zip(axes, ['G1', 'ABL1']):
    rs = [r for r in new if r['panel'] == panel]; labels = [r['method'].replace('B_cum(128)', r'$B_{\mathrm{cum}}(128)$').replace('B_cum(512)', r'$B_{\mathrm{cum}}(512)$') for r in rs]
    bars = ax.bar(labels, [r['hits'] / 160 * 100 for r in rs], color=['#397d4c' if panel == 'ABL1' and r['method'] in ['A1', 'A2'] else '#7696ae' for r in rs], width=.65)
    for bar, r in zip(bars, rs): ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{r['hits']}/160", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 104); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_title(panel + ' (same 160 nonlinear models)', fontsize=11); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True); ax.tick_params(axis='x', labelsize=9)
axes[0].set_ylabel('Prediction accuracy (%)'); fig.savefig(O / 'figure3_g1_abl1.pdf'); plt.close(fig)
print('rendered', sorted(p.name for p in O.glob('*.pdf')))
