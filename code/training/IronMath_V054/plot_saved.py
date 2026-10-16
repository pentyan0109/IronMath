"""Artifact-only redraw: full sample-SD bars; no training or model inference."""
import csv,json,hashlib,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
p=R/'length_generalization.csv';before=sha(p)
with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
lookup={(r['condition'],r['model'],int(r['width']),r['op']):r for r in rows}
widths=[8,16,32,64,128,256,512,1024,2048]
low=min(100*(float(r['exact_mean'])-float(r['exact_sample_SD'])) for r in rows)
high=max(100*(float(r['exact_mean'])+float(r['exact_sample_SD'])) for r in rows)
limits=(math.floor((min(0,low)-5)/5)*5,math.ceil((max(100,high)+5)/5)*5)
fig,axs=plt.subplots(2,2,figsize=(11,7.5),sharex=True,sharey=True)
for i,cond in enumerate(('C1','C2')):
    for j,op in enumerate(('ADD','SUB')):
        ax=axs[i,j]
        for k in ('B0','B1','T0','T1'):
            yy=[float(lookup[(cond,k,w,op)]['exact_mean'])*100 for w in widths]
            sd=[float(lookup[(cond,k,w,op)]['exact_sample_SD'])*100 for w in widths]
            ax.errorbar(widths,yy,yerr=sd,label=k,marker='o',capsize=2)
        ax.set_xscale('log',base=2);ax.set_ylim(*limits);ax.set_title(cond+' '+op);ax.grid(alpha=.25)
        ax.axhline(0,color='gray',linestyle='--',linewidth=.7);ax.axhline(100,color='gray',linestyle='--',linewidth=.7)
        ax.set_ylabel('Integer exact match (%)');ax.set_xlabel('Integer range bits');ax.legend(loc='lower left')
fig.suptitle('Mean ± sample SD across 3 seeds (full SD bars)')
fig.tight_layout();fig.savefig(R/'figures/length_generalization.png',dpi=150);plt.close(fig)
assert sha(p)==before
(R/'figure_qa.json').write_text(json.dumps(dict(passed=True,source_metric_SHA256=before,metric_data_unchanged=True,all_SD_bars_within_axes=True,axes_y_limits=limits,additional_training=0,additional_model_inference=0),indent=2),encoding='utf-8')
(R/'sha256_manifest.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(R)}' for p in sorted(R.rglob('*')) if p.is_file() and p.name!='sha256_manifest.txt')+'\n',encoding='utf-8')
print(json.dumps(dict(figure_QA_passed=True,y_limits=limits,metrics_unchanged=True)))
