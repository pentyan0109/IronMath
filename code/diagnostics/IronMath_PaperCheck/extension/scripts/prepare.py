from core import *
def main():
    assert not (R/'audit/STARTED.json').exists(),'Existing extension must not be overwritten'
    ps=process_snapshot('before');ps=ps if isinstance(ps,list) else [ps]
    other=[p for p in ps if 'extension/scripts' not in (p.get('CommandLine') or '').replace('\\','/')]
    if other:raise RuntimeError('Other Python/Julia process present; record and stop before CPU-intensive work: '+str(other))
    for folder in ['E1_finite_time','E2_causal_intervention','E3_independent_replication','audit','paper/v06','paper/v07']:(R/folder).mkdir(exist_ok=True,parents=True)
    paper=Path('inputs/main (7).tex');assert 'v0.6' in paper.read_text(encoding='utf-8').splitlines()[0]
    for source,target in [(paper,R/'paper/v06/main.tex'),(W/'paper/refs.bib',R/'paper/v06/refs.bib'),(W/'paper/jmlr2e.sty',R/'paper/v06/jmlr2e.sty')]:target.write_bytes(source.read_bytes())
    wanted=set([paper,W/'paper/main.tex',W/'paper/refs.bib',W/'paper/SOURCES.txt'])
    for folder in ['IronMath_V054','IronMath_V055','v054h_dynamics','v054t_transition','IronMath_D11','IronMath_G1','IronMath_ALL','IronMath_PaperCheck','IronMath_PaperCheck/stage2','IronMath_PaperCheck/addon/abl1','IronMath_PaperCheck/addon/fp64']:
        for p in (W/folder).iterdir():
            if p.is_file() and p.suffix in ['.py','.md','.csv','.json'] and not p.name.startswith('WORKSPACE'):wanted.add(p)
    wanted.update((W/'IronMath_V053/data').glob('*.json'));wanted.add(W/'v054t_transition/data/probes.jsonl')
    for seed in range(3):wanted.add(W/f'IronMath_V054/checkpoints/C1_B0_s{seed}/best_validation.pt')
    wanted.update((W/'IronMath_PaperCheck/addon/abl1/scripts').glob('*.py'))
    wanted.add(W/'v054r_replication/data/V051_vs_V054_weights.json')
    files=[dict(path=str(p),bytes=p.stat().st_size,SHA256=sha(p)) for p in sorted(wanted)]
    dump('audit/INPUT_MANIFEST.json',dict(utc=utc(),files=files,missing=[]))
    dump('audit/ENVIRONMENT.json',dict(utc=utc(),python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=2,device='cpu',disk_free=shutil.disk_usage(W).free))
    # Short synthetic compute pilot only; no Test outcomes or study model used.
    setup(97531);m=Residual();opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01);x=torch.zeros(16,9,3);y=torch.zeros(16,9,dtype=torch.long)
    tt=time.perf_counter();cc=time.process_time()
    for i in range(10):opt.zero_grad();l=F.cross_entropy(m(x).flatten(0,1),y.flatten());l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
    stepwall=(time.perf_counter()-tt)/10;stepcpu=(time.process_time()-cc)/10
    sm=torch.jit.script(m).eval();h=torch.zeros(12,545);xx=torch.zeros(12,3);yy=torch.zeros(12,dtype=torch.long);tt=time.perf_counter()
    with torch.no_grad():sm.evolution(xx,h,yy,1000)
    diagwall=time.perf_counter()-tt
    estimate=20*1200*stepwall*1.7+23*20000*diagwall/1000+1200
    dump('audit/RESOURCE_ESTIMATE.json',dict(utc=utc(),scratch_pilot_seed=97531,scratch_updates=10,study_models=20,train_step_wall_seconds=stepwall,train_step_CPU_seconds=stepcpu,batch12_1000_iteration_wall_seconds=diagwall,total_estimated_seconds=estimate,peak_RAM_bytes=memory(),RAM_limit_bytes=2*1024**3,total_wall_limit_seconds=7200,storage_estimate_bytes=700*1024**2,results_not_used_for_budget=True))
    if estimate>7200 or shutil.disk_usage(W).free<1024**3:raise RuntimeError('Resource estimate outside fixed limit; no study training started')
    # Verify exact V054 parameter initialization, before study outcomes.
    initial=[]
    original={int(r['seed']):r['initial_sha256'] for r in rows(W/'IronMath_V054/cpu_memory.csv') if r['stage']=='training' and r['condition']=='C1' and r['model']=='B0'}
    for seed in [0,1,2]:setup(seed);v=Residual();h=modelhash(v);assert h==original[seed];initial.append(dict(seed=seed,hash=h,matches=True))
    dump('audit/INITIALIZATION_REPRODUCTION.json',initial)
    md('audit/MASTER_PLAN.md','''# E1〜E3 実行計画

既存原稿・結果は不変。全成果物はextensionのみ。E1（理論/合成/保存軌道＋必要なreadonly再生）→E2規則凍結と介入→E3規則凍結と20モデル学習→評価→独立監査→新規v07ドラフトの順。CPU2threads、process peak2GiB、全体wall7200秒を上限。結果に応じた追加seed・時間延長・閾値探索は禁止。資源不足・障害は未完として報告。scratch seed97531の10更新は資源試験のみ、モデル性能を評価せず、研究20モデルに混ぜない。
元V054の構造・初期化を既存initial_SHAと一致検査。E3はseed100〜109×データ条件0/1。データ条件は同じ既存Train母集団から独立した問題提示乱数（101000+2i+c）を使い、独立生成コーパスとは呼ばない。Test/Validationの分割は変更しない。各1200更新・AdamW .002/wd.01/clip1、ADD8+SUB8、200更新ごとの等演算Validation CE最小で選択。
V054Hには状態ベクトルは残っておらず、E1-Cの距離/Jacobian/線形化とE2のため既存3checkpointを読み取り専用で再生する。新旧の無介入マージン・破綻時刻の一致を先に確認。高精度参照は数学的証明とは呼ばない。E1の初等恒等式・不等式に新規定理の優先性を主張しない。
''')
    dump('audit/STARTED.json',dict(utc=utc(),start_epoch=time.time(),input_manifest_SHA256=sha(R/'audit/INPUT_MANIFEST.json'),plan_SHA256=sha(R/'audit/MASTER_PLAN.md')))
    print('ESTIMATE seconds',round(estimate),'RAM MiB',round(memory()/1024**2),'free GiB',round(shutil.disk_usage(W).free/1024**3,2),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:stage_failure('prepare',e);raise
