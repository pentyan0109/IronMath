"""Stage 0: preregistration config, asset paths, environment, preservation manifest (before), V046 ZIP metadata (read-only),
main-evaluation data generation with SHA. No model execution here."""
import datetime,json,os,platform,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from common import E,WORK,V054,SEEDS,ckpt_path,sha_file,dump,generate,save_set,load_set,verify_rows,annotate,counts

AUDIT_DIRS=['IronMath_V0','IronMath_V4','IronMath_V052','IronMath_V053','IronMath_V054','IronMath_Abacus']
WIDTHS=[2048,4096,8192,12000,16384,32768,65000]

def manifest():
    return {str(p.relative_to(WORK)):sha_file(p) for d in AUDIT_DIRS for p in sorted((WORK/d).rglob('*')) if p.is_file()}

def ps(cmd):
    r=subprocess.run(['powershell','-NoProfile','-Command',cmd],capture_output=True,text=True,timeout=120);return r.stdout.strip()

def main():
    assert not (E/'config.json').exists(),'config already registered; prep must not rerun'
    now=datetime.datetime.now().isoformat(timespec='seconds')
    config=dict(
        experiment='V054E addition endurance (inference only)',registered_at=now,
        model='V054 C1 B0 (budget C1 = 1200 updates; B0 = independent-task binary residual GRU with op flag), Validation-selected best_validation.pt, seeds 0/1/2, trained on 8-bit only',
        operation='ADD only',io='LSB-first, hidden zero-initialised per problem, input (a_i,b_i,0) for W+1 steps incl. guard 0, output W+1 bits, argmax',
        widths=WIDTHS,width_2048_set='IronMath_V053/data/ADD_Test2048.json as used by V054 (512 problems)',
        problems_per_width=512,category_composition=dict(V050_part=dict(counts()[0]),V053_fill=dict(counts()[1]),note='same construction as V054 Test (V051 Test 256 from V050 generator + V053 fill 256 cycling 6 kinds)'),
        seeds_data=dict(main='80000+width',boundary='81000+search_index',confirmation='82000+width',probe_65000='83000+65000 (not specified in instruction; fixed here before any run)'),
        pass_rule='3-seed mean integer exact-match rate >= 0.95',
        primary=dict(mean95_endurance_width='largest evaluated width whose 3-seed mean exact rate >= 95%',allseed95_endurance_width='largest evaluated width where all 3 seeds >= 95%',reported_separately=['main','confirmation'],boundary_search='exploratory, reported separately'),
        transition='first main width whose 3-seed mean < 95% -> stop main widths beyond it (unevaluated), go to boundary search',
        boundary_search=dict(bisection_between='W_p (last pass) and W_f (first fail)',midpoint='round((lo+hi)/2) to nearest multiple of 64',max_iterations=3,new_set_per_midpoint=True,monotonicity_assumed=False,W_b='largest passing width among W_p and midpoints'),
        confirmation=dict(width='W_b, or 65000 if no failure',set='new set seed 82000+W, same composition and size',edge_overlap_rule='the 16 deterministic V050 boundary edge cases are fixed functions of W and therefore identical to those in the set already evaluated at W; they are kept (not re-drawn) and confirmation exact rate is reported both on all problems and excluding those 16; all other problems are checked for duplicates against earlier sets at the same width'),
        reduction_rule='if main-evaluation estimate > 3.5 h: halve each category count (ceil, min 8) for widths >= 16384; if still over, run widths ascending and mark widths not reached within budget as unevaluated',
        reduction_subset='halved set = first ceil(n/2) (min 8) problems of each origin-category group of the already generated main set, in generation order (no regeneration)',
        time_budget_hours=5.0,main_estimate_limit_hours=3.5,
        estimate_method='per-bit-per-problem wall cost measured on the adopted engine at batch 128 (2048-bit reproduction and 4096-bit check), scaled linearly in sequence length; 8-problem 4096/8192 timings (batch 8) reported alongside as linearity check',
        probe_65000=dict(condition='only if main, boundary and confirmation are all finished and time remains',per_seed=32,composition='random 16, propagation 8, first 8 V050 boundary edges (includes (2^L-1)+1 and (2^L-1)+(2^L-1))',use='descriptive only, never used for endurance judgement'),
        reference_prediction=dict(formula='p2048^(L/2048) per seed, p2048 = V054 C1 B0 2048-bit ADD exact rate per seed',use='reference only, not used for pass/fail'),
        hidden_trace=dict(rows='first 16 problems of each width set x each seed',every_steps=256,stats=['L2 norm mean','L2 norm max','fraction |h|>0.99','residual term |tanh(gate)|*||h|| mean']),
        hazard_intervals=[[0,2048],[2048,8192],[8192,16384],[16384,32768],[32768,65001]],
        hazard_definition='event = first wrong output bit position (LSB=0); exact sequences right-censored at output length; hazard = events in interval / bits exposed in interval by sequences error-free at interval entry',
        first_error_phase='V050/V051 definition from carry-in/carry-out at the first wrong position: start(cin=0,cout=1) middle(1,1) end(1,0) none(0,0); exact problems = correct',
        float64_replay='first up to 10 failed problems per seed (main evaluation, ascending width then set order), weights and computation in float64, exploratory',
        engine_batch=128,threads=2,precision='float32 (checkpoint standard)',
        unchanged=['no training','no checkpoint write','no structure change','1-bit GRU step','V055 untouched','no V050/V051 comparison'])
    dump('config.json',config)

    # asset paths
    assets=dict(report=dict(path='IronMath_V054/REPORT_V054.md',sha256=sha_file(V054/'REPORT_V054.md'),
                quote_B0='B0はV053と同じ残差GRUで演算flagによってADD/SUBを学習する「独立タスク」であり、別々の重みを持つ2モデルではない。',
                quote_C1='C1は1200更新・各演算9600提示、C2は学習CPU90秒。 / Validation選択checkpointの実測値',
                C1_interpretation='C1 = training budget condition (1200 updates); not an evaluation condition. Checkpoint = best_validation.pt (Validation-selected), per run.py main().'),
        checkpoints={f'seed{s}':dict(path=str(ckpt_path(s).relative_to(WORK)),sha256=sha_file(ckpt_path(s))) for s in SEEDS},
        model_code=dict(path='IronMath_V054/model.py',sha256=sha_file(V054/'model.py'),imported=['Residual','batch']),
        evaluation_code=dict(path='IronMath_V054/run.py',sha256=sha_file(V054/'run.py'),note='evaluate(): torch.jit.script(Residual) on model.batch input, batch 128, argmax'),
        spec=dict(path='IronMath_V054/SPEC_REPRESENTATION.md',sha256=sha_file(V054/'SPEC_REPRESENTATION.md')),
        test_sets={f'ADD_Test{w}':dict(path=f'IronMath_V053/data/ADD_Test{w}.json',sha256=sha_file(WORK/f'IronMath_V053/data/ADD_Test{w}.json')) for w in [8,16,32,64,128,256,512,1024,2048]},
        saved_predictions=dict(path='IronMath_V054/predictions.jsonl.gz',sha256=sha_file(V054/'predictions.jsonl.gz'),subset='condition C1, model B0, op ADD, 9 widths x 512 x 3 seeds = 13824'),
        generators=dict(V050=dict(path='IronMath_V4/v050/src/prepare.py',sha256=sha_file(WORK/'IronMath_V4/v050/src/prepare.py')),V053=dict(path='IronMath_V053/dataset.py',sha256=sha_file(WORK/'IronMath_V053/dataset.py'))),
        phase_definition=dict(path='IronMath_V4/v050/src/evaluate.py',sha256=sha_file(WORK/'IronMath_V4/v050/src/evaluate.py')))
    dump('asset_paths.json',assets)

    # environment + concurrency
    procs=ps("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ForEach-Object { \"$($_.ProcessId)`t$($_.CreationDate)`t$($_.CommandLine)\" }")
    env=dict(started=now,python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),cpu=ps('(Get-CimInstance Win32_Processor).Name'),
             logical_processors=os.cpu_count(),threads=2,other_python_processes_at_start=[l for l in procs.splitlines() if l and str(os.getpid()) not in l.split('\t')[0]],
             note='timings are wall time; V055 run.py processes were running concurrently at start (observed only, not touched)')
    dump('environment.json',env)

    t=time.perf_counter();before=manifest();dump('preservation_before.json',dict(dirs=AUDIT_DIRS,excluded=['v054e_endurance','IronMath_V055'],files=len(before),seconds=time.perf_counter()-t,sha256=before))

    # V046 ZIP metadata (read-only)
    z=WORK/'IronMath_V4/v046/dagger_rollouts.csv.zip';st=z.stat()
    with zipfile.ZipFile(z) as f:members=[dict(name=i.filename,date_time=list(i.date_time),file_size=i.file_size,compress_size=i.compress_size,CRC=f'{i.CRC:08x}',create_system=i.create_system,create_version=i.create_version,extract_version=i.extract_version,compress_type=i.compress_type,comment=i.comment.decode(errors='replace')) for i in f.infolist()];comment=f.comment.decode(errors='replace')
    acl=ps(f"(Get-Acl -LiteralPath '{z}').Owner")
    item=ps(f"Get-Item -LiteralPath '{z}' | Select-Object CreationTime,LastWriteTime,LastAccessTime,Attributes | ConvertTo-Json")
    streams=ps(f"Get-Item -LiteralPath '{z}' -Stream * | Select-Object Stream,Length | ConvertTo-Json")
    sync=ps("Get-Process | Where-Object { $_.ProcessName -match 'OneDrive|Dropbox|GoogleDrive|googledrivesync|iCloud|Box|Sync' } | Select-Object ProcessName,Id | ConvertTo-Json")
    onedrive=dict(env_OneDrive=os.environ.get('OneDrive'),env_OneDriveConsumer=os.environ.get('OneDriveConsumer'),path_under_onedrive=bool(os.environ.get('OneDrive')) and str(z).lower().startswith(os.environ.get('OneDrive','~').lower()))
    csv_exists=(WORK/'IronMath_V4/v046/dagger_rollouts.csv').exists()
    traces=dict(storage_audit_20261014='storage_audit_20261014/STORAGE_DELETE_CANDIDATES.csv and archive_sha256.csv list IronMath_V4/v046/dagger_rollouts.csv (253046374 bytes, SHA 548d22c6...) with existing ZIP dagger_rollouts.csv.zip (109538970 bytes) whose member SHA equals the original CSV SHA; directory files dated 2026-10-14 18:25-18:35',
                v051_records='IronMath_V4/v051/REPORT_V051.md, preservation_deviation.json, SHA_difference_investigation.json, src/verify_archive.py record that the CSV was replaced by the ZIP during the V051 run and that V051 code has no compression/deletion step',
                note='records only; no file was opened for writing; cause not determined')
    dump('v046_zip_metadata.json',dict(path=str(z.relative_to(WORK)),size=st.st_size,sha256=sha_file(z),stat=dict(st_ctime=st.st_ctime,st_mtime=st.st_mtime,st_atime=st.st_atime,ctime_iso=datetime.datetime.fromtimestamp(st.st_ctime).isoformat(),mtime_iso=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(),atime_iso=datetime.datetime.fromtimestamp(st.st_atime).isoformat()),
         powershell_item=item,owner=acl,alternate_streams=streams,zip_comment=comment,members=members,original_csv_present=csv_exists,sync_software_processes=sync,onedrive=onedrive,traces=traces,cause='not determined (not asserted)'))

    # data: main sets (generated before any model execution)
    manifest_data={}
    src=json.loads((WORK/'IronMath_V053/data/ADD_Test2048.json').read_text())
    rows=[annotate(dict(id=r['id'],op='ADD',a=r['a'],b=r['b'],width=2048,category=r['category'],origin=r['origin'])) for r in src]
    for r,s in zip(rows,src):assert r['longest_propagation']==s['longest_propagation'] and r['answer']==s['answer']
    verify_rows(rows);manifest_data['main_2048']=dict(sha256=save_set('main_2048',rows),n=len(rows),seed='V054 existing Test',source_sha256=sha_file(WORK/'IronMath_V053/data/ADD_Test2048.json'))
    for w in WIDTHS[1:]:
        t=time.perf_counter();rows=generate(w,80000+w,'main');verify_rows(rows)
        manifest_data[f'main_{w}']=dict(sha256=save_set(f'main_{w}',rows),n=len(rows),seed=80000+w,generate_verify_seconds=time.perf_counter()-t)
        print('DATA',w,len(rows),round(time.perf_counter()-t,1),flush=True)
    dump('data/manifest.json',manifest_data)
    print('PREP DONE',flush=True)

if __name__=='__main__':main()
