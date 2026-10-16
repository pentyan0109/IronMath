"""Freeze the complete replication protocol before quality tests and training."""
import copy
from common import *
assert (R/'PRIMARY_LENGTHS_APPROVED.json').exists(), 'Await research director clarification; no training'
assert (R/'WORKSPACE_BEFORE.json').exists() and not read('WORKSPACE_BEFORE.json')['errors']
assert not (R/'TRAINING_STARTED.json').exists()
reg=copy.deepcopy(json.loads((W/'IronMath_FF1/PREREGISTRATION.json').read_text(encoding='utf-8')))
reg['experiment']='IronMath FF1-R';reg['registered_utc']=now()
reg['conditions']['seeds']=list(range(10,20))
reg['primary_lengths']=read('PRIMARY_LENGTHS_APPROVED.json')['lengths']
reg['long_evaluation']['primary_truth']='all R positions including prefix R, all categories, at primary_lengths; all 7 lengths are evaluated and reported'
reg['long_evaluation']['generate_after']='A,D,B,C combined prediction freeze, never before'
reg['D']=dict(repetitions=16,criterion='all final100 movement values finite and maximum<1e-6, endpoints finite, and no nonfinite movement/margin/endpoint during recorded repetitions; any established failure => failure; incomplete coverage without established failure => missing',inputs='new diagnostic traces only; no readout correctness, distances, A predictions, or long results; margin finite/nonfinite only, never its value or sign',same_as_FF1A=True)
reg['primary']=dict(H1='A_hits strictly greater than B, C_all_success, C_all_failure, D; ties unsupported',H1_overrides=['all 40 actual outcomes identical => indeterminate','either A prediction class fewer than 5 models => indeterminate'],H2='finite-wrong models >=5 and A correct failure count > D count => supported; <5 => indeterminate; otherwise unsupported',H3='descriptive decomposition of convergence, stability, distance and readout at A/D disagreements; not causal',McNemar='exact two-sided binomial p=0.5 with n=A_only_correct+D_only_correct; no discordances => p=1',CI='paired correctness difference A-D, 40-model bootstrap10000, numpy.default_rng(80000), percentile2.5/97.5; alpha.05 descriptive, not H1 decision',C='save both uniform predictions; compare against both')
reg['order']=['workspace SHA before','registration','quality','train all40','A and D diagnostics','B short evaluation','freeze A B C D together','generate long datasets','long evaluation','independent saved-output audit','workspace SHA after','report and stop']
reg['source_copy_audit_SHA256']=sha(R/'SOURCE_COPY_AUDIT.json')
reg['code_SHA256']={p.name:sha(p) for p in sorted(R.glob('*.py'))}
reg['clarifications']='User FF1-R supplement incorporated; A distance is only two-attractor flag, not model-success gate; primary lengths approval separately preserved'
reg['prior_FF1_primary_result_not_changed']=True
for mid,h,l,s in model_ids():
    dump(f'configs/{mid}.json',dict(model=mid,H=h,L=l,seed=s,train_seed=10000+s,validation_seed=20000+s,training=reg['training'],model_specification=reg['model']))
import platform,importlib.metadata
dump('ENVIRONMENT.json',dict(utc=now(),python=sys.version,platform=platform.platform(),machine=platform.machine(),packages={n:importlib.metadata.version(n) for n in ['torch','numpy']},torch_threads=2,torch_interop_threads=1,device='cpu',precision='float32'))
dump('PREREGISTRATION.json',reg)
dump('PREREGISTRATION_SHA256.json',dict(SHA256=sha(R/'PREREGISTRATION.json'),utc=now()))
md='# FF1-R 事前登録\n\n登録時刻：'+reg['registered_utc']+'\n\n新規seed10〜19、H32/64×L8/16、40モデル。学習・モデル・診断A・評価コードはFF1のコピー。common.pyのモデルseed範囲だけ変更し、診断コードのSHA一致を保存。\n\n主判定長さ：'+str(reg['primary_lengths'])+'。全7長さを評価する。定義・全パラメータ・生成規則・監査・判定不能条件は下の完全JSONに凍結。方式Dのマージンは有限性だけを使用し、正誤・符号・大きさは使わない。距離はAのモデル予測に使わない。\n\n```json\n'+json.dumps(reg,ensure_ascii=False,indent=2)+'\n```\n'
(R/'PREREG_FF1R.md').write_text(md,encoding='utf-8')
dump('PREREG_FF1R_SHA256.json',dict(SHA256=sha(R/'PREREG_FF1R.md'),utc=now()))
print('PREREGISTERED',sha(R/'PREREGISTRATION.json'),flush=True)
