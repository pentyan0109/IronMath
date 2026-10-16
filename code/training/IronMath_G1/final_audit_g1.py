"""Post-exit saved-file audit/report notes only; not experimental code or model execution."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import hashlib,json,csv,math,datetime,shutil
import numpy as np
R=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(n):return json.loads((R/n).read_text(encoding='utf-8'))
def rows(n):
    with (R/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
assert not (R/'FINAL_POSTEXIT_AUDIT.json').exists()
lines=(R/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
for line in lines:
    h,p=line.split('  ',1);assert sha(R/p)==h,p
for n in ['PREREGISTRATION.json','PREREG_G1.md','THEORY_G1.md']:
    key=n.rsplit('.',1)[0]+'_SHA256.json';assert sha(R/n)==load(key)['SHA256']
for n in ['PREREG_G1.md','THEORY_G1.md','REPORT_G1.md']:
    raw=(R/n).read_bytes();s=raw.decode('utf-8',errors='strict');assert s.encode('utf-8')==raw and '\ufffd' not in s and '???' not in s
assert sha(R/'REPORT_G1.md')==load('REPORT_UTF8_CHECK.json')['SHA256']
metrics=load('g1_metrics.json');models=rows('g1_results.csv');checks=[]
for m in models:
    for n in ['A','D','actual_success']:m[n]=m[n]=='True'
for group,mm in [('nonlinear_pooled',[m for m in models if m['architecture']!='linear'])]+[(a,[m for m in models if m['architecture']==a]) for a in ['residual_gru','gru','lstm','rnn','linear']]:
    b=sum(m['A']==m['actual_success'] and m['D']!=m['actual_success'] for m in mm);c=sum(m['D']==m['actual_success'] and m['A']!=m['actual_success'] for m in mm);n=b+c
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
    d=np.array([int(m['A']==m['actual_success'])-int(m['D']==m['actual_success']) for m in mm]);indices=np.random.default_rng(81000).integers(0,len(mm),size=(10000,len(mm)))
    ci=np.percentile(d[indices].mean(1),[2.5,97.5]).tolist();saved=metrics['pooled_nonlinear'] if group=='nonlinear_pooled' else metrics['by_architecture'][group]
    assert p==saved['McNemar_exact_two_sided_p'] and ci==saved['bootstrap95_CI']
    checks.append(dict(group=group,b=b,c=c,p=p,CI=ci))
linear_models=sorted({r['model'] for r in models if r['architecture']=='linear'});distances=[];alltau=[];rho_checks=0;points=0
for mid in linear_models:
    rec=load(f'diagnostic_records/{mid}.json')
    for neutral in ['N','R']:
        with np.load(R/'linear_analytic'/f'{mid}_{neutral}.npz') as z:
            lam=z['lambdas'];tau=z['taus'];q=z['q'];point=z['analytic_point'];ends=z['numerical_endpoints']
        assert np.all((lam>0)&(lam<1));assert np.array_equal(tau,-1/np.log(lam));assert np.array_equal(point,q/(1-lam));points+=1
        if neutral=='N':alltau.extend(tau.tolist())
        rr=[r for r in rec['details'] if r['neutral']==neutral]
        for r in rr:
            if r['converged']:
                assert math.isclose(r['spectral_radius'],float(lam.max()),rel_tol=1e-12);rho_checks+=1
                j=2*r['prefix']+r['value'];diff=float(np.linalg.norm(ends[j].astype(np.float64)-point))
                expected=next(a for a in rec['linear_analytic'] if a['neutral']==neutral and a['prefix']==r['prefix'] and a['value']==r['value'])['analytic_numerical_L2_difference']
                assert math.isclose(diff,expected,rel_tol=1e-12,abs_tol=1e-15);distances.append(diff)
            else:assert next(a for a in rec['linear_analytic'] if a['neutral']==neutral and a['prefix']==r['prefix'] and a['value']==r['value'])['analytic_numerical_L2_difference'] is None
integrity=load('g1_integrity_audit.json');assert integrity['passed'] and integrity['workspace']['original_files_unchanged']
assert integrity['checkpoint_count']==3000 and load('SAVED_OUTPUT_AUDIT.json')['independent_regraded_items']==95200
old_report=(R/'REPORT_G1.md').read_text(encoding='utf-8');(R/'REPORT_G1_PIPELINE.md').write_text(old_report,encoding='utf-8')
shutil.copyfile(R/'REPORT_UTF8_CHECK.json',R/'REPORT_UTF8_CHECK_PIPELINE.json')
nonconv_success=[m['model'] for m in models if m['actual_success'] and m['all16_converged']=='False']
tau_median=float(np.median(alltau));tau_max=float(np.max(alltau));diff_max=float(np.max(distances)) if distances else None
note='\n## 最終確認と解釈の留保（判定変更なし）\n\n'
note+=f'線形40モデルの計2560成分の時定数は、中央値{tau_median:.6g}、最大{tau_max:.6g}ステップ。N/Rの解析的固定点80件、収束終点の照合{len(distances)}件を保存配列から独立再計算し、報告値との一致を確認した。解析的固定点と数値終点のL2差の最大値は{diff_max:.6g}。これはFP32反復と実数上の固定点の差であり、0と仮定しない。各モデルの最大・中央値と全成分はCSVに保存している。\n\n'
note+='線形40モデルでは、短系列方式Bも全件を失敗と予測し、一律失敗Cも40/40だった。P2の支持だけから、Aが線形モデルでB/Cより優れるとも、長さだけが失敗の原因とも主張できない。\n\n'
note+=f'主判定に成功したが診断16本が全収束ではなかったモデルは{len(nonconv_success)}件（{", ".join(nonconv_success)}）。数値的非収束と有限長での誤答は同一ではない。非線形合算でAには偽陽性11件・偽陰性9件が残り、予測は完全ではない。\n\n'
note+='McNemarのP1合算は事前登録された主検定。構造別p値は多重比較補正前の記述値として扱う。ブートストラップの反復単位は指定どおりモデルであり、同じseedを構造・学習長間で共有する設計上の依存の可能性には留意する。評価問題はFF1と共通であり、新しい問題集合への再現とは区別する。\n\n'
ws=integrity['workspace'];note+=f"保存監査では開始前の{ws['original_count']}ファイルが全て不変だった。一方、既存ディレクトリ内の新規ファイル{len(ws['added_files_in_existing_locations'])}件を検出し、パスをWORKSPACE_COMPARISON.jsonに保存した。新規最上位ディレクトリは名前と作成時刻だけを同JSONに記録した。作成元は断定せず、G1からそれらの内容を書き換えていない。\n"
text=old_report+note;(R/'REPORT_G1.md').write_text(text,encoding='utf-8');actual=(R/'REPORT_G1.md').read_text(encoding='utf-8',errors='strict');assert actual==text and '\ufffd' not in actual and '???' not in actual
post=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,no_new_learning_or_model_inference=True,original_artifact_manifest_files_verified=len(lines),registration_and_theory_SHA_verified=True,McNemar_and_bootstrap_independently_recomputed=checks,linear_analytic_points_verified=points,linear_converged_rho_checked=rho_checks,linear_endpoint_differences_verified=len(distances),linear_tau_median_all_components=tau_median,linear_tau_max_all_components=tau_max,linear_endpoint_difference_max=diff_max,report_notes_only_no_outcome_or_rule_changes=True,report_UTF8_readback_exact=True,final_report_SHA256=sha(R/'REPORT_G1.md'),nonconvergent_diagnostic_but_actual_success=nonconv_success,existing_source_files_preserved=ws['original_count'])
(R/'FINAL_POSTEXIT_AUDIT.json').write_text(json.dumps(post,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'REPORT_UTF8_CHECK.json').write_text(json.dumps(dict(utc=post['utc'],SHA256=post['final_report_SHA256'],UTF8_roundtrip_exact=True,notes_added_after_pipeline_with_pipeline_version_preserved=True),ensure_ascii=False,indent=2),encoding='utf-8')
paths=sorted(p for p in R.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
(R/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+p.relative_to(R).as_posix()+'\n' for p in paths),encoding='utf-8')
print(json.dumps(post,ensure_ascii=False),flush=True)
