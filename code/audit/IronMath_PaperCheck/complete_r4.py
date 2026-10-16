"""Reference comparisons and checkpoint digest validation, independent code."""
from common import *
def run():
    c=Comparisons();stats=load(R/'memory_group_statistics.json');tau=load(R/'linear_tau_recomputed.json');ref=load(W/'IronMath_G1/FINAL_POSTEXIT_AUDIT.json')
    c.add('G1','linear_tau_median',ref['linear_tau_median_all_components'],tau['median'],'REPORT_G1.md / FINAL_POSTEXIT_AUDIT.json','-1/log(lambda), 2560 raw components')
    c.add('G1','linear_tau_max',ref['linear_tau_max_all_components'],tau['maximum'],'REPORT_G1.md / FINAL_POSTEXIT_AUDIT.json','-1/log(lambda)')
    ar=load(R/'arithmetic_summary.json')
    for r in rows(W/'v054e_endurance/summary.csv'):
        if r['category']=='ALL' and r['seed']=='mean' and r['set_type'] in ['main','confirmation']:
            v=next(x for x in ar['V054E']['means'] if x['set']==r['set_type'] and x['width']==int(r['width']))
            for oldkey,newkey in [('exact_rate','mean'),('exact_rate_sample_SD','sample_SD')]:c.add('V054E',f"{r['set_type']}/{r['width']}/{newkey}",float(r[oldkey]),v[newkey],'summary.csv / REPORT_V054E.md','independent prediction integer rescoring')
    for r in rows(W/'v054e_endurance/confirmation.csv'):
        v=next(x for x in ar['V054E']['means'] if x['set']=='confirmation' and x['width']==int(r['width']))
        c.add('V054E','independent_confirmation/mean',float(r['mean_exact']),v['mean'],'confirmation.csv','independent prediction integer rescoring')
    # Compare the actual Markdown display, not merely its companion summary.
    for line in (W/'v054e_endurance/REPORT_V054E.md').read_text(encoding='utf-8-sig').splitlines():
        parts=[x.strip() for x in line.strip().strip('|').split('|')]
        if line.startswith('|') and len(parts)==8 and parts[0].isdigit() and '±' in parts[5]:
            width=int(parts[0]);v=next(x for x in ar['V054E']['means'] if x['set']=='main' and x['width']==width)
            for seed in range(3):
                rr=next(x for x in ar['V054E']['counts'] if x['set']=='main' and x['width']==width and x['seed']==seed)
                c.add('V054E',f'report_display/{width}/seed{seed}',float(parts[seed+2]),round(rr['correct']/rr['N']*100,2),'REPORT_V054E.md main table','independent exact count rounded to displayed 2 decimals')
            for name,oldvalue,value in [('mean',parts[5].split('±')[0],v['mean']),('sample_SD',parts[5].split('±')[1],v['sample_SD'])]:
                c.add('V054E',f'report_display/{width}/{name}',float(oldvalue),round(value*100,2),'REPORT_V054E.md main table',f'independent integer exact rates; unrounded percentage={value*100:.17g}')
    for line in (W/'IronMath_G1/REPORT_G1.md').read_text(encoding='utf-8-sig').splitlines():
        pp=[x.strip() for x in line.strip().strip('|').split('|')]
        if line.startswith('|') and pp[0] in ['residual_gru','gru','lstm','rnn','linear'] and len(pp)==15:
            s=stats[f'G1/{pp[0]}/actual3']
            for idx,method in [(5,'A'),(6,'B'),(7,'C_all_success'),(8,'C_all_failure'),(9,'D')]:c.add('G1',f'report_display/{pp[0]}/{method}/hits',int(pp[idx]),s[method]['hits'],'REPORT_G1.md architecture table','raw packed outputs')
            c.add('G1',f'report_display/{pp[0]}/McNemar_p',float(pp[13]),s['paired_AD']['McNemar_exact_two_sided_p'],'REPORT_G1.md architecture table','independent exact paired test')
    for study in ['V058','V058b']:
        rr=rows(R/(study.lower()+'_independent_judgment_table.csv'))
        report=W/f'IronMath_{study}/REPORT_{study}.md'
        for line in report.read_text(encoding='utf-8-sig').splitlines():
            pp=[x.strip() for x in line.strip().strip('|').split('|')]
            if line.startswith('|') and len(pp)>=6 and pp[0] in ['64','128'] and any(x['checkpoint']==pp[1] for x in rr):
                r=next(x for x in rr if x['width']==pp[0] and x['checkpoint']==pp[1])
                for idx,key in [(2,'prediction'),(3,'O0_correct'),(4,'O1_correct'),(5,'actual_best')]:c.add(study,f'report_display/{pp[0]}/{pp[1]}/{key}',pp[idx],r[key],report.name,'independent raw integer rescoring')
    for study,file,key in [('FF1','IronMath_FF1A/ff1a_predictor_comparison.csv','group'),('FF1R','IronMath_FF1R/ff1r_predictor_comparison.csv','subset')]:
        for definition in (['actual3'] if study=='FF1' else ['actual3','actual7']):
            s=stats[f'{study}/all/{definition}']
            for r in rows(W/file):
                if r[key]!='all40':continue
                method='C_all_failure' if r['method']=='C' else r['method']
                for k in ['hits','TP','FP','TN','FN']:c.add(study,definition+'/'+r['method']+'/'+k,int(r[k]),s[method][k],file,'packed predictions + invalid masks + independent labels')
    audits=rows(R/'arithmetic_checkpoint_hash_checks.csv')
    for study,filepath,container in [('V054E','v054e_endurance/asset_paths.json','checkpoints'),('V054R','v054r_replication/asset_paths.json','files'),('D10','decimal_d10/asset_paths.json','files')]:
        for key,v in load(W/filepath)[container].items():
            if Path(v['path']).suffix.lower() not in ['.pt','.pth','.safetensors']:continue
            p=Path(v['path']);p=p if p.is_absolute() else W/p
            audits.append(dict(study=study,asset=key,path=str(p),expected=v['sha256'],actual=sha(p)))
    for g in ['control','intervention']:
        for seed in range(10):
            p=W/f'IronMath_D11/checkpoints/{g}_s{seed}';ref=load(p/'training_record.json')
            audits.append(dict(study='D11',asset=f'{g}_s{seed}',path=str(p/'best_validation.pt'),expected=ref['best_checkpoint_file_SHA256'],actual=sha(p/'best_validation.pt')))
    audits=list({(a['study'],a['asset'],a['path']):a for a in audits}.values())
    for a in audits:a['match']=a['actual'].lower()==a['expected'].lower()
    assert all(a['match'] for a in audits)
    table('arithmetic_checkpoint_hash_checks.csv',audits)
    allrows=rows(R/'r4_memory_recomputation.csv')+rows(R/'r4_arithmetic_recomputation.csv')+c.data
    table('r4_recomputation.csv',allrows)
    bad=[r for r in allrows if str(r['match']).lower()=='false'];unreported=[r for r in allrows if r['match']=='not_reported']
    text='# R4：不一致と再検証の限界\n\n'
    text+=f'照合項目 {len(allrows)}件。数値・判定の不一致 {len(bad)}件。旧集計で未報告の追加記述値 {len(unreported)}件。\n\n'
    for r in bad:text+=f"- {r['experiment']} / {r['metric']}: 報告 {r['reported_value']}、再計算 {r['recomputed_value']}。元資料 {r['reported_source']}。{r['raw_source']}。両方を保持する。原因の候補は表示時の切り捨て／丸め処理の差。元の表示生成経路は未確定で、どちらかへ修正しない。\n"
    text+='''
## 欠測と検証可能な範囲

- G1・FF1・FF1-R：入力記号から正解の記憶値を新規実装で復元し、保存された予測bit列と非有限マスクを全位置で展開した。R以外の非有限も失敗分類に反映した。学習済みモデルはロード・実行していない。
- V054EとV058/V058b：保存された予測整数を、Pythonの独立した整数加算・乗算と比較した。データセットの正解も整数演算と照合した。
- V054R・D10・D11の耐久Probe：保存された各試行のexactフラグから再集計した。生の予測bit/桁列が保存されていないため、そのフラグ自体の独立再採点はできない。固定点の読み出し正誤は出力と正解の数値から再判定した。スペクトル半径は保存値を利用し、ヤコビアンの新規計算は行っていない。
- D11の評価ログは成功時のpredictionがnullで、誤答時に予測整数を保存する実装である。保存整数がある試行だけ独立にa+bと照合し、nullの試行は保存exactフラグを集計した。全試行の独立再採点が完了したとは扱わない。件数はd11_prediction_coverage.csv。
- D11のH1/H2の差の平均と95%区間は元のanalysis_summary.jsonに明示されていないため、報告値をNOT_REPORTEDとした。今回の区間は事後の記述値（seed単位、10,000回、seed66000）であり、H3の事前登録区間と区別する。
- FF1には有限誤答型に関する事前登録H2はない。FF1-RのH2規則をFF1に当てた数値は参考の事後集計であり、FF1の登録仮説へ追加しない。
- FF1-Rの3長さ定義と7長さ定義を併記し、元の全7長さの判定を変更しない。
- 一致は既存報告書に対応する保存集計の精密値と比較した結果。丸めた報告書の表示値、欠測、生ログのexactフラグに依存する項目を、独立した再推論による再現と同一視しない。
'''
    (R/'r4_discrepancies.md').write_text(text,encoding='utf-8')
    dump('R4_COMPLETION.json',dict(comparisons=len(allrows),discrepancies=len(bad),unreported_descriptive_values=len(unreported),checkpoint_hashes=len(audits)+len(rows(R/'checkpoint_hash_checks.csv')),all_checkpoint_hashes_match=True))
if __name__=='__main__':run()
