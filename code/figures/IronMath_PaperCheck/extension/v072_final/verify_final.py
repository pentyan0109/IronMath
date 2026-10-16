"""Independent scalar norm check, limited-edit verification, preservation and UTF-8 audit."""
from final_revision import *
import math
def main():
    rr=rows(O/'E3_ENDPOINT_NORMS.csv');assert len(rr)==240 and len({(r['model'],r['index']) for r in rr})==240;checked=[]
    for mid in sorted({r['model'] for r in rr}):
        z=np.load(src(X/f'E3_independent_replication/diagnostics/{mid}_trajectory.npz'),allow_pickle=False)['end']
        for r in [a for a in rr if a['model']==mid]:
            independent=math.sqrt(math.fsum(float(v)*float(v) for v in z[int(r['index'])]));assert math.isclose(independent,float(r['endpoint_L2_norm']),abs_tol=1e-12,rel_tol=0);checked.append(independent)
    norms=json.loads((O/'E3_ENDPOINT_NORMS_SUMMARY.json').read_text(encoding='utf-8'));assert math.isclose(norms['median'],statistics.median(checked),abs_tol=1e-12) and math.isclose(norms['minimum'],min(checked),abs_tol=1e-12) and math.isclose(norms['maximum'],max(checked),abs_tol=1e-12)
    s=(O/'main_v0.7.2.tex').read_text(encoding='utf-8');old=(P/'main_v0.7.2.tex').read_text(encoding='utf-8');edits=json.loads((O/'LOCAL_EDITS.json').read_text(encoding='utf-8'));replay=old
    for e in edits:replay=replay.replace(e['old'],e['new'],1)
    assert replay==s and '\\CHECK' not in s
    assert r'hold lengths $k = 256$, 1{,}024 and 4{,}096' in s and r'hold lengths $k \ge 256$' in s
    assert all(r['matches']=='0' for r in rows(O/'INITIAL_HASH_COMPARISON.csv'))
    for name in ['figure1_arithmetic_margins.pdf','figure3_g1_abl1.pdf']:assert sha(O/'figures'/name)==sha(P/'figures'/name)
    manifest=json.loads((O/'INPUT_SHA256.json').read_text(encoding='utf-8'));manifest.update({p:v for p,v in INPUTS.items() if not Path(p).is_relative_to(O)});changed=[p for p,v in manifest.items() if not Path(p).is_relative_to(O) and sha(p)!=v['SHA256']];assert not changed
    oldmanifest=json.loads((P/'SHA256_MANIFEST.json').read_text(encoding='utf-8'));oldchanges=[p for p,v in oldmanifest['files'].items() if sha(P/p)!=v['SHA256']];assert not oldchanges
    build=json.loads((O/'BUILD_SUMMARY.json').read_text(encoding='utf-8'));assert build['CHECK_remaining']==build['undefined_references']==build['error']==0
    write('FINAL_REVISION_REPORT.md',f'''# v0.7.2 最終修正完了

指定7項目の処理を完了。新学習・新モデル推論・新Test生成/評価はいずれも0。保存データの集計と指定本文の局所修正だけ。元原稿と過去成果物は上書きしていない。

1. 短い保持長で誤答を観測しなかった試行は右打切りで、旧集計でも時刻の平均・SD・範囲に入れていなかった。k=Tの最終位置での観測は欠測ではない。本文のTiming括弧を共通の長いk（E3:256/1024、旧:256/1024/4096）へ限定。最大観測時刻78/88を全て超え、同じ保存履歴ごとの観測時刻は不変。旧k256/1024/4096を使う5節・k>=256を使う8節と矛盾なし。詳細：TIMING_CENSOR_RECHECK.md、各CSV。
2. 新しい10初期重みSHAは、V053・V054 B0/B1・V055のいずれとも一致0。この結果を本文に1文追記。T0/T1は比較から除外。INITIAL_HASH_COMPARISON.csv。
3. 全240個の保存E3終点（各545次元）のL2ノルム：中央値9.005443255066913、範囲[6.418488700025291,12.391761565776502]。本文に9.0054,[6.4185,12.3918]を転記。240ベクトルをmath.fsumによる別計算でも1e−12許容で一致確認。終点の生成・モデル実行はなし。E3_ENDPOINT_NORMS.csv/summaryとCHECK_FILL_LOG_FINAL.md。
4. 図1キャプションを指定文言・$(a,b)$表記へ変更。図1自体は前版PDFからbyte一致で保全。
5. 図2凡例のみShared endpointへ変更。全棒の数値は前版CSV不変、PDF再レンダリングを目視確認。
6. 謝辞のand carried out the literature searchへ変更。
7. 公式JMLRスタイルで32ページ生成。CHECK残存0・未定義参照0・エラー0、BibTeX警告0。一般警告{build['warning']}・Overfull{build['overfull']}・Underfull{build['underfull']}はBUILD_REPORT.md/BUILD_WARNINGS.csvに全件保存。指定外のレイアウト・表の変更はせず、これらを解消済みとは扱わない。

LOCAL_EDITS.jsonを旧原稿へ順に適用すると新原稿の全文と一致する。CHECKを埋めた後の不要なCHECKマクロ定義・例示コメントだけを併せて除去した。旧数値や未指定の主張を書き換えていない。図3も前版PDFとbyte一致。提供bibと実物JMLRスタイルは前版とbyte一致。

全32ページのコンタクトシートと変更図2を目視確認。全報告書はUTF-8で読み直し。元データ・保存tensor・旧原稿・旧成果物のSHA不変を確認して停止。再現コードはfinal_revision.py→build_final.py→verify_final.py（元成果物を保持する場合は別の新規出力先で実行）。
''')
    utf=[]
    for p in O.glob('*.md'):
        text=p.read_text(encoding='utf-8');assert '\ufffd' not in text;utf.append(dict(file=p.name,SHA256=sha(p),UTF8_readback=True))
    dump('INPUT_SHA256.json',manifest);dump('FINAL_QA.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CHECK_remaining=0,undefined_references=0,errors=0,endpoint_norm_vectors_independently_checked=240,allowed_edits_verified=True,section5_8_lengths_consistent=True,initial_hash_matches=0,original_input_changes=changed,prior_delivery_file_changes=oldchanges,prior_delivery_files_checked=len(oldmanifest['files']),new_training=0,new_inference=0,new_Test_evaluation=0,figure2_visual_QA=True,manuscript_pages_contact_sheet_reviewed=32,UTF8_reports=utf,stopped=True))
    dump('SHA256_MANIFEST.json',dict(self_excluded=True,files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,SHA256=sha(p)) for p in O.rglob('*') if p.is_file() and p.name!='SHA256_MANIFEST.json'}));print('Final verification PASS: CHECK0, undefined0, error0; original files unchanged; STOP',flush=True)
if __name__=='__main__':main()
