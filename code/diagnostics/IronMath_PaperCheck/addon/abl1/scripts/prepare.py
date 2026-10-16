"""ABL1 inventory and prospective analysis-rule freeze. No model execution."""
from pathlib import Path
import os,sys,json,csv,hashlib,datetime,subprocess,platform,time
R=Path(__file__).resolve().parents[1];W=R.parents[2]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(name,obj):
    p=R/name;p.parent.mkdir(exist_ok=True,parents=True)
    assert not p.exists(),f'Will not overwrite {p}'
    if name.endswith('.md'):
        p.write_text(obj,encoding='utf-8');assert p.read_text(encoding='utf-8')==obj and '\ufffd' not in obj
    else:p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def processes(stage):
    command="Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^(python|julia)'} | Select-Object Name,ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3"
    s=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True)
    write('processes_'+stage+'.json',dict(utc=utc(),returncode=s.returncode,processes=json.loads(s.stdout) if s.stdout.strip() else [],error=s.stderr.strip()))
RULES=r'''# ABL1 固定規則

ABL1はG1の主結果の後に追加した探索的検証であり、G1の事前登録P1・P2を変更しない。

## 対象・真値
G1の既存200モデル（5構造×L8/L16×seed20〜39）。主な解釈は非線形160、線形40は別表、全200も記述する。追加学習・追加推論・checkpointのロードは行わず、元コードはテキストとして読むだけ。入力不足・構造不整合は記録して停止し、推測で補完しない。
真値はG1の主定義、2048・8192・65000の全カテゴリ・全R正解。全7長さの失敗分類と主3長さの失敗を区別する。保存bit列・非有限マスクから入力記号の最終書き込み値を独立復元して再採点し、g1_results.csvと照合する。陽性=長系列成功。

## 軌道条件
diagnostic_g1.pyとmodels.pyの定義を採用。共通の前置き4本（seed30000、8記号）の後のW0/W1直後状態。NとRを同じ出発点の複製から独立に20000回反復。4前置き×2値×2中立入力=16軌道。t=1（最初の中立入力の処理後）〜20000を対象とし、猶予区間を設けない。Nではh_tにRを1回入力した読み出し（反復の軌道自体は変えない）、Rではh_tのhead。
G1のconvは終点が有限、最後の100移動量が全て有限かつ最大<1e-6。rhoはconv時だけ保存され、非収束のrho欠測は適格性を示さない。距離>=1e-3は二点フラグにのみ使われ、A0の成功判定には含まれない。

## A0〜A5
- A0: main_PREDICTIONS_FROZEN.jsonのA。16軌道すべてでconv・保存rho<1・有限で保持値に一致する終点読み出しを要求する元論理を再構成し、完全一致を検査。過去結果を上書きしない。
- A1: 16終点すべてreadout_finiteかつreadout==value。conv/rhoを使わない。
- A2: 保存marginsとlogit_finiteを用い、t=1〜20000の全16軌道で一度も不正・誤読がなければ成功。二値argmaxは同点でクラス0になるためvalue0はmargin>=0、value1はmargin>0。元G1のlogit_finiteはN直後と試行R読み出しのANDで、元の有効性ガードを維持する。このガードとmarginの矛盾も別途記録する。marginのNaNは不正、無限marginかつlogit_finite=Trueは有限logitの減算overflowなので符号で判定する。logit_finite=Falseは不正。state_finiteの失敗数も別に記録するが、A2へ別の隠れ状態条件を加えない。終点のreadoutラベルとも最終marginの符号・tieを照合する。
- A3: 全16でconvかつ終点読み出し正解。rho条件なし。
- A4: 全16でconvかつrho有限・<1。読み出し条件なし。
- A5: 凍結Dを使用。Dは全16convかつ全軌道のendpoints/movements/marginsが有限、全state_finite/logit_finite=True。A4のrho条件とDの軌道全体の有限性条件は異なり、同一視しない。
非収束時のrho=Noneと非有限時のreadout=Noneは元設計由来の欠測。A3/A4のANDのconvが偽ならその軌道は偽と定義できるため、rhoを補完しない。他の必須値の欠落・非有限対照の曖昧さが判定を阻む場合は結果欠測とし、対象を密かに除外しない。

## 指標・比較
全200、非線形160、線形40、各構造40を別集計。accuracy、hits、TP/TN/FP/FN、TPR/TNR、balanced accuracy=(TPR+TNR)/2、MCC、実測と予測のクラス数。真の陽性または陰性が0なら該当率とbalanced accuracyはNA。MCCの分母0はNA、便宜的に0へ置換しない。
比較順:(A0,A1),(A0,A2),(A0,A3),(A0,A4),(A0,A5),(A1,A3),(A3,A0)。両予測の不一致と左だけ/右だけ正解をモデル別に保存。正答率差=左−右。全モデルを記述単位とし、主要な区間はseedクラスタ。各group内でseed20〜39を復元抽出し、同じseedの全アーキテクチャ・両Lをまとめて複製。抽出された全モデルの平均対応差を計算。10000回、numpy.random.default_rng(87000)、percentile 2.5/97.5。比較・groupごとに同じ初期seedで再始動。20seed、非線形各seed8モデル、全200各seed10モデル、構造別各seed2モデル。seed別の正答率・対応差も保存。
McNemar両側正確検定は二項n=b+c,p=.5の両裾（n=0ならp=1）。モデル依存を考慮しない補助解析で、主判定には使わない。区間は多重性未補正の探索的区間。

## 解釈規則（結果を見る前に固定）
「明確な差」はseedクラスタ95%区間が0を除く場合。A0対A1/A2がともに正で0を除くときに限りケース1の限定的証拠とする。区間が0を含む場合はケース2/5として「明確な差を検出できなかった」と記す。予測が完全一致した場合はこの200モデル上での同一判定と記し、一般的同等性は主張しない。簡易診断が上回る場合も同じ基準で報告する。
収束の増分はA3−A1、rhoの増分はA0−A3で評価し、改善・悪化モデルをともに記録する。全体の区間が0を含んでも、個別の判定変化・誤分類の入替は隠さない。rho条件で変わるモデルが0でも、局所安定性の一般的な不要性は主張しない。

## 失敗様式・FP64
発散は全7長さのどの位置でも非有限logitがあること。有限誤答は主判定失敗かつ非有限なし。重複ラベルとして診断非収束、終点誤読、途中誤読、rho条件不成立、混合/書き換えカテゴリ失敗、即時書き換えR失敗、その他カテゴリ失敗を保存する。混合入力等が原因だとは断定しない。FP64の既存200モデルの保存表は別の感度分析に限り、A0/A5を置換しない。反復収束・読み出し・rho条件のどれが変化したかを記録する。

## コスト
新しいモデル実行の計測はしない。S1/S2の既存ログのstate iteration/readout/jacobian_eig/prep/totalと直接テスト時間を引用する。既存iterateは途中読み出し計算も含み、状態更新単独時間を分離できない。jacobianとeigはログで結合されており分離NA。A1〜A5各専用パイプラインの総実行費用は未計測とし、既存A費用を同一と推測しない。保存軌道からの今回の追加解析wall/CPU時間は別に計測し、オンラインモデル推論費用と比較しない。
形式的な状態更新数は1モデル320000（16×20000）、N途中読み出しの試行R更新は別に160000（8×20000）。直接テストは各長68系列、Bcum128は15232、Bcum512は50048。観測入力・開始状態・順序が異なり、step数を同一CPU費用へ換算しない。

## 保全
INPUT_MANIFESTの全入力を実行前後でSHA256照合。既存原稿・凍結予測・checkpoint・元結果を変更しない。MarkdownはUTF-8読み直し検証。原規則・入力manifest・環境・予測定義のhashと時刻をFREEZE.jsonへ保存した後に初めてABL1結果を集計する。最後に報告して停止する。
'''
def prepare():
    import numpy as np,torch
    torch.set_num_threads(2);R.mkdir(exist_ok=True,parents=True);processes('before')
    if (R/'FREEZE.json').exists():raise RuntimeError('ABL1 already frozen; do not overwrite')
    write('RULES_ABL1.md',RULES)
    write('PREDICTIONS_ABL1.md','''# ABL1 予測と判定の登録

本検証は査読後の探索的分析。既存G1の結果は既知であり、事前登録した確認実験とは呼ばない。
実行前にA0優位・簡易読み出しの優位・差の未検出・完全同一のいずれも排除しない。
P-readout: A1/A2のみでA0の予測情報の多くが再現されるか、ケース1/2/5の両側同一基準で評価。
P-convergence: A3対A1の判定変化と正誤の入替を調べる。
P-stability: A0対A3の判定変化と正誤の入替を調べる。
予測アルゴリズム・評価範囲・seedクラスタ区間と判定規則はRULES_ABL1.mdに記載。
ここにABL1の未知の比較値を予め書いたり、特定の勝者を指定したりしない。
''')
    ps=subprocess.run(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json'],capture_output=True,text=True)
    write('ENVIRONMENT.json',dict(utc=utc(),python=sys.version,numpy=np.__version__,pytorch=torch.__version__,platform=platform.platform(),CPU=json.loads(ps.stdout) if ps.stdout.strip() else platform.processor(),cpu_query_error=ps.stderr.strip(),torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),new_model_inference=False,new_training=False))
    roots=[W/'IronMath_ALL',W/'IronMath_G1',W/'IronMath_PaperCheck']
    inventory=[]
    for root in roots:
        for folder,subs,names in os.walk(root):
            subs[:]=[x for x in subs if Path(folder)/x!=R and x not in ['__pycache__','.git']]
            for name in names:
                p=Path(folder)/name;inventory.append(dict(path=p.relative_to(W).as_posix(),bytes=p.stat().st_size))
    with (R/'FILE_INVENTORY.csv').open('w',encoding='utf-8',newline='') as f:q=csv.DictWriter(f,['path','bytes']);q.writeheader();q.writerows(inventory)
    selected=set()
    g=W/'IronMath_G1'
    for dirname in ['assets','diagnostic_records','diagnostic_traces','evaluation_records','main_long_data','saved_predictions']:
        selected.update(p for p in (g/dirname).iterdir() if p.is_file())
    frozen=read(g/'main_PREDICTIONS_FROZEN.json')['models'];assert len(frozen)==200
    for r in frozen:
        a=read(g/'assets'/f"{r['model']}.json");p=g/a['path'];assert p.is_file();selected.add(p)
    for p in g.iterdir():
        if p.is_file() and p.suffix in ['.md','.py','.csv','.json']:selected.add(p)
    for folder in [W/'IronMath_ALL',W/'IronMath_PaperCheck',W/'IronMath_PaperCheck/stage2',W/'IronMath_PaperCheck/addon/fp64',W/'IronMath_PaperCheck/addon/fresh_eval']:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ['.md','.csv','.json'] and not p.name.startswith('WORKSPACE'):selected.add(p)
    selected.update((W/'IronMath_PaperCheck/stage2/records').glob('*.json'))
    selected.update(p for p in (W/'paper').glob('*') if p.is_file())
    files=[];started=utc()
    for i,p in enumerate(sorted(selected),1):
        entry=dict(path=p.relative_to(W).as_posix(),bytes=p.stat().st_size,SHA256=sha(p),suffix=p.suffix)
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:entry['arrays']={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}
        elif p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as f:entry['columns']=next(csv.reader(f),[])
        elif p.suffix=='.json' and p.parent.name=='diagnostic_records':
            d=read(p);entry['detail_records']=len(d['details']);entry['missing_fields']={k:sum(k not in x for x in d['details']) for k in ['neutral','prefix','value','converged','readout','readout_finite','spectral_radius']}
            entry['null_fields']={k:sum(x.get(k) is None for x in d['details']) for k in ['readout','spectral_radius']}
        files.append(entry)
        if i%200==0:print('Inventoried',i,'/',len(selected),flush=True)
    write('INPUT_MANIFEST.json',dict(started_utc=started,completed_utc=utc(),inventory_file='FILE_INVENTORY.csv',inventory_count=len(inventory),selected_input_count=len(files),files=files,notes=['Selected 200 Validation checkpoints; other checkpoints are enumerated in FILE_INVENTORY only.','Nonconvergent rho null and nonfinite readout null are structural, not imputed.']))
    write('FREEZE.json',dict(utc=utc(),before_any_ABL1_comparison=True,files={n:dict(SHA256=sha(R/n),mtime_ns=(R/n).stat().st_mtime_ns) for n in ['RULES_ABL1.md','PREDICTIONS_ABL1.md','INPUT_MANIFEST.json','ENVIRONMENT.json']}))
    print('FROZEN',len(files),'inputs',flush=True)
if __name__=='__main__':prepare()
