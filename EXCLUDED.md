# EXCLUDED — 含めなかったもの

ファイル単位の一覧は `_EXCLUDED_FILES.csv`（元のパス・サイズ・理由）、checkpoint は `CHECKPOINTS.csv`（ハッシュと役割）。原本はワークスペースに残っており、どれも変更・削除していない。

## 1. 理由別

| 理由 | ファイル数 | MB |
|---|---:|---:|
| exploratory or non-evidence experiment: only reports and preregistration documents are archived | 2148 | 4051.2 |
| whole-workspace file manifest (lists unrelated projects and absolute paths); comparison summaries kept | 20 | 224.7 |
| process list of the personal computer | 17 | 0.0 |
| whole-workspace preservation log | 6 | 0.0 |

## 2. 理由×フォルダ

| 理由 | フォルダ | ファイル数 | MB |
|---|---|---:|---:|
| exploratory or non-evidence experiment: only reports and preregistration documents are archived | IronMath_V4 | 1154 | 1970.7 |
| exploratory or non-evidence experiment: only reports and preregistration documents are archived | IronMath_V0 | 721 | 1743.1 |
| exploratory or non-evidence experiment: only reports and preregistration documents are archived | IronMath_V052 | 67 | 109.6 |
| exploratory or non-evidence experiment: only reports and preregistration documents are archived | IronMath_V057 | 76 | 77.4 |
| exploratory or non-evidence experiment: only reports and preregistration documents are archived | IronMath_V058 | 42 | 72.1 |
| exploratory or non-evidence experiment: only reports and preregistration documents are archived | IronMath_V058b | 47 | 71.9 |
| exploratory or non-evidence experiment: only reports and preregistration documents are archived | IronMath_Abacus | 41 | 6.4 |
| process list of the personal computer | IronMath_PaperCheck | 13 | 0.0 |
| process list of the personal computer | IronMath_FF1 | 2 | 0.0 |
| process list of the personal computer | IronMath_FF1R | 1 | 0.0 |
| process list of the personal computer | IronMath_G1 | 1 | 0.0 |
| whole-workspace file manifest (lists unrelated projects and absolute paths); comparison summaries kept | IronMath_PaperCheck | 9 | 134.5 |
| whole-workspace file manifest (lists unrelated projects and absolute paths); comparison summaries kept | IronMath_G1 | 3 | 26.2 |
| whole-workspace file manifest (lists unrelated projects and absolute paths); comparison summaries kept | IronMath_FF1R | 3 | 25.7 |
| whole-workspace file manifest (lists unrelated projects and absolute paths); comparison summaries kept | IronMath_FF1 | 3 | 25.0 |
| whole-workspace file manifest (lists unrelated projects and absolute paths); comparison summaries kept | IronMath_ALL | 2 | 13.3 |
| whole-workspace preservation log | IronMath_PaperCheck | 6 | 0.0 |

## 3. 含めていない checkpoint（一覧とSHA256のみ）

| 役割 | ファイル数 | MB |
|---|---:|---:|
| exploratory | 736 | 4633.7 |
| other | 4044 | 1892.1 |
| supporting | 59 | 639.1 |

`primary`（論文の主張に直接使う324ファイル、414MB）は `checkpoints/` に含めた。

## 4. 対象外のワークスペース

研究ワークスペース（minimum-ai-v0）のうち、IronMath 以外のフォルダ（「ミニマムAI」の V1〜V19・KS 系列、alice、demo0、TinyStories-JP、paper/ など）と、ホームフォルダの他プロジェクト・個人ファイルは、走査対象に入れていない。IronMath 内でも次は対象外：IronMath_Discovery（論文で使っていない後続探索）、storage_audit_*、パス `.venv`、各種キャッシュ（`__pycache__`, `mpl_cache`, `runtime_cache`, ブラウザ検査用プロファイル）。

## 5. 書き換えたもの（コピーのみ）

ユーザー名を含む絶対パスを `<HOME>` に置換したテキストファイル：124 件（`_SANITIZED_FILES.csv` に置換前後の SHA256）。2回目の検査で、TeX ログの折り返しで分断されたパスや「PC名\\ユーザー名」形式の所有者表記が残っていたので、ユーザー名を `<USER>`、PC名を `<HOST>` に置き換えた（同じ一覧に含む）。論文に記載された著者のメールアドレスはそのまま残した。コードでは TeX の実行ファイルの場所を `tex_bin`、ダウンロードフォルダを `inputs` に置き換えた（組版用スクリプトのみ）。事前登録ファイルはバイト単位でそのままで、ユーザー名を含むものはなかった。gzip 内のユーザー名：検出なし。

## 6. 個人情報・認証情報の検査

全テキストファイルについて、メールアドレス・APIキー・トークン・秘密鍵のパターンを検査した（`_PRIVACY_FLAGS.csv`、22 件）。検出されたメールアドレスは LaTeX パッケージ（natbib・booktabs）の作者連絡先と、コード中の `m.core.weight_ih@m.projection`（行列積の式）だけで、個人情報ではない。「secret-like」の2件は図スクリプト中の CHECK 印を探す文字列（`token=` で始まる）で、認証情報ではない。著者のメールアドレスは論文記載のものだけ。

## 7. 再現への影響

- 除外したワークスペース目録（WORKSPACE_BEFORE/AFTER.json, manifest_before/after.tsv）は「実験の前後でワークスペースが変わっていない」ことの記録で、論文の数値の再計算には使わない。比較結果（WORKSPACE_COMPARISON.json, preservation_comparison.json 等）は含めた。
- プロセス一覧（processes_*.json, process_checks.jsonl）は同時実行の有無の確認記録で、数値には使わない。同時実行の事実は ALL_DEVIATIONS.md と各報告書に残っている。
- 探索段階（V0〜V053）と乗算練習（V057/V058/V058b）のコード・結果は、論文で証拠として使っていない。報告書と事前登録文書は含めた。
- primary 以外の checkpoint は、論文の数値の再集計には不要（保存済みの予測・軌道・結果から再集計できる）。学習の途中経過や選択手順を checkpoint から検証したい場合は必要になる。
