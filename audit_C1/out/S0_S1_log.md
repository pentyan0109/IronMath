# S0–S1 記録
- 承認された設計文書の SHA256：DESIGN 28e56fb1…09e6、METHOD 61ea5ce8…ebcc、FREEZE_PLAN f1331f21…aaf（設計コミット f468e4a）
- 環境：Linux コンテナ、4 CPU、15 GB。Python 3（システム）、PyTorch 2.14.1+cpu（元の実験と同じ版）、NumPy 2.5.3
- リリース取得：part3 の tar.gz SHA256 269b8ace…5150、part4 の tar.gz SHA256 f049fe59…be4。いずれも SHA256SUMS.txt と一致。part2 は v054t の結果の CSV だけを再展開した
- 既存ファイル：変更なし（git status は clean。新しいファイルは audit_C1/ 以下だけ）
- S1：単体テスト9件すべて PASS（tests.py：合成データ・手計算の例・ダミーモデルだけを使用）。コードの SHA256 は S1_code_sha256.txt。analyze.py（主指標の集計）もこのコミットで凍結した
- 設計からの補足（真値開封前の実装上の決定）：
  1. B3＝P1 から開始状態−の軌道を除いた版を、副次ベースラインとして追加した。指示書 §4 の「同じ固定入力マージン情報を使う単純なベースライン」にあたり、D-2 の追加情報がどれだけ寄与しているかを切り分ける。
  2. 開始状態−の prefix で誤答があっても、文脈は除外しない。METHOD §2.4 が除外の対象にしているのは c 側の prefix だけ。件数は contexts.csv に記録する。
  3. 真値と B2 は、保持区間の誤答だけを数える（prefix の誤答は S4 の除外で扱う）。
- 費用：このセッションから Claude の使用額を確認する手段はない（後述）。
