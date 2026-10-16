# Environment

## Original runs
- OS: Microsoft Windows 11 Home 10.0.26200
- CPU: Intel(R) Core(TM) i5-10210U @ 1.60GHz, 4 cores / 8 logical processors; RAM 8 GB; GPU: Intel UHD Graphics (not used; all runs on CPU)
- Python 3.13.14 (venv), PyTorch 2.14.1+cpu, NumPy 2.5.3, Matplotlib 3.11.2, SciPy 1.18.1, Pillow 12.3.0 (`versions_venv.txt`, full list `pip_freeze_venv.txt`)
- Thread counts and per-run environment records: `results/<experiment>/ENVIRONMENT.json` (or `environment.json`), e.g. `results/IronMath_G1/ENVIRONMENT.json`, `results/IronMath_PaperCheck/extension/audit/ENVIRONMENT.json`.
- Figures of v0.7.3/v0.7.4 were drawn with Matplotlib 3.10.8 under Python 3.14; the paper was typeset with MiKTeX pdfTeX 1.40.25.

## Recreating
Full environment (training/inference code): `python -m venv venv` then `pip install -r pip_freeze_venv.txt` (PyTorch CPU wheels: add `--index-url https://download.pytorch.org/whl/cpu` for torch). SudachiPy/sudachidict in the freeze belong to an unrelated part of the workspace and are not needed.

Verification only (no PyTorch): `pip install -r requirements_verify.txt`, plus `pdftoppm` (Poppler) for the figure comparison.
