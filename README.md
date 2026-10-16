# IronMath reproducibility archive (paper v0.7.4)

Archive accompanying "How Small Recurrent Networks Fail at Length Generalization: Shared Endpoints, Transient Memory, and Diagnostic Ablations" (T. Ikegami, manuscript v0.7.4). Local, unpublished version prepared 2026-10-17. It lets a reader trace each table, figure and headline number of the paper to the saved records it came from, and re-run the main re-aggregations without training or running any model.

**Published** at https://github.com/pentyan0109/IronMath. The repository holds the documentation, maps, code and environment files; the complete archive (about 1.98 GB) is attached to the release [`paper-v0.7.4`](https://github.com/pentyan0109/IronMath/releases/tag/paper-v0.7.4) as four `.tar.gz` parts:

| Part | Content |
|---|---|
| `IronMath_archive_v074_part1_core.tar.gz` | documentation, maps, `code/`, `preregistration/`, `records/`, `environment/`, `tools/`, `MANIFEST_SHA256.txt` |
| `IronMath_archive_v074_part2_results.tar.gz` | `results/` |
| `IronMath_archive_v074_part3_evaluation_sets.tar.gz` | `evaluation_sets/` |
| `IronMath_archive_v074_part4_checkpoints.tar.gz` | `checkpoints/` (the 324 primary checkpoints) |

Extract all four into the same directory (`tar -xzf <part>` for each); they unpack into `archive_v074/`. `SHA256SUMS.txt` in the release lists the hashes of the parts, and `MANIFEST_SHA256.txt` those of every extracted file (`sha256sum -c MANIFEST_SHA256.txt` inside `archive_v074/`). The contents are released under the MIT License (see "License" and `LICENSE`).

## Layout

| Path | Content |
|---|---|
| `PAPER_MAP.md` / `.csv` | Paper table/figure/number → archived source file (row/column hint) → scripts that reference it |
| `FILE_MAP.csv` | Every archived file: archive path, original path in the research workspace, SHA-256 before/after copy |
| `CHECKPOINTS.csv` | Every checkpoint of the series (experiment, model, seed, role, parameters, bytes, SHA-256, initial-weight SHA-256 where recorded) |
| `checkpoints/` | The `primary` checkpoints only (see "Checkpoints") |
| `EXCLUDED.md` | What was left out and why |
| `ARCHIVE_VERIFY.md` | Result of re-running the checks below from a fresh copy of this archive |
| `environment/` | Software versions, `pip freeze`, OS/CPU, and how to recreate the environment |
| `code/training/` | Training code (V054, V055, M3, D10, D11, FF1, FF1-R, G1, E3 (`IronMath_PaperCheck/extension/scripts/e3.py`); V053 for hash comparison) |
| `code/evaluation/` | Evaluation-set generators and evaluation code (V054E, V054T, fresh G1 evaluation) |
| `code/diagnostics/` | Diagnostic and analysis code (V054H, V054R, D10-F, FF1-A, ABL1, timing baselines, break timing, FP64, S1/S2, E1, E2) |
| `code/audit/` | Independent re-scoring and number checks (R1–R4, re-verification, IronMath_ALL, v03/v072/v073 audits) and `v074_verify/` |
| `code/figures/` | Figure scripts of v0.7.2/v0.7.3 and `v074_render/` (re-renders Figures 1–3 from the archived CSVs) |
| `preregistration/` | Preregistration documents, frozen predictions, frozen rules and their recorded hashes, **byte-identical** to the originals |
| `results/` | Saved results (CSV/JSON/JSONL/NPZ, saved predictions and diagnostic traces) behind the paper's numbers |
| `evaluation_sets/` | Saved evaluation (and, for D11, training-presentation) data, with generator code in `code/` |
| `records/` | Experiment instructions and reports, deviation records (`IronMath_ALL/ALL_DEVIATIONS.md`), audit reports, exploratory-phase reports |
| `tools/assemble_workspace.py` | Rebuilds the original workspace layout from this archive |
| `MANIFEST_SHA256.txt` | SHA-256 of every file in the archive |

Within each category the original folder structure is kept: `<category>/<experiment folder>/<original relative path>`. Experiment folders: `IronMath_V054` (V054), `v054e_endurance`, `v054t_transition`, `v054h_dynamics`, `v054r_replication`, `IronMath_V055`, `m3_checker` (M3), `decimal_d10` (D10), `d10_frequency` (D10-F), `IronMath_D11`, `IronMath_FF1`, `IronMath_FF1A`, `IronMath_FF1R`, `IronMath_G1`, `IronMath_PaperCheck` (paper checks R1–R4, S1/S2, re-verification, break timing, ABL1, FP64, fresh evaluation, timing baselines, and `extension/` with E1–E3 and the v0.7.x audits), `IronMath_ALL` (series index, all predictions, seeds and hashes, deviations). The exploratory folders (`IronMath_V0`, `IronMath_V4`, `IronMath_Abacus`, `IronMath_V052`) and the multiplication exercises excluded from the evidence (`IronMath_V057`, `IronMath_V058`, `IronMath_V058b`) contribute reports and preregistration documents only.

## Reproduce the checks (no training, no inference)

Requirements: Python 3.13, the packages in `environment/requirements_verify.txt`, and `pdftoppm` (Poppler; MiKTeX and TeX Live ship it) for the figure comparison.

```
python -m venv venv
venv/Scripts/python -m pip install -r environment/requirements_verify.txt      # (venv/bin/python on Linux/macOS)
venv/Scripts/python code/audit/v074_verify/verify_archive.py . verify_out
venv/Scripts/python code/figures/v074_render/render_figures.py . verify_out/figures
venv/Scripts/python code/figures/v074_render/compare_figures.py results/IronMath_PaperCheck/extension/v074_figures verify_out/figures verify_out/figure_compare pdftoppm
```

`verify_archive.py` recomputes (a) G1 correct counts of A/D/B/B_cum(128)/B_cum(512) and ABL1 A1/A2 on the 160 nonlinear models from frozen predictions, saved diagnostic records/traces and per-length scores; (b) the E3 240-case readout × outcome table and 46/80 shared endpoints; (c) the E2 per-context summary; (e) the SHA-256 of preregistration documents against the recorded values. (d) is the figure re-rendering. Expected values and the outcome of a run on a fresh copy are in `ARCHIVE_VERIFY.md`.

To run an original script unchanged, first rebuild the original layout: `python tools/assemble_workspace.py . ../workspace`, then run the script from there. Scripts that train, run inference or load models also need the checkpoint files.

## Environment of the original runs

Windows 11 Home (10.0.26200), Intel Core i5-10210U (4 cores / 8 threads), 8 GB RAM, CPU only (no CUDA). Python 3.13.14, PyTorch 2.14.1+cpu, NumPy 2.5.3, Matplotlib 3.11.2, SciPy 1.18.1 (`environment/`). Most experiments ran with 1–2 threads; per-experiment `ENVIRONMENT.json` / `environment.json` files are kept in `results/`. Figures of v0.7.3/v0.7.4 were drawn with Matplotlib 3.10.8 under Python 3.14; small font-rasterization differences between versions are expected.

## Checkpoints

`CHECKPOINTS.csv` lists all checkpoints with their roles:
- `primary` = the checkpoints the paper's claims rest on (V054 B0 C1 three seeds, V055 F0–F2, D10, the 200 selected G1 models, the 80 selected FF1/FF1-R models, the 20 selected E3 models);
- `supporting` = other checkpoints used in reported comparisons;
- `other` = intermediate, final-state or resume states that were not selected;
- `exploratory` = exploratory phase.

This version contains the 324 `primary` checkpoints (414 MB) under `checkpoints/<original path>`, each checked against its SHA-256 in `CHECKPOINTS.csv`. The other checkpoints (supporting 639 MB, other 1.89 GB, exploratory 4.63 GB; 7.58 GB for all 5,163) are listed with hashes only and are not published.

## Privacy and modifications

The archive contains no credentials and no personal information other than the author's name and the e-mail address printed in the paper. Absolute paths containing the local user name were replaced by `<HOME>` (remaining user and computer names by `<USER>` and `<HOST>`) in copies of text files (originals unchanged; each change is listed in `_SANITIZED_FILES.csv` with SHA-256 before and after). In code, absolute paths to the TeX binaries and the download folder were replaced by `tex_bin` and `inputs`. Preregistration files are byte-identical copies and contain no such paths. Whole-workspace file manifests and process lists of the personal computer were excluded (`EXCLUDED.md`).

## Roles

Experiments were planned by GPT (OpenAI) as research lead, implemented and run by Claude Code (Anthropic) and Codex (OpenAI) from written instructions, and reviewed by Claude (Anthropic); preregistration documents were approved by the author before being frozen. The archive was assembled by Claude Code. See the paper's acknowledgments.

## License

All contents of this repository and of the paper-v0.7.4 release (code, data, checkpoints and documents) are released under the MIT License; see LICENSE.

Third-party files kept in the release for exact re-builds (`jmlr2e.sty`, `natbib.sty`, `booktabs.sty`, `plainnat.bst`) remain under their own licenses.

## Contact

Tetsuya Ikegami, pentyan0109@gmail.com
