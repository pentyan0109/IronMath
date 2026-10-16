"""Rebuild the original workspace layout from the archive, so the original scripts can find their inputs.

Usage: python tools/assemble_workspace.py <archive root> <target dir>
Each file listed in FILE_MAP.csv is copied to <target dir>/<original_path>. The original scripts locate inputs relative to
their own folder (e.g. Path(__file__).parents[...]), so they expect this layout. Checkpoints are not in the archive; scripts
that load checkpoints (training, inference, diagnostics on models) need them placed at the paths in CHECKPOINTS.csv.
Text files whose username was replaced contain '<HOME>' where an absolute path appeared; such paths only occur in logs
and records, and in a few paper-build scripts (tex_bin / inputs placeholders), see _SANITIZED_FILES.csv.
"""
import csv, shutil, sys
from pathlib import Path

A = Path(sys.argv[1]).resolve(); T = Path(sys.argv[2]).resolve()
if T.exists() and any(T.iterdir()): sys.exit(f'{T} is not empty')
n = 0
with open(A / 'FILE_MAP.csv', encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        if not r['original_path'] or r['original_path'].startswith('('): continue
        dst = T / r['original_path']; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(A / r['archive_path'], dst); n += 1
print('files placed:', n)
