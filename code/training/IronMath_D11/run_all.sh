#!/usr/bin/env bash
# IronMath D11 reproduction. Run from minimum-ai-v0/IronMath_D11 in Git Bash. 4 lanes x 2 threads (timings are reference only).
set -euo pipefail
PY=../.venv/Scripts/python.exe
mkdir -p logs
lane() {  # lane STAGES... over a list of "group seed" pairs given in $PAIRS
  local name=$1; shift
  for gs in $PAIRS; do
    g=${gs%:*}; s=${gs#*:}
    for st in "$@"; do $PY -B d11.py $st $g $s >> logs/${name}.log 2>&1; done
  done
}
phase() {  # phase STAGES...: 20 models split over 4 lanes, wait for all
  local pids=()
  for L in 0 1 2 3; do
    PAIRS=""
    for s in 0 1 2 3 4 5 6 7 8 9; do for g in control intervention; do
      j=0; [ "$g" = intervention ] && j=1
      i=$(( s * 2 + j ))
      [ $(( i % 4 )) -eq $L ] && PAIRS="$PAIRS $g:$s"
    done; done
    PAIRS="$PAIRS" lane "$1_lane$L" "$@" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
}
case "${1:-all}" in
  train) $PY -B d11.py gen_train > logs/gen_train.log 2>&1; phase train ;;
  test)  $PY -B d11.py gen_test > logs/gen_test.log 2>&1; phase eval probes equiv fixed; $PY -B d11.py analyze > logs/analyze.log 2>&1 ;;
esac
