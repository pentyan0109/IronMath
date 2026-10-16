#!/usr/bin/env bash
# Inference-only re-verification. Run from minimum-ai-v0 in Git Bash: bash IronMath_PaperCheck/reverify/run_all.sh
# Free disk space is checked at the start of every stage and every model run; below 1 GiB -> STOP (recorded in logs/disk.log).
set -uo pipefail
PY=.venv/Scripts/python.exe
RVD=IronMath_PaperCheck/reverify
mkdir -p $RVD/logs
disk() {  # disk STAGE
  local free; free=$($PY -B -c "import shutil;print(shutil.disk_usage('.').free)")
  echo "$(date '+%Y-%m-%d %H:%M:%S') $1 free_bytes=$free free_GiB=$($PY -B -c "print(round($free/2**30,3))")" >> $RVD/logs/disk.log
  if [ "$free" -lt 1073741824 ]; then echo "STOP $1 free disk below 1 GiB" | tee -a $RVD/logs/disk.log > $RVD/STOPPED_DISK.txt; exit 3; fi
}
procs() {  # procs LABEL : record python processes
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 2" > $RVD/logs/processes_$1.json
}
run() { (cd $RVD && ../../$PY -B "$@"); }
lane() {  # lane NAME JOBS... ; each job "exp args"
  local name=$1; shift
  for job in "$@"; do disk "infer $job"; run infer_decimal.py $job >> $RVD/logs/$name.log 2>&1 || { echo "FAILED $job" >> $RVD/logs/$name.log; return 1; }; done
}
case "${1:-all}" in
  all)
    procs start
    disk preserve_before; run preserve.py before > $RVD/logs/preserve_before.log 2>&1 || exit 1
    disk inference
    # D11: 3 lanes x 2 threads (as the original 2 threads per process); D10 + V054R: 1 lane x 1 thread
    L0=(); L1=(); L2=(); i=0
    for s in 0 1 2 3 4 5 6 7 8 9; do for g in control intervention; do
      case $((i % 3)) in 0) L0+=("d11 $g $s");; 1) L1+=("d11 $g $s");; 2) L2+=("d11 $g $s");; esac; i=$((i+1)); done; done
    lane d11_lane0 "${L0[@]}" & p0=$!
    lane d11_lane1 "${L1[@]}" & p1=$!
    lane d11_lane2 "${L2[@]}" & p2=$!
    ( lane d10 "d10 0" "d10 1" "d10 2" && for m in V051_A1 V055_F0 V055_F1 V055_F2; do disk "infer v054r $m"; run infer_v054r.py $m >> $RVD/logs/v054r.log 2>&1 || { echo "FAILED $m" >> $RVD/logs/v054r.log; exit 1; }; done ) & p3=$!
    rc=0; for p in $p0 $p1 $p2 $p3; do wait $p || rc=1; done
    [ $rc -eq 0 ] || { echo "inference failed" ; exit 1; }
    disk grade; run grade.py > $RVD/logs/grade.log 2>&1 || exit 1
    disk preserve_after; run preserve.py after > $RVD/logs/preserve_after.log 2>&1 || exit 1
    procs end
    echo DONE ;;
esac
