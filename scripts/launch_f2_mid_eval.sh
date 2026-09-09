#!/usr/bin/env bash
# F2 mid-result eval (timeline + F2@75k stratified), same protocol as F3 board.
# Never uses GPU3 (live F2 train). Prefer GPU0.
set -u
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
OUT=$ROOT/reports/f03_test16_strat
LOG=$OUT/logs
GPU=${1:-0}
mkdir -p "$LOG"
cd "$ROOT"

if [ "$GPU" = "3" ]; then
  echo "[guard] refusing GPU3 (F2 training)"
  exit 2
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
echo "[guard] using GPU${GPU} free=${FREE} MiB"

"$PY" scripts/eval_f03_test16_strat.py timeline --arm F2 --html-only
"$PY" scripts/eval_f03_test16_strat.py gallery

echo "[1/3] F2 timeline on GPU${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_f03_test16_strat.py timeline --arm F2 --device cuda:0 \
  >"$LOG/F2_timeline.stdout" 2>&1
ec1=$?
echo "[timeline] exit=$ec1"

echo "[2/3] F2@75k full stratified"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_f03_test16_strat.py generate --method F2_75000 --device cuda:0 \
  >"$LOG/F2_75000.stdout" 2>&1
ec2=$?
echo "[F2_75000] exit=$ec2"

echo "[3/3] metrics + gallery"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_f03_test16_strat.py metrics --lpips --device cuda:0 \
  >"$LOG/metrics_f2.stdout" 2>&1
ec3=$?
"$PY" scripts/eval_f03_test16_strat.py gallery
"$PY" scripts/eval_f03_test16_strat.py timeline --arm F2 --html-only

if [ $ec1 -eq 0 ] && [ $ec2 -eq 0 ] && [ $ec3 -eq 0 ]; then
  echo F2_MID_DONE >"$OUT/F2_MID_DONE.txt"
  echo "[done] $OUT/timeline_f2.html"
  exit 0
fi
echo "[fail] timeline=$ec1 generate=$ec2 metrics=$ec3" >&2
exit 1
