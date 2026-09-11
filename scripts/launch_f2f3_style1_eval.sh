#!/usr/bin/env bash
# F2/F3 style-oneshot eval: Es=永, Δ still ref8. GPU0 only. Does not overwrite mean8 preds.
set -u
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
OUT=$ROOT/reports/f03_test16_strat
LOG=$OUT/logs
GPU=${1:-0}
mkdir -p "$LOG"
cd "$ROOT"

if [ "$GPU" = "3" ]; then
  echo "[guard] refusing GPU3"
  exit 2
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
echo "[guard] GPU${GPU} free=${FREE} MiB"

echo "[1/4] F2_75000_s1"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_f03_test16_strat.py generate \
  --method F2_75000_s1 --device cuda:0 \
  >"$LOG/F2_75000_s1.stdout" 2>&1
ec1=$?
echo "[F2_75000_s1] exit=$ec1"

echo "[2/4] F3_80k_s1"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_f03_test16_strat.py generate \
  --method F3_80k_s1 --device cuda:0 \
  >"$LOG/F3_80k_s1.stdout" 2>&1
ec2=$?
echo "[F3_80k_s1] exit=$ec2"

echo "[3/4] metrics (s1 only, merge)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_f03_test16_strat.py metrics \
  --only F2_75000_s1,F3_80k_s1 --lpips --device cuda:0 \
  >"$LOG/metrics_s1.stdout" 2>&1
ec3=$?
echo "[metrics_s1] exit=$ec3"

echo "[4/4] gallery Mode D"
"$PY" scripts/eval_f03_test16_strat.py gallery
ec4=$?
echo "[gallery] exit=$ec4"
echo "DONE ec1=$ec1 ec2=$ec2 ec3=$ec3 ec4=$ec4"
exit $(( ec1 || ec2 || ec3 || ec4 ))
