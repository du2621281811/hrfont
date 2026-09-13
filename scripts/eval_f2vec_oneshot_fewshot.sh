#!/usr/bin/env bash
# F2-VEC@40k test16 stratified: few-shot (Es=ref8) then one-shot (Es=永, Δ still ref8).
# Does not overwrite F2-DELTARSI preds. GPU default 2.
set -u
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
OUT=$ROOT/reports/f03_test16_strat
LOG=$OUT/logs
GPU=${1:-2}
mkdir -p "$LOG"
cd "$ROOT"

if [ "$GPU" = "1" ]; then
  echo "[guard] refusing GPU1"
  exit 2
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
echo "[guard] GPU${GPU} free=${FREE} MiB"

echo "[1/3] F2VEC_40000 fewshot"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_f03_test16_strat.py generate \
  --method F2VEC_40000 --device cuda:0 \
  >"$LOG/F2VEC_40000.stdout" 2>&1
ec1=$?
echo "[F2VEC_40000] exit=$ec1"

echo "[2/3] F2VEC_40000_s1 oneshot"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_f03_test16_strat.py generate \
  --method F2VEC_40000_s1 --device cuda:0 \
  >"$LOG/F2VEC_40000_s1.stdout" 2>&1
ec2=$?
echo "[F2VEC_40000_s1] exit=$ec2"

echo "[3/3] metrics (F2VEC only, merge)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/eval_f03_test16_strat.py metrics \
  --only F2VEC_40000,F2VEC_40000_s1 --lpips --device cuda:0 \
  >"$LOG/metrics_f2vec.stdout" 2>&1
ec3=$?
echo "[metrics_f2vec] exit=$ec3"
echo "DONE ec1=$ec1 ec2=$ec2 ec3=$ec3"
exit $(( ec1 || ec2 || ec3 ))
