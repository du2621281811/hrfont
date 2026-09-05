#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniforge3/envs/boogu/bin/python
ROOT=/root/projects/hrfont
EF=$ROOT/scripts/eval_framework
LOG=$ROOT/logs/e12
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONUNBUFFERED=1
cd "$EF"
for SEED in 3407 3408 3409; do
  echo "[$(date -Is)] START phi_s2 seed=$SEED gpu=$CUDA_VISIBLE_DEVICES"
  $PY train_style_encoder.py --config "$ROOT/configs/e12_phi_s2_v2_s${SEED}.yaml" \
    2>&1 | tee "$LOG/phi_s2_v2_s${SEED}.log"
  echo "[$(date -Is)] START id_cls seed=$SEED"
  $PY train_id_cls.py --config "$ROOT/configs/e12_id_cls_v2_s${SEED}.yaml" \
    2>&1 | tee "$LOG/id_cls_v2_s${SEED}.log"
done
echo "[$(date -Is)] E12 formal v2 all seeds done"
