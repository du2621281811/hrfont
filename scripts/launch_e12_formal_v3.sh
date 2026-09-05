#!/usr/bin/env bash
# E12 cache_v3 pipeline: phi_s2 -> id_cls -> T1-T4, for each seed.
# Applies S1 (char axis), S2 (typeface-group split + cross-group negatives),
# S4 (62-class ID-CLS), S5 (val-AUC early stop). S3 (more real typefaces) is
# still open and is the reason gates may remain out of reach.
set -uo pipefail
PY=/root/miniforge3/envs/boogu/bin/python
ROOT=/root/projects/hrfont
EF=$ROOT/scripts/eval_framework
LOG=$ROOT/logs/e12
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONUNBUFFERED=1
cd "$EF"
for SEED in 3407 3408 3409; do
  echo "[$(date -Is)] phi_s2 v3 seed=$SEED gpu=$CUDA_VISIBLE_DEVICES"
  $PY train_style_encoder.py --config "$ROOT/configs/e12_phi_s2_v3_s${SEED}.yaml" \
    2>&1 | tee "$LOG/phi_s2_v3_s${SEED}.log"
  echo "[$(date -Is)] id_cls v3 seed=$SEED"
  $PY train_id_cls.py --config "$ROOT/configs/e12_id_cls_v3_s${SEED}.yaml" \
    2>&1 | tee "$LOG/id_cls_v3_s${SEED}.log"
  echo "[$(date -Is)] T1-T4 v3 seed=$SEED"
  $PY self_tests.py --config "$ROOT/configs/e12_self_tests_v3_s${SEED}.yaml" \
    2>&1 | tee "$LOG/self_tests_v3_s${SEED}.log" || echo "[$(date -Is)] gate_failed seed=$SEED"
done
echo "[$(date -Is)] E12 v3 all seeds done"
