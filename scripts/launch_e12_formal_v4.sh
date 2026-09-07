#!/usr/bin/env bash
# E12 cache_v4: S3 26 typefaces + v3 charset, then phi_s2 -> id_cls -> T1-T4.
# GPU3 only. Do not set CUDA_VISIBLE_DEVICES to 2 (F3 exclusive).
set -uo pipefail
PY=/root/miniforge3/envs/boogu/bin/python
ROOT=/root/projects/hrfont
EF=$ROOT/scripts/eval_framework
LOG=$ROOT/logs/e12
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
if [[ "$CUDA_VISIBLE_DEVICES" == "2" ]]; then
  echo "REFUSE: GPU2 is F3 exclusive" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
cd "$EF"
for SEED in 3407 3408 3409; do
  echo "[$(date -Is)] phi_s2 v4 seed=$SEED gpu=$CUDA_VISIBLE_DEVICES"
  $PY train_style_encoder.py --config "$ROOT/configs/e12_phi_s2_v4_s${SEED}.yaml" \
    2>&1 | tee "$LOG/phi_s2_v4_s${SEED}.log"
  echo "[$(date -Is)] id_cls v4 seed=$SEED"
  $PY train_id_cls.py --config "$ROOT/configs/e12_id_cls_v4_s${SEED}.yaml" \
    2>&1 | tee "$LOG/id_cls_v4_s${SEED}.log"
  echo "[$(date -Is)] T1-T4 v4 seed=$SEED"
  $PY self_tests.py --config "$ROOT/configs/e12_self_tests_v4_s${SEED}.yaml" \
    2>&1 | tee "$LOG/self_tests_v4_s${SEED}.log" || echo "[$(date -Is)] gate_failed seed=$SEED"
done
echo "[$(date -Is)] E12 v4 all seeds done"
