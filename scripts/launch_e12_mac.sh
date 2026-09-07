#!/usr/bin/env bash
# Run E12 v4 on Apple Silicon / any machine without assuming CUDA.
# Usage:
#   bash scripts/launch_e12_mac.sh 3407
#   bash scripts/launch_e12_mac.sh all
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
EF="$ROOT/scripts/eval_framework"
CACHE="$ROOT/artifacts/e12/cache_v4"
SEED_ARG="${1:-3407}"

if [[ ! -d "$CACHE" ]]; then
  echo "missing $CACHE — pull git (cache_v4) or rebuild from fonts_s3" >&2
  exit 1
fi
n=$(find "$CACHE" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
if [[ "$n" -lt 20 ]]; then
  echo "cache_v4 looks incomplete (dirs=$n, expect ~26)" >&2
  exit 1
fi

run_one() {
  local SEED="$1"
  local PHI_OUT="$ROOT/runs/e12_phi_s2_v4_mac_s${SEED}"
  local ID_OUT="$ROOT/runs/e12_id_cls_v4_mac_s${SEED}"
  local ST_OUT="$ROOT/runs/e12_self_tests_v4_mac_s${SEED}"
  echo "[$(date -Is)] E12 mac seed=$SEED device=auto cache=$CACHE"
  cd "$EF"
  $PY train_style_encoder.py \
    --config "$ROOT/configs/e12_phi_s2_v4_s${SEED}.yaml" \
    --set "data.cache_dir=$CACHE" \
    --set "train.device=auto" \
    --set "train.fp16=false" \
    --set "output.dir=$PHI_OUT"
  $PY train_id_cls.py \
    --config "$ROOT/configs/e12_id_cls_v4_s${SEED}.yaml" \
    --set "data.cache_dir=$CACHE" \
    --set "train.device=auto" \
    --set "output.dir=$ID_OUT"
  $PY self_tests.py \
    --config "$ROOT/configs/e12_self_tests_v4_s${SEED}.yaml" \
    --set "data.cache_dir=$CACHE" \
    --set "train.device=auto" \
    --set "model.phi_checkpoint=$PHI_OUT/best.pt" \
    --set "model.id_checkpoint=$ID_OUT/best.pt" \
    --set "output.dir=$ST_OUT" \
    || echo "[$(date -Is)] gate_failed seed=$SEED (expected until T2 is fixed)"
  echo "[$(date -Is)] done seed=$SEED → $ST_OUT/self_tests.json"
}

if [[ "$SEED_ARG" == "all" ]]; then
  for s in 3407 3408 3409; do run_one "$s"; done
else
  run_one "$SEED_ARG"
fi
echo "[$(date -Is)] E12 mac launch finished"
