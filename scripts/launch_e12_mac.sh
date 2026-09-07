#!/usr/bin/env bash
# Run E12 v4 on Apple Silicon (MPS) or CPU. From repo root:
#   bash scripts/launch_e12_mac.sh 3407
#   bash scripts/launch_e12_mac.sh all
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EF="$ROOT/scripts/eval_framework"
CACHE="$ROOT/artifacts/e12/cache_v4"
PY="${PYTHON:-python3}"
DEVICE="${E12_DEVICE:-auto}"
SEEDS_ARG="${1:-3407}"

if [[ ! -d "$CACHE" ]]; then
  echo "missing $CACHE — pull latest main (cache_v4 is in git) or rebuild with e12_build_cache_v4.sh" >&2
  exit 1
fi
if [[ ! -f "$CACHE/manifest.json" ]]; then
  echo "cache_v4 has no manifest.json" >&2
  exit 1
fi

if [[ "$SEEDS_ARG" == "all" ]]; then
  SEEDS=(3407 3408 3409)
else
  SEEDS=("$SEEDS_ARG")
fi

cd "$EF"
for SEED in "${SEEDS[@]}"; do
  echo "[$(date -Is)] phi_s2 v4 seed=$SEED device=$DEVICE"
  $PY train_style_encoder.py --config "$ROOT/configs/e12_phi_s2_v4_s${SEED}.yaml" \
    --set "data.cache_dir=$CACHE" \
    --set "output.dir=$ROOT/runs/e12_phi_s2_v4_s${SEED}" \
    --set "train.device=$DEVICE" \
    --set "train.fp16=false"
  echo "[$(date -Is)] id_cls v4 seed=$SEED"
  $PY train_id_cls.py --config "$ROOT/configs/e12_id_cls_v4_s${SEED}.yaml" \
    --set "data.cache_dir=$CACHE" \
    --set "output.dir=$ROOT/runs/e12_id_cls_v4_s${SEED}" \
    --set "train.device=$DEVICE"
  echo "[$(date -Is)] T1-T4 v4 seed=$SEED"
  set +e
  $PY self_tests.py --config "$ROOT/configs/e12_self_tests_v4_s${SEED}.yaml" \
    --set "data.cache_dir=$CACHE" \
    --set "model.phi_checkpoint=$ROOT/runs/e12_phi_s2_v4_s${SEED}/best.pt" \
    --set "model.id_checkpoint=$ROOT/runs/e12_id_cls_v4_s${SEED}/best.pt" \
    --set "output.dir=$ROOT/runs/e12_self_tests_v4_s${SEED}" \
    --set "train.device=$DEVICE"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[$(date -Is)] gate_failed seed=$SEED (exit $rc) — expected until T2 is fixed"
  fi
done
echo "[$(date -Is)] E12 mac run finished"
