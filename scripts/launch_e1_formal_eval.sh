#!/usr/bin/env bash
# Multi-GPU launcher for E1 formal eval generation.
# Does not touch training; shards fonts across GPUs.
set -euo pipefail
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
cd "$ROOT"

EVAL_ID="${1:?eval_id required}"
CHARS="${CHARS:-stratified}"
SPLIT="${SPLIT:-test}"
SEEDS="${SEEDS:-3407}"
METHODS="${METHODS:-P1 E1_100k E1_valbest}"
# Prefer free cards; default 2 and 3 (avoid heavily occupied 0/1)
GPUS="${GPUS:-2,3}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
LOGDIR="$ROOT/runs/E1-FTV2-A-S3407/eval_formal/$EVAL_ID"
mkdir -p "$LOGDIR"

echo "[launch] eval_id=$EVAL_ID chars=$CHARS split=$SPLIT seeds=$SEEDS methods=$METHODS gpus=$GPUS n=$N"

i=0
for gpu in "${GPU_ARR[@]}"; do
  free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  echo "[launch] gpu=$gpu free=${free}MiB shard=$i/$N"
  if [[ "$free" -lt 5000 ]]; then
    echo "[launch] WARN gpu $gpu free<5000MiB, still launching (may OOM)"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/eval_e1_formal.py" generate \
    --eval-id "$EVAL_ID" \
    --methods $METHODS \
    --split "$SPLIT" \
    --chars "$CHARS" \
    --seeds "$SEEDS" \
    --shard "$i/$N" \
    --device cuda:0 \
    >"$LOGDIR/launch_gpu${gpu}_shard${i}.stdout" 2>&1 &
  echo $! >"$LOGDIR/launch_gpu${gpu}_shard${i}.pid"
  i=$((i + 1))
done

echo "[launch] started $N workers; logs under $LOGDIR"
wait
echo "[launch] all workers finished"
"$PY" "$ROOT/scripts/eval_e1_formal.py" metrics --eval-id "$EVAL_ID" --lpips --device cuda:0
"$PY" "$ROOT/scripts/eval_e1_formal.py" verify --eval-id "$EVAL_ID"
"$PY" "$ROOT/scripts/eval_e1_formal.py" gallery --eval-id "$EVAL_ID"
ln -sfn "$LOGDIR" "$ROOT/data/e1_formal_eval"
echo "[launch] metrics+verify+gallery done → $LOGDIR"
echo "[launch] browse: http://127.0.0.1:8777/e1_formal_eval/  (http.server --directory data)"
