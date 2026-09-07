#!/usr/bin/env bash
# Representative F0/F3 eval on idle GPU2. Never uses GPU3 (F2).
# P1 and F0 share GPU2 in parallel (per-item seed, results invariant).
# F3 runs after both finish (higher VRAM / cache).
set -u
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
OUT=$ROOT/reports/f03_test16_strat
LOG=$OUT/logs
mkdir -p "$LOG"
cd "$ROOT"

gpu_free() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

if [ "$(gpu_free 3)" -lt 4000 ]; then
  echo "[guard] GPU3 looks occupied (F2). Will not use it."
fi
FREE2=$(gpu_free 2)
echo "[guard] GPU2 free=${FREE2} MiB · F2 stays on GPU3"

"$PY" scripts/eval_f03_test16_strat.py gallery

run_one() {
  local mid=$1
  echo "[launch] $mid on GPU2"
  CUDA_VISIBLE_DEVICES=2 "$PY" scripts/eval_f03_test16_strat.py generate \
    --method "$mid" --device cuda:0 \
    >"$LOG/${mid}.stdout" 2>&1
  return $?
}

# Parallel Official + F0 if GPU2 has enough headroom; else serial.
P1_EC=0
F0_EC=0
if [ "$FREE2" -ge 14000 ]; then
  echo "[launch] parallel P1 + F0_100k on GPU2"
  CUDA_VISIBLE_DEVICES=2 "$PY" scripts/eval_f03_test16_strat.py generate --method P1 --device cuda:0 \
    >"$LOG/P1.stdout" 2>&1 &
  echo $! >"$LOG/P1.pid"
  CUDA_VISIBLE_DEVICES=2 "$PY" scripts/eval_f03_test16_strat.py generate --method F0_100k --device cuda:0 \
    >"$LOG/F0_100k.stdout" 2>&1 &
  echo $! >"$LOG/F0_100k.pid"
  wait "$(cat "$LOG/P1.pid")" || P1_EC=$?
  wait "$(cat "$LOG/F0_100k.pid")" || F0_EC=$?
else
  echo "[launch] serial P1 then F0 (GPU2 free ${FREE2})"
  run_one P1; P1_EC=$?
  run_one F0_100k; F0_EC=$?
fi

if [ "$P1_EC" -ne 0 ]; then
  echo "[retry] P1 serial after fail ec=$P1_EC"
  run_one P1 || exit 1
fi
if [ "$F0_EC" -ne 0 ]; then
  echo "[retry] F0 serial after fail ec=$F0_EC"
  run_one F0_100k || exit 1
fi

run_one F3_80k || exit 1

echo "[metrics]"
CUDA_VISIBLE_DEVICES=2 "$PY" scripts/eval_f03_test16_strat.py metrics --lpips --device cuda:0 \
  >"$LOG/metrics.stdout" 2>&1 || exit 1

"$PY" scripts/eval_f03_test16_strat.py gallery
echo DONE >"$OUT/DONE.txt"
echo "[done] $OUT"
