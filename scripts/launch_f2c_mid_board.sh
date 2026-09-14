#!/usr/bin/env bash
# Fast mid-board for F2-CLEAN: same sampler/protocol, idle GPUs only.
# Does NOT touch GPU0 (F2-CLEAN training). Result-safe font shards (seed 3407 per item).
set -euo pipefail
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
OUT=$ROOT/reports/f03_test16_strat
LOG=$OUT/logs
mkdir -p "$LOG"

STEPS="${STEPS:-5000,10000,15000}"
cd "$ROOT"

echo "[html] stub compare page"
"$PY" scripts/eval_f03_test16_strat.py timeline --arm F2C --html-only

echo "[launch] F2C timeline shards on GPU1 + GPU2 · steps=$STEPS"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/eval_f03_test16_strat.py timeline \
  --arm F2C --device cuda:0 --shard 0/2 --steps "$STEPS" \
  >"$LOG/F2C_timeline_0of2.stdout" 2>&1 &
echo $! >"$LOG/F2C_timeline_0of2.pid"

CUDA_VISIBLE_DEVICES=2 "$PY" scripts/eval_f03_test16_strat.py timeline \
  --arm F2C --device cuda:0 --shard 1/2 --steps "$STEPS" \
  >"$LOG/F2C_timeline_1of2.stdout" 2>&1 &
echo $! >"$LOG/F2C_timeline_1of2.pid"

echo "pids $(cat "$LOG/F2C_timeline_0of2.pid") $(cat "$LOG/F2C_timeline_1of2.pid")"
echo "board: $OUT/timeline_f2_clean.html"
echo "also: http://127.0.0.1:8790/  (link after copy) · f03 board on :19000 if up"
