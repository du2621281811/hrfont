#!/usr/bin/env bash
# Wait until G0c training is done (or no G0c train.py), then run G v0913 shot board.
set -euo pipefail
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
LOG=$ROOT/reports/g_v0913_shot/launch.log
DONE=$ROOT/runs/G0c-F0-V0913-BS256-A-S3407/DONE.json
mkdir -p "$(dirname "$LOG")"

echo "[$(date -Iseconds)] waiter start" | tee -a "$LOG"

while true; do
  if [[ -f "$DONE" ]]; then
    echo "[$(date -Iseconds)] G0c DONE.json present" | tee -a "$LOG"
    break
  fi
  if ! pgrep -f 'G0c-F0-V0913-BS256-A-S3407.*train.py' >/dev/null 2>&1; then
    # process gone — wait a bit for DONE flush
    sleep 30
    if [[ -f "$DONE" ]] || ! pgrep -f 'G0c-F0-V0913-BS256-A-S3407' >/dev/null 2>&1; then
      echo "[$(date -Iseconds)] G0c processes gone" | tee -a "$LOG"
      break
    fi
  fi
  # progress crumb
  step=$(python3 - <<'PY'
from pathlib import Path
import re
p=Path("/root/projects/hrfont/runs/G0c-F0-V0913-BS256-A-S3407.console.log")
if not p.exists():
    print("?"); raise SystemExit
parts=re.split(rb"[\r\n]+", p.read_bytes())
for line in reversed(parts[-50:]):
    m=re.search(rb"(\d+)/20000", line)
    if m:
        print(m.group(1).decode()); break
else:
    print("?")
PY
)
  echo "[$(date -Iseconds)] waiting G0c step=${step}/20000" | tee -a "$LOG"
  sleep 60
done

# free GPU settle
sleep 20
echo "[$(date -Iseconds)] linking dirty F2/F2RL + generating G methods" | tee -a "$LOG"

# 1) reuse dirty baselines (no GPU)
$PY "$ROOT/scripts/eval_g_v0913_shot_board.py" --methods F2_80000,F2_80000_k1,F2RL_40000,F2RL_40000_k1 >>"$LOG" 2>&1

# 2) generate G methods — one process per GPU (8)
# Order: shorter / needed first
METHODS=(G0b_s1 G0c_s1 G1_s1 G1_s8 G2_s1 G2_s8 G2RL_s1 G2RL_s8)
# pilot after G2RL to avoid loading clash on same code cwd — run on freed GPUs in second wave
PIDS=()
for i in "${!METHODS[@]}"; do
  mid=${METHODS[$i]}
  echo "[$(date -Iseconds)] launch $mid on cuda:$i" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$i $PY "$ROOT/scripts/eval_g_v0913_shot_board.py" \
    --device cuda:0 --methods "$mid" >>"$LOG" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do
  wait "$pid" || echo "[$(date -Iseconds)] WARN pid $pid failed" | tee -a "$LOG"
done

# pilot on GPU0/1
echo "[$(date -Iseconds)] launch pilot_s1 / pilot_s8" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 $PY "$ROOT/scripts/eval_g_v0913_shot_board.py" --device cuda:0 --methods pilot_s1 >>"$LOG" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 $PY "$ROOT/scripts/eval_g_v0913_shot_board.py" --device cuda:0 --methods pilot_s8 >>"$LOG" 2>&1 &
p1=$!
wait "$p0" || true
wait "$p1" || true

# final html + metrics
echo "[$(date -Iseconds)] html-only finalize" | tee -a "$LOG"
$PY "$ROOT/scripts/eval_g_v0913_shot_board.py" --html-only >>"$LOG" 2>&1
echo "[$(date -Iseconds)] ALL DONE board=$ROOT/reports/g_v0913_shot/index.html" | tee -a "$LOG"
