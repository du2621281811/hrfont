#!/usr/bin/env bash
# Detach G0b 8-GPU train from Cursor/SSH. Does not overwrite F0-CLEAN-* or failed G0.
set -euo pipefail
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
RUN=G0b-F0-V0913-BS256-A-S3407
LOG=$ROOT/runs/${RUN}.console.log
PIDF=$ROOT/reports/watchdog_g/g0_launcher.pid
mkdir -p "$ROOT/reports/watchdog_g" "$(dirname "$LOG")"
if ls /proc/[0-9]*/cmdline >/dev/null 2>&1; then
  if tr '\0' ' ' < /proc/*/cmdline 2>/dev/null | grep -F 'train.py' | grep -F "$RUN" | grep -vq grep; then
    echo "already running"
    exit 0
  fi
fi
setsid "$PY" "$ROOT/scripts/launch_g0.py" --yes \
  >>"$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "started launcher pid=$(cat "$PIDF")  log=$LOG"
echo "GPUs 0-7  8x32=256  max_steps=10000  lr=1e-5  constant_with_warmup 500"
