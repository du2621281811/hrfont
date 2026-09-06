#!/usr/bin/env bash
# Start (or restart) the F0–F3 live training monitor on :8787.
set -euo pipefail
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
LOG=$ROOT/logs/f123_monitor.log
PIDF=$ROOT/logs/f123_monitor.pid
mkdir -p "$ROOT/logs"

if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$old/cmdline" | grep -q f123_monitor.py; then
      echo "already running pid=$old"
      exit 0
    fi
  fi
fi

# Drop a leftover listener on 8787 only if it is our monitor.
if command -v fuser >/dev/null 2>&1; then
  :
fi

nohup "$PY" "$ROOT/scripts/f123_monitor.py" --port 8787 --bind 0.0.0.0 --auto-resume \
  >>"$LOG" 2>&1 &
echo $! >"$PIDF"
sleep 0.4
echo "started pid=$(cat "$PIDF")  http://0.0.0.0:8787/  snapshot $ROOT/reports/f123_dashboard/"
