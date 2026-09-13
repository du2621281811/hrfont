#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
LOG=$ROOT/reports/watchdog_g/watchdog.stdout
PIDF=$ROOT/reports/watchdog_g/watchdog.pid
mkdir -p "$ROOT/reports/watchdog_g" "$ROOT/reports/g_dashboard"

if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$old/cmdline" | grep -q watchdog_g.py; then
      echo "already running pid=$old"
      exit 0
    fi
  fi
fi

setsid "$PY" "$ROOT/scripts/watchdog_g.py" >>"$LOG" 2>&1 < /dev/null &
echo $! >"$PIDF"
sleep 0.6
echo "started pid=$(cat "$PIDF")  $ROOT/reports/g_dashboard/"
