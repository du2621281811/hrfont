#!/usr/bin/env bash
# Detached overnight supervisor for F0/F2-CLEAN + E12-b.
set -euo pipefail
ROOT=/root/projects/hrfont
WD="$ROOT/reports/watchdog_tonight_20260914"
PY=/root/miniforge3/envs/boogu/bin/python
mkdir -p "$WD" "$ROOT/reports/f0f2_clean_v0913/logs" "$ROOT/reports/e12_b"

# Single instance
if [[ -f "$WD/watchdog.pid" ]]; then
  old=$(cat "$WD/watchdog.pid" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    cmd=$(tr '\0' ' ' < /proc/"$old"/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -q 'watchdog_tonight_f0f2_e12b'; then
      echo "already running pid=$old"
      exit 0
    fi
  fi
fi

# Drop stale STOP
rm -f "$WD/STOP"

nohup setsid "$PY" -u "$ROOT/scripts/watchdog_tonight_f0f2_e12b.py" \
  >>"$WD/watchdog.console.log" 2>&1 < /dev/null &
echo "started pid=$! -> $WD"
sleep 2
tail -n 30 "$WD/watchdog.console.log" || true
cat "$WD/status.json" 2>/dev/null || true
