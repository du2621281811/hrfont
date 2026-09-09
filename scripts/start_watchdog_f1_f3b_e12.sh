#!/bin/bash
# Detach F1/F3b/E12 watchdog from any controlling terminal / Cursor session.
set -euo pipefail
ROOT=/root/projects/hrfont
WD=$ROOT/reports/watchdog_f1_f3b_e12
mkdir -p "$WD"
# Stop previous watchdog python only (exact script path).
while read -r pid; do
  [ -n "$pid" ] || continue
  if tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q 'scripts/watchdog_f1_f3b_e12.py'; then
    kill "$pid" 2>/dev/null || true
  fi
done < <(pgrep -f 'scripts/watchdog_f1_f3b_e12.py' || true)
sleep 1
cd "$ROOT"
# Double-fork via setsid; redirect all stdio; ignore hangup.
nohup setsid /root/miniforge3/envs/boogu/bin/python -u \
  "$ROOT/scripts/watchdog_f1_f3b_e12.py" \
  >>"$WD/watchdog.stdout" 2>&1 < /dev/null &
echo $! > "$WD/launcher.pid"
sleep 2
if [ -f "$WD/watchdog.pid" ]; then
  echo "WATCHDOG_OK pid=$(cat "$WD/watchdog.pid")"
else
  echo "WATCHDOG_MISSING_PIDFILE"
  tail -50 "$WD/watchdog.stdout" || true
  exit 1
fi
