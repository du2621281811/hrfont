#!/usr/bin/env bash
# Detach V100 hub from Cursor/SSH. Does not touch F0 or :8791 watchdog.
set -euo pipefail
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
DIR=$ROOT/reports/v100_hub
mkdir -p "$DIR"
if [ -f "$DIR/hub.pid" ]; then
  old=$(cat "$DIR/hub.pid" || true)
  if [ -n "${old:-}" ] && [ -d "/proc/$old" ]; then
    if tr '\0' ' ' < "/proc/$old/cmdline" | grep -q serve_v100_hub.py; then
      echo "already running pid=$old  http://127.0.0.1:19000/"
      exit 0
    fi
  fi
fi
# Replace any leftover copy of this script only.
while read -r pid; do
  [ -z "$pid" ] && continue
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q serve_v100_hub.py; then
    kill "$pid" 2>/dev/null || true
  fi
done < <(pgrep -f 'scripts/serve_v100_hub.py' || true)
sleep 0.3
setsid "$PY" "$ROOT/scripts/serve_v100_hub.py" --port 19000 \
  >>"$DIR/hub.stdout" 2>&1 < /dev/null &
echo $! > "$DIR/hub.pid"
sleep 0.2
echo "started pid=$(cat "$DIR/hub.pid")  http://127.0.0.1:19000/"
echo "tunnel: ssh -L 19000:127.0.0.1:19000 root@172.18.41.23"
