#!/bin/bash
# Detach F2-RL128 train + watchdog from Cursor/SSH. Does not stop a live run.
set -euo pipefail
ROOT=/root/projects/hrfont
WD=$ROOT/reports/watchdog_f2_rl128
PY=/root/miniforge3/envs/boogu/bin/python
mkdir -p "$WD"

detach_train() {
  local pid=""
  local p cmd
  while read -r p; do
    [ -n "$p" ] || continue
    cmd=$(tr '\0' ' ' < /proc/"$p"/cmdline 2>/dev/null || true)
    case "$cmd" in
      *FontDiffuser/train.py*F2RL*F2-RL128-A-S3407*) pid=$p; break ;;
    esac
  done < <(pgrep -f 'FontDiffuser/train.py' || true)
  if [ -z "$pid" ]; then
    echo "TRAIN_PID_NONE (watchdog will resume from last_state if needed)"
    return 0
  fi
  local sid
  sid=$(awk '/^NSsid:/{print $2}' /proc/"$pid"/status)
  if [ "$sid" = "$pid" ]; then
    echo "TRAIN_ALREADY_SESSION_LEADER pid=$pid"
    return 0
  fi
  if ! command -v gdb >/dev/null 2>&1; then
    echo "GDB_MISSING; train pid=$pid stays in session $sid (watchdog is the safety net)"
    return 0
  fi
  echo "DETACH_TRAIN pid=$pid old_sid=$sid"
  gdb -p "$pid" --batch \
    -ex 'set pagination off' \
    -ex 'call (long)setsid()' \
    -ex 'call (void *)signal(1, (void *)1)' \
    -ex 'detach' \
    -ex 'quit' >>"$WD/detach.gdb.log" 2>&1 || true
  sid=$(awk '/^NSsid:/{print $2}' /proc/"$pid"/status 2>/dev/null || echo missing)
  echo "TRAIN_AFTER_DETACH pid=$pid sid=$sid"
  # Reparent train to init so a Cursor recursive-child kill cannot reach it.
  local ppid
  ppid=$(awk '/^PPid:/{print $2}' /proc/"$pid"/status)
  if [ -n "$ppid" ] && [ "$ppid" != "1" ]; then
    local pcmd
    pcmd=$(tr '\0' ' ' < /proc/"$ppid"/cmdline 2>/dev/null || true)
    if echo "$pcmd" | grep -q 'launch_cn2west_f123.py'; then
      echo "ORPHAN_TRAIN kill_launcher ppid=$ppid"
      kill -KILL "$ppid" 2>/dev/null || true
      sleep 1
      echo "TRAIN_PPID_NOW $(awk '/^PPid:/{print $2}' /proc/"$pid"/status 2>/dev/null || echo dead)"
    fi
  fi
}

# Stop previous F2-RL128 watchdog only.
while read -r pid; do
  [ -n "$pid" ] || continue
  if tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null | grep -q 'scripts/watchdog_f2_rl128.py'; then
    # SIGTERM is ignored by the watchdog; use SIGKILL for replace.
    kill -KILL "$pid" 2>/dev/null || true
  fi
done < <(pgrep -f 'scripts/watchdog_f2_rl128.py' || true)
sleep 1

detach_train

cd "$ROOT"
nohup setsid "$PY" -u "$ROOT/scripts/watchdog_f2_rl128.py" \
  >>"$WD/watchdog.stdout" 2>&1 < /dev/null &
echo $! > "$WD/launcher.pid"
sleep 2
if [ -f "$WD/watchdog.pid" ]; then
  echo "WATCHDOG_OK pid=$(cat "$WD/watchdog.pid")"
  cat "$WD/status.json" 2>/dev/null || true
else
  echo "WATCHDOG_MISSING_PIDFILE"
  tail -50 "$WD/watchdog.stdout" || true
  exit 1
fi
