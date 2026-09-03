#!/usr/bin/env bash
# Outer loop: never exits; respawns keepalive if it dies. No crontab required.
# Touch reports/hrfont_overnight/STOP to halt outer (train checks the same file).
set -u
trap '' HUP
ROOT=/root/projects/hrfont
REP=$ROOT/reports/hrfont_overnight
LOG=$REP/outer_forever.log
PIDF=$REP/outer.pid
STOP=$REP/STOP
mkdir -p "$REP"
echo $$ >"$PIDF"
echo "$(date -Is) outer start pid=$$" >>"$LOG"
alive() { pgrep -f "/root/projects/hrfont/scripts/hrfont_keepalive.sh" >/dev/null 2>&1; }
while true; do
  if [[ -f "$STOP" ]]; then
    echo "$(date -Is) STOP; outer exit" >>"$LOG"
    exit 0
  fi
  if ! alive; then
    echo "$(date -Is) keepalive missing; start" >>"$LOG"
    setsid bash "$ROOT/scripts/hrfont_keepalive.sh" </dev/null >>"$REP/keepalive.stdout" 2>&1 &
  fi
  sleep 60
done
