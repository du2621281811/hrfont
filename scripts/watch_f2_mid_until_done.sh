#!/usr/bin/env bash
# Offline watchdog: keep F2 mid-eval running until DONE. No network needed.
# Re-launches parallel runner on crash; skip-existing keeps PNGs stable.
set -u
ROOT=/root/projects/hrfont
OUT=$ROOT/reports/f03_test16_strat
LOG=$OUT/logs
RUNNER=$ROOT/scripts/launch_f2_mid_eval_parallel.sh
mkdir -p "$LOG"
export PYTHONUNBUFFERED=1
cd "$ROOT"

echo "[watch] $(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ starting" | tee -a "$LOG/watchdog.log"
echo $$ >"$LOG/watchdog.pid"

is_done() {
  local f
  f=$(find "$OUT/preds/F2_75000" -name '*.png' 2>/dev/null | wc -l)
  [ "$f" -ge 752 ] || return 1
  /root/miniforge3/envs/boogu/bin/python - <<'PY'
from pathlib import Path
root = Path("/root/projects/hrfont/reports/f03_test16_strat/preds")
steps = [5000,10000,20000,30000,40000,50000,60000,70000,75000]
for s in steps:
    mid = "F2_75000" if s == 75000 else f"F2_{s}"
    d = root / mid
    n = sum(1 for _ in d.rglob("*.png")) if d.is_dir() else 0
    if s == 75000:
        if n < 752: raise SystemExit(1)
    else:
        if n < 256: raise SystemExit(1)
raise SystemExit(0)
PY
}

runner_alive() {
  pgrep -f 'launch_f2_mid_eval_parallel.sh' >/dev/null
}

while true; do
  if is_done; then
    date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/F2_MID_DONE.txt"
    echo F2_MID_DONE >>"$OUT/F2_MID_DONE.txt"
    echo "[watch] $(date -u +%Y-%m-%dT%H:%M:%SZ) COMPLETE" | tee -a "$LOG/watchdog.log"
    # refresh pages once more
    /root/miniforge3/envs/boogu/bin/python scripts/eval_f03_test16_strat.py gallery >>"$LOG/watchdog.log" 2>&1 || true
    /root/miniforge3/envs/boogu/bin/python scripts/eval_f03_test16_strat.py timeline --arm F2 --html-only >>"$LOG/watchdog.log" 2>&1 || true
    # metrics if missing
    if [ ! -f "$OUT/metrics_summary.json" ] || ! grep -q F2_75000 "$OUT/metrics_summary.json" 2>/dev/null; then
      CUDA_VISIBLE_DEVICES=0 /root/miniforge3/envs/boogu/bin/python scripts/eval_f03_test16_strat.py metrics --lpips --device cuda:0 >>"$LOG/watchdog.log" 2>&1 || true
      /root/miniforge3/envs/boogu/bin/python scripts/eval_f03_test16_strat.py gallery >>"$LOG/watchdog.log" 2>&1 || true
    fi
    exit 0
  fi

  if ! runner_alive; then
    echo "[watch] $(date -u +%Y-%m-%dT%H:%M:%SZ) relaunch parallel runner" | tee -a "$LOG/watchdog.log"
    nohup bash "$RUNNER" >>"$LOG/parallel_runner.stdout" 2>&1 &
    echo $! >"$LOG/parallel_runner.pid"
  fi

  # progress breadcrumb
  t=$(find "$OUT/preds" -maxdepth 1 -type d -name 'F2_*' -print0 2>/dev/null | xargs -0 -I{} find {} -name '*.png' 2>/dev/null | wc -l)
  f=$(find "$OUT/preds/F2_75000" -name '*.png' 2>/dev/null | wc -l)
  echo "[watch] $(date -u +%Y-%m-%dT%H:%M:%SZ) timeline_pngs~$t full=$f f2train=$(cat $ROOT/runs/F2-DELTARSI-A-S3407/heartbeat.json 2>/dev/null)" >>"$LOG/watchdog.log"

  # ensure local http for offline browse (ignore fail)
  if ! pgrep -f 'http.server 8767' >/dev/null; then
    nohup python3 -m http.server 8767 --bind 0.0.0.0 --directory "$OUT" >/tmp/f03_8767.log 2>&1 &
  fi

  sleep 90
done
