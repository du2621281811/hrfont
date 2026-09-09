#!/usr/bin/env bash
# Parallel F2 mid-eval on GPU0/1/2. Per-item seed3407 → shard-invariant PNGs.
# Never uses GPU3. Safe to re-run (skip-existing). Offline-friendly.
set -u
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
OUT=$ROOT/reports/f03_test16_strat
LOG=$OUT/logs
export PYTHONUNBUFFERED=1
mkdir -p "$LOG"
cd "$ROOT"
LOCK=$LOG/parallel_runner.lock

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[parallel] another runner holds $LOCK; exit"
  exit 0
fi

refuse_gpu3() {
  if [ "${1:-}" = "3" ]; then
    echo "refusing GPU3"; exit 2
  fi
}

count_timeline() {
  "$PY" - <<'PY'
from pathlib import Path
root = Path("/root/projects/hrfont/reports/f03_test16_strat/preds")
steps = [5000,10000,20000,30000,40000,50000,60000,70000,75000]
n = 0
for s in steps:
    mid = "F2_75000" if s == 75000 else f"F2_{s}"
    d = root / mid
    # timeline uses 16 fonts × 16 chars; count any pngs present for progress
    if d.is_dir():
        n += min(256, sum(1 for _ in d.rglob("*.png")))
print(n)
PY
}

timeline_complete() {
  "$PY" - <<'PY'
from pathlib import Path
root = Path("/root/projects/hrfont/reports/f03_test16_strat/preds")
steps = [5000,10000,20000,30000,40000,50000,60000,70000,75000]
ok = True
for s in steps:
    mid = "F2_75000" if s == 75000 else f"F2_{s}"
    d = root / mid
    n = sum(1 for _ in d.rglob("*.png")) if d.is_dir() else 0
    if n < 256:
        ok = False
        break
print("1" if ok else "0")
PY
}

count_full() {
  find "$OUT/preds/F2_75000" -name '*.png' 2>/dev/null | wc -l
}

timeline_target() {
  "$PY" - <<'PY'
from pathlib import Path
run = Path("/root/projects/hrfont/runs/F2-DELTARSI-A-S3407")
steps = [5000,10000,20000,30000,40000,50000,60000,70000,75000]
have = sum(1 for s in steps if (run/f"global_step_{s}"/"unet.pth").is_file())
print(16 * 16 * have)
PY
}

echo "[parallel] $(date -u +%Y-%m-%dT%H:%M:%SZ) start"
"$PY" scripts/eval_f03_test16_strat.py timeline --arm F2 --html-only
"$PY" scripts/eval_f03_test16_strat.py gallery

# ---- Phase A: timeline shards on GPU 0/1/2 (stagger to warm Ec page cache) ----
T_TARGET=$(timeline_target)
T_HAVE=$(count_timeline)
echo "[parallel] timeline png(capped) $T_HAVE / $T_TARGET"
if [ "$(timeline_complete)" != "1" ]; then
  for spec in "0:0/3" "1:1/3" "2:2/3"; do
    gpu=${spec%%:*}; shard=${spec##*:}
    refuse_gpu3 "$gpu"
    echo "[timeline] launch GPU$gpu shard=$shard"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_f03_test16_strat.py timeline \
      --arm F2 --device cuda:0 --shard "$shard" \
      >"$LOG/F2_timeline_g${gpu}_s${shard//\//of}.stdout" 2>&1 &
    echo $! >"$LOG/F2_timeline_g${gpu}.pid"
    # stagger: let first worker mmap Ec before others contend
    sleep 45
  done
  wait || true
  "$PY" scripts/eval_f03_test16_strat.py timeline --arm F2 --html-only
fi
T_HAVE=$(count_timeline)
echo "[parallel] timeline after phase A: $T_HAVE / $T_TARGET complete=$(timeline_complete)"
if [ "$(timeline_complete)" != "1" ]; then
  echo "[parallel] timeline incomplete; retry serial gaps on GPU0"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval_f03_test16_strat.py timeline \
    --arm F2 --device cuda:0 --shard 0/1 \
    >"$LOG/F2_timeline_retry.stdout" 2>&1 || true
fi

# ---- Phase B: full F2@75k stratified, 3-way font shard ----
F_TARGET=752
F_HAVE=$(count_full)
echo "[parallel] F2_75000 png $F_HAVE / $F_TARGET"
if [ "$F_HAVE" -lt "$F_TARGET" ]; then
  for spec in "0:0/3" "1:1/3" "2:2/3"; do
    gpu=${spec%%:*}; shard=${spec##*:}
    refuse_gpu3 "$gpu"
    echo "[full] launch GPU$gpu shard=$shard"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/eval_f03_test16_strat.py generate \
      --method F2_75000 --device cuda:0 --shard "$shard" \
      >"$LOG/F2_75000_g${gpu}_s${shard//\//of}.stdout" 2>&1 &
    echo $! >"$LOG/F2_75000_g${gpu}.pid"
    sleep 20
  done
  wait || true
fi
F_HAVE=$(count_full)
if [ "$F_HAVE" -lt "$F_TARGET" ]; then
  echo "[parallel] full incomplete; retry gaps GPU0"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval_f03_test16_strat.py generate \
    --method F2_75000 --device cuda:0 --shard 0/1 \
    >"$LOG/F2_75000_retry.stdout" 2>&1 || true
fi

# ---- Phase C: metrics + gallery ----
echo "[parallel] metrics"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval_f03_test16_strat.py metrics --lpips --device cuda:0 \
  >"$LOG/metrics_f2.stdout" 2>&1 || true
"$PY" scripts/eval_f03_test16_strat.py gallery
"$PY" scripts/eval_f03_test16_strat.py timeline --arm F2 --html-only

T_HAVE=$(count_timeline)
F_HAVE=$(count_full)
echo "[parallel] final timeline=$T_HAVE/$T_TARGET full=$F_HAVE/$F_TARGET tl_ok=$(timeline_complete)"
if [ "$(timeline_complete)" = "1" ] && [ "$F_HAVE" -ge "$F_TARGET" ]; then
  date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/F2_MID_DONE.txt"
  echo F2_MID_DONE >>"$OUT/F2_MID_DONE.txt"
  echo "[parallel] DONE"
  exit 0
fi
echo "[parallel] NOT DONE yet" >&2
exit 1
