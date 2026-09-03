#!/usr/bin/env bash
# Keep HR-Font Stage A 96 alive with NO wall-clock deadline.
# Restarts on crash. Does not kill native256 walkaway / GPU1.
# Touch reports/hrfont_overnight/STOP to halt cleanly.
set -u
trap '' HUP
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight
LOG=$REP/keepalive.log
PIDF=$REP/keepalive.pid
STOP=$REP/STOP
mkdir -p "$REP" "$ROOT/runs/e2_stageA96"

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }

if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    if [[ "$old" != "$$" ]]; then
      hb "keepalive already running pid=$old; exit"
      exit 0
    fi
  fi
fi
echo $$ >"$PIDF"

alive() { pgrep -f "$1" >/dev/null 2>&1; }

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | "$PY" -c '
import sys
cands=[]
for line in sys.stdin:
    parts=[x.strip() for x in line.split(",")]
    if len(parts)<2: continue
    i,f=int(parts[0]),int(parts[1])
    if i==1: continue
    cands.append((f,i))
cands.sort(reverse=True)
print(cands[0][1] if cands and cands[0][0]>=6000 else "")
'
}

start_train() {
  if alive "hrfont_e2_stageA96_train.py"; then
    hb "train already running"
    return 0
  fi
  if [[ -f "$STOP" ]]; then
    hb "STOP present; not starting train"
    return 0
  fi
  gpu=$(pick_gpu)
  if [[ -z "$gpu" ]]; then
    hb "no GPU with >=6GB free (skip GPU1); wait"
    return 1
  fi
  hb "start Stage A 96 on GPU $gpu"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
    nohup "$PY" -u "$ROOT/scripts/hrfont_e2_stageA96_train.py" \
    >>"$REP/e2_stageA.stdout" 2>&1 &
  echo $! >"$REP/e2_stageA.pid"
  hb "train pid=$(cat "$REP/e2_stageA.pid") gpu=$gpu"
}

hb "keepalive start pid=$$ (NO 09:00 cutoff; runs through tomorrow+)"
start_train || true
backoff=20

while true; do
  if [[ -f "$STOP" ]]; then
    hb "STOP file seen; leave train running unless it exits; keepalive halt"
    exit 0
  fi
  if ! alive "hrfont_e2_stageA96_train.py"; then
    hb "train not running; restart in ${backoff}s"
    sleep "$backoff"
    if start_train; then
      backoff=20
    else
      backoff=$(( backoff < 300 ? backoff * 2 : 300 ))
    fi
  else
    backoff=20
  fi
  "$PY" - <<'PY' >>"$LOG" 2>&1 || true
from pathlib import Path
import json, time, subprocess
rep=Path("/root/projects/hrfont/reports/hrfont_overnight")
out=Path("/root/projects/hrfont/runs/e2_stageA96")
cks=sorted(out.glob("step_*.pt"))
try:
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.free,utilization.gpu","--format=csv,noheader"], text=True)
except Exception as e:
    gpu=str(e)
st={
  "t": time.strftime("%Y-%m-%d %H:%M:%S"),
  "e0": True,
  "e1_smoke": (rep/"e1_smoke.json").exists(),
  "train": "e2_stageA96",
  "deadline": None,
  "train_alive": True,
  "train_ckpt": str(out/"last.pt") if (out/"last.pt").exists() else (str(cks[-1]) if cks else None),
  "n_ckpt": len(cks),
  "gpu": gpu.strip().splitlines() if isinstance(gpu,str) else [],
}
# merge step from train STATUS if present
prev=rep/"STATUS.json"
if prev.exists():
    try:
        old=json.loads(prev.read_text())
        for k in ("step","loss","mem_max_miB","fatal"):
            if k in old: st[k]=old[k]
    except Exception:
        pass
(rep/"STATUS.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
print("heartbeat", st["t"], "ckpt", st["train_ckpt"], "step", st.get("step"))
PY
  sleep 120
done
