#!/usr/bin/env bash
# Keep HR-Font Stage A formal v2 alive. Stop via: touch reports/hrfont_overnight/STOP_FORMAL
set -u
trap '' HUP
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight
LOG=$REP/keepalive_formal.log
PIDF=$REP/keepalive_formal.pid
STOP=$REP/STOP_FORMAL
mkdir -p "$REP" "$ROOT/runs/e2_stageA96_formal"

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }

if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null && [[ "$old" != "$$" ]]; then
    hb "formal keepalive already running pid=$old; exit"
    exit 0
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
print(cands[0][1] if cands and cands[0][0]>=8000 else "")
'
}

start_train() {
  if alive "hrfont_e2_stageA96_formal_train.py"; then
    hb "formal train already running"
    return 0
  fi
  if [[ -f "$STOP" ]]; then
    hb "STOP_FORMAL present; not starting"
    return 0
  fi
  gpu=$(pick_gpu)
  if [[ -z "$gpu" ]]; then
    hb "no GPU >=8GB free; wait"
    return 1
  fi
  hb "start formal Stage A on GPU $gpu"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
    nohup "$PY" -u "$ROOT/scripts/hrfont_e2_stageA96_formal_train.py" \
    >>"$REP/e2_stageA_formal.stdout" 2>&1 &
  echo $! >"$REP/e2_stageA_formal.pid"
  hb "formal train pid=$(cat "$REP/e2_stageA_formal.pid") gpu=$gpu"
}

hb "formal keepalive start pid=$$"
start_train || true
backoff=20

while true; do
  if [[ -f "$STOP" ]]; then
    hb "STOP_FORMAL seen; keepalive halt"
    exit 0
  fi
  if ! alive "hrfont_e2_stageA96_formal_train.py"; then
    hb "formal train not running; restart in ${backoff}s"
    sleep "$backoff"
    if start_train; then backoff=20; else backoff=$(( backoff < 300 ? backoff * 2 : 300 )); fi
  else
    backoff=20
  fi
  sleep 120
done
