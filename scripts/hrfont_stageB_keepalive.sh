#!/usr/bin/env bash
# Keep Stage B SupportAdapter training alive.
set -u
trap '' HUP
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight
LOG=$REP/keepalive_stageB.log
PIDF=$REP/keepalive_stageB.pid
STOP=$REP/STOP_STAGE_B
mkdir -p "$REP" "$ROOT/runs/e2_stageB96"

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }

if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" 2>/dev/null || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null && [[ "$old" != "$$" ]]; then
    hb "stageB keepalive already pid=$old; exit"
    exit 0
  fi
fi
echo $$ >"$PIDF"

alive() { pgrep -f "$1" >/dev/null 2>&1; }

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | "$PY" -c '
import sys
c=[]
for line in sys.stdin:
 p=[x.strip() for x in line.split(",")]
 if len(p)<2: continue
 i,f=int(p[0]),int(p[1])
 if i==1: continue
 c.append((f,i))
c.sort(reverse=True)
print(c[0][1] if c and c[0][0]>=8000 else "")
'
}

MAX_B="${HRFONT_STAGE_B_MAX:-30000}"
STAGEA_CKPT="${HRFONT_STAGEA_CKPT:-$ROOT/runs/e2_stageA96_formal/last.pt}"

start_train() {
  if alive "hrfont_e2_stageB96_train.py"; then
    hb "stageB train already running"
    return 0
  fi
  if [[ -f "$STOP" ]] || [[ -f "$REP/STOP_ALL" ]]; then
    hb "STOP set; not starting stageB"
    return 0
  fi
  gpu=$(pick_gpu)
  if [[ -z "$gpu" ]]; then
    hb "no GPU; wait"
    return 1
  fi
  hb "start Stage B gpu=$gpu max=$MAX_B init=$STAGEA_CKPT"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$gpu" \
  HRFONT_STAGE_B_MAX="$MAX_B" \
  HRFONT_STAGEA_CKPT="$STAGEA_CKPT" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
    nohup "$PY" -u "$ROOT/scripts/hrfont_e2_stageB96_train.py" \
    >>"$REP/e2_stageB.stdout" 2>&1 &
  echo $! >"$REP/e2_stageB.pid"
}

hb "stageB keepalive start"
start_train || true
backoff=20
while true; do
  if [[ -f "$STOP" ]] || [[ -f "$REP/STOP_ALL" ]]; then
    hb "STOP; halt"
    exit 0
  fi
  if ! alive "hrfont_e2_stageB96_train.py"; then
    # done if reached max steps
    if [[ -f "$ROOT/runs/e2_stageB96/last.pt" ]]; then
      step=$("$PY" -c "import torch; print(int(torch.load('$ROOT/runs/e2_stageB96/last.pt',map_location='cpu',weights_only=False).get('step',0)))" 2>/dev/null || echo 0)
      if [[ "$step" -ge "$MAX_B" ]]; then
        hb "stageB reached max=$MAX_B; done"
        exit 0
      fi
    fi
    hb "stageB not running; restart in ${backoff}s"
    sleep "$backoff"
    if start_train; then backoff=20; else backoff=$(( backoff<300 ? backoff*2 : 300 )); fi
  else
    backoff=20
  fi
  sleep 120
done
