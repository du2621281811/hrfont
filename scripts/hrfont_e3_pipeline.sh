#!/usr/bin/env bash
# E3 pipeline: train RSI←Ec(style) control @10k → official eval → update mentor page.
# Survives disconnect: run via nohup.
set -euo pipefail
ROOT=/root/projects/hrfont
REP=$ROOT/reports/hrfont_overnight
PY=/root/miniforge3/envs/boogu/bin/python
LOG=$REP/e3_pipeline.log
CK=$ROOT/runs/e3_rsi_style_control/step_10000.pt
MET=$REP/formal_eval/metrics_e3_10000.json
DONE=$REP/E3_COMPLETE

hb() { echo "$(date -Iseconds) $*" | tee -a "$LOG"; }

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' '$1!=1 {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'
}

hb "E3 pipeline start"

if [[ -f "$DONE" ]]; then
  hb "E3_COMPLETE exists — skip"
  exit 0
fi

GPU=$(pick_gpu)
hb "train GPU=$GPU"

if [[ ! -f "$CK" ]]; then
  hb "launch E3 train 10k"
  CUDA_VISIBLE_DEVICES=$GPU HRFONT_E3_MAX=10000 \
    "$PY" -u "$ROOT/scripts/hrfont_e3_rsi_style_train.py" >>"$LOG" 2>&1
  hb "train done"
else
  hb "ckpt exists $CK"
fi

[[ -f "$CK" ]] || CK=$ROOT/runs/e3_rsi_style_control/last.pt
if [[ ! -f "$CK" ]]; then
  hb "ERROR no E3 checkpoint"
  exit 1
fi

GPU=$(pick_gpu)
hb "eval GPU=$GPU"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$ROOT/scripts/hrfont_e2_official_eval.py" \
  --rsi-style --ckpt "$CK" --out "$MET" --skip-ft >>"$LOG" 2>&1
hb "eval done"

"$PY" "$ROOT/scripts/hrfont_protocol_audit_build.py" >>"$LOG" 2>&1 || true
"$PY" "$ROOT/scripts/hrfont_formal_preview_build.py" >>"$LOG" 2>&1 || true

touch "$DONE"
hb "E3 pipeline COMPLETE"
