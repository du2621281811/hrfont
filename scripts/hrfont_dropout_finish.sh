#!/usr/bin/env bash
# Finish dropout evals R3+R4 after R1/R2, summarize, monitor Stage B
set -u
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight
ABL=$REP/dropout_ablation
LOG=$REP/dropout_finish.log

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }

wait_eval() {
  local id=$1
  local out="$ABL/eval_${id}.json"
  while [[ ! -f "$out" ]]; do
    if pgrep -f "official_eval.py.*${id}" >/dev/null; then
      sleep 120
    else
      hb "WARN $id eval died without output"
      return 1
    fi
  done
  hb "$id done"
}

hb "dropout_finish start"
for id in R1_cfg01_d01 R2_cfg01_d00; do
  wait_eval "$id" || true
done

for id in R3_style025_d025 R4_cfg025_d025; do
  out="$ABL/eval_${id}.json"
  ck="$ROOT/runs/dropout_ablation/${id}/step_10000.pt"
  [[ -f "$out" ]] && continue
  hb "eval $id"
  CUDA_VISIBLE_DEVICES=2 HRFONT_STAGEA_CKPT="$ck" \
    "$PY" "$ROOT/scripts/hrfont_e2_official_eval.py" --ckpt "$ck" --out "$out" --skip-ft >>"$LOG" 2>&1
done

"$PY" "$ROOT/scripts/hrfont_e2_dropout_ablation_summarize.py" >>"$LOG" 2>&1
"$PY" "$ROOT/scripts/hrfont_expert_validate.py" >>"$LOG" 2>&1
hb "dropout_finish COMPLETE"
