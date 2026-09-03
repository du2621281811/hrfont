#!/usr/bin/env bash
# Supervise Stage B eval + downstream until PIPELINE_COMPLETE.
# Survives disconnects; relaunches failed steps. Stop: touch reports/hrfont_overnight/STOP_ALL
set -u
trap '' HUP INT
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight
FE=$REP/formal_eval
LOG=$REP/posteval_supervisor.log
PIDF=$REP/posteval_supervisor.pid
DONE=$REP/PIPELINE_COMPLETE
STOP=$REP/STOP_ALL
MET=$FE/metrics_stageB_25000.json

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }

if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" 2>/dev/null || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null && [[ "$old" != "$$" ]]; then
    hb "posteval_supervisor already pid=$old"
    exit 0
  fi
fi
echo $$ >"$PIDF"

eval_running() { pgrep -f "[p]ython.*hrfont_e2_stageB_official_eval.py" >/dev/null 2>&1; }

run_stageB_eval() {
  hb "launch Stage B official eval"
  for gpu in 3 2 0; do
    if CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$ROOT/scripts/hrfont_e2_stageB_official_eval.py" \
        --out "$MET" >>"$REP/eval_stageB.log" 2>&1; then
      hb "Stage B eval OK gpu=$gpu"
      return 0
    fi
    hb "WARN Stage B eval failed gpu=$gpu"
  done
  return 1
}

run_downstream() {
  hb "gap stratified"
  "$PY" "$ROOT/scripts/hrfont_gap_stratified_analysis.py" >>"$LOG" 2>&1 || hb "WARN gap analysis"

  if [[ ! -f "$FE/style_metrics.json" ]]; then
    hb "style metrics"
    for gpu in 2 3; do
      if CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/scripts/hrfont_style_metrics.py" >>"$REP/style_metrics.log" 2>&1; then
        hb "style_metrics OK gpu=$gpu"
        break
      fi
      hb "WARN style_metrics failed gpu=$gpu"
    done
  fi

  hb "formal preview + handoff"
  "$PY" "$ROOT/scripts/hrfont_formal_preview_build.py" >>"$LOG" 2>&1 || true
  "$PY" "$ROOT/scripts/hrfont_write_handoff.py" >>"$LOG" 2>&1 || true
  "$PY" "$ROOT/scripts/hrfont_expert_validate.py" >>"$LOG" 2>&1 || true
  touch "$DONE"
  hb "PIPELINE_COMPLETE"
}

hb "posteval_supervisor start pid=$$"

while true; do
  if [[ -f "$STOP" ]]; then
    hb "STOP_ALL — exit"
    exit 0
  fi
  if [[ -f "$DONE" ]]; then
    hb "already complete"
    exit 0
  fi

  if [[ -f "$MET" ]]; then
    hb "metrics exist — run downstream"
    run_downstream
    exit 0
  fi

  if eval_running; then
    hb "Stage B eval running…"
  else
    hb "Stage B eval not running — (re)launch"
    if ! run_stageB_eval; then
      hb "eval failed all GPUs; retry in 120s"
      sleep 120
      continue
    fi
    run_downstream
    exit 0
  fi

  sleep 120
done
