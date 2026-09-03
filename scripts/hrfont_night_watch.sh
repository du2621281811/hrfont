#!/usr/bin/env bash
# Top-level overnight watchdog: keep formal train + supervisor + dropout ablation alive.
# Stop all: touch reports/hrfont_overnight/STOP_ALL
set -u
trap '' HUP
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight
ABL=$REP/dropout_ablation
LOG=$REP/night_watch.log
PIDF=$REP/night_watch.pid
STOP=$REP/STOP_ALL
mkdir -p "$REP" "$ABL" "$REP/formal_eval"

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }

if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" 2>/dev/null || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null && [[ "$old" != "$$" ]]; then
    hb "night_watch already running pid=$old; exit"
    exit 0
  fi
fi
echo $$ >"$PIDF"

alive() { pgrep -f "$1" >/dev/null 2>&1; }

write_status() {
  local sb_step="—" sb_loss="—" n_abl=0 r1="pending" orch="—"
  if [[ -f "$REP/STATUS_STAGE_B.json" ]]; then
    sb_step=$("$PY" -c "import json; d=json.load(open('$REP/STATUS_STAGE_B.json')); print(d.get('step','—'))" 2>/dev/null || echo "—")
    sb_loss=$("$PY" -c "import json; d=json.load(open('$REP/STATUS_STAGE_B.json')); print(d.get('loss','—'))" 2>/dev/null || echo "—")
  elif [[ -f "$ROOT/runs/e2_stageB96/last.pt" ]]; then
    sb_step=$("$PY" -c "import torch; print(int(torch.load('$ROOT/runs/e2_stageB96/last.pt',map_location='cpu',weights_only=False).get('step',0)))" 2>/dev/null || echo "—")
  fi
  [[ -f "$ABL/eval_R1_cfg01_d01.json" ]] && r1=done
  for id in R1_cfg01_d01 R2_cfg01_d00 R3_style025_d025 R4_cfg025_d025; do
    [[ -f "$ABL/eval_${id}.json" ]] && n_abl=$((n_abl + 1))
  done
  if [[ -f "$REP/PIPELINE_COMPLETE" ]]; then orch=complete; elif alive "hrfont_overnight_orchestrator.sh"; then orch=running; else orch=idle; fi
  cat >"$REP/STATUS_OVERNIGHT.md" <<EOF
# HR-Font 无人值守状态

更新时间：$(date -Is)

## 一眼进度

| 任务 | 状态 |
|------|------|
| Stage A @80k | ✅ 完成 · ft=0.081 · A=0.087 |
| Dropout 消融 eval | **${n_abl}/4**（R1=${r1}） |
| Stage B SupportAdapter | **${sb_step} / 25000** loss=${sb_loss} |
| 编排器 | **${orch}** |

## 回来后只看

\`\`\`bash
cat reports/hrfont_overnight/READ_ME_FIRST.md
cat reports/hrfont_overnight/REVIEW_SUMMARY_AND_PLAN.md
cat reports/hrfont_overnight/formal_preview/index.html
\`\`\`

## 日志

- \`orchestrator_unattended.log\` · \`keepalive_stageB.log\` · \`night_watch.log\`

## 紧急停止

\`\`\`bash
touch reports/hrfont_overnight/STOP_ALL
\`\`\`
EOF
}

ensure_keepalive() {
  if [[ -f "$REP/STOP_FORMAL" ]] || [[ -f "$STOP" ]]; then return 0; fi
  if ! alive "hrfont_formal_keepalive.sh"; then
    hb "restart formal keepalive"
    nohup bash "$ROOT/scripts/hrfont_formal_keepalive.sh" >>"$REP/keepalive_formal.log" 2>&1 &
  fi
}

ensure_supervisor() {
  if [[ -f "$STOP" ]]; then return 0; fi
  local phase
  phase=$( "$PY" -c "
import json
from pathlib import Path
p=Path('$REP/SUPERVISOR.json')
print(json.loads(p.read_text()).get('phase','') if p.exists() else '')
" 2>/dev/null || true)
  if [[ "$phase" == "all_complete" || "$phase" == "stage_b_stopped" ]]; then return 0; fi
  if ! alive "hrfont_experiment_supervisor.py"; then
    hb "restart experiment supervisor (A→B pipeline)"
    nohup "$PY" -u "$ROOT/scripts/hrfont_experiment_supervisor.py" >>"$REP/supervisor.log" 2>&1 &
  fi
}

ensure_stageB_keepalive() {
  if [[ -f "$STOP" ]] || [[ -f "$REP/STOP_STAGE_B" ]]; then return 0; fi
  local step=0 max=25000
  if [[ -f "$ROOT/runs/e2_stageB96/last.pt" ]]; then
    step=$("$PY" -c "import torch; print(int(torch.load('$ROOT/runs/e2_stageB96/last.pt',map_location='cpu',weights_only=False).get('step',0)))" 2>/dev/null || echo 0)
  fi
  [[ -f "$REP/STAGE_B_PLAN.json" ]] && max=$("$PY" -c "import json; print(json.load(open('$REP/STAGE_B_PLAN.json')).get('max_steps',25000))" 2>/dev/null || echo 25000)
  if [[ "$step" -ge "$max" ]] && ! alive "hrfont_e2_stageB96_train.py"; then return 0; fi
  if ! alive "hrfont_stageB_keepalive.sh" && ! alive "hrfont_overnight_orchestrator.sh"; then
    hb "restart Stage B keepalive"
    export HRFONT_STAGEA_CKPT="$ROOT/runs/e2_stageA96_formal/step_80000.pt"
    export HRFONT_STAGE_B_MAX="$max"
    nohup bash "$ROOT/scripts/hrfont_stageB_keepalive.sh" >>"$REP/keepalive_stageB.log" 2>&1 &
  fi
}

ensure_orchestrator() {
  if [[ -f "$STOP" ]]; then return 0; fi
  if [[ -f "$REP/PIPELINE_COMPLETE" ]]; then return 0; fi
  if ! alive "hrfont_overnight_orchestrator.sh"; then
    hb "restart unattended orchestrator"
    nohup bash "$ROOT/scripts/hrfont_overnight_orchestrator.sh" >>"$REP/orchestrator_unattended.log" 2>&1 &
  fi
}

ensure_posteval() {
  if [[ -f "$STOP" ]] || [[ -f "$REP/PIPELINE_COMPLETE" ]]; then return 0; fi
  if ! alive "hrfont_posteval_supervisor.sh"; then
    hb "restart posteval supervisor (B eval → gap → metrics → handoff)"
    nohup bash "$ROOT/scripts/hrfont_posteval_supervisor.sh" >>"$REP/posteval_supervisor.log" 2>&1 &
  fi
}

maybe_refresh_verdict() {
  if [[ -f "$REP/formal_eval/metrics_step_80000.json" ]]; then
    n=0
    for id in R1_cfg01_d01 R2_cfg01_d00 R3_style025_d025 R4_cfg025_d025; do
      [[ -f "$ABL/eval_${id}.json" ]] && n=$((n + 1))
    done
    if [[ "$n" -ge 3 ]]; then
      "$PY" "$ROOT/scripts/hrfont_e2_dropout_ablation_summarize.py" >>"$ABL/orchestrator.log" 2>&1 || true
    fi
  fi
}

hb "night_watch start pid=$$"
write_status

while true; do
  if [[ -f "$STOP" ]]; then
    hb "STOP_ALL — night_watch halt"
    write_status
    exit 0
  fi
  ensure_orchestrator
  ensure_posteval
  ensure_stageB_keepalive
  maybe_refresh_verdict
  write_status
  sleep 300
done
