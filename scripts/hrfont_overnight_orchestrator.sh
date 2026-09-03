#!/usr/bin/env bash
# Full unattended pipeline (REVIEW_SUMMARY_AND_PLAN.md §7).
# Stage B train → B eval → gap → style metrics → formal preview → handoff
set -u
trap '' HUP INT
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight
ABL=$REP/dropout_ablation
FE=$REP/formal_eval
LOG=$REP/orchestrator_unattended.log
PIDF=$REP/orchestrator_unattended.pid
DONE=$REP/PIPELINE_COMPLETE
CK80=$ROOT/runs/e2_stageA96_formal/step_80000.pt
MAX_B=25000

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }
decide() { echo "$(date -Is) $*" | tee -a "$REP/AUTO_DECISIONS.log"; }

write_status() {
  local sb_step="—" phase="running"
  [[ -f "$REP/STATUS_STAGE_B.json" ]] && sb_step=$("$PY" -c "import json;print(json.load(open('$REP/STATUS_STAGE_B.json')).get('step','—'))" 2>/dev/null || echo "—")
  [[ -f "$DONE" ]] && phase=complete
  local b_eval=— gap=— sm=— prev=—
  [[ -f "$FE/metrics_stageB_25000.json" ]] && b_eval=done
  [[ -f "$FE/gap_stratified.json" ]] && gap=done
  [[ -f "$FE/style_metrics.json" ]] && sm=done
  [[ -f "$REP/formal_preview/index.html" ]] && prev=done
  cat >"$REP/STATUS_OVERNIGHT.md" <<EOF
# HR-Font 无人值守状态

更新时间：$(date -Is)

| 任务 | 状态 |
|------|------|
| Stage A @80k | ✅ ft=0.081 A=0.087 |
| Stage B 训练 | **${sb_step} / ${MAX_B}** |
| Stage B eval | **${b_eval}** |
| Gap 分层 | **${gap}** |
| 风格指标原型 | **${sm}** |
| Formal 预览页 | **${prev}** |
| 编排器 | **${phase}** |

回来后看：\`READ_ME_FIRST.md\` · \`REVIEW_SUMMARY_AND_PLAN.md\` · \`formal_preview/index.html\`
EOF
}

if [[ -f "$PIDF" ]]; then
  old=$(cat "$PIDF" 2>/dev/null || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null && [[ "$old" != "$$" ]]; then
    hb "orchestrator already pid=$old"
    exit 0
  fi
fi
echo $$ >"$PIDF"

[[ -f "$DONE" ]] && { hb "PIPELINE_COMPLETE — exit"; exit 0; }
[[ -f "$REP/STOP_ALL" ]] && { hb "STOP_ALL — exit"; exit 0; }

hb "=== orchestrator v2 start ==="
write_status

# ── Dropout R1 + summarize (idempotent) ──
if [[ ! -f "$ABL/eval_R1_cfg01_d01.json" ]]; then
  hb "R1 eval"
  ck="$ROOT/runs/dropout_ablation/R1_cfg01_d01/step_10000.pt"
  CUDA_VISIBLE_DEVICES=2 HRFONT_STAGEA_CKPT="$ck" \
    "$PY" "$ROOT/scripts/hrfont_e2_official_eval.py" \
      --ckpt "$ck" --out "$ABL/eval_R1_cfg01_d01.json" --skip-ft \
    >>"$ABL/R1_cfg01_d01.eval.log" 2>&1 || hb "WARN R1 eval"
fi
"$PY" "$ROOT/scripts/hrfont_e2_dropout_ablation_summarize.py" >>"$LOG" 2>&1 || true
"$PY" "$ROOT/scripts/hrfont_dropout_auto_verdict.py" >>"$LOG" 2>&1 || true

# ── Stage B train ──
if [[ ! -f "$REP/STOP_STAGE_B" ]]; then
  export HRFONT_STAGEA_CKPT="$CK80"
  export HRFONT_STAGE_B_MAX="$MAX_B"
  if ! pgrep -f "hrfont_stageB_keepalive.sh" >/dev/null; then
    hb "launch Stage B keepalive"
    nohup bash "$ROOT/scripts/hrfont_stageB_keepalive.sh" >>"$REP/keepalive_stageB.log" 2>&1 &
    sleep 5
  fi
  hb "wait Stage B -> $MAX_B"
  while [[ ! -f "$REP/STOP_ALL" ]]; do
    step=0
    [[ -f "$ROOT/runs/e2_stageB96/last.pt" ]] && \
      step=$("$PY" -c "import torch;print(int(torch.load('$ROOT/runs/e2_stageB96/last.pt',map_location='cpu',weights_only=False).get('step',0)))" 2>/dev/null || echo 0)
    write_status
    if [[ "$step" -ge "$MAX_B" ]] && ! pgrep -f "hrfont_e2_stageB96_train.py" >/dev/null; then
      hb "Stage B done step=$step"
      break
    fi
    if ! pgrep -f "hrfont_stageB_keepalive.sh" >/dev/null && ! pgrep -f "hrfont_e2_stageB96_train.py" >/dev/null; then
      hb "WARN relaunch keepalive"
      nohup bash "$ROOT/scripts/hrfont_stageB_keepalive.sh" >>"$REP/keepalive_stageB.log" 2>&1 &
    fi
    sleep 300
  done
fi

[[ -f "$REP/STOP_ALL" ]] && exit 0

# ── Stage B official eval (~90min) ──
if [[ ! -f "$FE/metrics_stageB_25000.json" ]]; then
  hb "Stage B official eval"
  for gpu in 3 2 0; do
    if CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/scripts/hrfont_e2_stageB_official_eval.py" \
        --out "$FE/metrics_stageB_25000.json" >>"$REP/eval_stageB.log" 2>&1; then
      decide "Stage B eval OK gpu=$gpu"
      break
    fi
    hb "WARN Stage B eval failed gpu=$gpu retry"
  done
fi

# ── Gap stratified ──
if [[ ! -f "$FE/gap_stratified.json" ]]; then
  hb "gap stratified analysis"
  "$PY" "$ROOT/scripts/hrfont_gap_stratified_analysis.py" >>"$LOG" 2>&1 || hb "WARN gap analysis"
fi

# ── Style metrics prototype (64-grid subset) ──
if [[ ! -f "$FE/style_metrics.json" ]]; then
  hb "style metrics prototype"
  for gpu in 2 3; do
    if CUDA_VISIBLE_DEVICES=$gpu "$PY" "$ROOT/scripts/hrfont_style_metrics.py" >>"$REP/style_metrics.log" 2>&1; then
      decide "style_metrics OK gpu=$gpu"
      break
    fi
    hb "WARN style_metrics failed gpu=$gpu"
  done
fi

# ── Formal preview page ──
hb "formal preview build"
"$PY" "$ROOT/scripts/hrfont_formal_preview_build.py" >>"$LOG" 2>&1 || true

# ── Handoff ──
hb "write handoff"
"$PY" "$ROOT/scripts/hrfont_write_handoff.py" >>"$LOG" 2>&1
"$PY" "$ROOT/scripts/hrfont_expert_validate.py" >>"$LOG" 2>&1 || true

touch "$DONE"
decide "PIPELINE_COMPLETE — see READ_ME_FIRST.md formal_preview/"
write_status
hb "=== orchestrator v2 COMPLETE ==="
