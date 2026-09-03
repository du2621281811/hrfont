#!/usr/bin/env bash
# Recover pipeline: eval Stage A + dropout ablations + Stage B
set -u
trap '' HUP
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight
ABL=$REP/dropout_ablation
LOG=$REP/recover_pipeline.log
mkdir -p "$REP/formal_eval" "$ABL"

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }

run_eval() {
  local gpu=$1 ckpt=$2 out=$3
  hb "EVAL gpu=$gpu ckpt=$(basename "$ckpt") -> $(basename "$out")"
  CUDA_VISIBLE_DEVICES=$gpu HRFONT_STAGEA_CKPT="$ckpt" \
    "$PY" "$ROOT/scripts/hrfont_e2_official_eval.py" \
      --ckpt "$ckpt" --out "$out" >>"$LOG" 2>&1
  return $?
}

hb "=== recover pipeline start ==="

# 1) Stage A @80k official eval (GPU3)
CK80="$ROOT/runs/e2_stageA96_formal/step_80000.pt"
if [[ ! -f "$REP/formal_eval/metrics_step_80000.json" ]]; then
  run_eval 3 "$CK80" "$REP/formal_eval/metrics_step_80000.json" || hb "WARN: 80k eval failed rc=$?"
else
  hb "skip 80k eval (exists)"
fi

# 2) Dropout ablation evals (GPU2) — weights already trained
for id in R1_cfg01_d01 R2_cfg01_d00 R3_style025_d025 R4_cfg025_d025; do
  out="$ABL/eval_${id}.json"
  ck="$ROOT/runs/dropout_ablation/${id}/step_10000.pt"
  [[ -f "$ck" ]] || ck="$ROOT/runs/dropout_ablation/${id}/last.pt"
  if [[ -f "$out" ]]; then
    hb "skip $id eval (exists)"
    continue
  fi
  if [[ ! -f "$ck" ]]; then
    hb "WARN: missing ckpt for $id"
    continue
  fi
  run_eval 2 "$ck" "$out" || hb "WARN: $id eval failed rc=$?"
done

# 3) Summarize dropout
"$PY" "$ROOT/scripts/hrfont_e2_dropout_ablation_summarize.py" >>"$LOG" 2>&1
hb "dropout summarize done"

# 4) Plan + launch Stage B from 80k eval
MET="$REP/formal_eval/metrics_step_80000.json"
if [[ -f "$MET" ]]; then
  "$PY" - <<'PY' >>"$LOG" 2>&1
import json, os, subprocess
from pathlib import Path
ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
met = json.loads((REP / "formal_eval/metrics_step_80000.json").read_text())
ft = met.get("ft_cnstyle", {}).get("L1_mean")
a = met.get("stageA_formal", {}).get("L1_mean")
ck = str(ROOT / "runs/e2_stageA96_formal/step_80000.pt")
plan = {"stageA_ckpt": ck, "ft_l1": ft, "stageA_l1": a}
if ft is None or a is None:
    plan.update(max_steps=20000, reason="eval 缺指标；默认 Stage B")
elif a < ft:
    plan.update(max_steps=40000, reason=f"Stage A 胜 ft (Δ={ft-a:.4f})")
elif a <= ft * 1.08:
    plan.update(max_steps=25000, reason=f"Stage A 接近 ft ({a:.4f} vs {ft:.4f})")
else:
    plan.update(max_steps=15000, reason=f"Stage A 落后 ft → 短训 Stage B")
(REP / "STAGE_B_PLAN.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
print("STAGE_B_PLAN", plan)
# update supervisor
sup = json.loads((REP / "SUPERVISOR.json").read_text()) if (REP / "SUPERVISOR.json").exists() else {}
sup.update(phase="stage_b", best_eval={"step": 80000, "ft_l1": ft, "stageA_l1": a, "delta": (ft-a) if ft and a else None},
           stage_b_plan=plan, stage_b_max=plan["max_steps"], eval_done=list(set(sup.get("eval_done", []) + [80000])))
(REP / "SUPERVISOR.json").write_text(json.dumps(sup, indent=2, ensure_ascii=False))
PY
  hb "Stage B plan written"
  # Stop formal keepalive (Stage A done)
  touch "$REP/STOP_FORMAL" 2>/dev/null || true
  export HRFONT_STAGEA_CKPT="$CK80"
  export HRFONT_STAGE_B_MAX="$(python3 -c "import json; print(json.load(open('$REP/STAGE_B_PLAN.json'))['max_steps'])")"
  if ! pgrep -f "hrfont_stageB_keepalive.sh" >/dev/null; then
    hb "launch Stage B keepalive max=$HRFONT_STAGE_B_MAX"
    nohup bash "$ROOT/scripts/hrfont_stageB_keepalive.sh" >>"$REP/keepalive_stageB.log" 2>&1 &
  fi
else
  hb "WARN: no 80k metrics — Stage B not launched"
fi

hb "=== recover pipeline COMPLETE ==="
