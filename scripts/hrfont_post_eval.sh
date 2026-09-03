#!/usr/bin/env bash
# Wait for 80k eval, then dropout evals + Stage B launch
set -u
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight
ABL=$REP/dropout_ablation
LOG=$REP/post_eval.log
MET=$REP/formal_eval/metrics_step_80000.json
CK80=$ROOT/runs/e2_stageA96_formal/step_80000.pt

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }

hb "post_eval watcher start"

while [[ ! -f "$MET" ]]; do
  if ! pgrep -f "official_eval.py.*80000" >/dev/null; then
    hb "80k eval process gone but no metrics — check eval_80k.log"
    sleep 30
    if [[ ! -f "$MET" ]]; then
      hb "retry 80k eval"
      CUDA_VISIBLE_DEVICES=3 "$PY" "$ROOT/scripts/hrfont_e2_official_eval.py" \
        --ckpt "$CK80" --out "$MET" >>"$REP/eval_80k.log" 2>&1 &
    fi
  fi
  sleep 60
done

hb "80k eval done — $(python3 -c "import json; d=json.load(open('$MET')); print('ft',d['ft_cnstyle']['L1_mean'],'A',d['stageA_formal']['L1_mean'])")"

for id in R1_cfg01_d01 R2_cfg01_d00 R3_style025_d025 R4_cfg025_d025; do
  out="$ABL/eval_${id}.json"
  ck="$ROOT/runs/dropout_ablation/${id}/step_10000.pt"
  [[ -f "$ck" ]] || ck="$ROOT/runs/dropout_ablation/${id}/last.pt"
  if [[ -f "$out" ]]; then hb "skip $id"; continue; fi
  hb "eval $id"
  CUDA_VISIBLE_DEVICES=2 HRFONT_STAGEA_CKPT="$ck" \
    "$PY" "$ROOT/scripts/hrfont_e2_official_eval.py" --ckpt "$ck" --out "$out" --skip-ft >>"$LOG" 2>&1 || hb "WARN $id failed"
done

"$PY" "$ROOT/scripts/hrfont_e2_dropout_ablation_summarize.py" >>"$LOG" 2>&1

# Stage B plan + launch
"$PY" - <<'PY' >>"$LOG" 2>&1
import json, os, subprocess
from pathlib import Path
ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
met = json.loads((REP / "formal_eval/metrics_step_80000.json").read_text())
ft, a = met["ft_cnstyle"]["L1_mean"], met["stageA_formal"]["L1_mean"]
ck = str(ROOT / "runs/e2_stageA96_formal/step_80000.pt")
plan = {"stageA_ckpt": ck, "ft_l1": ft, "stageA_l1": a, "delta": ft - a}
if a < ft:
    plan.update(max_steps=40000, reason=f"Stage A 胜 ft (Δ={ft-a:.4f})")
elif a <= ft * 1.08:
    plan.update(max_steps=25000, reason=f"Stage A 接近 ft")
else:
    plan.update(max_steps=15000, reason="Stage A 落后 ft → 短训")
(REP / "STAGE_B_PLAN.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
sup = json.loads((REP / "SUPERVISOR.json").read_text()) if (REP / "SUPERVISOR.json").exists() else {}
sup.update(phase="stage_b", best_eval={"step": 80000, "ft_l1": ft, "stageA_l1": a, "delta": ft-a},
           stage_b_plan=plan, stage_b_max=plan["max_steps"])
(REP / "SUPERVISOR.json").write_text(json.dumps(sup, indent=2, ensure_ascii=False))
print("plan", plan)
PY

export HRFONT_STAGEA_CKPT="$CK80"
export HRFONT_STAGE_B_MAX="$(python3 -c "import json; print(json.load(open('$REP/STAGE_B_PLAN.json'))['max_steps'])")"
rm -rf "$ROOT/runs/e2_stageB96" 2>/dev/null || true
if ! pgrep -f "hrfont_stageB_keepalive" >/dev/null; then
  hb "launch Stage B max=$HRFONT_STAGE_B_MAX"
  nohup bash "$ROOT/scripts/hrfont_stageB_keepalive.sh" >>"$REP/keepalive_stageB.log" 2>&1 &
fi
hb "post_eval COMPLETE"
