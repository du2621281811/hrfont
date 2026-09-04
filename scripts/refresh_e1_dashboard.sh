#!/usr/bin/env bash
# Dynamic E1 dashboard refresh — never touches training GPU3.
# - Every 60s: loss curve only (CPU, --skip-samples)
# - Every 300s or when a new ckpt appears: sample on idle GPU2
set -euo pipefail
ROOT=/root/projects/hrfont
OUT=$ROOT/runs/E1-FTV2-A-S3407
VIZ=$OUT/viz
PY=/root/miniforge3/envs/boogu/bin/python
LOG=$VIZ/refresh.log
mkdir -p "$VIZ"
SAMPLE_GPU=2
TICK=0

pick_idle_sample_gpu() {
  # Prefer GPU with most free memory among {0,1,2} (never 3 = train)
  $PY - <<'PY'
import subprocess
out=subprocess.check_output([
  "nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"
], text=True)
best,best_free=2,-1
for line in out.strip().splitlines():
    idx,free=line.split(",")
    idx=int(idx.strip()); free=int(free.strip())
    if idx==3:  # training GPU — never use
        continue
    if free>best_free:
        best,best_free=idx,free
print(best)
PY
}

has_new_ckpt() {
  $PY - <<'PY'
import json
from pathlib import Path
viz=Path("/root/projects/hrfont/runs/E1-FTV2-A-S3407/viz/samples")
run=Path("/root/projects/hrfont/runs/E1-FTV2-A-S3407")
root=Path("/root/projects/hrfont")
n_val=len([l for l in (root/"manifests/pipeline_v3_val_stems.txt").read_text().splitlines() if l.strip()])
n_test=len([l for l in (root/"manifests/pipeline_v3_test_stems.txt").read_text().splitlines() if l.strip()])
expected_n=(n_val+n_test)*4  # A/a/G/0
expected_fonts={"val": n_val, "test": n_test}

def incomplete(tag: str) -> bool:
    marker=viz/tag/"_done.json"
    if not marker.is_file():
        return True
    try:
        prev=json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return True
    return not (
        int(prev.get("expected_n", -1)) == expected_n
        and int(prev.get("n", 0)) >= expected_n
        and prev.get("fonts") == expected_fonts
    )

need=False
# formal 5k ckpts + P1
if incomplete("step0_P1"):
    need=True
for p in run.glob("global_step_*"):
    if (p/"unet.pth").is_file():
        try:
            step=int(p.name.split("_")[-1])
        except ValueError:
            continue
        if incomplete(f"step{step}"):
            need=True
# mid-interval last_state only if not on a 5k milestone
last=run/"last_state"/"trainer_state.pt"
if last.is_file():
    try:
        import torch
        st=torch.load(str(last), map_location="cpu", weights_only=False)
        step=int(st.get("global_step",0))
        if step>0 and step%5000!=0 and incomplete(f"step{step}_last"):
            need=True
    except Exception:
        pass
# missing val-loss for any formal ckpt
val_hist={}
vp=Path("/root/projects/hrfont/runs/E1-FTV2-A-S3407/viz/val_loss_history.json")
if vp.is_file():
    try:
        for r in json.loads(vp.read_text(encoding="utf-8")):
            val_hist[int(r["step"])]=r
    except Exception:
        pass
need_steps={0}
for p in run.glob("global_step_*"):
    if (p/"unet.pth").is_file():
        try:
            need_steps.add(int(p.name.split("_")[-1]))
        except ValueError:
            pass
if any(s not in val_hist or int(val_hist[s].get("n",0))<=0 for s in need_steps):
    need=True
print("yes" if need else "no")
PY
}

echo "[$(date -Is)] viz_refresher_start" >>"$LOG"
while true; do
  TICK=$((TICK+1))
  # always refresh loss (CPU)
  if $PY "$ROOT/scripts/build_e1_train_dashboard.py" --skip-samples >>"$LOG" 2>&1; then
    echo "[$(date -Is)] loss_ok tick=$TICK" >>"$LOG"
  else
    echo "[$(date -Is)] loss_fail tick=$TICK" >>"$LOG"
  fi

  DO_SAMPLE=0
  if [ $((TICK % 5)) -eq 0 ]; then DO_SAMPLE=1; fi
  if [ "$(has_new_ckpt)" = "yes" ]; then DO_SAMPLE=1; fi

  if [ "$DO_SAMPLE" = "1" ]; then
    SAMPLE_GPU=$(pick_idle_sample_gpu)
    # bail if free < 6GB on chosen card
    FREE=$($PY - <<PY
import subprocess
out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits","-i","$SAMPLE_GPU"], text=True)
print(int(out.strip()))
PY
)
    if [ "$FREE" -lt 6000 ]; then
      echo "[$(date -Is)] skip_sample gpu=$SAMPLE_GPU free=${FREE}MiB" >>"$LOG"
    else
      echo "[$(date -Is)] sample_start gpu=$SAMPLE_GPU free=${FREE}MiB" >>"$LOG"
      if CUDA_VISIBLE_DEVICES=$SAMPLE_GPU $PY "$ROOT/scripts/build_e1_train_dashboard.py" --gpu "$SAMPLE_GPU" --device cuda:0 --max-fonts 0 --eval-val-loss >>"$LOG" 2>&1; then
        echo "[$(date -Is)] sample_ok gpu=$SAMPLE_GPU" >>"$LOG"
      else
        echo "[$(date -Is)] sample_fail gpu=$SAMPLE_GPU" >>"$LOG"
      fi
    fi
  fi
  sleep 60
done
