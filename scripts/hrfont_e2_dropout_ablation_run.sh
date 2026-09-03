#!/usr/bin/env bash
# Dropout ablation orchestrator — GPU 2/3 only (GPU0 = formal main, skip GPU1).
set -u
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
REP=$ROOT/reports/hrfont_overnight/dropout_ablation
LOG=$REP/orchestrator.log
mkdir -p "$REP"

hb() { echo "$(date -Is) $*" | tee -a "$LOG"; }

run_one() {
  local gpu=$1 run_id=$2 drop_cfg=$3 drop_delta=$4 drop_mode=$5
  hb "START $run_id gpu=$gpu cfg=$drop_cfg d=$drop_delta mode=$drop_mode"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=$gpu \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
    "$PY" -u "$ROOT/scripts/hrfont_e2_dropout_ablation_train.py" \
      --run-id "$run_id" --drop-cfg "$drop_cfg" --drop-delta "$drop_delta" \
      --drop-mode "$drop_mode" --max-steps 10000 \
    >>"$REP/${run_id}.stdout" 2>&1
  local rc=$?
  hb "DONE $run_id rc=$rc — eval"
  CUDA_VISIBLE_DEVICES=$gpu HRFONT_STAGEA_CKPT="$ROOT/runs/dropout_ablation/${run_id}/step_10000.pt" \
    "$PY" "$ROOT/scripts/hrfont_e2_official_eval.py" \
      --ckpt "$ROOT/runs/dropout_ablation/${run_id}/last.pt" \
      --out "$REP/eval_${run_id}.json" 2>>"$REP/${run_id}.stdout" || \
  CUDA_VISIBLE_DEVICES=$gpu HRFONT_STAGEA_CKPT="$ROOT/runs/dropout_ablation/${run_id}/last.pt" \
    "$PY" "$ROOT/scripts/hrfont_e2_official_eval.py" \
      --ckpt "$ROOT/runs/dropout_ablation/${run_id}/last.pt" \
      --out "$REP/eval_${run_id}.json" >>"$REP/${run_id}.stdout" 2>&1
  hb "EVAL $run_id written eval_${run_id}.json"
}

hb "orchestrator start"

# Wave 1 parallel: GPU2 + GPU3
run_one 2 R1_cfg01_d01 0.1 0.1 both &
pid1=$!
run_one 3 R2_cfg01_d00 0.1 0.0 both &
pid2=$!
wait $pid1 || hb "R1 failed"
wait $pid2 || hb "R2 failed"

# Wave 2 parallel
run_one 2 R3_style025_d025 0.25 0.25 style_only &
pid3=$!
run_one 3 R4_cfg025_d025 0.25 0.25 both &
pid4=$!
wait $pid3 || hb "R3 failed"
wait $pid4 || hb "R4 failed"

# Summarize
"$PY" "$ROOT/scripts/hrfont_e2_dropout_ablation_summarize.py" >>"$LOG" 2>&1
hb "orchestrator COMPLETE"
