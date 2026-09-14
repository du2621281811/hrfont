#!/usr/bin/env bash
# After G queue + v0913 board: 8-GPU RL-pilot + TC on G2 + TC on G2-RL.
# Shared: build TC cache + H pretrain (1 GPU). Then serial 8-GPU jobs.
set -euo pipefail
ROOT=/root/projects/hrfont
PY=/root/miniforge3/envs/boogu/bin/python
LOGDIR=$ROOT/reports/g_tc_8gpu_queue
mkdir -p "$LOGDIR" "$ROOT/artifacts/tc_v2_cache" "$ROOT/artifacts/tc_v2_head"
LOG=$LOGDIR/queue.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Iseconds)] queue start"

G2=$ROOT/runs/G2-F2-V0913-A-S3407/global_step_10000
G2RL=$ROOT/runs/G2-RL-V0913-A-S3407/global_step_10000
TC_CACHE=$ROOT/artifacts/tc_v2_cache
TC_HEAD_DIR=$ROOT/artifacts/tc_v2_head
TC_HEAD=$TC_HEAD_DIR/tc_head.pth
DATA=$ROOT/data/fontdiffuser-p253-t295-s338-cn2west-v2
SPLIT=$ROOT/manifests/split_v3_228_16_16.json
CLEAN=$ROOT/manifests/v0913_clean
EC=$ROOT/artifacts/g0/ec_multiscale

# --- 0) TC cache + H pretrain (shared, 1 GPU) ---
if [[ ! -f $TC_CACHE/manifest.json ]]; then
  echo "[$(date -Iseconds)] build_tc_v2_cache"
  CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/build_tc_v2_cache.py \
    --data-root "$DATA" --split-manifest "$SPLIT" \
    --clean-map "$CLEAN" --out "$TC_CACHE" \
    --resolution 96 --batch-size 32 --device cuda:0
else
  echo "[$(date -Iseconds)] tc cache exists, skip"
fi

if [[ ! -f $TC_HEAD ]]; then
  echo "[$(date -Iseconds)] train_tc_v2_head"
  CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/train_tc_v2_head.py \
    --tc-cache "$TC_CACHE" --ec-cache "$EC" --clean-map "$CLEAN" \
    --out "$TC_HEAD_DIR" --steps 2000 --batch-size 256 \
    --lr 3e-4 --warmup-steps 100 --eval-interval 100 \
    --nshot-min 1 --nshot-max 8 --seed 3407 --device cuda:0
else
  echo "[$(date -Iseconds)] tc head exists, skip"
fi

# --- 1) 8-GPU RL pilot from G2 (no TC), local_lr=1e-4 ---
echo "[$(date -Iseconds)] G-RL-pilot-8gpu"
$PY $ROOT/scripts/launch_g_joint_8gpu.py --yes \
  --run_id G-RL-pilot-8gpu-V0913-A-S3407 \
  --arm F2RL --parent "$G2" --max_steps 5000 --port 29540 \
  --no-tc --local_lr 1e-4 \
  | tee -a $LOGDIR/pilot8.console.log

# --- 2) 8-GPU TC on G2 parent ---
echo "[$(date -Iseconds)] G-TC-G2-8gpu"
$PY $ROOT/scripts/launch_g_joint_8gpu.py --yes \
  --run_id G-TC-G2-8gpu-V0913-A-S3407 \
  --arm F2 --parent "$G2" --max_steps 5000 --port 29541 \
  --tc --tc_cache "$TC_CACHE" --tc_head "$TC_HEAD" \
  | tee -a $LOGDIR/tc_g2.console.log

# --- 3) 8-GPU TC on G2-RL parent ---
echo "[$(date -Iseconds)] G-TC-G2RL-8gpu"
$PY $ROOT/scripts/launch_g_joint_8gpu.py --yes \
  --run_id G-TC-G2RL-8gpu-V0913-A-S3407 \
  --arm F2RL --parent "$G2RL" --max_steps 5000 --port 29542 \
  --tc --tc_cache "$TC_CACHE" --tc_head "$TC_HEAD" \
  | tee -a $LOGDIR/tc_g2rl.console.log

# decision note
cat >> $ROOT/reports/G_QUEUE_DECISIONS_20260914.md <<'EOF'

### D12 · 8-GPU pilot + 双 TC（用户 2026-09-14）

- **决定：** 看板推理完成后，串行启动：
  1. 共享 `build_tc_v2_cache` + `train_tc_v2_head`（单卡）
  2. `G-RL-pilot-8gpu-V0913-A-S3407`：G2@10k warm-start，F2RL，无 TC，local_lr=1e-4，**8×bs8=64**，+5k
  3. `G-TC-G2-8gpu-V0913-A-S3407`：G2@10k + TC，arm F2，8 卡，+5k
  4. `G-TC-G2RL-8gpu-V0913-A-S3407`：G2-RL@10k + TC，arm F2RL，8 卡，+5k
- **lr：** 绝对 1e-5（UNet）/ 1e-4（local 或 TC head），**不**随 batch 放大。
- **注意：** 与既有单卡 `G-RL-pilot`（全局 bs8）配方不同，结果不可直接当同设定复现。
EOF

echo "[$(date -Iseconds)] queue ALL DONE"
