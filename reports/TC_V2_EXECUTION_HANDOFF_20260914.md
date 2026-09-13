# TC-v2 / Group G 执行交接

日期：2026-09-14。设计权威文件：[`G_STYLE_COMPLETION_PLAN_20260914.md`](G_STYLE_COMPLETION_PLAN_20260914.md)。实现与验证记录：[`TC_V2_LUNA_IMPLEMENTATION_REVIEW_20260914.md`](TC_V2_LUNA_IMPLEMENTATION_REVIEW_20260914.md)。本文件是待执行命令，不表示执行机已启动这些任务。

## 1. 先决条件与资产

1. 拉取本轮提交，核实当前 GPU / watchdog / G0、G2 状态；不自动停止任何已有作业。
2. 选定 **clean G2** checkpoint，而不是旧 dirty F2；核实其 parent G0 的 Es/Ec 权重 SHA 与 `artifacts/g0` 缓存一致，三个缓存的 `progress.json` 均完成。TC 不重训 Es/Ec。
3. 保留每个独立 run 的 parent 路径、parent step、权重 SHA、Git SHA、缓存 manifest 与新增步数。新实验使用新目录；不得复用已有 G2/G2-PRL 的输出目录。
4. 在仓库根目录执行。沿用执行机已有 PyTorch/diffusers/accelerate 环境；VGG16 权重须可本地读取或预先下载，TensorBoard 须可用。下面路径是显式占位符，执行者先替换，不能照字面运行。

```bash
TC_PY=/root/miniforge3/envs/boogu/bin/python
TC_DATA=/root/projects/hrfont/data/fontdiffuser-p253-t295-s338-cn2west-v2
TC_SPLIT=/root/projects/hrfont/manifests/split_v3_228_16_16.json
TC_CLEAN=/root/projects/hrfont/manifests/v0913_clean
TC_ES=/root/projects/hrfont/artifacts/g0/es_spatial
TC_EC=/root/projects/hrfont/artifacts/g0/ec_multiscale
TC_LOCAL=/root/projects/hrfont/artifacts/g0/es_local
TC_CACHE=/absolute/new/tc-cache
TC_HEAD=/absolute/new/tc-head-run
TC_PARENT=/absolute/selected/clean-G2-checkpoint
TC_OUTPUT=/absolute/new/pilot-run
TC_GPU=0
```

## 2. CPU 回归与真实 GPU smoke

```bash
"$TC_PY" scripts/test_review_regressions.py
"$TC_PY" scripts/test_tc_v2_integration_review.py
"$TC_PY" tests/test_tc_v2.py
PYTHONPATH=code/variants/cn2west_f123_rsi/FontDiffuser "$TC_PY" scripts/test_identity_safe_rsi.py
```

本机 CPU 合成测试不能替代真实资产 smoke。执行机先完成缓存提取，然后以新临时 run 运行下面 joint 命令的 **2 步版本**（`max_train_steps=2`、`state_interval=1`），验证 fp16 forward/backward、finite loss、checkpoint 写出/严格恢复，以及专用 sampler 的真实单字输出；通过才开 2k/5k pilot。smoke 目录不能继续充当正式 run。

## 3. VGG 描述缓存与 H 预训练

```bash
CUDA_VISIBLE_DEVICES="$TC_GPU" "$TC_PY" scripts/build_tc_v2_cache.py \
  --data-root "$TC_DATA" --split-manifest "$TC_SPLIT" \
  --clean-map "$TC_CLEAN" --out "$TC_CACHE" \
  --resolution 96 --batch-size 32 --device cuda:0

CUDA_VISIBLE_DEVICES="$TC_GPU" "$TC_PY" scripts/train_tc_v2_head.py \
  --tc-cache "$TC_CACHE" --ec-cache "$TC_EC" --clean-map "$TC_CLEAN" \
  --out "$TC_HEAD" --steps 2000 --batch-size 256 \
  --lr 3e-4 --warmup-steps 100 --eval-interval 100 \
  --nshot-min 1 --nshot-max 8 --seed 3407 --device cuda:0
```

可给缓存命令添加 `--vgg-weights /absolute/vgg16-state-dict.pth` 使用本地完整 torchvision 权重。缓存只用 clean train target 拟合标准化统计，val target 用于验证，test target 不缓存。跨机器移动缓存时，保留原 manifest；用 `--clean-map` 指向新机器上的同内容映射，不手改 manifest 路径破坏 fingerprint。

H 的 val loss 不是图像效果结论。正式推进前另做固定 true-1 / true-8 readout，与 train-only 同字符均值、简单 Ref 预测比较；当前预训练脚本的固定 Ref val loss 仅是学习健康度读数，不是完整论文消融评估。

## 4. 统一 pilot 命令

下面展示 **TC 联合训练**。`TC_PARENT` 在本阶段必须改成阶段 1 胜出的 clean 底座 checkpoint；若胜出的是 RL，用 `F2RL`，否则用 `F2`。从相同 parent 新开两个 run，TC 与续训锚点共享数据、种子、学习率日程和新增步数。

```bash
CUDA_VISIBLE_DEVICES="$TC_GPU" "$TC_PY" code/variants/cn2west_f123_rsi/FontDiffuser/train.py \
  --arm F2RL --rsi_source delta --no-support --freeze_encoders \
  --encoder_runtime cache_only --warm_start_from "$TC_PARENT" --no-parity_check \
  --data_root "$TC_DATA" --split_manifest "$TC_SPLIT" --v0913_clean_map "$TC_CLEAN" \
  --es_cache_path "$TC_ES" --ec_cache_path "$TC_EC" --es_local_cache_path "$TC_LOCAL" \
  --tc_enabled --tc_cache_path "$TC_CACHE" --tc_head_ckpt "$TC_HEAD" \
  --tc_loss_coefficient 0.01 --tc_learning_rate 1e-4 \
  --resolution 96 --style_image_size 96 --content_image_size 96 \
  --nshot_min 1 --nshot_max 8 --source_drop 0.25 --drop_prob 0.1 \
  --train_batch_size 8 --gradient_accumulation_steps 1 --mixed_precision fp16 \
  --learning_rate 1e-5 --lr_scheduler constant_with_warmup --lr_warmup_steps 200 \
  --max_train_steps 5000 --ckpt_interval 1000 --state_interval 1000 --best_min_step 1000 \
  --perceptual_coefficient 0.01 --offset_coefficient 0.5 --max_grad_norm 1.0 \
  --seed 3407 --log_interval 100 --experience_name G-TC-pilot \
  --output_dir "$TC_OUTPUT"
```

- **阶段 1a / G-RL-pilot：** 从选定 clean G2 开始，`--arm F2RL`，删除 TC 开关/缓存/预训练输入参数与 TC loss/lr，改 `--no-tc_enabled`，增加 `--local_learning_rate 1e-4`。这只提高新 local projection 的学习率，不提高已有 UNet/RSI 的学习率。
- **阶段 1b / G-base-continue：** 同一 parent、同样预算，用 `--arm F2 --no-tc_enabled`，独立输出目录。
- **阶段 2a / G-TC-pilot：** 使用上面的命令，从阶段 1 胜出的 parent 开始，加载预训练 H，W_out 零初始化。
- **阶段 2b / G-best-continue：** 同一胜出 parent，保持相同 arm，删除 TC 输入/loss/lr，使用 `--no-tc_enabled`，独立输出目录。已训练 local projection 与 RSI 权重必须保留。

阶段2若沿用已训练的RL投影，默认跟随底座 `learning_rate=1e-5`；只有阶段1新建local projection才显式设1e-4。TC的H/W仍使用1e-4。`local_learning_rate`默认不指定，旧F/G命令的学习率行为保持不变。

2k 首次视觉 readout，值得则继续到5k；不因 `best/` 的 diffusion loss 更低自动认定其生成最好。主方案延长采用另一个明确登记的阶段；**warm-start** 重置 optimizer/scheduler/step，**resume** 才恢复同一实验的状态，二者不能混用。正式 TC resume 必须带匹配的 TC/cache binding 与非空 H/W 权重。

## 5. 同采样器视觉验收

不要把专用 DDPM 的 TC 输出与旧 DPM 的 baseline 输出直接归因比较；它们必须通过同一个专用入口、相同 scheduler/步数/CFG/ref/噪声生成。该入口当前是最小单字采样工具，不是完整批量论文 evaluator。

```bash
TC_FONT=replace-with-clean-val-font-stem
TC_CP=replace-with-valid-target-codepoint
TC_SAMPLE=/absolute/new/val-sample.png
CUDA_VISIBLE_DEVICES="$TC_GPU" "$TC_PY" scripts/sample_tc_v2.py \
  --arm F2RL --checkpoint "$TC_OUTPUT/best" --tc_enabled \
  --data-root "$TC_DATA" --split val --font "$TC_FONT" --content-cp "$TC_CP" \
  --refs u6C38 u548C u4E66 u98CE u9AA8 u97F5 u5929 u5730 \
  --es_cache_path "$TC_ES" --ec_cache_path "$TC_EC" --es_local_cache_path "$TC_LOCAL" \
  --tc_cache_path "$TC_CACHE" --split_manifest "$TC_SPLIT" --v0913_clean_map "$TC_CLEAN" \
  --output "$TC_SAMPLE" --steps 20 --guidance_scale 7.5 --seed 3407 --device cuda:0
```

以上示例 Ref 必须核实存在于该字体的缓存，不能凭示例假定所有字体都齐全。true-1 只传一个 Ref（例如 `--refs u6C38`），不是把一个 Ref 重复8次。baseline 使用同一入口的 `--no-tc_enabled`，换成对应 checkpoint / 新输出路径，arm 与训练保持一致。

先固定8个 clean val字体、每字体至多12个合法目标字做2k面板；进入主训练前扩展至完整有效val。PNG与JSON均保留，汇总笔触、端点、风格一致性及可读性，再补齐分语种L1/LPIPS/SSIM。20步DDPM首先用于一致性 smoke；若其自身未达到可用采样质量，统一提高所有对照的步数后再做方法选择，不能把采样器退化当作TC失败。

## 6. 仍需 PI / 执行者作出的选择

- 依据执行机真实进度确定 clean G2 parent 及可用 GPU；本轮未替 PI 选择未知 checkpoint，也未重排活跃任务。
- 阶段 1 的 clean val 决定最终使用 F2 还是 F2RL；现有肉眼 F2-RL > F2-PRL 只决定优先级。
- 阶段 2 是否延长，取决于同起点、同采样器视觉收益；代码正确与合成 overfit 不保证真实效果提升。

如真实资产 smoke 失败，修复后先重复 smoke，不直接用长训练掩盖问题。若特征预测改善但图像不改善，先检查注入幅度/梯度与训推条件，再考虑更换描述；不恢复 Set-Delta，不同时改 alpha、Ec/Es 或 renderer。
