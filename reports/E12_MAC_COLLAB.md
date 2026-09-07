# 在 MacBook（Apple Silicon）上跑 E12

结论：**可以训。** E12 是小编码器（φ_s2 + id_cls），不是 FontDiffuser UNet。框架已支持 `mps`（`device_from_config`：cuda → mps → cpu）。M 系列（含 M5）用 MPS 即可；无独显 NVIDIA。

## 1. 要同步什么

| 路径 | 体积 | 用途 | 是否已进 git |
|---|---|---|---|
| `artifacts/e12/cache_v4/` | ~18 MB | **训练+自检输入**（已渲染 PNG） | 本提交起同步 |
| `configs/e12_*_v4_s*.yaml` | 很小 | 配置 | 已同步 |
| `scripts/eval_framework/` | 代码 | 训练/自检 | 已同步 |
| `reports/e12_v4_results/` | 很小 | 本机已跑过的门结果 | 本提交同步 |
| `artifacts/e12/fonts_s3/` | ~279 MB | 仅当要**重建** cache | **不进 git**（见 §4） |

有 `cache_v4` 就够复训与自检，**不必**拷贝 fonts_s3。

## 2. 环境

```bash
# Python 3.10+，建议独立 venv
pip install torch torchvision  # 官方 macOS / arm64 轮子即可
pip install pillow pyyaml numpy scikit-learn  # 按 import 补齐
cd /path/to/hrfont
```

确认 MPS：

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## 3. 一键（推荐）

```bash
# 从仓库根目录；device=auto 会选 mps
bash scripts/launch_e12_mac.sh 3407
# 或三种子
bash scripts/launch_e12_mac.sh all
```

脚本会：

1. 用 `--set` 把绝对路径改成相对本机仓库根；
2. `train.device=auto`（或显式 `mps`）；
3. `train.fp16=false`（MPS 上关闭 cuda GradScaler 路径，更稳）；
4. 写出 `runs/e12_*_v4_s${SEED}/`。

单步手动示例：

```bash
ROOT=$(pwd)
EF=$ROOT/scripts/eval_framework
SEED=3407
cd "$EF"
python train_style_encoder.py --config "$ROOT/configs/e12_phi_s2_v4_s${SEED}.yaml" \
  --set data.cache_dir=$ROOT/artifacts/e12/cache_v4 \
  --set output.dir=$ROOT/runs/e12_phi_s2_v4_s${SEED} \
  --set train.device=mps --set train.fp16=false
python train_id_cls.py --config "$ROOT/configs/e12_id_cls_v4_s${SEED}.yaml" \
  --set data.cache_dir=$ROOT/artifacts/e12/cache_v4 \
  --set output.dir=$ROOT/runs/e12_id_cls_v4_s${SEED} \
  --set train.device=mps
python self_tests.py --config "$ROOT/configs/e12_self_tests_v4_s${SEED}.yaml" \
  --set data.cache_dir=$ROOT/artifacts/e12/cache_v4 \
  --set model.phi_checkpoint=$ROOT/runs/e12_phi_s2_v4_s${SEED}/best.pt \
  --set model.id_checkpoint=$ROOT/runs/e12_id_cls_v4_s${SEED}/best.pt \
  --set output.dir=$ROOT/runs/e12_self_tests_v4_s${SEED} \
  --set train.device=mps
```

门结果对比：打开 `reports/E12_SELFTEST_V4_REVIEW_20260907.md` 与 `reports/e12_v4_results/`。

## 4. 若需要重建 cache（可选）

```bash
# 自行下载 26 套 OFL 字库到 artifacts/e12/fonts_s3/（见 scripts/e12_fetch_external_fonts.py）
bash scripts/e12_build_cache_v4.sh
```

字体体积大且含第三方许可，默认不进 git。训练机若另传 `fonts_s3` tar，放同路径即可。

## 5. 预期与注意

- 单 seed φ_s2 约数千 step；M 系列通常数十分钟级（视散热/功耗），远小于 F123 UNet。
- **数值不必与 CUDA 逐位一致**；门限判定（过/不过）应大体同向。若 T2 仍 ~0.75–0.80，与服务器结论一致。
- 不要改门限 YAML 事后放宽。
- F0–F3 生成器训练仍依赖 CUDA + 大数据集，**不要**指望在 Mac 上复现 F123；E12 可并行。
