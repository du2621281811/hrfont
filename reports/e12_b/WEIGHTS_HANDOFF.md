# E12-b 权重交接（合作者）

**日期**：2026-09-14  
**用途**：同域家族打分（φ + membership）；协议见 `reports/EXPERIMENT_E12B_V0913_20260914.md`。

## 权重不在 git

仓库忽略 `runs/**` 与 `*.pth`。权重通过 GitHub Release 分发：

- **Release tag**：`e12-b-v0913-20260914`
- **URL**：https://github.com/du2621281811/hrfont/releases/tag/e12-b-v0913-20260914
- **Asset**：`e12-b-v0913-best-20260914.tar.gz`（仅 `best.pt` + 小配置/曲线，不含中间 `step_*.pt`）

下载后解压到仓库根目录下的 `runs/`：

```bash
cd /path/to/hrfont
mkdir -p runs
tar -xzf e12-b-v0913-best-20260914.tar.gz -C runs/
# 得到：
#   runs/e12_phi_s2_b_s3407/best.pt
#   runs/e12_membership_b_s3407/best.pt
```

## SHA256（必核）

```
83cb0a92adc8a234fc2d63ac14bf1895fb2c2e5dc280fe4caa4cf2f266e21d97  runs/e12_phi_s2_b_s3407/best.pt
5828cc56d7cb666416088e898804b4c6f058cae699aee4ff99f287f739e84d44  runs/e12_membership_b_s3407/best.pt
```

## 配置路径

`configs/e12_*_b_s3407.yaml` 里是本机绝对路径。合作者请改成自己的 `$ROOT`，或：

```bash
ROOT=/path/to/hrfont
python scripts/score_preds_e12_b.py --device cuda:0
# 脚本内默认读 $ROOT/runs/e12_phi_s2_b_s3407/best.pt 与 membership best.pt
# （若脚本写死 /root/projects/hrfont，请先改 ROOT 常量或加软链）
```

## 已在 git 的配套

| 项 | 路径 |
|----|------|
| 合同 | `reports/EXPERIMENT_E12B_V0913_20260914.md` |
| 状态 | `reports/e12_b/STATUS.md` |
| 配置 | `configs/e12_phi_s2_b_s3407.yaml`、`configs/e12_membership_b_s3407.yaml` |
| 打分 | `scripts/score_preds_e12_b.py` |
| 建 cache | `scripts/build_e12_cache_v0913_b.py` |
| 可用性 | `manifests/v0913_clean/fonts.tsv` |
| 论文表 | `reports/e12_paper/` |

## 可选：特征 cache

`artifacts/e12/cache_v0913_b/`（约 332MB）**未**进本 Release。  
- 只打分已有 PNG：**可不下 cache**  
- 要重训 / 内部 holdout：用 `build_e12_cache_v0913_b.py` 重建，或另要 cache 包

## 不要做的事

- 不要把 `best.pt` / 全量 `runs/` `git add` 进 main  
- 不要用 E12-b 数字冒充「外部零重叠」评测（见实验合同 §7）
