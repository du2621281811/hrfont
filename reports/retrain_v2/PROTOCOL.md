# retrain_v2 风格参考协议（决策稿）

**日期：** 2026-08-07  
**状态：** 已决策，待按清单整改 / 重训  
**任务声称：** 中文 few-shot 风格 → 生成拉丁(P1) / 假名(P2)；测集 Demo-8；指标 L1↓ / SSIM↑

---

## 1. 决策：主实验用什么原则

三选一结论：

| 原则 | 是否作主协议 | 说明 |
|------|:------------:|------|
| **A. 和官方对齐** | 否（附录） | 各方法官方默认本就不同（1/3/4/6/8 shot、方向可能相反），无法支撑「同一任务横比」 |
| **B. 训测对齐** | **必要** | 训练见什么风格脚本，测试就必须是同一类；禁止「西文训、中文测」 |
| **C. 各方法统一** | **主实验靠它** | 同一任务、同一参考池、同一测集，才能选基模 / 给 mentor |

**主协议 = C（跨方法统一任务） ∩ B（单方法训测一致）**  
**官方对齐 = 附录 / 兼容说明**（写清差异即可，不进主表混比）

一句话：要横比、选底座 → **统一任务 + 训测对齐**；不要为「忠实官方」而接受西文训、中文测。

---

## 2. 主协议（锁定）

对所有进入**主结果表**的方法：

1. **风格参考池（固定）：** `永和书风骨韵天地`（8 个中文）  
2. **内容：** P1 拉丁 / P2 假名（训=评字符集）  
3. **训 = 测脚本一致：** 训练风格必须来自 **同字体中文**（测用固定 ref8 /「永」；训允许在同字体更大中文池里随机抽，**不允许**用 P1∪P2 西文/假名当主训练 style）  
4. **测集：** Demo-8 holdout  
5. **字体预算（主表）：** 同一档；建议横比用 **42**；**253 仅作放大实验单独列**  
6. **张数（n_ref）：**  
   - 目标统一为 **8**（与 GAR / 公共参考字一致）  
   - 若架构锁死（如 GAN `style_channel=6`、FD 原生 1-shot）：**表头必须标明实际 shot**，禁止假装全是 8-shot；但仍须是 **中文** style  

**GT 只用于打分，不是模型输入。**

---

## 3. 现状审计（2026-08-07）

**完整逐方法问题清单（风格 + content + 字体预算 + 状态）：**  
→ [`EXPERIMENT_AUDIT.md`](EXPERIMENT_AUDIT.md)

公共参考字磁盘：`data/retrain_v2/font/.../chinese/<font>/`（测用 ref8；FD 训可扩到 `style_cn_pool.txt`）；西文 GT：`.../english/`。

### 3.1 各方法训 / 测风格怎么选

| 方法 | 训练风格参考 | 测试风格参考 | 与主协议 |
|------|--------------|--------------|----------|
| MF / FTrans / FCA / GAS | 随机 **6 CN**（`chinese/`，`style_channel=6`） | 固定 **6 CN**：`永和书风骨韵` | 脚本 OK；张数 6≠8 |
| LF-Font | 随机 **3** 张，池=**P1∪P2（同脚本，无 CN）** | 固定 **8 CN** | **违例：训测脚本不一致** |
| FontDiffuser 旧 `ft/` | **1** 张随机同字体（**P1∪P2，无 CN**） | 固定 **1 CN**：`永` | **违例：最严重** |
| FontDiffuser `ft_cnstyle`@42 / `ft_p253_cnstyle` | **1** 张同字体中文池（~338） | 固定 **1 CN**：`永` | ✅ 已评测；主表用 @42/25k；253 单列 |
| MX-Font | 随机 **3** 张，池≈P1∪P2+8CN（多数仍同脚本） | 固定 **8 CN** | **违例：训测基本不一致** |
| GAR-Font | 固定全部 **8 CN** | 固定 **8 CN** | **最符合主协议**；主表若用 p253 则字体预算不公 |

### 3.2 其它问题（摘要；细节见 EXPERIMENT_AUDIT）

**高：** FD 旧跑西文训；LF/MX 训测脚本不一致；MX 训 content 带风格；GAR 主行用 253 不公。  
**中：** GAN 6-shot 叙事；数据盘扩到 ~260 字体后重训须过滤 stem。  
**低：** LF P2 训 DejaVu / 测 Noto；FD 排除同字 GT 当 style（正确）。

### 3.3 与官方的关系

| 方法 | 官方大致默认 | 我们现状 | 结论 |
|------|--------------|----------|------|
| GAN 系 | 常 `style_channel=6`，方向仓库默认可能 english2chinese | 我们强制 CN→西文 + 6 CN | 任务向合理；非 100% 官方默认 |
| LF | 训 `n_in_s=3` 同脚本；测一长串 CN ref | 训同脚本、测 8 CN | 部分像官方训法，但与**本任务主协议**冲突 |
| FD | 训 1 张随机同字体；测 1-shot | 机制像官方，但训无 CN | 官方机制 ≠ 本任务正确协议 |
| MX | 训 `n_in_s=3`；val 常 `n_ref=4` | 训 3、测 8 | 部分像官方训法；与主协议冲突 |
| GAR | `n_ref=8` | 固定 8 CN 训测 | 数量对齐；固定集合是我们为可复现做的锁定 |

**不要用「向官方对齐」为 FD/LF/MX 的西文训辩解**——那是另一套实验问题。

---

## 4. 整改优先级（待执行）

1. **FontDiffuser：** 训练 style 改为 CN 池（已改代码，见下节）；旧 `ft/`（西文 style）保留作对照，新跑用 `ft_cnstyle` / `ft_p253_cnstyle`。  
2. **LF / MX：** 训练 style 改为 **仅 CN 池**；`n_in` 尽量贴近测（目标 8，或表上标明 3 并两端统一）；LF 若跨语仍不可修，主表降级或剔除。  
3. **n_ref 叙事：** 全表标明实际 1/6/8；能升 8 的升，不能的显式标注。  
4. **公平横比：** 主表 GAR 用 **P0/42**；**ft_p253** 单列「放大实验」。  
5. **文档 / 图库文案：** 禁止再写「八方法均为 8-shot CN 训练」除非整改完成。

---

## 4.1 FontDiffuser 重训步骤（CN style）

**原因：** 旧数据只有 `TargetImage`（西文 GT），仓库默认从同文件夹随机抽 style → 训练无 CN。

**已改：**
- `scripts/expand_retrain_v2_cn_style_pool.py` → 把 `style_cn_pool.txt` 同字体中文渲进 `chinese/`
- `scripts/build_retrain_v2_fontdiffuser_data.py` → `StyleImage/<font>/` = 该字体全部 `chinese/*.png`（训时随机抽，不限 8）
- `code/FontDiffuser/dataset/font_dataset.py` → 若存在 `StyleImage/`，只从中抽 style
- `scripts/retrain_v2_finetune_fontdiffuser.py` → 支持 `--run_name` / `--data_root` / `--rebuild_data`
- **评测仍固定 1×「永」**（与旧对照一致）

**推荐命令（42 字、协议对齐，先跑这档）：**

```bash
# 1) 重建数据（写入 StyleImage）
/root/miniforge3/envs/boogu/bin/python \
  /root/projects/hrfont/scripts/build_retrain_v2_fontdiffuser_data.py

# 2) 从官方 ckpt 微调 30k（约 1.7h / 1 GPU）
/root/miniforge3/envs/boogu/bin/python \
  /root/projects/hrfont/scripts/retrain_v2_finetune_fontdiffuser.py \
  --gpu 0 --max_steps 30000 --run_name ft_cnstyle --rebuild_data

# 3) 评测（写入 p*_eval_cnstyle/，不覆盖旧结果）
/root/miniforge3/envs/boogu/bin/python \
  /root/projects/hrfont/scripts/retrain_v2_eval_fontdiffuser.py \
  --ckpt_dir /root/projects/hrfont/runs/ft_cnstyle/global_step_30000 \
  --tag cnstyle
```

**放大到 253（与 GAR-p253 同字体池）：**

```bash
/root/miniforge3/envs/boogu/bin/python \
  /root/projects/hrfont/scripts/build_retrain_v2_fontdiffuser_data.py \
  --use-p253-list --dst /root/projects/hrfont/data/fontdiffuser_p253

/root/miniforge3/envs/boogu/bin/python \
  /root/projects/hrfont/scripts/retrain_v2_finetune_fontdiffuser.py \
  --gpu 0 --max_steps 60000 --batch_size 4 \
  --data_root /root/projects/hrfont/data/fontdiffuser_p253 \
  --run_name ft_p253_cnstyle
```

测时仍用 **1×「永」**（FD 原生 1-shot）；训练时从同字体 **中文池**（`StyleImage/`，可大于 8）随机 1 张。

---

## 5. 报告口径（给 mentor）

- **已完成、可展示：** Demo-8 图库与指标；须口头/脚注说明 FD/LF/MX **训练风格协议与测试不完全一致**（整改前）。  
- **选基模：** 在协议未统一前，勿把「FD@42 很好」与「GAR@253」直接当同条件结论；比 42 档用 GAR-P0 vs FD/MX/GAN；比放大用双方都到 253 且 **CN style 训测对齐** 后再定唯一底座。  
- **官方复现：** 如需，另开「官方设定」附录，不与主表混排。

---

## 6. 相关入口

| 文档 / 页 | 路径 |
|-----------|------|
| 本决策 | `reports/retrain_v2/PROTOCOL.md` |
| **各方法问题审计** | `reports/retrain_v2/EXPERIMENT_AUDIT.md` |
| **当前实验执行计划** | `reports/retrain_v2/EXPERIMENT_PLAN.md` |
| 结果摘要 | `reports/retrain_v2/RETRAIN_V2_SUMMARY.md` |
| 训练手册 | `reports/retrain_v2/TRAINING_PLAYBOOK.md` |
| 图库 | `reports/retrain_v2/gallery.html` |
| GAR p253 | `reports/retrain_v2/gar_p253/` |
| GAR P0 对照 | `reports/retrain_v2/gar_p0_train/`、`GAR_P0_CHECK.md` |
