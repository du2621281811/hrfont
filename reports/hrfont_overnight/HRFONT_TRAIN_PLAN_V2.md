# HR-Font 跨语训练计划 v2（Stage A / B）

**状态：** 待你 review 后启动 Stage A 重训  
**日期：** 2026-09-02  
**取代：** overnight `pilot_v1`（像素混 Δ、render_fit@96、IoU gap、support 平铺 token）— 旧结果仅归档对照，不进 v2 主表  
**上位设计：** [`ICLR2027_HRFONT.md`](../ICLR2027_HRFONT.md)（主张与符号）  
**横比基线：** `ft_cnstyle@25k`（官方 RSI 接线跨语微调）

---

## 0. 一页结论

| 已拍板 | 内容 |
|--------|------|
| **D1** | 结构编码域统一用 **B₀=FZKTJW** 作 Δ 减数、RS-Gap、Support 风格差；**渲染与预处理与 ft 相同** |
| **D2** | Stage A：**特征混 Δ**，禁止像素混图再过 Ec |
| **D3** | Bank / Target / Style / Content 烘焙链与 **`ft_cnstyle` 完全一致**（`build_retrain_v2_dataset.render` @128） |
| **D4** | Support 门控：**RS-Gap**（ref 子空间结构残差），弃 IoU |
| **D5** | Support 三臂：**域内 / 全域 / 混合**；**严禁**当前目标字体入池、**严禁** q=c |
| **D6** | Support：**加权组合向量** \( \sum W_{q,s}(E_c(I_{s,q})-E_c(I_{B_0},q)) \) 再经 Adapter |
| **P1/P2** | **保留**，本计划不锁死（见 §9） |
| **训练** | **你 review 本文后再开 Stage A v2**；B 在 A v2 达标后 |

---

## 1. 故事线（论文与实验共用）

```text
Baseline（官方 FD 零微调）
  → 会生成，但 RSI 用汉字骨架扭拉丁 → 跨语病态

ft_cnstyle（跨语 SFT，官方接线）
  → 模型「认识」中文 style + 拉丁 content/GT 联合分布
  → RSI 仍看 Ec(汉字) → 形变源错误

Stage A v2（设计师：学「变化方向」）
  → 在像目标风格的库字体里，学「c 相对中性底该怎么变」
  → Δ 特征进 RSI；Es 仍只看用户汉字 R（真风格）

Stage B v2（设计师：学「用好 ref」）
  → ref8 盖不住 c 的结构知识时，检索锚点字 q 并**组合**注入
  → 类比：设计师先定永/和/书…，再为难字补口、日、o 等
```

---

## 2. 与 ft 对齐的渲染与预处理（D1 + D3）

### 2.1 统一函数

| 项 | 规范 |
|----|------|
| 渲染 | `scripts/build_retrain_v2_dataset.render(font_path, ch, size=128)` |
| 画布 | **128×128** 灰度 `L`，二分搜最大字号，margin **8px** |
| 入库 | PNG → **RGB JPG** quality=95（与 `build_retrain_v2_fontdiffuser_data.py` 一致） |
| 进网络 | `Resize(96, BILINEAR)` → `ToTensor` → `Normalize(mean=0.5, std=0.5)` |
| Content（与 ft 横比） | **DejaVu Sans** → `fontdiffuser/train/ContentImage/uXXXX.jpg`（不改源） |
| 中性结构底 B₀（D1） | **`data/FZKTJW.TTF`**，同一 `render()` @128 烘焙，供 Δ 减数 / RS-Gap / Support 差分 |

### 2.2 v2 Bank 重建（替代 `e0_bank/r96` + `render_fit`）

**输出根目录（新建，不覆盖 pilot）：** `data/hrfont/e0_bank_v2/`

| 子目录 | 内容 | 渲染字体 |
|--------|------|----------|
| `content_dejavu/` | P1 全字符 | DejaVu（= ft ContentImage 同源，可软链或复制） |
| `content_b0/` | P1 + ref8 + donors | FZKTJW（B₀ 结构底） |
| `target/<font>/` | P1 拉丁 GT | 各训练/Demo-8 字体 TTF |
| `style/<font>/` | ref8 八字 + 扩展中文池 | 各字体（训时随机，评测锁「永」） |
| `cache/` | `ec_es_r96.pt`、`rs_gap.json`、encoder 向量 | 离线缓存 |

**脚本规划：** `scripts/hrfont_e0_build_bank_v2.py`（待实现）

**剔除规则（继承 E0）：** 假拉丁（与 DejaVu NCC>0.985）、空墨迹、缺 cmap。

**Demo-8：** 可渲染供评测，**永不进入** α 池与 support 库字体集合 \(\mathcal{P}\)。

---

## 3. Stage A v2：特征混 Δ（D2）

### 3.1 符号

- 当前训练样本：目标字体 \(F\)，目标字 \(c\)（P1 拉丁），风格参考 \(R\)（中文，训时随机 ref8 或扩展池）
- 候选库 \(\mathcal{P}\)：42 训练字体，**leave-one-out：\(F \notin \mathcal{P}\)**
- \(\alpha\)：用 \(F\) 的 ref8 经 \(E_s\) 得 proto，与其余字体 cosine → top-\(M=3\) → softmax(\(/\tau_\alpha\))，\(\tau_\alpha=0.07\)
- \(E_c^{(l)}\)：content encoder 第 \(l\) 尺度特征（含最终 spatial map）

### 3.2 特征混（禁止像素混）

对每个尺度 \(l\)：

\[
\mathrm{mix}^{(l)} = \sum_{s \in \mathrm{top}\text{-}M} \tilde\alpha_s \, E_c^{(l)}\!\bigl(I(B_s, c)\bigr)
\]

\[
\mathrm{neutral}^{(l)} = E_c^{(l)}\!\bigl(I(B_0, c)\bigr)
\]

\[
\Delta^{(l)} = \mathrm{mix}^{(l)} - \mathrm{neutral}^{(l)}
\]

**RSI 结构支路：** `delta_res[l] = Δ^(l)`（多尺度列表），**替换**官方 `Ec(style汉字)`。

**MCA 身份支路：** 仍用 \(E_c(C)\)，\(C=\) DejaVu 上的 \(c\)（与 ft 一致）。

**风格支路：** 仍用 \(E_s(R)\)，\(R\) 为当前字体中文 style 图。

### 3.3 训练样本字段

| 字段 | 来源 |
|------|------|
| `content` | `ContentImage/u{c}.jpg`（DejaVu） |
| `style` | `style/<F>/` 随机 1 张中文（**禁止**选到与 \(c\) 同形的西文 GT 路径） |
| `target` | `target/<F>/u{c}.jpg` |
| `alpha_fonts` | top-3 库字体名（不含 \(F\)） |
| `delta_res` | 在线算或读 cache（推荐 **在线 Ec**，保证特征混精确） |

### 3.4 损失与 dropout（与 ft / pilot 对齐）

| 项 | 值 |
|----|-----|
| 初始化 | `ft_cnstyle@25k`：`unet` + 两 encoder 加载；**重初始化**全部 `OffsetRefStrucInter` |
| 冻结 | `style_encoder`、`content_encoder` |
| 可训 | `UNet`（含 RSI 偏移头） |
| 损失 | `MSE(noise) + 0.01×VGG_percep + 0.5×offset` |
| CFG dropout | content+style 联合 blank **0.1** |
| Δ dropout | `delta_res ← 0` **0.25** |
| 优化 | AdamW `lr=1e-5`，`batch=1`，`accum` 按显存 |
| 步数 | **80_000**（与 pilot A 同量级，便于对照） |
| 输出 | `runs/e2_stageA96_v2/` |

### 3.5 脚本规划

- 训练：`scripts/hrfont_e2_stageA96_v2_train.py`
- 评测：`scripts/hrfont_e2_official_eval.py` 扩展 `FontDiffuserDeltaDPM_v2`（特征混 Δ 推理路径）
- 停止文件：`reports/hrfont_overnight/STOP_STAGE_A_V2`

### 3.6 Stage A 验收（过线再开 B）

| 检查 | 标准 |
|------|------|
| 协议 | 推理 `_hidden` 与训练同一公式；无像素混 |
| 相对 ft | Demo-8×P1 L1 **≤ 1.10× ft**（93% 水平，pilot 约 1.073） |
| 消融趋势 | 无 Δ / RSI 仍看汉字（E3）应差于 A v2 |
| 视觉 | 复杂字（a、e、g、@）不崩 |

---

## 4. RS-Gap 门控（D4）

### 4.1 定义（替代 `gap_b0_iou.json`）

在 **B₀=FZKTJW** 上编码（与 Δ 同一中性底）：

- \(v_c = \mathrm{L2norm}\big(\mathrm{GAP}(E_c(I(B_0,c)))\big)\)
- \(V_R = [v_{r_1},\ldots,v_{r_8}]\)，\(r_k \in\) ref8
- 投影：\(P_R = V_R(V_R^\top V_R)^{-1}V_R^\top\)（或 QR）
- \(v_c^{\mathrm{miss}} = v_c - P_R v_c\)
- **RS-Gap：** \(\mathrm{gap}(c) = \|v_c^{\mathrm{miss}}\|_2 / (\|v_c\|_2 + \epsilon)\)

### 4.2 与 cover / MMR 统一

- \(\mathrm{cover}(q) = \cos(v_c^{\mathrm{miss}}, v_q)\)，\(q \neq c\)
- MMR：\(\lambda=0.7\)，top-\(K=3\)，\(\mathrm{cover}(q)\ge\theta=0.25\)

### 4.3 阈值 γ

- **不在 Demo-8 上调参**
- 在 **calib4**（4 套训练字体子集）上扫 \(\gamma\)，使 support 开启率约 **30%–60%**（避免 pilot 的 97% 全开）
- 默认起点 \(\gamma=0.35\)，以 calib 上 write_ok / L1 为准锁定
- 冻结写入：`data/hrfont/e0_bank_v2/cache/rs_gap_calib.json`

### 4.4 缓存

- `content_vec_b0[c]`、`ref8_vecs` 在 bank v2 建库时写入 `ec_es_r96.pt`
- 弃用：`freeze_gap()` 的 mask IoU（仅保留作附录对照）

---

## 5. Stage B v2：Support 三臂 + 组合注入（D5 + D6）

### 5.1 硬性排除（你要求的「同 content 字体」）

以下规则 **三臂共用**：

| 规则 | 说明 |
|------|------|
| **E1** | 当前目标字体 \(F\) **不得**出现在 α 池、Δ 混库、Support 取图字体集合（leave-one-out） |
| **E2** | 支撑字 **\(q \neq c\)**（禁止用目标字自身作 support） |
| **E3** | Support 图 **不得**来自目标字体 \(F\)：只从 \(\mathcal{P}\setminus\{F\}\) 的 top-\(M\) 库字体取 \(I(B_s,q)\) |
| **E4** | 域内臂：\(q\) 不得为当前 **style 输入** 那一字（若训时 style=`永`，则 \(q\neq\)「永」可选入池，但 eval 1-shot 永时 support 不应再喂「永」图——**support 与 style 参考字去重**） |

### 5.2 三臂定义（D5）

| 臂 ID | 候选集 \(\mathcal{Q}\) | 行为 |
|-------|------------------------|------|
| **B-in** | ref8 ∪ donors_18（`口回日目田一二十土川人八乙了中申`） | 仅域内中文供形 + 设计锚点 |
| **B-global** | P1 全表（62）∪ donors_18 | 全域检索（含拉丁 o、a…） |
| **B-hybrid** | 先 **B-in** MMR；若 \(\max_q \mathrm{cover}(q) < \theta\) 再 **B-global** 重搜 | 默认主方法候选 |

**门控：** 仅当 \(\mathrm{gap}(c)\ge\gamma\) 才启用 Support；否则前向 = Stage A v2。

### 5.3 组合注入（D6，替代 pilot 平铺 token）

对每个选中 \((q, s)\)：

\[
f_{q,s} = \mathrm{GAP}\bigl(E_c(I(B_s,q)) - E_c(I(B_0,q))\bigr)
\]

\[
W_{q,s} = \tilde\alpha_s \cdot \mathrm{cover}(q)
\]

\[
v_{\mathrm{sup}} = \sum_{q,s} W_{q,s}\, f_{q,s} \Big/ \bigl(\sum W + \epsilon\bigr)
\]

\[
\mathrm{tokens} = \mathrm{SupportAdapter}(v_{\mathrm{sup}})\ \Rightarrow\ \text{拼入 } E_s \text{ 的 cross-attn 序列}
\]

- **SupportAdapter**：Linear→GELU→Linear→LayerNorm，末层 **零初始化**（起步 ≡ A v2）
- **训练 dropout**：20% 整段丢弃 Support（CFG）
- **低 gap**：\(v_{\mathrm{sup}}\) 为空，前向同 A

### 5.4 训练配置

| 项 | 值 |
|----|-----|
| 初始化 | Stage A v2 best ckpt |
| 冻结 | UNet + 两 encoder |
| 可训 | **仅 SupportAdapter** |
| 损失 | `MSE(noise)` |
| 步数 | 25_000 / 臂（或三臂共训一个 Adapter + 条件标签，二选一实现时见 §8） |
| 输出 | `runs/e2_stageB96_v2_{in,global,hybrid}/` |

### 5.5 评测协议（与 ft 公平）

- Content：live DejaVu `render()` @96（与 pilot 相同，待 P1 项可改读盘）
- Style：Demo-8 测试字体 +「永」1-shot
- GT：`unified_v1/renders/128/gt_latin` → 96
- 采样：DPM++ 20 步，CFG 7.5，seed=123
- 主指标：L1；辅：write_ok、**RS-Gap 三分位分层**、B−A 增益曲线

---

## 6. 实验矩阵（v2 主表）

| ID | 方法 | RSI 结构源 | Support | 备注 |
|----|------|------------|---------|------|
| M0 | 官方 FD 零微调 | Ec(汉字) | 无 | 附录 |
| M1 | **ft_cnstyle@25k** | Ec(汉字) | 无 | 主基线 |
| M2 | **Stage A v2** | **Δ 特征混** | 无 | 本计划重训 |
| M3 | A v2 + **B-in** | Δ | 域内组合 | |
| M4 | A v2 + **B-global** | Δ | 全域组合 | |
| M5 | A v2 + **B-hybrid** | Δ | 混合组合 | 主方法候选 |
| M6 | E3：RSI 仍看汉字 | Ec(汉字) | 无 | 消融，@10k 可短训 |

**旧 pilot（A formal / B support）→ 归档 `pilot_v1`，不与 v2 混表。**

---

## 7. 实施顺序（你 review 后执行）

```text
Phase 0  文档锁定（本文）                    ← 当前
Phase 1  hrfont_e0_build_bank_v2.py         渲库 + cache + RS-Gap 表
Phase 2  hrfont_e2_stageA96_v2_train.py     特征混 Δ 训练
Phase 3  A v2 eval + E3 短消融              过验收线
Phase 4  hrfont_support_v2.py             RS-Gap + 三臂检索 + 组合注入
Phase 5  B-in / B-global / B-hybrid 训练
Phase 6  分层评测 + 更新 formal_preview
```

**明确：在你确认本文前，不启动 Phase 2 训练。**

---

## 8. 待实现文件清单

| 文件 | 职责 |
|------|------|
| `scripts/hrfont_e0_build_bank_v2.py` | ft 同源渲染建库 |
| `scripts/hrfont_delta_feature.py` | 特征混 Δ、多尺度 residual |
| `scripts/hrfont_rs_gap.py` | RS-Gap、cover、MMR |
| `scripts/hrfont_support_v2.py` | pick_support_v2（三臂 + E1–E4） |
| `scripts/hrfont_e2_stageA96_v2_train.py` | Stage A 训练 |
| `scripts/hrfont_e2_stageB96_v2_train.py` | Stage B 训练（`--arm in\|global\|hybrid`） |
| `scripts/hrfont_e2_official_eval_v2.py` | 统一评测 |
| `reports/hrfont_overnight/PROTOCOL_AUDIT_V2.json` | 机读协议（训/评分列） |

---

## 9. 保留项（P1 / P2，本计划不锁）

### P1 — 影响公平性与叙事，后续消融

| ID | 议题 | 选项 |
|----|------|------|
| P1-7 | Style 训练采样 | ref8 随机 vs 全中文池随机（α 仍 ref8 proto） |
| P1-8 | 评测 ref | 1-shot「永」vs 8-ref 平均 Es |
| P1-9 | Δ 是否进 MCA | 仅 RSI vs Identity+Change 双路 |
| P1-10 | Stage B 可训范围 | 仅 Adapter vs 浅调 RSI |
| P1-11 | E3 对照 | RSI←Ec(汉字) @10k |

### P2 — 资源与增强

| ID | 议题 |
|----|------|
| P2-12 | α 池：42 kNN vs 12 medoid |
| P2-13 | 指标：write_ok、设计师 2AFC |
| P2-14 | 域内是否含拉丁 q |
| P2-15 | 256 hybrid 分辨率支路（数据先行，训练排在 A v2 之后） |

---

## 10. 风险与对照

| 风险 | 缓解 |
|------|------|
| DejaVu(C) vs B₀(Δ) 双中性底 | 主实验保持 C=DejaVu 与 ft 公平；附录可加「C 也改 B₀」 |
| 特征混在线算 Ec 慢 | 预缓存 top-3 的 \(E_c(I(B_s,c))\) 与 \(E_c(I(B_0,c))\) |
| γ 仍导致 support 过多 | calib4 用分位数门控（top 40% gap）替代固定 0.35 |
| B 三臂训练成本 | 可共用一个 Adapter、三臂混合 batch；或先 hybrid 单臂 |

---

## 11. 入口与归档

| 资源 | 路径 |
|------|------|
| **本计划** | `reports/hrfont_overnight/HRFONT_TRAIN_PLAN_V2.md` |
| 设计正文 | `reports/ICLR2027_HRFONT.md` |
| pilot 结果 | `reports/hrfont_overnight/formal_preview/`（标注 pilot_v1） |
| ft 权重 | `runs/ft_cnstyle/global_step_25000` |
| v2 数据（待建） | `data/hrfont/e0_bank_v2/` |
| v2 A 权重（待训） | `runs/e2_stageA96_v2/` |

---

## 12. Review 检查清单（请你勾选）

- [ ] B₀=FZKTJW 仅用于 Δ/gap/support 差分，Content 输入仍 DejaVu（与 ft 横比）是否接受？
- [ ] 特征混 Δ 公式（§3.2）是否与你的「变化方向」表述一致？
- [ ] Support 排除 E1–E4 是否覆盖「同 content 字体」顾虑？
- [ ] 三臂 B-in / B-global / B-hybrid 是否都要跑满 25k，还是先 hybrid？
- [ ] Stage A 80k 步数与验收线（§3.6）是否同意？
- [ ] **确认后回复「按 v2 开训」**，再执行 Phase 1–2。

---

*生成：2026-09-02 · HR-Font 跨语 v2 训练计划*
