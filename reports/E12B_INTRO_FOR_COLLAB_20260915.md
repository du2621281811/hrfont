# E12-b 介绍（给合作者）· 数据说清楚

日期：2026-09-15  
定位：**同域家族风格打分器**（协议 A PNG + `v0913_clean` 可用性）  
不是：字形生成器；不是外部零重叠评测器（那是另一条 E12 线）

合同：`reports/EXPERIMENT_E12B_V0913_20260914.md`  
权重 Release：https://github.com/du2621281811/hrfont/releases/tag/e12-b-v0913-20260914  
交接：`reports/e12_b/WEIGHTS_HANDOFF.md`

---

## 1. 它是什么、输出什么

两段模型，串起来用：

| 组件 | 一句话 | 输出 |
|------|--------|------|
| **φ**（phi） | 把字形图编成风格向量；同字体的中文与拉丁应靠近 | **Ours φ / Style Score**（与 ref8 原型的余弦） |
| **membership** | 冻住 φ，再学「这张西文 + 这组中文 ref，是不是同族」 | **mem 概率**（辅助） |

打分时（生成器结果上）：

1. 取目标字体 **8 个中文 StyleImage**（永和书风骨韵天地）→ φ → 均值作原型  
2. 生成图 / GT 过 φ → 与原型比余弦 → **主指标**  
3. 同一对输入过 membership → **辅指标**（主表看 latin+digit）

---

## 2. 数据总览（先抓这张表）

| 层级 | 是什么 | 规模 | 进 git？ |
|------|--------|------|----------|
| 原始 PNG | 协议 A `train/StyleImage` + `TargetImage` | ~576MB 全 train 池 | 否（数据仓本地） |
| 可用性 | `manifests/v0913_clean/fonts.tsv` + `pairs_train.tsv` | 260 字体行；train 对 56,429 | **是** |
| **训练 cache** | `artifacts/e12/cache_v0913_b/` | **223 字体 · 79,175 张 · ~332MB** | **否**（可重建） |
| φ 权重 | `runs/e12_phi_s2_b_s3407/best.pt` | **129MB** | **否 → Release** |
| membership 权重 | `runs/e12_membership_b_s3407/best.pt` | **47MB** | **否 → Release** |
| 配置 / 说明 | `configs/e12_*_b_*.yaml`、本文件、STATUS | 小 | **是** |

**打分不读 cache**，只读测试侧 StyleImage + 生成图/GT。

---

## 3. 数据怎么流（四步）

```
① 按 v0913_clean 筛字体与语种
② 筛过的 PNG → 建成 cache（只服务训练）
③ cache 上训 φ → 冻 φ 训 membership
④ 打分：test Style(ref8) + 生成图（不读 cache）
```

| 步骤 | 用哪些数据 | 不用哪些 |
|------|------------|----------|
| ① 筛 | train 字体；`fonts.tsv` bucket；`pairs_train` 允许的 (font,char) | 5 个 exclude；某语种 `keep=0` 的格子 |
| ② cache | 协议 A 已有 PNG，**不重渲** | 生成图；val16/test16 字体 |
| ③ 训 | 见 §4（字符表锁死 = v51） | 正式 test16 选 ckpt；生成图 |
| ④ 打分 | 应用集字体（如 test16）的 Style + pred/GT | cache；训练 holdout 字体列表 |

---

## 4. 每一次训练的数据范围（核心）

字体：cache **223** → 族级 **70/15/15**（seed 3407）≈ **train 156 / val 33 / test 34**  
（这是打分器自己的 holdout，**不是** 数据集的 val16/test16。）

### 4.1 训 φ

| 项 | 内容 |
|----|------|
| 字体 | holdout **train ≈156** |
| **中文** | **104** 字：`永和书风骨韵天地一二三四五六七八九十人大小上下中山水火木金土日月年时分口目手足心生学国家民工力田车马鸟鱼虫石花草树林春夏秋冬东西南北左右前后来去出入开关高低长短多少新旧好坏黑白红黄蓝绿雨雪云电光明暗江海河湖原` |
| **拉丁** | **52** 字：`A–Z` + `a–z` |
| 一条样本 | 同字体 **1 张中文 + 1 张拉丁** |
| 索引规模 | 约 **4836** 个 (font, 中文)；拉丁按 index 轮换 |
| 训练量 | max **5000** steps，batch **64**，对称 InfoNCE |
| **明确不含** | 数字、扩展拉丁、假名、注音（不当 φ 主对比字符） |

### 4.2 训 membership（冻住 φ）

| 项 | 内容 |
|----|------|
| 字体 | 同上 **train ≈156**（episode 内抽同族/异族） |
| **query** | **62** 字：`A–Z` + `a–z` + `0–9` |
| **ref** | **8** 汉字：`永和书风骨韵天地`（与生成协议 ref8 对齐） |
| 一条样本 | **1** query 图 + **8** ref 图；偶数同族 / 奇数异族 |
| 训练量 | **20000** episodes，batch **32**，BCE + val 上温度缩放 |
| **明确不含** | 假名、注音、扩展拉丁、其余汉字（不当 membership 主门） |

### 4.3 打分时（对照，非训练）

| 项 | 内容 |
|----|------|
| ref | 目标字体 StyleImage **同一 ref8** |
| query | GT 或方法生成图 |
| **主表语种** | **latin + digit**，且字体 bucket 允许（`keep=1`） |
| 假名 / 注音 | 仅诊断子集；注音还要求 bucket=`all_scripts` |

---

## 5. 字体可用性 bucket（筛与 keep 同一套）

来源：`manifests/v0913_clean/fonts.tsv`

| bucket | train 字体数 | 该字体允许的语种 |
|--------|--------------|------------------|
| `all_scripts` | 130 | 汉 + 拉丁 + 数字 + 假名 + 注音 |
| `no_bopomofo` | 58 | 上列 **无注音** |
| `han_latin_digit` | 35 | **仅** 汉 + 拉丁(含扩展) + 数字 |
| `exclude` | 5 | 不进 cache、不训练、不打分 → 故为 **223** |

数据集 val16/test16 字体集合仍在，但 **不进 E12-b 训练**；打分时同样按 bucket keep。

---

## 6. Cache 里各语种存量（原料，≠ 训练监督表）

路径：`artifacts/e12/cache_v0913_b/`（~332MB）

| 语种 | 字符种数 | PNG 条数 |
|------|----------|----------|
| 汉字 | 102 | 22,746 |
| 数字 | 10 | 2,230 |
| 拉丁大写 / 小写 / 扩展 | 26 / 26 / 27 | 5,798 / 5,798 / 6,021 |
| 平假名 / 片假名 | 83 / 86 | 15,604 / 16,168 |
| 注音 | 37 | 4,810 |
| **合计** | — | **79,175** |

注意：cache 里假名/注音很多，是因为大量 `all_scripts`/`no_bopomofo` 字体被导入了；**φ / membership 主监督并不吃它们当主字符表**（见 §4）。

---

## 7. 和生成器评测怎么接

- 生成器（F0/F2 等）在 **test16** 上出图  
- E12-b 用同一字体的 **ref8 中文 Style** + 生成西文图打分  
- 只报 **`keep=1`** 的格子；主对比子集 = **latin+digit**  
- 不把 E12-b 数字写成「外部零重叠」结论；人评 Spearman 仍待填

---

## 8. 合作者最小使用

```bash
git pull
gh release download e12-b-v0913-20260914 -p e12-b-v0913-best-20260914.tar.gz
mkdir -p runs && tar -xzf e12-b-v0913-best-20260914.tar.gz -C runs/
# 核 SHA：见 reports/e12_b/WEIGHTS_HANDOFF.md
python scripts/score_preds_e12_b.py --device cuda:0
```

重训才需要 cache：跑 `scripts/build_e12_cache_v0913_b.py`（需本地协议 A PNG），或另要 cache 包。

---

## 9. 一句话备忘

> **数据：223 清洗 train → 7.9 万 cache。**  
> **φ：~156 族 × 104 中文 ↔ 52 拉丁。**  
> **membership：同族划分 × query 62（字母数字）+ ref 8 汉字；假名注音不进主门。**  
> **打分：不读 cache；主看 latin+digit + keep。**
