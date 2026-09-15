# E12-b 内部各语种数据范围（给合作者）

日期：2026-09-15  
权威合同：`reports/EXPERIMENT_E12B_V0913_20260914.md`  
可用性表：`manifests/v0913_clean/fonts.tsv`

---

## 1. 字体池（按可用性 bucket）

训练 cache 只用 **train 可用字体 223**（原 train228 去掉 5 个 `exclude`）。  
数据集 val16 / test16 **不进训练**；仅打分时按同一 bucket 规则 keep。

| split | all_scripts | no_bopomofo | han_latin_digit | exclude | 合计 |
|-------|-------------|-------------|-----------------|---------|------|
| train（进 E12-b） | 130 | 58 | 35 | 5（踢掉） | **223 可用** |
| val（数据划分） | 9 | 5 | 2 | — | 16 |
| test（数据划分） | 7 | 6 | 3 | — | 16 |

**bucket 含义（打分 keep）：**

| bucket | 允许计分的语种 |
|--------|----------------|
| `all_scripts` | 汉字 + 拉丁 + 数字 + 假名 + 注音 |
| `no_bopomofo` | 上列 **除注音** |
| `han_latin_digit` | **仅** 汉字 + 拉丁（含扩展）+ 数字 |
| `exclude` | 不进训练、不打分 |

`keep=0` 的 (font,char) **不计指标**。

---

## 2. Cache 里各语种有多少（训练原料）

路径：`artifacts/e12/cache_v0913_b/`（约 332MB，**不在 git**）

| 语种 | 唯一字符数 | cache PNG 条数 |
|------|------------|----------------|
| 汉字 han | 102 | 22,746 |
| 数字 digit | 10 | 2,230 |
| 拉丁大写 | 26 | 5,798 |
| 拉丁小写 | 26 | 5,798 |
| 拉丁扩展 | 27 | 6,021 |
| 平假名 | 83 | 15,604 |
| 片假名 | 86 | 16,168 |
| 注音 | 37 | 4,810 |
| **合计** | — | **79,175** |

说明：假名/注音条数多，是因为 `all_scripts` / `no_bopomofo` 字体多；`han_latin_digit` 字体 **没有** 假名/注音条目。

---

## 3. 模型实际监督用哪些语种

### φ（跨语种风格编码）

| 角色 | 字符范围 | 数量 |
|------|----------|------|
| 中文侧 | config `chinese_chars`（含永和书风骨韵天地 + 扩展） | **104** |
| 西文侧 | `A–Z` `a–z` | **52** |

- 训练目标：同字体「中文 ↔ 拉丁」拉近  
- cache 里假名/注音可存在，但 **φ 主对比对是中文×拉丁**（抄 v51）

### membership（同族判别）

| 角色 | 字符范围 | 数量 |
|------|----------|------|
| query | `A–Z` `a–z` `0–9` | **62** |
| ref | `永和书风骨韵天地`（与生成 ref8 对齐） | **8** |

- **假名 / 注音不进 membership 主门**（与 v51 一致）  
- 对 membership 而言，假名/注音是 OOD；主表请看 **latin+digit**

---

## 4. 打分时报什么语种

| 用途 | 语种范围 |
|------|----------|
| **主表 / 与生成器对齐** | 拉丁 + 数字（membership 训练域） |
| φ Style Score | 同左；ref 仍为 ref8 汉字 StyleImage |
| 假名 | 仅 `all_scripts` + `no_bopomofo` 字体可报（诊断） |
| 注音 | 仅 `all_scripts`（诊断） |
| 扩展拉丁 | 按字体 bucket：`han_latin_digit` 及以上才 keep |

---

## 5. 一句话给合作者

> E12-b 内部：**223 清洗 train 字体**；φ 主学 **中文↔拉丁**；membership 主学 **西文/数字 query + 8 汉字 ref**；假名/注音可在 cache/诊断里出现，**不当 membership 主监督**；正式报分默认 **latin+digit + keep**。

权重下载：https://github.com/du2621281811/hrfont/releases/tag/e12-b-v0913-20260914  
交接：`reports/e12_b/WEIGHTS_HANDOFF.md`
