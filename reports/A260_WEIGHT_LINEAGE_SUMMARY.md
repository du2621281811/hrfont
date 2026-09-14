# A/260 字库字重 / 同字型分布速览

- 总字体: **260**（train228+val16+test16）
- 去掉明确字重后缀后的设计基数: **257**
- 同一设计 ≥2 字重: **2** 组 / **5** stems
- 名称带明确字重后缀: **83** (31.9%)
- 无明确后缀（一名一品）: **177**

可视化: `reports/e12_eye_probe_v4_s3407/a260_weight_situation.png`

## 字重后缀计数

| 后缀 | 约义 | 次数 |
|---|---|---:|
| `(none)` | 无明确后缀 | 177 |
| `EB` | ExtraBold | 19 |
| `H` | Heavy/Black | 16 |
| `B` | Bold | 12 |
| `L` | Light | 9 |
| `T` | Thin | 7 |
| `Cu` | 粗 | 4 |
| `EL` | ExtraLight | 4 |
| `R` | Regular | 4 |
| `UL` | UltraLight | 2 |
| `Italic` | 斜体 | 1 |
| `Bold` | Bold | 1 |
| `Xi` | 细 | 1 |
| `UB` | UltraBold | 1 |
| `M` | Medium | 1 |
| `DB` | DemiBold | 1 |

## 同字型多字重（池内真实并存）

- **FZSiNTJW**: `FZSiNTJW-H`, `FZSiNTJW-UB`, `FZSiNTJW-UL`
- **FZXinZYHJW**: `FZXinZYHJW_EB`, `FZXinZYHJW_UL`

## 结论

- A/260 **几乎不做成套字重覆盖**：260 stem ≈ 257 个不同设计。
- 只有 **2 组** 在池里同时有多个字重。
- 约四成名字带 -B/-EB/-H 等，但多数是「这个产品刚好是粗体版」，旁边没有细体兄弟。
- 方法库 = **广覆盖字型**；不是 Noto 那种 **字重阶梯**。
