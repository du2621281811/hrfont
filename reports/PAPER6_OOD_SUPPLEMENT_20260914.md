# 补充测试 6 套（Paper6 OOD）— 给合作者

**日期**：2026-09-14  
**状态**：可供肉眼观察；**未**做中西文风格一致性审查；**不可**直接当正式指标集。

## 这是什么

从方正池里挑出的 **6 套不在原 260**（train228 / val16 / test16）的字体，约定归入 **测试集（新增）**，与原 test16 并列作 OOD 展示。

| stem | 显示名 | 风格标签（粗分） |
|------|--------|------------------|
| `FZHanZKTTJW` | 方正憨仔卡通体 简 | cartoon |
| `FZWangDXCJW` | 方正王铎行草 简 | xingkai |
| `FZZJ-ZTMBZKJW` | 方正字迹-朱涛毛笔正楷简体 | shouxie |
| `FZZJ-OYCDXKJW` | 方正字迹-欧阳长迪行楷 简 | xingkai |
| `FZPangPHJW` | 方正胖胖黑 简 | cartoon |
| `FZHPJW` | 方正琥珀简体 | display |

清单文件：`reports/paper6_0914_f0f2/stems.txt`

## 重要限制（请先读）

1. **未检查中西文风格一致性**  
   未逐套确认西文（数字 / Latin / 扩展）与中文 ref 在笔画、对比度、装饰、字重上是否同一设计意图。部分字体西文可能偏「凑字符」或与中文风格脱节。
2. **用途**  
   - ✅ 肉眼观察生成质量、脏/净臂差异、明显失败模式  
   - ❌ 在未进一步确认前，不要把本 6 套上的 L1/SSIM/LPIPS/E12/CLIP 等数字写进正式表或对外结论
3. **若要算指标**  
   需先做风格一致性审查（至少：中文 ref8 vs 西文 GT 的肉眼配对 + 记录不合格 stem/字符），再决定保留子集或整套剔除。

## 看哪里

| 资源 | 路径 |
|------|------|
| F0/F2 脏 vs CLEAN 看板 | `reports/paper6_0914_f0f2/index.html` |
| 本说明（同目录） | `reports/paper6_0914_f0f2/README.md` |
| 元数据 | `reports/paper6_0914_f0f2/meta.json` |
| 筛字过程（更广 shortlist） | `reports/paper_fonts_0914_screen/` |
| 生成脚本 | `scripts/eval_paper6_0914_f0f2.py` |

协议对齐 `timeline_f2_clean`：DPM++20 / CFG7.5 / seed3407 / 8-shot；**未**使用合作者 G2 / G2-RL。

## 与正式 test16 的关系

- 正式分层评测、论文主表仍以 **原 test16** 为准。  
- 本 6 套是 **补充 OOD 观察集**，不进入训练/验证，也不自动并入 test16 stems 列表。
