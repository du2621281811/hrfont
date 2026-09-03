# Mentor 只看这一页

**v2 训练计划（待 review 后开训）：** [`HRFONT_TRAIN_PLAN_V2.md`](HRFONT_TRAIN_PLAN_V2.md) — 特征混 Δ、ft 同源渲染、RS-Gap、Support 三臂+组合注入。下方网页为 **pilot_v1** 结果。

**网页：** http://172.19.45.13:19000/reports/hrfont_overnight/formal_preview/

页面结构（给 Mentor）：
1. **任务是什么** — 输入输出、改什么
2. **数据划分** — 42 训练 / 8 测试 / 976 格
3. **三个实验怎么设** — 训练步数、训什么、RSI 接什么
4. **各实验输入渲染** — Content/Style/Δ/Support 来源、脚本、train vs eval（§④ + `PROTOCOL_AUDIT.json`）
5. **评测怎么做** — 5 步流程 + ft vs A 公平性
6. **结果** — L1 表 + 分字体 + gap + 版本绑定

**主表数字以 metrics JSON 为准**（A@80k: 8/28 06:02 · B@25k: 8/29 02:07）。8/29 01:10 的 B eval 无效。

## 发给 Mentor（离线）

**不要**只下载网页另存为单个 `.html`（图片会丢，打不开或空白）。

请发整个文件夹或离线包（生成命令：`python scripts/hrfont_formal_preview_export.py`）：

| 文件 | 说明 |
|------|------|
| `formal_preview/export/HR-Font_Mentor汇报_离线单页.html` | **推荐**：单文件，图已内嵌，双击即用 |
| `formal_preview/export/HR-Font_Mentor汇报_完整包.zip` | 解压后打开 `index.html`（需保留 png 等同目录） |
| `formal_preview/export/打开说明.txt` | 给 Mentor 的简短说明 |

**E3 对照实验**（RSI 仍看汉字 @10k）若未完成，见 `e3_pipeline.log` / `STATUS_OVERNIGHT.md`。

其它长文档备查，勿作主入口。
