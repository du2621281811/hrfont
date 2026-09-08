# 公司侧字体工具（外部评测候选）

把三套**现成公司工具**放到仓库里，方便合作者先看代码与接口，再决定能不能接进 HR-Font / E12 风格评测。  
**结论尚未做：** 本目录只做归档与说明，不代表已验证可用。

| 目录 | 来源 zip | 体积（原包） | 用户备注（待验证） |
|---|---|---|---|
| [`poster_font_0724/`](./poster_font_0724/) | `/root/0724.zip` | ~207 MB | 海报字体识别；**单字**；要框；边界略松据说较鲁棒；含检测+识别+backbone；**主攻中文** |
| [`get_stroke_embedding/`](./get_stroke_embedding/) | `get_stroke_embedding.zip` | ~104 MB | 笔触/骨架 embedding API；**可能支持西文**；效果未知 |
| [`similar_recommend/`](./similar_recommend/) | `similar_recommend.zip` | ~913 MB | 相似字体推荐；**仅中文**；大小比例鲁棒一般 |

## Git 里有什么 / 没有什么

- **有：** 源码、配置、小样例图、字体列表/阈值等文本（合计约 2 MB）。
- **没有：** `.pth` / `.pt` 权重、大量 `.ttf` 字库、运行日志、`__pycache__`。  
  GitHub 单文件 100 MB 限制；整包约 1.2 GB，不适合进主仓库。

**完整原包（含权重）**：[GitHub Release `external-eval-company-fonts-20260908`](https://github.com/du2621281811/hrfont/releases/tag/external-eval-company-fonts-20260908)  
训练机本地仍保留：

```text
/root/0724.zip
/root/projects/get_stroke_embedding.zip
/root/projects/similar_recommend.zip
```

## 与 HR-Font 的关系（评测视角）

我们缺的是**过门的风格打分器**（E12 T2 仍失败）。这三套更像「识别 / 检索 / embedding」产品线：

1. **poster_font_0724** — CRAFT 检字框 + CoatNet/ArcFace 类识别；若只能单字且偏中文，对「生成拉丁像不像目标字体」帮助有限，但可作中文侧对照或检测预处理。
2. **get_stroke_embedding** — 轻量 stroke/skeleton embedding + 相似度 API；若西文可用，最值得先做 **F0/F3 生成图 vs GT** 的小规模探针。
3. **similar_recommend** — 库内相似推荐（CoatNet 权重极大）；脚本注明比例鲁棒一般，作跨语种生成评估要谨慎。

建议合作者顺序：先读各子目录入口代码 → 用 Release 权重起服务 → 用 `reports/f03_test16_strat` 里少量 GT/F0/F3 图做手工/脚本对比 → 再决定是否写正式对接。

## 快速入口

| 包 | 入口 |
|---|---|
| poster | `poster_font_0724/detectionapi/founder_detection.py`、`recognitionapi/founder_recognition.py`、`recognitionapi/backbone/networks.py` |
| stroke | `get_stroke_embedding/README.md`、`app.py`（`/embed/stroke`、`/similarity/stroke`） |
| similar | `similar_recommend/font_recommend_service_1020/service.py`、`font_manage_service_1020/networks.py` |

权重还原示例（stroke）：把 Release / 原 zip 里的 `weights/*.pt` 放回 `get_stroke_embedding/weights/`。  
poster / similar 的 `.pth` 路径以各服务 `service_config` / `backup/` 约定为准。
