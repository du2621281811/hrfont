# NOTES（交接备忘）

录入日期：2026-09-08。来自业务侧口述，**未在本机跑通评测**。

## poster_font_0724（0724.zip）

- 场景：海报字体识别。
- 约束：单字；需要检出框；中文为主。
- 听说对「框得不太严」相对鲁棒。
- 结构：`detectionapi`（CRAFT）+ `recognitionapi`（CoatNet 等）+ `backbone`。
- 原包权重约：`craft_mlt_25k.pth` ~83 MB，`coatNet1_weight_*.pth` ~134 MB，`ArcFace_weight_*.pth` ~16 MB。

## get_stroke_embedding

- 独立 Feature Embedding API（VGG19 前段 + style-token adapter + 骨架）。
- 明确不做 SD/UNet/VAE/CLIP。
- 用户认为可能覆盖西文；效果未知 → **优先小样本探针**。
- 权重：`stroke_embedding.pt` / `stroke_embedding_test.pt` 各约 59 MB。

## similar_recommend

- 相似字体推荐服务（manage + recommend）。
- 用户：只中文；对大小比例鲁棒一般。
- 原包含超大 CoatNet 权重（单文件数百 MB）及多套 TTF，仅 Release/本机保留。
