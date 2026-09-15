# I2 中文构建审核 — 2026-09-15

审核 Git：`fefab546d185a602e740ba40fe1de02c81fefcc1`。脚本：`scripts/prepare_i_cn.py`；原构建器：`scripts/build_cn2west_v2_proto_abc.py`。

## 结论与待确认事项

**当前 338 张 I2 中文中性图符合原 Protocol A 的绘制规则，并且与“295 目标字 + 338 中文字统一定字号”的原构建器输出逐像素完全一致（338/338）。**

但它们是新增中性内容资产，不是历史中文 ContentImage 的复制。旧 ContentImage 原本只有 295 个目标字。当前环境重绘这 295 字，仅 86 字与历史 PNG 逐像素相同；原因还没有定位。因此，结论是“算法与统一字号一致”，不是“历史渲染环境已精确复现”。

I1 不使用这批新增中文中性图，继续运行。正式 I2 已设置 `I2_RENDER_REVIEW_HOLD`，等待 PI 审阅本页；不删除旧数据，不重绘 V0913 主任务，不自动解除 HOLD。

## 构建规则，供逐项 review

| 项目 | 原 Git Protocol A | I2 当前构建与审计 |
|---|---|---|
| 目标字体的 GT/ref | 同一字体的全部 target295 + style338 一起确定字号 | **直接复用旧 StyleImage PNG**，不重新绘制 GT/ref |
| 中性内容字体 | NotoSansCJK-Regular.ttc | 相同路径；当前字体文件 SHA 见下 |
| 字号选择 | 最大字号，使每字 textbbox 的宽和高都不超过 84 | 原准备脚本读取历史 content_size=83；审计重新用原 find_size_A 验证联合633字仍为83 |
| 画布与边距 | 96×96；内框84×84，对应6px目标边距 | 相同 |
| 居中 | x=(96−bbox宽)//2−bbox.left；y=(96−bbox高)//2−bbox.top | 调用同一个 render_glyph_AB，无手工改中心 |
| 越界 | bbox宽或高大于84时逐字号缩小 | 复用同一函数；不是逐字放大撑满画布 |
| 像素 | 白底黑字，灰度绘制后复制成RGB | 相同，无128→96缩放、无旋转/描边增强 |
| 内容特征 | 冻结 I0 的 Ec，正常图像归一化 | CPU生成，单独保存 features.pt，校验Ec/PNG/特征哈希 |
| 辅助样本 | — | 仅允许的训练字体；中文目标随机；1–8个同字体ref，严格排除目标字符 |

### 为什么字号是83，而不是84？

原 `find_size_A` 实测：只看旧 target295 → **83**；只看 Han338 → **84**；看二者联合633 → **83**。

我们建议采用“**扩展统一中性字符库存，同时保持原83字号**”。这样新增汉字不会因为单独自适应而整体放大。原 `render_content` 历史上只接收 target295；把联合库存作为扩展域是这次新增资产的明确约定，不冒称历史上已渲染了633字。

审计使用独立 Pillow12.2.0 / FreeType2.14.3，未替换训练环境 Pillow；字体 SHA256：
`5dcd1c336cc9344cb77c03a0cd8982ca8a7dc97d620fd6c9c434e02dcb1ceeb3`。

## 验证证据

- [完整逐字机器结果](render_audit_20260915.json)：338/338像素相同；338/338 PNG哈希与已生成清单一致；当前字体哈希一致。
- [中文中性图预览](I2_RENDER_PREVIEW.html)：原始96px显示；全部338原图已在 `reports/experiments/I/CN_CONTENT/` 归档。
- 原295字重绘：86完全相同；最差 `u311B`（ㄛ），平均绝对灰度差8.168945/255。它不只一定是抗锯齿差，不能在没有证据时归因于版本或字体二进制。
- 本次审核只增加独立审计脚本及报告，没有修改运行中的模型/训练脚本，也没有替换任何训练 PNG 或缓存。

复核命令（在 I 独立快照中，CPU）：

```bash
PYTHONPATH=render_deps:. /root/miniforge3/envs/boogu/bin/python \
  scripts/audit_i2_render_20260915.py \
  --out /root/projects/hrfont/reports/i_20260915/render_audit_20260915.json
```

## 请 PI 确认

推荐：接受 **Protocol A算法一致 + 联合库存统一83字号 + 新增汉字独立哈希版本**，通过图片审阅后再解除正式 I2 HOLD。

如果“一致”要求的是**历史环境逐像素复现**，则保留 HOLD，先向原构建执行机索取字体二进制哈希、Pillow/FreeType/布局引擎信息和原始生成配置。不能用“图不空、边界合法”替代这项要求，也不能为了继续排程直接覆盖旧中性图。
