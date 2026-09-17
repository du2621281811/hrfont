# 0917独立划分与v2融合数据集 Review

已将合作者 `main@4ececb4300bba7aed703d1505f8021afbf7a1537` 合并到工作分支。正式0917名单为232字体，225仅为满覆盖子集。0917先独立完成字体分组与train/val/test划分，再与0913的同名split合并。

[逐字体清单](font_inventory.tsv) · [机器验证结果](VERIFICATION.json) · [0917划分清单](../../manifests/v0917_split/split.json) · [v2划分清单](../../manifests/v2/split.json)

完整review图册（984张PNG）保存在V100 `/root/data1/hrfont_dataset_v2_20260917/review/`，已交付本地 `outputs/DATASET_V2_20260917_REVIEW/`。打开 `index.html` 审阅v2，打开 `v0917.html` 仅审阅0917；大图不进Git，可用本文命令重建。

## 划分结果

| 数据版本 | split | 名义字体 | 有效字体 | target GT | 西文 | 假名 | 注音 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0917独立 | train | 200 | 200 | 39,547 | 17,012 | 20,870 | 1,665 |
| 0917独立 | val | 16 | 16 | 3,208 | 1,370 | 1,690 | 148 |
| 0917独立 | test | 16 | 16 | 3,208 | 1,370 | 1,690 | 148 |
| v2融合 | train | 428 | 423 | 95,976 | 36,859 | 52,642 | 6,475 |
| v2融合 | val | 32 | 32 | 7,331 | 2,794 | 4,056 | 481 |
| v2融合 | test | 32 | 32 | 7,088 | 2,794 | 3,887 | 407 |

0913名义划分仍为228/16/16；原来被排除的5个train字体仍被排除，未重新启用。0913的64,432条有效target样本逐行保留，原有Target/Style/Content PNG不重渲染。v2共492个名义字体，487个有效字体。

## GT与中文参考规则

- `latin`只授权数字10＋大小写字母52；`latin_ext`另授权27个扩展字；`hiragana`、`katakana`、`zhuyin`分别授权83、86、37字。未勾选语种即使TTF有cmap，也不生成target GT。
- 7个Ext不完整字体全部保留，70个已知缺失font×char组合全部跳过。未使用225字体子集代替232全集。
- 实际渲染新增45,963个target GT，自动渲染拒绝0个；逐字检查cmap、glyph ID、空图、`.notdef`和触边。无替代字体、无自动繁简转换。
- 4个字体缺少346个简体中文参考字符，真实参考池合计78,070张。它们依然保留在232中；中文参考池仅包含该TTF确实支持且成功渲染的字符。

| 字体 | 可用中文参考 | 原固定ref8缺字 |
|---|---:|---|
| FZXLB | 250 / 338 | 书风韵 |
| FZZJ-MJZCFU | 252 / 338 | 书风 |
| ZKTBanQTFU | 252 / 338 | 书风 |
| ZKTMingXTFU | 252 / 338 | 书风 |

`style_pool.json`是逐字体实际参考池。所有字体共同可用的8字备选为 **永和骨天地山水人**，仅作为v2后续协议选项；没有修改已有K族实验的ref8。不能直接把原固定ref8套到上表4字体。

`donor_train_by_cp.json`给出每个目标字符在train中可用的donor字体，后续训练必须优先按此逐字过滤。粗粒度的`donor_train.json`仅供分组索引，不能据其推定该字体覆盖所有扩展拉丁字。

## 分组、泄漏与冻结

- seed=3407。新字体最终形成192个分组，目标200/16/16；分组考虑官方family、TTF name16、明确字重后缀和ASCII62＋中文ref8的完全相同图像。
- 与旧字体存在明确关系的新字体跟随旧split，其余整组划分并按语种覆盖、缺字状态平衡。划分不使用生成模型的质量或测试指标。
- `fzsj_1275635`与`fzsj_2017088`的ASCII62＋中文8图像相同，放在同一split；这是一项分组依据，不等于断言其全部字符相同。
- 新增分组跨split为0。0913自身已有以下两组历史跨split关系，按你的要求保留，不声称v2完全不存在历史泄漏：

  - test, train：FZDeSHJW_507R, FZDeSHJW_508R, FZDeSHJW_509R, FZDeSHJW_510M, FZDeSHJW_511M, FZDeSHJW_512B, FZDeSHJW_513B, FZDeSHJW_514H, FZDeSHJW_515H。
  - train, val：FZYouHJW_508R, FZYouHJW_509R, FZYouHJW_510M, FZYouHJW_511M, FZYouHJW_512B, FZYouHJW_513B, FZYouHK_508R, FZYouHK_509R, FZYouHK_510M, FZYouHK_511M, FZYouHK_512B, FZYouHK_513B。

旧TTF文件在V100不可用，因此旧版完整家族谱系无法穷尽核验。图像指纹与元数据分组可发现已识别的重复/同系列关系，不能证明不存在其他未识别近似字体。

## 渲染与验证

0913保留原像素；0917使用Protocol A的96×96 RGB、6px边界、每字体统一字号、居中、无resize。字号在“获准且存在的target＋存在的中文338池”上计算；没有让缺字框或未获准语种决定字号。
新渲染运行库：Pillow 11.3.0 / FreeType 2.13.3；旧summary记录Pillow 12.2.0 / FreeType 2.14.3。两批不能据此声称跨运行库逐像素等价，旧数据本身没有重渲染。

全量验证状态 **PASS**：
- 153,197个原Target/Style/Content文件的SHA256与融合引用核对一致；原manifest和split文件哈希未变。
- 124,033张新增目标/参考PNG逐文件哈希、96×96与RGB检查通过。
- 三个split互斥，0913归属不变，旧pair逐字段相同，v2恰好等于0913与独立0917的并集；文件目录中的GT集合与pair清单一致。
- 缺字、未勾选语种不进入target；逐字donor只来自train；参考池与实际PNG一致。
- 划分重算一致；额外负向检查确认会阻断新分组同时连接多个旧split。

上述是机器检查。完整语义正确性仍需你在review页审阅；不能把cmap/非空/哈希通过写成人工确认了每个字形。字号偏小、原排除字体、缺字字体已标记，网页“仅缺字或异常”可筛选。

## 数据位置与复现

V100数据根目录（合法PNG的链接视图，引用现有只读来源，不重复改写旧图）：

```text
/root/data1/hrfont_dataset_v2_20260917/v0917
/root/data1/hrfont_dataset_v2_20260917/v2
/root/data1/hrfont_dataset_v2_20260917/manifests/v0917
/root/data1/hrfont_dataset_v2_20260917/manifests/v2
```

需要保留 `/root/projects/hrfont/data/fontdiffuser-p253-t295-s338-cn2west-v2` 和本次 `rendered/` 目录。迁移时必须保留或重建引用，不能只复制软链接文本。原TTF位于本次 `source/ttf/`，原ZIP和字体文件不进Git。

```bash
python scripts/build_v0917_v2_dataset.py render --root <build-root> --contracts <frozen-contract-root> --workers 3
python scripts/build_v0917_v2_dataset.py assemble --root <build-root> --contracts <frozen-contract-root> --old-root <original-0913-png-root>
python scripts/build_v0917_v2_dataset.py verify --root <build-root> --contracts <frozen-contract-root> --old-root <original-0913-png-root>
python scripts/review_v0917_v2_dataset.py --root <build-root> --contracts <frozen-contract-root>
```

构建只使用低优先级CPU，没有启动v2训练，也没有更改正在执行的K3及其数据。新训练前需接入新split、pair、逐字donor与参考池，并重新定义v2的matched评估协议。

## 输入身份

- ZIP SHA256：`25de9f1759df0b6ae86cb85464aefa4f6707400ff6ebb42331cf8c00394c6b14`
- 正式232清单SHA256：`9b5308a0d6a389d279e4424bba5ab8cedfabb5f8663bf98049bcfa4abe56ecce`
- 0913 split SHA256：`7aca8a4724142c35b38129847eb5d0be199cfc4824160c5cb69ccced309a2f37`
- `manifests/*/INDEX.json`列出衍生清单的全部哈希；`split_audit.json`记录分组依据；`old_source_sha256.tsv`记录所有旧文件指纹。

交付检查：984张完整字形图册均通过本地PNG完整性检查，manifest哈希一致；桌面和手机浏览器的字体数、split/来源/检索筛选、缺字提示与原尺寸图片加载通过。另抽查16个重点字体的实际GT；这是视觉抽查，不替代你对全部字形的review。记录见`BROWSER_QA.json`、`DELIVERY_FILES_QA.json`、`VISUAL_SPOT_CHECK.json`。
