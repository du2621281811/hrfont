# 补充集协作同步前 QA

- 输入已选：**232**
- 通过（可同步）：**225** 字 / **224** 家
- 未通过：**7**

## 训练 target（295）
- `ascii_digits`: 10
- `ascii_letters`: 52
- `latin_ext_letters`: 27
- `hiragana`: 83
- `katakana`: 86
- `bopomofo`: 37

## 预览 vs 全表
审核页每语种只渲约 8 个探针字；本 QA 按 **全量 target bucket** 查 cmap，并对命中字形做协议 A 墨迹抽检（拒空心/tofu）。

## 覆盖汇总（勾选语种上）
- `ascii_digits`: 全满 230/230
- `ascii_letters`: 全满 230/230
- `latin_ext_letters`: 全满 199/206
- `hiragana`: 全满 144/144
- `katakana`: 全满 143/143
- `bopomofo`: 全满 53/53

## 问题类型
- `target_cmap_incomplete`: 7

## 未通过字体
- `fzsj_1966717` scripts=['latin', 'latin_ext'] → target_cmap_incomplete
- `fzsj_1966721` scripts=['latin', 'latin_ext'] → target_cmap_incomplete
- `fzsj_1966722` scripts=['latin', 'latin_ext'] → target_cmap_incomplete
- `fzsj_2923481` scripts=['latin', 'latin_ext', 'hiragana', 'katakana', 'zhuyin'] → target_cmap_incomplete
- `fzsj_2923966` scripts=['latin', 'latin_ext', 'hiragana', 'katakana', 'zhuyin'] → target_cmap_incomplete
- `FZSJ-WULXEM` scripts=['latin', 'latin_ext'] → target_cmap_incomplete
- `FZXLB` scripts=['latin', 'latin_ext', 'hiragana', 'katakana'] → target_cmap_incomplete

## 假图风险（探针看起来有字，但 target 不全或有空墨）
- 案例数: 7
  - `fzsj_1966717` script=`latin_ext`
  - `fzsj_1966721` script=`latin_ext`
  - `fzsj_1966722` script=`latin_ext`
  - `fzsj_2923481` script=`latin_ext`
  - `fzsj_2923966` script=`latin_ext`
  - `FZSJ-WULXEM` script=`latin_ext`
  - `FZXLB` script=`latin_ext`

## 给合作者
- 用 `manifest_verified.json` / `stems_verified.txt`（已剔除问题字）
- **不要**直接发未校验的原始 232 若含上述失败项
- TTF 仍不进 git，按 `font_file` + sha1 前缀对齐本地方正库

