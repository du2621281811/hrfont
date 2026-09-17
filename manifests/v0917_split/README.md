# 0917独立划分

划分及GT清单已经构建，完整性验证见 [报告](../../reports/dataset_v2_20260917/REPORT.md)。

- `split.json`：固定字体划分；原0913的归属不变。
- `pairs_*.tsv`：唯一合法target GT清单。不要直接扫描旧dirty数据根作为训练列表。
- `fonts.tsv`：来源、split、可用性、缺字信息；旧5个排除字体未被重新启用。
- `style_pool.json`：该字体真实存在的中文参考码点；禁止替代字体或自动繁简转换。
- `reference_compatibility.json`：原ref8不兼容字体及所有字体共同可用的8字备选；未应用到现有K族实验。
- `donor_train_by_cp.json`：逐字符train donor，训练时必须以此排除缺字与未授权语种。
- `donor_train.json`：仅粗粒度分组索引，不代表每个字体覆盖组内所有码点。
- `sample_weights.json`：兼容的50/38/12抽样权重配置；未启动训练。
- `INDEX.json`：派生数据统计及清单SHA256。

PNG在V100 `/root/data1/hrfont_dataset_v2_20260917/v0917`。字体文件与PNG不进Git。后续新训练须适配新的pair、逐字donor与参考池，不能仅替换data_root就复用旧固定字表/缓存。
