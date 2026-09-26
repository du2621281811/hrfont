# K7-B 32K 源码归档

这是为合作者核查 K7-B 32K 实验实现准备的独立源码快照。

- 训练运行：`K7-B-V3-K0-S3407-50K-R1`
- 核查 checkpoint：`global_step_32000`（EMA）
- 来源目录：服务器 `/root/projects/hrfont_k7_v3_20260922`
- 来源身份：`K7_CODE_IDENTITY.json` 中记录的 `k7-b-v3-20260922-r15-overlay-checkpoints+ec-lru-return-v2+dynamic-es-sharded-cache`
- Git 根仓库基线：`2952d0efe091f900a11599ebdd14645f9f6c74cf`

`source_snapshot/` 按原始 `K7_CODE_IDENTITY.json` 的文件清单收录源码；`provenance/` 保存训练配置、32K checkpoint 元数据和身份清单，供交叉核对。归档不包含 EMA/模型权重、字体数据、bank、缓存或推理图片。服务器原工作树保持不变；此归档位于单独分支。

本快照是源码审阅材料，不是脱离原训练环境即可直接运行的完整打包环境。运行仍需原有数据、预训练资产与依赖环境。
