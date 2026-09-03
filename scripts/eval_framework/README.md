# HR-Font E12 evaluation framework

该目录实现独立的跨语系风格评测器、字符分类器、membership verifier、T1–T4 自测和推理指标。训练数据必须是项目字体池之外的公开/外部字体；严禁使用项目 261 字体或 Demo-8。`fonts_dir` 是运行时决定，不写死进仓库。

## GPU server

先准备字体目录并构建一次可续跑 cache：

```bash
cd scripts/eval_framework
python build_cache.py --fonts_dir /srv/public_fonts --cache_dir /scratch/e12_cache \
  --chars 永和书风骨韵天地ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz \
  --canvas 96 --strategy fixed_size --fixed_size 80
```

也可用 `--fonts_manifest /path/fonts_manifest.txt`（每行一个绝对路径或相对 manifest 的路径）。同一 cache 目录的字体 SHA、字符或渲染参数变化会 fail closed，请换新目录。`per_font_height_fit` 可替换 `fixed_size`。

复制并修改 `configs/*.yaml` 中的 cache/checkpoint/output 路径，然后执行：

```bash
python train_style_encoder.py --config configs/phi_s2.yaml --set train.seed=3407
python train_id_cls.py --config configs/id_cls.yaml --set train.seed=3407
python train_membership.py --config configs/membership.yaml
python self_tests.py --config configs/self_tests.yaml
```

正式 φ_s2 和 ID-CLS 对 3407/3408/3409 三个 seed 分别运行。默认单 GPU、无需 DDP；CUDA 上可开启 fp16，指标始终转为 float32。所有训练目录保存 resolved YAML、canonical JSON 和 SHA-256。self-tests 正式门为 T1 median≥0.6、T2 AUC≥0.90、T3≥0.90（可按冻结协议改为设计稿的 0.85）、T4 正向显著下降。

评测脚本同样使用 YAML（字段可参考 `smoke_test.py` 的 `style`/`identity` 配置）：

```bash
python eval_style.py --config /path/eval_style.yaml
python eval_identity.py --config /path/eval_identity.yaml
```

`eval_style.py` 输出逐样本 SC-R/SC-GT、Rank@1、MRR、margin 和按 script 聚合；`eval_identity.py` 输出 top-1/top-5、macro accuracy、混淆矩阵、paired retention/rescue/failure。当前 cache 输入模式把外部真实渲染作为 GT 正控；正式生成图评测应制作同 schema 的只读 cache，不能覆盖生成结果。

## Local smoke test

macOS 上直接运行：

```bash
python smoke_test.py
```

它使用 `/System/Library/Fonts` 中最多四个 CJK 系统字体，缓存 3 个汉字和 3 个 Latin 字符，实际执行 φ_s2 5 steps、ID-CLS 5 steps、membership 3 steps、放宽门槛的 T1–T4 以及两类 eval。`_smoke_out/`、cache、权重和 runs 均为本地临时产物，已在本目录 `.gitignore` 排除，不应提交。

## Calibration

membership 选择 temperature scaling（无需 sklearn），校准集必须是 held-out families。正式开跑前 PI 需冻结：外部字体目录/许可清单、canvas 与渲染策略、cache 位置、以及是否维持 temperature scaling（若要 isotonic，应另开配置与配对比较）。
