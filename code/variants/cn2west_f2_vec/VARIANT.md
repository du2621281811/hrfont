# Variant `cn2west_f2_vec`

F2 mean-Δ 上挂可微渲染矢量头（VecFontSDF-lite）的 **side study**。
不覆盖 `F2-DELTARSI-A-S3407`，不进论文主表。

派生自 `cn2west_f123_rsi`（Δ / cache / identity-safe RSI 沿用）。
设计：[`reports/F2_VEC_MULTITASK_DESIGN_20260913.md`](../../../reports/F2_VEC_MULTITASK_DESIGN_20260913.md)。

| arm | `--rsi_source` | `--support` | 额外 |
|---|---|---|---|
| F2VEC | `delta` | off | `VecHead` 16 二次基元；前 5k 只热矢量头；之后 `x0_hint` 以 0.15 进头（不改 UNet 结构） |

推理 oneshot/fewshot 走与 F2 相同的 DPM 栅格路径：`scripts/eval_f03_test16_strat.py --method F2VEC_40000`（few-shot Es=ref8）或 `F2VEC_40000_s1`（one-shot Es=永，Δ 仍 ref8）。

```bash
python scripts/launch_cn2west_f2_vec.py --smoke
python scripts/launch_cn2west_f2_vec.py --yes
CUDA_VISIBLE_DEVICES=2 python scripts/eval_f03_test16_strat.py generate --method F2VEC_40000 --device cuda:0
CUDA_VISIBLE_DEVICES=2 python scripts/eval_f03_test16_strat.py generate --method F2VEC_40000_s1 --device cuda:0
```
