# TC-v2 Luna implementation review

Date: 2026-09-14

## Implemented

- `src/tc_v2.py`: frozen-VGG descriptor contract (896 dims), train-only standardizer, fail-closed cache reader, 896+707 cross-attention head, and zero-initialized shared 896-to-1024 projection broadcast to the existing nine global tokens.
- `train.py`: TC is opt-in (`--tc_enabled`), independent of Es/alpha/Delta/donor features; target rows are read only for training loss; CFG rows zero the TC residual; H/adapter remain outside the cache `no_grad` block; separate TC learning-rate parameter group; checkpoint save/load; strict TC resume and weight-only warm-start.
- `scripts/build_tc_v2_cache.py`: non-destructive VGG cache builder. Clean-map is required, target teachers are train/val only, test references may be cached, and the manifest records VGG/input/split/clean-map fingerprints and train statistics.
- `scripts/train_tc_v2_head.py`: frozen-cache H pretraining with weighted clean-pair sampling, fixed validation references, warmup, validation readout, and cache/Ec fingerprints.
- `scripts/sample_tc_v2.py`: dedicated cache-backed DDPM sampler for actual G9 + local-L128 + Mean-Delta + TC conditioning, with 1/8-shot refs, CFG, clean donor filtering, checkpoint/cache SHA checks, deterministic seed, and no target-GT reads.
- `tests/test_tc_v2.py`: CPU contracts for H gradients/masks, zero-init/broadcast/global-token injection, cache manifest validation, and split handling.

## Deliberate boundary

The legacy `sample.py` / DPM path was not modified to pretend it supports G-RL+TC. It remains the old sampler and rejects TC flags. `scripts/sample_tc_v2.py` is the dedicated G-RL+TC path; no GT target descriptor is exposed to it.

## Checks run

```text
PYTHONPATH=code/variants/cn2west_f123_rsi/FontDiffuser \
  HF_HOME=/private/tmp/hrfont-tc-validation.HRM6IJ/hf \
  TRANSFORMERS_CACHE=/private/tmp/hrfont-tc-validation.HRM6IJ/hf/hub \
  /private/tmp/hrfont-tc-validation.HRM6IJ/venv/bin/python tests/test_tc_v2.py
Ran 3 tests ... OK

/private/tmp/hrfont-tc-validation.HRM6IJ/venv/bin/python -m py_compile \
  code/variants/cn2west_f123_rsi/FontDiffuser/src/tc_v2.py \
  code/variants/cn2west_f123_rsi/FontDiffuser/src/model.py \
  code/variants/cn2west_f123_rsi/FontDiffuser/train.py \
  code/variants/cn2west_f123_rsi/FontDiffuser/sample.py \
  scripts/build_tc_v2_cache.py scripts/train_tc_v2_head.py
passed

Inline fake-cache CPU probe: `tc_conditions CPU grad/CFG OK`.
Inline clean-map probe: clean target filtering and test-target exclusion OK.
Train module/parser import: passed in the isolated Python 3.11 environment.
```

GPU training, real VGG cache extraction, full diffusion joint training, and G-RL+TC visual sampling were not started. The dedicated sampler was syntax/import checked but not run against real assets. Existing unrelated F/G artifacts, watchdogs, official/ours code, and caches were not modified or deleted. This file is an implementation review, not evidence of experiment results.

## Execution handoff

完整命令统一维护在 [`TC_V2_EXECUTION_HANDOFF_20260914.md`](TC_V2_EXECUTION_HANDOFF_20260914.md)，不再保留缺数据/缓存/日程参数的简化模板。该文件覆盖缓存、H预训练、clean RL与同起点续训、TC联合训练、真实GPU smoke、同采样器的1/8-shot验收。所有命令尚未在执行机启动。

`--warm_start_from` is weight-only and mutually exclusive with `--resume_from`; a TC resume requires non-empty `tc_head.pth`, `tc_global_adapter.pth`, and matching `tc_binding.json`. Baseline warm-start may omit TC modules; a TC-to-TC warm-start must preserve their cache binding.

## 独立 Luna review 与主代理复核

实现与独立 review 分别由两个 `gpt-5.6-luna` 代理完成。review 代理修改独立测试及旧测试夹具，生产修复由实现代理完成；主代理负责整合、复跑检查、文档与 Git。

本轮发现并修复的关键问题包括：

- 共享896→1024投影广播到现有global9，纠正896→9×1024的实现偏差；local/down-path不替换。
- 修复Ec pooled query误取标量；冻结缓存提取与可训练H/W分开，避免 `no_grad` 切断学习。
- 按clean pair映射构建teacher与train-only统计，禁止test target进入缓存；缓存大小、key、进度、维度和非有限统计校验失败时拒绝读取。
- 补齐CFG残差屏蔽、可变Ref mask、梯度裁剪、H预训练warmup顺序与固定验证输入。
- 补齐短pilot的best门槛、weight-only warm-start、非空TC权重与cache/Ec绑定恢复检查。
- 不把旧DPM包装器当成可用TC路径；专用DDPM入口使用完整G9/local/Mean-Delta/TC条件，也支持关闭TC的对照，固定噪声并使用inference mode。

最终通过：TC模块 **3/3**；独立集成检查 **12/12**（包括tiny batch overfit、H/W梯度、CFG、clean mapping、TC warm-start/resume绑定、参数组兼容性与采样接口）；既有review回归 **7/7**；CPU identity-safe RSI检查。训练、缓存构建、H预训练和专用采样的 `--help` 均通过，相关文件语法与 `git diff --check` 通过。独立测试使用真实helper/模块、模拟缓存及AST静态检查，**不是完整UNet在真实数据上的联合训练或真实checkpoint采样**。

最终补充：只有显式设置 `--local_learning_rate` 才拆出local参数组，默认保持旧RL/PRL参数顺序和optimizer组数；strict TC resume拒绝缺失local权重或trainer state；sampler检查实际PNG/JSON输出是否已存在，不覆盖旧结果。代码与文档交付不触发任何训练任务。

ml-training-recipes 在本轮具体落实为冻结编码器与梯度检查、小样本合成学习验证、短pilot、同起点续训锚点、weight-only warm-start与同实验resume分离。生成效果是否改善仍由执行交接中的真实GPU smoke及clean val图像决定。
