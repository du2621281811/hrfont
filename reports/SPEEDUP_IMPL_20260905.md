# HR-Font Stage A 加速实现报告（2026-09-05）

基线：`6616836`。本次只改变缓存构建/读取与批处理方式；K=10、τ=0.07、数据、训练 RNG 调用顺序、CFG/Δ draw 顺序和验证 cadence 均未改。

## 按文件说明

- `code/variants/cn2west_stage_a/FontDiffuser/train.py:139`：Δ 路径从离线余弦表按 `(font, ref chars)` 聚合整批 `[B,228]` 分数，逐行 mask 自身后只调用一次 `torch.topk`；非 train 的验证字体保留旧 Es 计算作为低频 fallback。`official` 分支与 `delta_enabled=false` 分支仍在任何新逻辑前返回。
- `code/variants/cn2west_stage_a/FontDiffuser/train.py:176`：收集整批 top-10 的 `(train_font, cp)` union；target 与 neutral 均去重、逐尺度合批 H2D，再按原邻居顺序调用 `mix_cached_delta`。权重转 CPU 后仍按旧函数逐项 `acc.add_(..., alpha=float(weight))`，保持 Δ 累加/减 neutral 语义。
- `scripts/hrfont_feature_cache.py:114`：新增只读 mmap `CosineTable`，验证 dtype、shape、文件字节数、字体/字符顺序、Es checkpoint SHA 与 `pooled.dat` SHA；启动时逐页预热。
- `scripts/hrfont_feature_cache.py:183`：新增 `EcCache.features_many()`；memmap handle 继续常驻，输入键稳定去重，每个尺度一次 indexed read。
- `scripts/hrfont_build_e1_caches.py:240`：新增 `--which cosine`。构建前 fail-closed 校验 Es manifest 记录的 style encoder SHA 与实际 checkpoint；以 fp32 重新归一化、fp32 matmul，最后仅落盘 fp16，并记录量化最大绝对误差。
- `scripts/e2_smoke_test.py:61`：新增固定 seed 的 512-sample 旧 `compute_alpha` vs fp16 离线分数 top-10 exact-set parity；打印 agreement 与最多 5 个边界翻转样例。
- `scripts/e2_smoke_test.py:97`：新增旧逐键 Ec fetch vs `features_many` 去重 fetch 的多尺度 Δ allclose parity。
- `code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py:34` 与三个 YAML：只增加余弦表路径和可选 Es-cache SHA 参数。E1c/E2b official 路径不会实例化或读取余弦表。

## 表格式与规格校正

数据文件默认 `artifacts/e2/es_cosine_e1_100k.f16`，裸 little-endian/native NumPy `float16`，C-order，layout 为 `C[query_font, style_char, library_font]`；伴随元数据为同路径加 `.json`。JSON 保存 shape、完整 fonts/library_fonts/chars 顺序及索引 map、`pooled.dat` SHA-256、Es checkpoint SHA-256、构建参数、fp32→fp16 最大绝对误差和时间戳。

实际 shape 是 **[228,338,228]**，文件约 35.3 MiB。任务文字同时要求 `[228,338,227]`、运行时按全局 `f` 索引 mask、并与当前 `_LibraryEs` 的 228-font 候选顺序一致；三者不能同时成立。HEAD 的旧行为是 228 候选、运行时 leave-one-out 后剩 227，因此实现保留 228 列并在 top-k 前把 self 列置为 `-inf`。这才与旧 leave-one-out 语义一致。

## 本地验证结果

- `PYTHONPYCACHEPREFIX=/tmp/hrfont-pycache-speedup python3 -m py_compile ...`：PASS（全部 7 个相关 Python 文件）。
- `python3 scripts/hrfont_delta_v2.py smoke`：PASS。
- `python3 scripts/e2_smoke_test.py`：PASS。
  - 合成、固定 seed 的 512-sample top-10 exact-set agreement：**509/512 = 99.414062%**。
  - 边界翻转：sample 30，old-only 107/new-only 27，10/11 margin `9.2312694e-06`；sample 117，old-only 66/new-only 122，margin `3.3549964e-05`；sample 273，old-only 64/new-only 159，margin `1.1086464e-05`。
  - batched/deduplicated Ec Δ parity：PASS（`rtol=1e-6, atol=1e-7`）。
  - 9-token cache-only forward：PASS。
- `git diff --check`：PASS。

本 sandbox 没有 E1@100k checkpoint、Es/Ec 正式 artifacts，因此**未构建正式余弦表、未在真实 228×338 Es cache 上复测 512-sample agreement、未做 CUDA 吞吐/显存测试，也未启动训练**。以上 99.414062% 是 smoke 的确定性合成检查，不能冒充真实缓存结果。

## Cursor / 机器侧重启 80k 前清单

```bash
cd ~/projects/hrfont

# 1. 构建正式表；会先验证 Es manifest 中 checkpoint SHA，失败即终止
python3 scripts/hrfont_build_e1_caches.py --which cosine \
  --es-out artifacts/e2/es_spatial_e1_100k \
  --cosine-out artifacts/e2/es_cosine_e1_100k.f16

# 2. 将 meta 的 es_cache_sha256 填入 configs/e2_stage_a_s3407.yaml 的
#    data.cosine_table_es_cache_sha256（不要给 E1c/E2b 强制使用）
python3 -c 'import json; print(json.load(open("artifacts/e2/es_cosine_e1_100k.f16.json"))["es_cache_sha256"])'

# 3. 静态与 smoke；记录真实机器输出
PYTHONPYCACHEPREFIX=/tmp/hrfont-pycache-speedup python3 -m py_compile \
  code/variants/cn2west_stage_a/FontDiffuser/train.py \
  code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py \
  code/variants/cn2west_stage_a/FontDiffuser/configs/fontdiffuser.py \
  scripts/hrfont_feature_cache.py scripts/hrfont_build_e1_caches.py \
  scripts/e2_smoke_test.py scripts/hrfont_delta_v2.py
python3 scripts/hrfont_delta_v2.py smoke
python3 scripts/e2_smoke_test.py

# 4. 用真实 Es cache 扩展/运行同一 512-sample parity，并确认 >=99%；
#    然后做短程独立 output-dir 吞吐测试，核对 draw_log 的 RNG 序列后再恢复 80k。
python3 code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py \
  --config configs/e2_stage_a_s3407.yaml --gpu 0 --yes \
  --max-steps 20 --output-dir runs/E2-SPEEDUP-SMOKE-S3407
```

开放风险：真实 Es 的 top-10 边界分布尚未测；val fonts 不在表的 228 个 query rows 内，验证 cadence 上仍走旧 α fallback（训练热路径已完全离线）；正式运行前必须填入 config SHA 并以短程吞吐/RNG 对照确认。
