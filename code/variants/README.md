# FontDiffuser 代码变体

```text
code/official/   → 官方干净，禁止改业务逻辑
code/ours/       → 历史补丁（含 Stroke-SCR 等），只读对照
code/variants/   → 新实验专用最小树，从 official 派生
```

## 规则

1. 新基模 / 新方法 **不得** 直接改 `official` 或在 `ours` 上继续堆功能。
2. 新建 `variants/<variant_id>/FontDiffuser/`：从 `official/FontDiffuser` 复制后只加本实验必需补丁。
3. 导出 diff：`diff -ru code/official/FontDiffuser code/variants/<id>/FontDiffuser > docs/patches/<id>.diff`（排除 ckpt / `__pycache__` / 标记文件）。
4. 在 `provenance/REGISTRY.md` 登记；训练入口显式指向该 variant。
5. `scripts/check_code_layout.py` 与 `scripts/pm_preflight.py` 会检查目录契约。

## 计划中的变体

- `cn2west_f2_vec`：F2 mean-Δ + 可微渲染矢量头（side study）。  
  启动：`python scripts/launch_cn2west_f2_vec.py --smoke` / `--yes`  
  设计：`reports/F2_VEC_MULTITASK_DESIGN_20260913.md`
- `cn2west_ft_v2`：CN→West 基模微调最小补丁（StyleImage PNG、无 Resize、P1 热启、SCR off）。  
  启动：`python scripts/launch_cn2west_ft_v2_e1.py --smoke` / `--yes`
