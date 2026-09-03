# `code/` — 官方 / 历史补丁 / 新变体必须分目录

合作者 **只 clone `hrfont`**。三层隔离，禁止混用：

| 路径 | 是什么 | 用途 |
|------|--------|------|
| [`official/FontDiffuser/`](official/FontDiffuser/) | 官方干净上游快照 | 只读对照 |
| [`ours/FontDiffuser/`](ours/FontDiffuser/) | 官方 + **历史**补丁 | 旧 FT / Stage A 恢复；默认只读 |
| [`variants/`](variants/) | 从 official 派生的**新实验**最小树 | 今后新训练入口 |

```text
❌ 不要把 ours/ 叫作「官方」
❌ 不要在 official/ 或 ours/ 上堆新功能
✅ 新实验 → variants/<id>/ + docs/patches/<id>.diff
✅ 历史对比：docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff
```

规则见 [`variants/README.md`](variants/README.md) 与 [`../docs/PROJECT_MANAGEMENT.md`](../docs/PROJECT_MANAGEMENT.md)。  
`ckpt/` 为本地 symlink，不进 Git。