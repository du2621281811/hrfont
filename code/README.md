# `code/` — 官方与我们的改动必须分目录

合作者 **只 clone `hrfont` 这一个仓** 即可。本目录下有两份 FontDiffuser，**名字不同，禁止混用**：

| 路径 | 是什么 | 用途 |
|------|--------|------|
| [`official/FontDiffuser/`](official/FontDiffuser/) | **官方干净源码**（上游 `main` 快照） | 对照、阅读、确认「官方长什么样」 |
| [`ours/FontDiffuser/`](ours/FontDiffuser/) | **官方 + 我们的补丁** | 本项目训练 / Stage A·B 的默认入口 |

```text
❌ 不要把 ours/ 叫作「官方」
❌ 不要在 official/ 里改代码做实验
✅ 对比改动：diff -ru code/official/FontDiffuser code/ours/FontDiffuser --exclude ckpt --exclude '*.txt'
✅ 或直接看 docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff
```

`ckpt/` 权重为本地 symlink，不进 Git。
