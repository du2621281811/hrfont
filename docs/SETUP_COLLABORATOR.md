# 合作者本机初始化（单个仓库）

```bash
git clone git@github.com:du2621281811/hrfont.git
cd hrfont
```

Git 传输用 SSH（`git@github.com:...`），不要用 HTTPS clone。网页浏览仍是 `https://github.com/du2621281811/hrfont`。新机器把本机 `~/.ssh/id_ed25519.pub` 加到该仓 **Deploy keys**（可写）或 GitHub 账号 SSH keys。

即可同时得到：

- 实验脚本与结果说明（`scripts/`、`PROJECT.md`、`reports/`、`provenance/`）
- 官方源码：`code/official/FontDiffuser/`
- 我们的改动版：`code/ours/FontDiffuser/`

**先读** [`EXPERIMENTS.md`](EXPERIMENTS.md) 与 [`../code/README.md`](../code/README.md)，确认不会把 `ours` 当成官方。

数据 / 权重本机路径：[`DATA_AND_WEIGHTS.md`](DATA_AND_WEIGHTS.md)。
