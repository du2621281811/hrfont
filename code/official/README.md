# 官方 FontDiffuser（干净上游）

**本目录 = 未经我们修改的官方源码快照。**

- 对应上游：https://github.com/yeungchenwa/FontDiffuser `main`
- Pin：见同目录 `UPSTREAM_PIN.txt`
- **禁止**在本目录做实验改动或当作训练入口
- 训练 / Stage A 请使用：`../../ours/FontDiffuser/`

与我们的补丁对比：

```bash
diff -ru code/official/FontDiffuser code/ours/FontDiffuser --exclude ckpt --exclude '*.txt'
# 或看 docs/patches/fontdiffuser-hrfont-local-patches-20260903.diff
```
