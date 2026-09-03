# 合作者本机初始化

```bash
git clone https://github.com/du2621281811/hrfont.git
cd hrfont
git clone -b hrfont/local-patches-20260903 https://github.com/du2621281811/fontdiffuser-hrfont.git code/FontDiffuser
cd code/FontDiffuser && git remote add upstream https://github.com/yeungchenwa/FontDiffuser.git && cd ../..
```

数据与权重：见 [`DATA_AND_WEIGHTS.md`](DATA_AND_WEIGHTS.md)。本机开发机已用 symlink 接好；其他机器需自行放置 `data/fontdiffuser` 与 `runs/` 或改脚本路径。

阅读顺序：[`OFFICIAL_VS_OURS.md`](OFFICIAL_VS_OURS.md) → 根目录 `PROJECT.md` → `COLLABORATOR_GUIDE.md`。

Compare 补丁：https://github.com/du2621281811/fontdiffuser-hrfont/compare/main...hrfont/local-patches-20260903
