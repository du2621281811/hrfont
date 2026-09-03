# 字库语种浏览 · 有无统计 · 数据集勾选

分类与「253 字库 cmap 实测」一致：汉字 / 拉丁 ASCII / 拉丁扩展 / 平假名 / 片假名 / 注音 / 韩文 / CJK标点 / 符号。

页面分两层，不要混读：

1. **字库有没有**（cmap）：顶部表 + 当前字体徽章  
2. **数据集能不能勾**（已渲染）：仅 Target P1∪P2 与 Style 汉字 338 可勾；韩文等不可勾

## 启动

```bash
python /root/projects/hrfont/scripts/serve_charset_picker.py
# http://127.0.0.1:8766/
```
