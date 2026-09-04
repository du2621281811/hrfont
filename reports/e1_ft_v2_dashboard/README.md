# E1-FTV2-A-S3407 训练看板（协作快照）

本目录是从训练机 `runs/E1-FTV2-A-S3407/viz/` **定期同步**的可分享快照（不含权重、不含完整 4 格 panel）。

## 本地查看

```bash
cd /path/to/hrfont
python -m http.server 8777 --directory reports
# 打开 http://127.0.0.1:8777/e1_ft_v2_dashboard/
```

或从仓库根目录：

```bash
python -m http.server 8777 --directory .
# http://127.0.0.1:8777/reports/e1_ft_v2_dashboard/
```

## 内容

- `index.html` / `loss.png` / `*_history.json` / `compare_index.json`
- `refs/{content,style,gt}/`：固定参考图
- `preds/<step>/`：各 checkpoint Pred（96×96）

训练仍在进行时，以训练机实时页为准；本快照可能略滞后。
