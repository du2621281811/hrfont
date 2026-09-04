#!/usr/bin/env bash
# Publish a collaborator-visible snapshot of the E1 training dashboard.
# Source of truth during training remains runs/.../viz (gitignored).
# This copies only the shareable web assets into reports/e1_ft_v2_dashboard/.
set -euo pipefail
ROOT=/root/projects/hrfont
SRC=$ROOT/runs/E1-FTV2-A-S3407/viz
DST=$ROOT/reports/e1_ft_v2_dashboard

if [[ ! -f "$SRC/index.html" ]]; then
  echo "missing dashboard source: $SRC/index.html" >&2
  exit 1
fi

mkdir -p "$DST"
rsync -a --delete \
  --exclude 'samples/' \
  --exclude '*.log' \
  --exclude 'full_sample_*.log' \
  --exclude 'refresh.log' \
  --exclude 'val_loss_eval.log' \
  --exclude 'refresher.pid' \
  --exclude '*.pid' \
  "$SRC/" "$DST/"

# Drop legacy strip refs (300x96) if any remain at refs/*.png
find "$DST/refs" -maxdepth 1 -type f -name '*.png' -delete 2>/dev/null || true

# Mark snapshot banner in HTML
python3 - "$DST" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import re, sys
dst = Path(sys.argv[1])
html_path = dst / "index.html"
html = html_path.read_text(encoding="utf-8")
stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
note = (
    '<div class="meta" style="background:#fff8e8;border:1px solid #e6d7a8;padding:8px 10px;margin:10px 0;border-radius:4px">'
    f'这是 Git 协作快照（{stamp}）。训练机实时页在本机 <code>data/e1_ft_v2_dashboard</code>→<code>runs/.../viz</code>。'
    ' 更新：<code>bash scripts/publish_e1_dashboard.sh</code> 后 commit/push。'
    '</div>'
)
if "Git 协作快照" not in html:
    html = html.replace("<main>", "<main>\n" + note, 1)
else:
    html = re.sub(
        r'<div class="meta" style="background:#fff8e8;[^>]*>.*?</div>',
        note,
        html,
        count=1,
        flags=re.S,
    )
html_path.write_text(html, encoding="utf-8")
print("stamped", html_path)
PY

cat >"$DST/README.md" <<'EOF'
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
EOF

python3 - "$DST" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys
dst = Path(sys.argv[1])
(dst / "SNAPSHOT.json").write_text(
    '{"source":"runs/E1-FTV2-A-S3407/viz","published_utc":"%s"}\n'
    % datetime.now(timezone.utc).isoformat(),
    encoding="utf-8",
)
print("published", dst)
PY

du -sh "$DST" "$DST/preds" "$DST/refs"
