#!/usr/bin/env python3
"""Plot F-arm training curves (train loss / val / rsi_gain) with 40k marker."""
import json, base64, io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = Path("/Users/xiaoweiliang/projects/hrfont/reports/training_logs")
arms = [
    ("F0 (RSI-free FT)", "F0-RSIFREE-FT-A-S3407", "#0F4D92"),
    ("F1 (official RSI)", "F1-OFFRSI-A-S3407", "#9A4D8E"),
    ("F2 (Δ-RSI)", "F2-DELTARSI-A-S3407", "#B64342"),
    ("F3 (Δ+support)", "F3-JOINT-DS-A-S3407", "#2E8B57"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
axes = axes.ravel()
for i, (label, arm, color) in enumerate(arms):
    ax = axes[i]
    for fname, key in [("train_log.jsonl", "loss"), ("train_loss.jsonl", "train_loss")]:
        fp = p / arm / fname
        if fp.exists():
            rows = [json.loads(l) for l in fp.read_text().strip().splitlines() if l.strip()]
            xs = [r["step"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, lw=1.2, color=color, label="train loss")
    fp = p / arm / "val_log.jsonl"
    if fp.exists():
        rows = [json.loads(l) for l in fp.read_text().strip().splitlines() if l.strip()]
        xs = [r["step"] for r in rows]
        ys = [r["val_loss"] for r in rows]
        ax.plot(xs, ys, "o--", ms=5, color="#E07B00", label="val loss")
        for r in rows:
            if r["step"] in (40000, 75000, 80000):
                ax.annotate(f"{r['step']//1000}k:{r['val_loss']:.4f}",
                            (r["step"], r["val_loss"]),
                            textcoords="offset points", xytext=(0, 10),
                            fontsize=8, color="#E07B00")
    ax.axvline(40000, color="#888", ls=":", lw=1)
    ylim = ax.get_ylim()
    if ylim[1] > 0:
        ax.text(40000, ylim[1] * 0.95, "40k", fontsize=8, color="#888", ha="right")
    ax.set_title(label, fontsize=11)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

fig2, ax2 = plt.subplots(figsize=(8, 3.5))
fp = p / "F3-JOINT-DS-A-S3407" / "train_log.jsonl"
rows = [json.loads(l) for l in fp.read_text().strip().splitlines() if l.strip()]
xs = [r["step"] for r in rows]
for j in range(len(rows[0]["rsi_gain"])):
    ax2.plot(xs, [r["rsi_gain"][j] for r in rows], lw=1, label=f"RSI head {j}")
ax2.axvline(40000, color="#888", ls=":")
ax2.set_title("F3 rsi_gain (RSI activation)")
ax2.set_xlabel("step")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)


def b64(fig_):
    buf = io.BytesIO()
    fig_.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig_)
    return base64.b64encode(buf.getvalue()).decode()


html = """<html><head><meta charset='utf-8'><style>
body{font-family:Menlo,monospace;background:#fff;color:#111;padding:16px}
h3{margin:8px 0}img{max-width:100%}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #bbb;padding:4px 8px}
</style></head><body>
<h3>F 臂训练曲线（2026-09-10，源 reports/training_logs/）</h3>
<img src='data:image/png;base64,__FIG1__'>
<img src='data:image/png;base64,__FIG2__'>
<p>注：F2 仓库日志截至 23.7k（执行机是否续跑未同步）；F1 仅旧段 0-300（当前 resume 日志未同步）；F3 为完整 80k。40k 竖虚线为参考。</p>
</body></html>"""
html = html.replace("__FIG1__", b64(fig)).replace("__FIG2__", b64(fig2))
out = Path("/Users/xiaoweiliang/projects/hrfont/reports/training_logs/F_ARM_CURVES_20260910.html")
out.write_text(html)
print("written", out)

f3v = [json.loads(l) for l in (p / "F3-JOINT-DS-A-S3407" / "val_log.jsonl").read_text().strip().splitlines()]
print("F3 val (>=35k):", [(r["step"], round(r["val_loss"], 4)) for r in f3v if r["step"] >= 35000])
f3t = [json.loads(l) for l in (p / "F3-JOINT-DS-A-S3407" / "train_log.jsonl").read_text().strip().splitlines()]
near40 = [r for r in f3t if abs(r["step"] - 40000) <= 100]
print("F3 train@~40k:", near40[0]["loss"] if near40 else None, " train@80k:", f3t[-1]["loss"])
f2v = [json.loads(l) for l in (p / "F2-DELTARSI-A-S3407" / "val_log.jsonl").read_text().strip().splitlines()]
print("F2 val:", [(r["step"], round(r["val_loss"], 4)) for r in f2v])
