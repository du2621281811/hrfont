#!/usr/bin/env python3
"""Build train / val / test curve JSON + SVG charts for mentor page.

- train: noise MSE + 0.5×offset on 42 fonts (Demo-8 excluded)
- val:   4 Demo-8 fonts, 32 cells, generation L1 (mid-sweep subset)
- test:  8 Demo-8 fonts, 64 cells, generation L1 (extended_demo8_eval)
"""
from __future__ import annotations

import json
import re
import statistics as stats
import time
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
OUT = REP / "mentor_preview"
LOG = REP / "e2_stageA.log"
META = ROOT / "data/hrfont/e0_bank/meta.json"


def parse_train_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    pat = re.compile(r"step=(\d+) loss=([\d.]+)(?: diff=([\d.]+))?")
    return [
        {"step": int(m.group(1)), "loss": float(m.group(2)), "diff": float(m.group(3) or m.group(2))}
        for m in pat.finditer(path.read_text(encoding="utf-8", errors="replace"))
    ]


def smooth_bins(points: list[dict], bin_size: int = 1000) -> list[dict]:
    if not points:
        return []
    max_step = points[-1]["step"]
    out = []
    for b in range(0, max_step + bin_size, bin_size):
        chunk = [p for p in points if b < p["step"] <= b + bin_size]
        if not chunk:
            continue
        out.append({
            "step": b + bin_size,
            "loss_mean": stats.mean(p["loss"] for p in chunk),
            "loss_std": stats.pstdev(p["loss"] for p in chunk) if len(chunk) > 1 else 0.0,
            "diff_mean": stats.mean(p["diff"] for p in chunk),
            "n": len(chunk),
        })
    return out


def nearest_train(train_smooth: list[dict], step: int) -> dict | None:
    if not train_smooth:
        return None
    return min(train_smooth, key=lambda r: abs(r["step"] - step))


def svg_dual_axis(
    train_smooth: list[dict],
    val_series: list[dict],
    test_series: list[dict],
    ft_val: float,
    ft_test: float,
) -> str:
    W, H, ml, mr, mt, mb = 800, 300, 58, 58, 36, 48
    iw = W - ml - mr
    ih = H - mt - mb

    all_steps = [r["step"] for r in train_smooth]
    for s in val_series + test_series:
        all_steps.append(s["step"])
    xmin, xmax = min(all_steps), max(all_steps)

    def xpx(step: int) -> float:
        return ml + (step - xmin) / max(xmax - xmin, 1) * iw

    t_losses = [r["loss_mean"] for r in train_smooth]
    t_lo, t_hi = min(t_losses) * 0.9, max(t_losses) * 1.05

    eval_l1 = [s["l1"] for s in val_series + test_series] + [ft_val, ft_test]
    e_lo, e_hi = min(eval_l1) * 0.95, max(eval_l1) * 1.05

    def y_train(v: float) -> float:
        return mt + (t_hi - v) / max(t_hi - t_lo, 1e-9) * ih

    def y_eval(v: float) -> float:
        return mt + (e_hi - v) / max(e_hi - e_lo, 1e-9) * ih

    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img">',
        f'<rect width="{W}" height="{H}" fill="#fff"/>',
        f'<text x="{ml}" y="22" fill="#12151a" font-size="14" font-weight="600" '
        f'font-family="IBM Plex Sans,sans-serif">训练 loss vs 验证/测试 L1（不同集合、不同量纲）</text>',
    ]

    for i in range(5):
        y = mt + ih * i / 4
        v = t_hi - (t_hi - t_lo) * i / 4
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-mr}" y2="{y:.1f}" stroke="#e8ecf1"/>')
        parts.append(
            f'<text x="{ml-6}" y="{y+4:.1f}" text-anchor="end" fill="#6b7280" font-size="10">train {v:.4f}</text>'
        )

    train_pts = " ".join(f"{xpx(r['step']):.1f},{y_train(r['loss_mean']):.1f}" for r in train_smooth)
    parts.append(f'<polyline fill="none" stroke="#9ca3af" stroke-width="2" points="{train_pts}"/>')

    if val_series:
        val_pts = " ".join(f"{xpx(s['step']):.1f},{y_eval(s['l1']):.1f}" for s in val_series)
        parts.append(f'<polyline fill="none" stroke="#1f6f8f" stroke-width="2.5" points="{val_pts}"/>')
        for s in val_series:
            cx, cy = xpx(s["step"]), y_eval(s["l1"])
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#1f6f8f"/>')

    if test_series:
        test_pts = " ".join(f"{xpx(s['step']):.1f},{y_eval(s['l1']):.1f}" for s in test_series)
        parts.append(
            f'<polyline fill="none" stroke="#0f7a4a" stroke-width="2.5" '
            f'stroke-dasharray="6 4" points="{test_pts}"/>'
        )
        for s in test_series:
            cx, cy = xpx(s["step"]), y_eval(s["l1"])
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#0f7a4a"/>')

    best_test = min(test_series, key=lambda s: s["l1"]) if test_series else None
    if best_test:
        cx, cy = xpx(best_test["step"]), y_eval(best_test["l1"])
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="#0f7a4a" stroke="#fff" stroke-width="1.5"/>')
        parts.append(
            f'<text x="{cx:.1f}" y="{cy-10:.1f}" text-anchor="middle" fill="#0f7a4a" '
            f'font-size="11" font-weight="700">test best {best_test["l1"]:.3f}@{best_test["step"]//1000}k</text>'
        )

    y_ft = y_eval(ft_test)
    parts.append(
        f'<line x1="{ml}" y1="{y_ft:.1f}" x2="{W-mr}" y2="{y_ft:.1f}" '
        f'stroke="#b45309" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{W-mr}" y="{y_ft-6:.1f}" text-anchor="end" fill="#b45309" font-size="11">ft test L1={ft_test:.3f}</text>'
    )

    for step in [10000, 40000, 70000, 100000, 130000, 140000]:
        if xmin <= step <= xmax:
            parts.append(
                f'<text x="{xpx(step):.1f}" y="{H-8}" text-anchor="middle" fill="#5c6570" font-size="10">{step//1000}k</text>'
            )

    parts += [
        f'<text x="{W-mr}" y="{mt+14}" text-anchor="end" fill="#9ca3af" font-size="11">灰=训练 total loss (42 fonts)</text>',
        f'<text x="{W-mr}" y="{mt+28}" text-anchor="end" fill="#1f6f8f" font-size="11">蓝=验证 L1 (4 Demo fonts, 32 cells)</text>',
        f'<text x="{W-mr}" y="{mt+42}" text-anchor="end" fill="#0f7a4a" font-size="11">绿虚=测试 L1 (8 Demo fonts, 64 cells)</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def svg_table_chart(title: str, series: list[dict], color: str, best_key: str = "l1") -> str:
    W, H, ml, mb = 800, 240, 58, 42
    iw = W - ml - 20
    ih = H - 56 - mb
    if not series:
        return f'<svg viewBox="0 0 {W} {H}"><text x="20" y="40">no data</text></svg>'
    steps = [s["step"] for s in series]
    vals = [s[best_key] for s in series]
    lo, hi = min(vals) * 0.95, max(vals) * 1.05
    xmin, xmax = min(steps), max(steps)

    def xpx(step):
        return ml + (step - xmin) / max(xmax - xmin, 1) * iw

    def ypx(v):
        return 36 + (hi - v) / max(hi - lo, 1e-9) * ih

    best = min(series, key=lambda s: s[best_key])
    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{W}" height="{H}" fill="#fff"/>',
        f'<text x="{ml}" y="22" fill="#12151a" font-size="14" font-weight="600">{title}</text>',
    ]
    for i in range(5):
        y = 36 + ih * i / 4
        v = hi - (hi - lo) * i / 4
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{W-20}" y2="{y:.1f}" stroke="#e8ecf1"/>')
        parts.append(f'<text x="{ml-6}" y="{y+4:.1f}" text-anchor="end" fill="#6b7280" font-size="10">{v:.3f}</text>')
    pts = " ".join(f"{xpx(s['step']):.1f},{ypx(s[best_key]):.1f}" for s in series)
    parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{pts}"/>')
    for s in series:
        cx, cy = xpx(s["step"]), ypx(s[best_key])
        r = 6 if s["step"] == best["step"] else 4
        fill = "#0f7a4a" if s["step"] == best["step"] else color
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="#fff" stroke-width="1.5"/>')
        parts.append(
            f'<text x="{cx:.1f}" y="{H-10}" text-anchor="middle" fill="#5c6570" font-size="10">{s["step"]//1000}k</text>'
        )
    if best:
        cx, cy = xpx(best["step"]), ypx(best[best_key])
        parts.append(
            f'<text x="{cx:.1f}" y="{cy-10:.1f}" text-anchor="middle" fill="#0f7a4a" '
            f'font-size="11" font-weight="700">{best[best_key]:.3f}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8"))
    demo8 = set(meta["demo8"])
    train_fonts = [f for f in meta["train_fonts"] if f not in demo8]

    raw = parse_train_log(LOG)
    train_smooth = smooth_bins(raw, 1000)

    trend_path = OUT / "trend/trend.json"
    ext_path = OUT / "extended_demo8_eval.json"
    trend = json.loads(trend_path.read_text()) if trend_path.exists() else {"series": []}
    ext = json.loads(ext_path.read_text()) if ext_path.exists() else {"results": {}}

    val_series = [
        {"step": r["step"], "l1": r["mean_l1_stageA"], "win_rate": r["win_rate"], "wins": r["wins"], "n": r["n"]}
        for r in trend.get("series", [])
    ]
    test_series = []
    for k, r in sorted(ext.get("results", {}).items(), key=lambda x: int(x[0])):
        test_series.append({
            "step": int(k),
            "l1": r["mean_l1_a"],
            "win_rate": r["win_rate"],
            "wins": r["wins"],
            "n": r["n"],
        })

    ft_val = val_series[0]["l1"] if val_series else 0.0
    if val_series:
        ft_val = trend["series"][0]["mean_l1_ft_cnstyle"]
    ft_test = next(iter(ext.get("results", {}).values()), {}).get("mean_l1_ft", ft_val)

    combined = []
    all_steps = sorted(set(s["step"] for s in val_series) | set(s["step"] for s in test_series))
    val_by = {s["step"]: s for s in val_series}
    test_by = {s["step"]: s for s in test_series}
    for step in all_steps:
        tr = nearest_train(train_smooth, step)
        row = {"step": step}
        if tr:
            row["train_loss"] = round(tr["loss_mean"], 6)
            row["train_diff"] = round(tr["diff_mean"], 6)
        if step in val_by:
            row["val_l1"] = round(val_by[step]["l1"], 4)
            row["val_win_rate"] = val_by[step]["win_rate"]
        if step in test_by:
            row["test_l1"] = round(test_by[step]["l1"], 4)
            row["test_win_rate"] = test_by[step]["win_rate"]
        combined.append(row)

    best_val = min(val_series, key=lambda s: s["l1"]) if val_series else None
    best_test = min(test_series, key=lambda s: s["l1"]) if test_series else None

    payload = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {
            "train": {
                "fonts": len(train_fonts),
                "demo8_excluded": len(demo8),
                "loss": "noise MSE + 0.5×offset (total logged as loss=)",
                "note": "Demo-8 fonts never sampled in StageASet",
            },
            "val": {
                "fonts": trend.get("protocol", {}).get("fonts", []),
                "cells": 32,
                "metric": "generation L1 vs GT (4/8 Demo-8 subset)",
            },
            "test": {
                "fonts": ext.get("fonts", list(demo8)),
                "cells": 64,
                "metric": "generation L1 vs GT (full Demo-8)",
            },
        },
        "train_smooth_1k": train_smooth,
        "val_series": val_series,
        "test_series": test_series,
        "combined_at_eval_steps": combined,
        "best_val": best_val,
        "best_test": best_test,
        "ft_baselines": {"val_l1": ft_val, "test_l1": ft_test},
        "train_end_step": raw[-1]["step"] if raw else 0,
        "train_end_loss": raw[-1]["loss"] if raw else None,
    }
    (OUT / "train_val_test_curves.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    charts = {
        "dual": svg_dual_axis(train_smooth, val_series, test_series, ft_val, ft_test),
        "val_l1": svg_table_chart("验证集 L1 — 4 Demo 字体 × 8 字 (32 格)", val_series, "#1f6f8f"),
        "test_l1": svg_table_chart("测试集 L1 — 全 Demo-8 × 8 字 (64 格)", test_series, "#0f7a4a"),
        "val_win": svg_table_chart(
            "验证集胜率 vs ft (32 格)", [{**s, "l1": s["win_rate"]} for s in val_series], "#1f6f8f", "l1"
        ),
        "test_win": svg_table_chart(
            "测试集胜率 vs ft (64 格)", [{**s, "l1": s["win_rate"]} for s in test_series], "#0f7a4a", "l1"
        ),
    }
    (OUT / "charts.json").write_text(json.dumps(charts, ensure_ascii=False), encoding="utf-8")
    print(f"wrote train_val_test_curves.json train_pts={len(train_smooth)} val={len(val_series)} test={len(test_series)}")
    if best_val:
        print(f"best val L1={best_val['l1']:.4f} @ {best_val['step']}")
    if best_test:
        print(f"best test L1={best_test['l1']:.4f} @ {best_test['step']}")


if __name__ == "__main__":
    main()
