#!/usr/bin/env python3
"""Gap-stratified L1 analysis from formal eval JSON + gap_b0_iou.json."""
from __future__ import annotations

import json
import statistics as stats
import time
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
OUT = REP / "formal_eval"
GAP = ROOT / "data/hrfont/e0_bank/gap_b0_iou.json"


def load_metrics(path: Path, key: str) -> dict[tuple[str, str], float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("results", {}).get(key, {}).get("rows") or d.get(key, {}).get("rows", [])
    if not rows and key == "stageA_formal":
        rows = d.get("results", {}).get("stageA_formal", {}).get("rows", [])
    out = {}
    for r in rows:
        out[(r["font"], r["char"])] = float(r["l1"])
    return out


def main() -> None:
    gap_rows = json.loads(GAP.read_text(encoding="utf-8"))["rows"]
    gap_map = {r["ch"]: float(r["gap"]) for r in gap_rows}
    gaps = sorted(gap_map.values())
    q33 = gaps[len(gaps) // 3]
    q66 = gaps[2 * len(gaps) // 3]

    def bucket(g: float) -> str:
        if g >= q66:
            return "high"
        if g >= q33:
            return "mid"
        return "low"

    m80 = OUT / "metrics_step_80000.json"
    ft = load_metrics(m80, "ft_cnstyle")
    a = load_metrics(m80, "stageA_formal")

    mb = OUT / "metrics_stageB_25000.json"
    b = load_metrics(mb, "stageB") if mb.exists() else {}

    cells = []
    for (font, ch), l1_ft in ft.items():
        if ch not in gap_map or (font, ch) not in a:
            continue
        l1_a = a[(font, ch)]
        g = gap_map[ch]
        cell = {
            "font": font,
            "char": ch,
            "gap": g,
            "bucket": bucket(g),
            "l1_ft": l1_ft,
            "l1_a": l1_a,
            "delta_ft_minus_a": l1_ft - l1_a,
            "a_wins": l1_a < l1_ft,
        }
        if (font, ch) in b:
            cell["l1_b"] = b[(font, ch)]
            cell["delta_a_minus_b"] = l1_a - b[(font, ch)]
            cell["b_wins_vs_a"] = b[(font, ch)] < l1_a
        cells.append(cell)

    summary = {}
    for bk in ("low", "mid", "high"):
        sub = [c for c in cells if c["bucket"] == bk]
        if not sub:
            continue
        summary[bk] = {
            "n": len(sub),
            "gap_range": [min(c["gap"] for c in sub), max(c["gap"] for c in sub)],
            "mean_l1_ft": stats.mean(c["l1_ft"] for c in sub),
            "mean_l1_a": stats.mean(c["l1_a"] for c in sub),
            "a_win_rate": sum(c["a_wins"] for c in sub) / len(sub),
        }
        if any("l1_b" in c for c in sub):
            bs = [c for c in sub if "l1_b" in c]
            summary[bk]["mean_l1_b"] = stats.mean(c["l1_b"] for c in bs)
            summary[bk]["b_win_rate_vs_a"] = sum(c.get("b_wins_vs_a", False) for c in bs) / len(bs)

    payload = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gap_thresholds": {"q33": q33, "q66": q66},
        "n_cells": len(cells),
        "summary_by_bucket": summary,
        "hypothesis": "high gap → Stage A/B gain vs ft should be larger",
        "cells": cells,
    }
    (OUT / "gap_stratified.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Gap 分层 L1 分析\n", f"更新时间：{payload['t']}\n\n", "| 桶 | n | gap 范围 | ft | A | A 胜率 |"]
    if summary.get("high", {}).get("mean_l1_b"):
        md[2] += " B | B胜A |"
    md[2] += "\n|-----|---|---------|-----|-----|--------|"
    if summary.get("high", {}).get("mean_l1_b"):
        md[2] += "---|------|"
    md[2] += "\n"
    for bk in ("low", "mid", "high"):
        s = summary.get(bk)
        if not s:
            continue
        line = (
            f"| {bk} | {s['n']} | {s['gap_range'][0]:.2f}–{s['gap_range'][1]:.2f} | "
            f"{s['mean_l1_ft']:.4f} | {s['mean_l1_a']:.4f} | {100*s['a_win_rate']:.0f}% |"
        )
        if "mean_l1_b" in s:
            line += f" {s['mean_l1_b']:.4f} | {100*s.get('b_win_rate_vs_a',0):.0f}% |"
        md.append(line)
    (OUT / "gap_stratified.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"gap_stratified n={len(cells)} buckets={list(summary.keys())}")


if __name__ == "__main__":
    main()
