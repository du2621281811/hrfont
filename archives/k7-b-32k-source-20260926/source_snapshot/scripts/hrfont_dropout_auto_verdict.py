#!/usr/bin/env python3
"""Auto-analyze dropout ablation and append verdict to VERDICT.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight/dropout_ablation"
FORMAL = ROOT / "reports/hrfont_overnight/formal_eval/metrics_step_80000.json"


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    rows = {}
    if FORMAL.exists():
        m = load(FORMAL)
        rows["R0@80k"] = {
            "stageA_l1": m["stageA_formal"]["L1_mean"],
            "ft_l1": m["ft_cnstyle"]["L1_mean"],
            "note": "formal main @80k (reference, not 10k)",
        }
    for rid in ["R1_cfg01_d01", "R2_cfg01_d00", "R3_style025_d025", "R4_cfg025_d025"]:
        p = REP / f"eval_{rid}.json"
        if p.exists():
            d = load(p)
            rows[rid] = {
                "stageA_l1": d["stageA_formal"]["L1_mean"],
                "ft_l1": d["ft_cnstyle"]["L1_mean"],
            }

    verdicts = []
    if "R2_cfg01_d00" in rows and "R3_style025_d025" in rows:
        r2, r3 = rows["R2_cfg01_d00"]["stageA_l1"], rows["R3_style025_d025"]["stageA_l1"]
        if r2 > r3 + 0.002:
            verdicts.append("Δ dropout 有必要：R2(无Δ drop) 差于 R3/R4")
        else:
            verdicts.append("Δ dropout 效应弱：需与 R0@80k 对照")

    if "R3_style025_d025" in rows:
        r3 = rows["R3_style025_d025"]["stageA_l1"]
        ft = rows["R3_style025_d025"]["ft_l1"]
        if r3 <= ft * 1.02:
            verdicts.append("R3 style_only @10k 接近 ft — content drop 在 10k 非必须（80k 主结论仍用 R0）")
        else:
            verdicts.append("content+style@0.1 在 10k 有趋势")

    if "R1_cfg01_d01" in rows and "R4_cfg025_d025" in rows:
        r1, r4 = rows["R1_cfg01_d01"]["stageA_l1"], rows["R4_cfg025_d025"]["stageA_l1"]
        if abs(r1 - r4) < 0.003:
            verdicts.append("Δ@0.1 vs Δ@0.25 @10k 差异小 — 保留 formal Δ@0.25 合理")

    best = min(
        ((k, v["stageA_l1"]) for k, v in rows.items() if k != "R0@80k"),
        key=lambda x: x[1],
        default=None,
    )
    out = {"rows": rows, "verdicts": verdicts, "best_10k": best[0] if best else None}
    (REP / "AUTO_VERDICT.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    extra = "\n## 自动结论（@10k，无需人工 review）\n\n"
    for v in verdicts:
        extra += f"- {v}\n"
    if best:
        extra += f"\n**10k 最优（除 R0@80k）**：{best[0]} L1={best[1]:.4f}\n"
    extra += "\n> R0 主结论以 formal @80k 为准；上表 R0 行是 80k，其余是 10k。\n"

    vpath = REP / "VERDICT.md"
    if vpath.exists():
        text = vpath.read_text(encoding="utf-8")
        if "## 自动结论" not in text:
            vpath.write_text(text + extra, encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
