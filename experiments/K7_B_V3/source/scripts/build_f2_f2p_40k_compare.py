#!/usr/bin/env python3
"""Build matched F2@40k vs F2-P@40k compare page from test16 preds + metrics."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PRED = ROOT / "reports/f03_test16_strat"
OUT = ROOT / "reports/f2p_40k_compare"
METRICS = PRED / "metrics_summary.json"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
STRAT = json.loads((PRED / "PROTOCOL.json").read_text(encoding="utf-8")) if (PRED / "PROTOCOL.json").is_file() else {}

MIDS = ("F2_40000", "F2P_40000")
LABELS = {"F2_40000": "F2@40k", "F2P_40000": "F2-P@40k"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fonts() -> list[str]:
    if STRAT.get("fonts"):
        return list(STRAT["fonts"])
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    return sorted(split["stems"]["test"])


def chars() -> list[str]:
    return list(STRAT.get("chars") or [])


def count_preds(mid: str) -> int:
    return sum(1 for _ in (PRED / "preds" / mid).rglob("*.png"))


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def metric_table(summary: dict) -> str:
    rows = []
    for mid in MIDS:
        m = summary.get("methods", {}).get(mid)
        if not m:
            rows.append(f"<tr><td>{LABELS[mid]}</td><td colspan=5>missing metrics</td></tr>")
            continue
        o = m["overall"]
        rows.append(
            "<tr>"
            f"<td><b>{LABELS[mid]}</b></td>"
            f"<td>{o.get('n', 0)}</td>"
            f"<td>{fmt(o.get('L1_mean'))}</td>"
            f"<td>{fmt(o.get('SSIM_mean'))}</td>"
            f"<td>{fmt(o.get('LPIPS_mean'))}</td>"
            f"<td>{fmt(o.get('coverage_mean'), 3)}</td>"
            f"<td>{fmt(o.get('blank_rate'), 3)}</td>"
            "</tr>"
        )
    return (
        "<table class='metrics'><thead><tr>"
        "<th>method</th><th>n</th><th>L1↓</th><th>SSIM↑</th><th>LPIPS↓</th><th>cov</th><th>blank</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<p class='note'>L1/SSIM/LPIPS vs GT are diagnostics — style claim is visual (refs / identity chars).</p>"
    )


def gallery(fs: list[str], cs: list[str], pick_chars: list[str], n_fonts: int = 8) -> str:
    blocks = []
    for stem in fs[:n_fonts]:
        cells = []
        for ch in pick_chars:
            cp = f"u{ord(ch):04X}"
            figs = []
            for mid in MIDS:
                rel = f"../f03_test16_strat/preds/{mid}/{stem}/{stem}+{cp}.png"
                p = PRED / "preds" / mid / stem / f"{stem}+{cp}.png"
                if p.is_file():
                    figs.append(
                        f"<figure><figcaption>{LABELS[mid]}</figcaption>"
                        f'<img src="{rel}" width="96" height="96" loading="lazy"/></figure>'
                    )
                else:
                    figs.append(f"<figure><figcaption>{LABELS[mid]}</figcaption><span class='miss'>∅</span></figure>")
            cells.append(f"<td><div class='ch'>{ch}</div><div class='pair'>{''.join(figs)}</div></td>")
        blocks.append(f"<tr><td class='font'><code>{stem}</code></td>{''.join(cells)}</tr>")
    head = "".join(f"<th>{c}</th>" for c in pick_chars)
    return (
        f"<div class='scroll'><table><thead><tr><th>font</th>{head}</tr></thead>"
        f"<tbody>{''.join(blocks)}</tbody></table></div>"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fs, cs = fonts(), chars()
    summary = {}
    if METRICS.is_file():
        summary = json.loads(METRICS.read_text(encoding="utf-8"))

    counts = {mid: count_preds(mid) for mid in MIDS}
    pick = [c for c in ("I", "l", "1", "o", "A", "a", "あ", "の", "永", "风") if c in cs] or cs[:8]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>F2@40k vs F2-P@40k · matched compare</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#f6f4f0;color:#1a1a1a;line-height:1.45}}
h1{{font-size:1.35rem;margin:0 0 .35rem}}
.meta,.note{{color:#555;font-size:.92rem;max-width:72ch}}
.metrics{{border-collapse:collapse;background:#fff;margin:12px 0}}
.metrics th,.metrics td{{border:1px solid #ddd;padding:6px 10px;text-align:right}}
.metrics th:first-child,.metrics td:first-child{{text-align:left}}
.scroll{{overflow-x:auto;background:#fff;border:1px solid #ddd;margin:12px 0 28px}}
table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #e6e6e6;padding:6px;vertical-align:top}}
th{{background:#eee}}
.pair{{display:flex;gap:4px}}
figure{{margin:0;text-align:center}}
figcaption{{font-size:10px;color:#666}}
.ch{{font-weight:600;margin-bottom:2px}}
.miss{{color:#aaa}}
.links a{{margin-right:12px}}
</style></head><body>
<h1>Matched ablation · F2@40k vs F2-P@40k</h1>
<p class="meta">Aggregator-only compare (same Δ-RSI, Mode B ref8, seed 3407, DPM++20 CFG7.5).
Generated {utc()}. Counts: F2={counts['F2_40000']}/752 · F2-P={counts['F2P_40000']}/752.</p>
<p class="links">
<a href="../style_domain_probe/">domain probe (train/val)</a>
<a href="../f03_test16_strat/index.html">fair-axes board</a>
</p>
<h2>Metrics (test16×47)</h2>
{metric_table(summary)}
<h2>Visual sample · identity / style probes</h2>
{gallery(fs, cs, pick, n_fonts=8)}
</body></html>
"""
    path = OUT / "index.html"
    path.write_text(html, encoding="utf-8")
    (OUT / "counts.json").write_text(json.dumps({"ts": utc(), "counts": counts}, indent=2) + "\n", encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
