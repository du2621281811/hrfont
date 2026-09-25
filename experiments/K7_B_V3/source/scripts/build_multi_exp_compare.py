#!/usr/bin/env python3
"""Multi-experiment visual+metrics board for P-line and main baselines.

Default columns: F0@100k, F2@40k, F2-P@40k, F3b-P@40k, F2@80k, F3@80k.
Reads preds/metrics from reports/f03_test16_strat/.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PRED = ROOT / "reports/f03_test16_strat"
OUT = ROOT / "reports/multi_exp_compare"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
PROTOCOL = PRED / "PROTOCOL.json"
METRICS = PRED / "metrics_summary.json"

# Matched P ablations first, then long-horizon context.
COLUMNS = [
    ("F0_100k", "F0@100k"),
    ("F2_40000", "F2@40k"),
    ("F2P_40000", "F2-P@40k"),
    ("F3bP_40000", "F3b-P@40k"),
    ("F2_80000", "F2@80k"),
    ("F3_80k", "F3@80k"),
]
# Fair matched pairs to highlight.
PAIR_NOTES = [
    ("F2@40k ↔ F2-P@40k", "aggregator only (mean vs per-ref)"),
    ("F2-P@40k ↔ F3b-P@40k", "support only (Δ+P vs Δ+P+own-font support)"),
    ("F2@80k / F3@80k", "long-horizon context (not matched to 40k)"),
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fonts() -> list[str]:
    if PROTOCOL.is_file():
        d = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if d.get("fonts"):
            return list(d["fonts"])
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    return sorted(split["stems"]["test"])


def chars() -> list[str]:
    if PROTOCOL.is_file():
        d = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if d.get("chars"):
            return list(d["chars"])
    return list("Il1oAaあのAGQa")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def count_preds(mid: str) -> int:
    p = PRED / "preds" / mid
    return sum(1 for _ in p.rglob("*.png")) if p.exists() else 0


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def metric_block(summary: dict) -> str:
    rows = []
    for mid, label in COLUMNS:
        m = summary.get("methods", {}).get(mid)
        if not m:
            rows.append(
                f"<tr><td>{label}</td><td>{count_preds(mid)}</td><td colspan=5 class='miss'>no metrics</td></tr>"
            )
            continue
        o = m["overall"]
        rows.append(
            "<tr>"
            f"<td><b>{label}</b></td>"
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
        "<p class='note'>像素指标是诊断；风格主张仍以 ref 对齐与 identity 字形为准。"
        " F2/F3@80k 与 40k 臂不可直接当 matched 总分榜。</p>"
    )


def pred_rel(mid: str, stem: str, ch: str, prefix: str = "preds") -> str:
    return f"{prefix}/{mid}/test/{stem}/test__{stem}__{cp_of(ch)}__s3407.png"


def pred_abs(mid: str, stem: str, ch: str) -> Path:
    return PRED / "preds" / mid / "test" / stem / f"test__{stem}__{cp_of(ch)}__s3407.png"


def gallery(fs: list[str], pick: list[str], n_fonts: int = 8, pred_prefix: str = "preds") -> str:
    head = "".join(f"<th>{c}</th>" for c in pick)
    blocks = []
    for stem in fs[:n_fonts]:
        cells = []
        for ch in pick:
            figs = []
            for mid, label in COLUMNS:
                rel = pred_rel(mid, stem, ch, prefix=pred_prefix)
                p = pred_abs(mid, stem, ch)
                if p.is_file():
                    figs.append(
                        f"<figure><figcaption>{label}</figcaption>"
                        f'<img src="{rel}" width="72" height="72" loading="lazy"/></figure>'
                    )
                else:
                    figs.append(
                        f"<figure><figcaption>{label}</figcaption><span class='miss'>∅</span></figure>"
                    )
            cells.append(f"<td><div class='ch'>{ch}</div><div class='pair'>{''.join(figs)}</div></td>")
        blocks.append(f"<tr><td class='font'><code>{stem}</code></td>{''.join(cells)}</tr>")
    return (
        f"<div class='scroll'><table><thead><tr><th>font</th>{head}</tr></thead>"
        f"<tbody>{''.join(blocks)}</tbody></table></div>"
    )


def render_html(
    *,
    pred_prefix: str,
    link_fair: str,
    link_multi: str,
    link_f2p: str,
    link_probe: str,
) -> str:
    fs, cs = fonts(), chars()
    pick = [c for c in ("I", "l", "1", "o", "A", "a", "あ", "の", "G", "Q") if c in cs] or cs[:8]
    summary = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.is_file() else {}
    counts = {mid: count_preds(mid) for mid, _ in COLUMNS}
    pairs = "".join(f"<li><b>{a}</b> — {b}</li>" for a, b in PAIR_NOTES)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>Multi-exp compare · F0 / F2 / F2-P / F3b-P / F3</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#f5f3ef;color:#1a1a1a;line-height:1.45}}
h1{{font-size:1.35rem;margin:0 0 .35rem}}
h2{{font-size:1.1rem;margin:1.4rem 0 .5rem}}
.meta,.note{{color:#555;font-size:.92rem;max-width:78ch}}
.metrics{{border-collapse:collapse;background:#fff;margin:12px 0}}
.metrics th,.metrics td{{border:1px solid #ddd;padding:6px 10px;text-align:right}}
.metrics th:first-child,.metrics td:first-child{{text-align:left}}
.scroll{{overflow-x:auto;background:#fff;border:1px solid #ddd;margin:12px 0 28px}}
table{{border-collapse:collapse;font-size:11px}}
th,td{{border:1px solid #e6e6e6;padding:5px;vertical-align:top}}
th{{background:#eee;position:sticky;top:0}}
.pair{{display:flex;flex-wrap:wrap;gap:3px;max-width:520px}}
figure{{margin:0;text-align:center}}
figcaption{{font-size:9px;color:#666;max-width:72px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.ch{{font-weight:600;margin-bottom:2px}}
.miss{{color:#aaa}}
.links a{{margin-right:14px}}
ul.pairs{{max-width:70ch}}
</style></head><body>
<h1>多实验对比 · Mode B ref8 · seed 3407</h1>
<p class="meta">生成于 {utc()} · test16×47 · DPM++20 CFG7.5<br/>
counts: {", ".join(f"{lab}={counts[mid]}" for mid, lab in COLUMNS)}</p>
<p class="links">
<a href="{link_fair}">fair-axes board</a>
<a href="{link_multi}">multi-shot board</a>
<a href="{link_f2p}">F2 vs F2-P</a>
<a href="{link_probe}">domain probe</a>
</p>
<h2>读法（合法对照）</h2>
<ul class="pairs">{pairs}</ul>
<h2>Metrics</h2>
{metric_block(summary)}
<h2>Visual · identity / style probes</h2>
{gallery(fs, pick, n_fonts=10, pred_prefix=pred_prefix)}
</body></html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {mid: count_preds(mid) for mid, _ in COLUMNS}

    # Standalone under reports/ (port 8773)
    path = OUT / "index.html"
    path.write_text(
        render_html(
            pred_prefix="../f03_test16_strat/preds",
            link_fair="../f03_test16_strat/index.html",
            link_multi="../f03_test16_strat/multi_shot_board.html",
            link_f2p="../f2p_40k_compare/",
            link_probe="../style_domain_probe/",
        ),
        encoding="utf-8",
    )

    # Same page inside f03 board root → works on :19000 and :8767
    board_page = PRED / "multi_exp_compare.html"
    board_page.write_text(
        render_html(
            pred_prefix="preds",
            link_fair="index.html",
            link_multi="multi_shot_board.html",
            link_f2p="index.html",
            link_probe="pi_highlights.html",
        ),
        encoding="utf-8",
    )

    (OUT / "counts.json").write_text(
        json.dumps({"ts": utc(), "counts": counts, "columns": COLUMNS}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("wrote", path)
    print("wrote", board_page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
