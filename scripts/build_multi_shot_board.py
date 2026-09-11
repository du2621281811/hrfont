#!/usr/bin/env python3
"""Build multi-shot comparison board: P1..F3 with each model's shot variants in one table.

Served via the local http.server (relative img paths, no base64)."""
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "reports" / "f03_test16_strat"
browse = json.loads((P / "browse_index.json").read_text(encoding="utf-8"))
metrics = json.loads((P / "metrics_items.json").read_text(encoding="utf-8"))
e12_items = json.loads((P / "e12_v51_items.json").read_text(encoding="utf-8"))
m_sum = browse["metrics"]["methods"]
e12_sum = browse["e12_v51"]["methods"]

METHODS = [
    ("P1", "P1 官方", "s1 永"),
    ("E1_100k", "E1@100k", "s1 永"),
    ("F0_100k", "F0@100k", "s1 永"),
    ("F1_30000", "F1@30k", "s1 永 mid"),
    ("F2_75000", "F2@75k", "s8 mean"),
    ("F2_75000_s1", "F2@75k", "s1 永"),
    ("F3_80k", "F3@80k", "s8 mean"),
    ("F3_80k_s1", "F3@80k", "s1 永"),
]

# per-item lookups
m_item = {}
for it in metrics:
    m_item[(it["method"], it["font"], it["cp"])] = it
e_item = {}
for it in e12_items:
    e_item[(it["method"], it["font"], it["cp"])] = it

rows = browse["items"] if "items" in browse else []
if not rows:
    # rebuild from metrics_items (already have font/char/cp/bucket/paths)
    rows = metrics
# dedupe to unique (font,char)
seen, uniq = set(), []
for it in rows:
    k = (it["font"], it["cp"])
    if k in seen:
        continue
    seen.add(k)
    uniq.append(it)
rows = uniq

fonts = browse["fonts"]


def chip(mid, font, cp, kind):
    it = m_item.get((mid, font, cp))
    if kind == "pix" and it:
        return f"L1 {it['L1']:.3f}<br>LP {it['LPIPS']:.3f}"
    if kind == "e12":
        e = e_item.get((mid, font, cp))
        if e:
            return f"φ {e['e12_phi_sc_r']:.3f}<br>m {e['e12_mem_logit']:.1f}"
    return ""


# header summary table
th = "<tr><th>方法</th><th>shot</th>" + "".join(
    f"<th>{c}</th>" for c in ["L1", "SSIM", "LPIPS", "φ SC-R", "mem"]) + "</tr>"
trs = []
for mid, label, shot in METHODS:
    r = m_sum[mid]["overall"]
    e = e12_sum.get(mid, {})
    trs.append(
        f"<tr><td>{label}</td><td>{shot}</td>"
        f"<td>{r['L1_mean']:.4f}</td><td>{r['SSIM_mean']:.4f}</td>"
        f"<td>{r['LPIPS_mean']:.4f}</td>"
        f"<td>{e.get('phi_sc_r', {}).get('mean', float('nan')):.3f}</td>"
        f"<td>{e.get('mem_prob', {}).get('mean', float('nan')):.3f}</td></tr>")
summary = f"<table class='sum'>{th}{''.join(trs)}</table>"

# column header for glyph tables
cols = "".join(
    f"<th class='mh'><div class='ml'>{label}</div><div class='ms'>{shot}</div></th>"
    for _, label, shot in METHODS)

nav = "".join(f"<a href='#f{i}'>{f}</a> " for i, f in enumerate(fonts))

# per-font tables: rows = chars, columns = GT + 8 methods
sections = []
for fi, font in enumerate(fonts):
    frows = [r for r in rows if r["font"] == font]
    frows.sort(key=lambda r: r["char"])
    body = []
    for r in frows:
        cp, ch = r["cp"], r["char"]
        gt = f"refs/gt/{font}+{cp}.png"
        cells = [f"<td class='gt'><img loading='lazy' src='{gt}' width='52'></td>"]
        for mid, _, _ in METHODS:
            img = f"preds/{mid}/test/{font}/test__{font}__{cp}__s3407.png"
            c_pix = chip(mid, font, cp, "pix")
            c_e12 = chip(mid, font, cp, "e12")
            cells.append(
                f"<td><img loading='lazy' src='{img}' width='52' "
                f"title='{mid}'><div class='ch'>{c_pix}</div>"
                f"<div class='ch e'>{c_e12}</div></td>")
        body.append(
            f"<tr><td class='chr'>{ch}<div class='bk'>{r['bucket']}</div></td>"
            f"{''.join(cells)}</tr>")
    sections.append(
        f"<h3 id='f{fi}'>{fi+1}. {font}</h3>"
        f"<table class='gly'><tr><th class='chh'>字</th><th class='chh'>GT</th>{cols}</tr>"
        f"{''.join(body)}</table>")

html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Multi-Shot Board — P1..F3</title>
<style>
body{{font-family:Menlo,monospace;background:#fff;color:#111;margin:14px}}
table.sum{{border-collapse:collapse;font-size:12px;margin:10px 0}}
table.sum td,table.sum th{{border:1px solid #bbb;padding:3px 7px;text-align:right}}
table.sum th{{background:#f2f2f2}}
h3{{margin:22px 0 6px}}
table.gly{{border-collapse:collapse;font-size:11px}}
table.gly td,table.gly th{{border:1px solid #ddd;padding:2px 4px;text-align:center;vertical-align:top}}
.ml{{font-weight:bold}}.ms{{color:#888;font-size:10px}}
.chr{{font-weight:bold;font-size:13px;min-width:2.2em}}
.bk{{color:#999;font-size:9px;font-weight:normal}}
.ch{{font-size:9px;color:#555;line-height:1.25}}
.ch.e{{color:#8a5a00}}
nav a{{margin-right:6px;font-size:12px}}
.note{{color:#666;font-size:12px;margin:6px 0}}
</style></head><body>
<h2>Multi-Shot Board：P1 → F3，各模型 shot 变体同表对照</h2>
<div class='note'>GT 仅作 positive control。L1/LPIPS/φ/mem 为逐字诊断（chips），φ=phi SC-R（越大越好），m=membership logit。图像经本地 http.server 加载。</div>
{summary}
<nav>{nav}</nav>
{''.join(sections)}
</body></html>"""
out = P / "multi_shot_board.html"
out.write_text(html, encoding="utf-8")
print("written", out, f"{len(rows)} rows, {len(fonts)} fonts, {out.stat().st_size//1024} KB")
