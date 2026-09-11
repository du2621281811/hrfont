#!/usr/bin/env python3
"""Build multi-shot comparison board (McCarthy dark-warm theme + filters).

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
    ("F1_80000", "F1@80k", "s1 永"),
    ("F2_80000", "F2@80k", "s8 mean"),
    ("F2_80000_s1", "F2@80k", "s1 永"),
    ("F3_80k", "F3@80k", "s8 mean"),
    ("F3_80k_s1", "F3@80k", "s1 永"),
]

m_item = {(it["method"], it["font"], it["cp"]): it for it in metrics}
e_item = {(it["method"], it["font"], it["cp"]): it for it in e12_items}

seen, uniq = set(), []
for it in metrics:
    k = (it["font"], it["cp"])
    if k in seen:
        continue
    seen.add(k)
    uniq.append(it)
rows = uniq
fonts = browse["fonts"]
BUCKETS = browse["buckets"]
# 精选子集：简单字形（如 1/c/n）各方法无差是正常的，默认只展示有区分度的字
CURATED = {
    "0", "2", "8",
    "A", "G", "M", "Q", "R", "W", "B", "O",
    "a", "o", "g", "e", "i", "l", "b", "q",
    "à", "é",
    "あ", "さ", "ん", "ア", "ン",
    "ㄅ", "ㄚ",
}


def chip(mid, font, cp, kind):
    if kind == "pix":
        it = m_item.get((mid, font, cp))
        return f"L1 {it['L1']:.3f} · LP {it['LPIPS']:.3f}" if it else ""
    e = e_item.get((mid, font, cp))
    return f"φ {e['e12_phi_sc_r']:.3f} · m {e['e12_mem_logit']:.1f}" if e else ""


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

cols = "".join(
    f"<th class='mh'><div class='ml'>{label}</div><div class='ms'>{shot}</div></th>"
    for _, label, shot in METHODS)

nav = "".join(f"<a href='#f{i}'>{f}</a> " for i, f in enumerate(fonts))
bucket_ui = "".join(
    f"<label><input type='checkbox' class='bkf' value='{b}' checked>{b}</label> "
    for b in BUCKETS)

sections = []
for fi, font in enumerate(fonts):
    frows = sorted([r for r in rows if r["font"] == font], key=lambda r: r["char"])
    body = []
    for r in frows:
        cp, ch = r["cp"], r["char"]
        gt = f"refs/gt/{font}+{cp}.png"
        cells = [f"<td class='gt'><img loading='lazy' src='{gt}' width='52'></td>"]
        for mid, _, _ in METHODS:
            img = f"preds/{mid}/test/{font}/test__{font}__{cp}__s3407.png"
            cells.append(
                f"<td><img loading='lazy' src='{img}' width='52' title='{mid}'>"
                f"<div class='ch'>{chip(mid, font, cp, 'pix')}</div>"
                f"<div class='ch e'>{chip(mid, font, cp, 'e12')}</div></td>")
        body.append(
            f"<tr data-bk='{r['bucket']}' data-keep='{1 if ch in CURATED else 0}'>"
            f"<td class='chr'>{ch}"
            f"<div class='bk'>{r['bucket']}</div></td>{''.join(cells)}</tr>")
    sections.append(
        f"<h3 class='fhead' id='f{fi}' data-font='{font}'>{fi+1}. {font} "
        f"<span class='fold'>▾</span></h3>"
        f"<table class='gly' data-font='{font}'>"
        f"<tr><th class='chh'>字</th><th class='chh'>GT</th>{cols}</tr>"
        f"{''.join(body)}</table>")

html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Multi-Shot Board — P1..F3</title>
<style>
:root{{
  --bg:#151310; --card:#1d1a16; --border:#3a342c;
  --fg:#d6cdb8; --muted:#8a8171; --accent:#9ab87a;
  --amber:#c8a24b; --red:#c06a5e; --head:#221e18;
}}
body{{font-family:Menlo,monospace;background:var(--bg);color:var(--fg);margin:14px}}
h2{{color:var(--fg)}} h3{{color:var(--accent);cursor:pointer;margin:20px 0 6px}}
h3:hover{{text-shadow:0 0 6px #9ab87a55}}
table.sum{{border-collapse:collapse;font-size:12px;margin:10px 0}}
table.sum td,table.sum th{{border:1px solid var(--border);padding:3px 7px;text-align:right;color:var(--fg)}}
table.sum th{{background:var(--head);color:var(--accent)}}
table.gly{{border-collapse:collapse;font-size:11px}}
table.gly td,table.gly th{{border:1px solid var(--border);padding:2px 4px;text-align:center;vertical-align:top}}
table.gly tr:nth-child(even) td{{background:#191613}}
table.gly th{{background:var(--head);color:var(--accent);position:sticky;top:40px;z-index:3}}
.ml{{font-weight:bold;color:var(--fg)}}.ms{{color:var(--muted);font-size:10px}}
.chr{{font-weight:bold;font-size:13px;min-width:2.2em;color:var(--fg)}}
.bk{{color:var(--muted);font-size:9px;font-weight:normal}}
.ch{{font-size:9px;color:var(--muted);line-height:1.3}}
.ch.e{{color:var(--amber)}}
nav a{{margin-right:6px;font-size:12px;color:var(--accent);text-decoration:none}}
nav a:hover{{text-decoration:underline}}
.fbar{{position:sticky;top:0;background:var(--card);padding:8px;border:1px solid var(--border);z-index:5}}
.fbar label{{margin-right:10px;font-size:12px;color:var(--fg);cursor:pointer}}
.fbar input{{accent-color:var(--accent)}}
button{{background:var(--head);color:var(--fg);border:1px solid var(--border);padding:4px 10px;margin-right:6px;cursor:pointer;font-family:inherit}}
button:hover{{border-color:var(--accent);color:var(--accent)}}
.note{{color:var(--muted);font-size:12px;margin:6px 0}}
.fold{{color:var(--muted)}}
.hidden{{display:none}}
</style></head><body>
<h2>Multi-Shot Board：P1 → F3，各模型 shot 变体同表对照</h2>
<div class='note'>GT 仅作 positive control。L1/LPIPS 逐字像素诊断；φ=phi SC-R（越大越好）、m=membership logit。点击字体标题折叠/展开；上方按字符桶过滤。</div>
<div class='fbar'>
{bucket_ui}
<button onclick="foldAll()">全部折叠</button><button onclick="unfoldAll()">全部展开</button>
<button id='btnChips'>隐藏数值</button>
<button id='btnCur'>显示全部字(47)</button>
</div>
{summary}
<nav>{nav}</nav>
{''.join(sections)}
<script>
const boxes=[...document.querySelectorAll('.bkf')];
boxes.forEach(b=>b.addEventListener('change',applyFilter));
let curOnly=true;
function applyFilter(){{
  const on=new Set(boxes.filter(b=>b.checked).map(b=>b.value));
  document.querySelectorAll('tr[data-bk]').forEach(tr=>{{
    const keepOk = !curOnly || tr.dataset.keep==='1';
    tr.classList.toggle('hidden', !(on.has(tr.dataset.bk) && keepOk));
  }});
}}
document.getElementById('btnCur').addEventListener('click',()=>{{
  curOnly=!curOnly;
  document.getElementById('btnCur').textContent=curOnly?'显示全部字(47)':'只看精选字(28)';
  applyFilter();
}});
applyFilter();
document.querySelectorAll('.fhead').forEach(h=>h.addEventListener('click',()=>{{
  const t=document.querySelector(`table.gly[data-font="${{h.dataset.font}}"]`);
  t.classList.toggle('hidden');
  h.querySelector('.fold').textContent=t.classList.contains('hidden')?'▸':'▾';
}}));
function foldAll(){{document.querySelectorAll('table.gly').forEach(t=>t.classList.add('hidden'));
 document.querySelectorAll('.fold').forEach(s=>s.textContent='▸');}}
function unfoldAll(){{document.querySelectorAll('table.gly').forEach(t=>t.classList.remove('hidden'));
 document.querySelectorAll('.fold').forEach(s=>s.textContent='▾');}}
let chipsOn=true;
document.getElementById('btnChips').addEventListener('click',()=>{{
  chipsOn=!chipsOn;
  document.querySelectorAll('.ch').forEach(c=>c.classList.toggle('hidden',!chipsOn));
  document.getElementById('btnChips').textContent=chipsOn?'隐藏数值':'显示数值';
}});
</script>
</body></html>"""
out = P / "multi_shot_board.html"
out.write_text(html, encoding="utf-8")
print("written", out, f"{len(rows)} rows, {out.stat().st_size//1024} KB")
