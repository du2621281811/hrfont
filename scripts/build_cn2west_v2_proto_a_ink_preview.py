#!/usr/bin/env python3
"""Scan proto-A dataset ink occupancy; build sorted preview page.

Metrics per PNG (96×96):
  ink_h / ink_w / ink_area ratios; max_dim = max(ink_h, ink_w)/96

Outputs under data/cn2west_v2_abc_review/proto_A_ink/:
  ink_report.json  — stats + bins + examples
  index.html       — fonts sorted by 永/median ink; probe gallery
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/projects/hrfont")
DS = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
CHARSET = json.loads((ROOT / "manifests/charset_cn2west_v2_planned.json").read_text(encoding="utf-8"))
OUT = ROOT / "data/cn2west_v2_abc_review/proto_A_ink"
CANVAS = 96

# ink max_dim bins (fraction of 96)
BINS = [
    ("empty", 0.00, 0.02, "空白/近空"),
    ("tiny", 0.02, 0.25, "极小 <25%"),
    ("small", 0.25, 0.50, "偏小 25–50%"),
    ("mid", 0.50, 0.70, "中等 50–70%"),
    ("large", 0.70, 0.85, "偏大 70–85%"),
    ("full", 0.85, 1.01, "接近满框 ≥85%"),
]


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def char_catalog() -> dict[str, dict]:
    """cp -> {ch, role, cat}"""
    cat: dict[str, dict] = {}
    for ch in CHARSET["style_han_338"]:
        cat[cp_of(ch)] = {"ch": ch, "role": "style", "cat": "style_han"}
    for key, chars in CHARSET["target"].items():
        for ch in chars:
            cat[cp_of(ch)] = {"ch": ch, "role": "target", "cat": key}
    # target_string may include extras already in target buckets
    for ch in CHARSET["target_string"]:
        cp = cp_of(ch)
        if cp not in cat:
            cat[cp] = {"ch": ch, "role": "target", "cat": "other"}
    return cat


def ink_of(path: Path) -> dict | None:
    try:
        a = np.asarray(Image.open(path).convert("L"))
    except Exception:
        return None
    ys, xs = np.where(a < 250)
    if len(ys) == 0:
        return {"h": 0.0, "w": 0.0, "area": 0.0, "max": 0.0, "empty": True}
    h = (ys.max() - ys.min() + 1) / CANVAS
    w = (xs.max() - xs.min() + 1) / CANVAS
    area = float(np.count_nonzero(a < 250)) / (CANVAS * CANVAS)
    return {"h": float(h), "w": float(w), "area": area, "max": float(max(h, w)), "empty": False}


def scan_font(job: dict) -> dict:
    stem, split = job["stem"], job["split"]
    root = Path(job["root"])
    catalog = job["catalog"]  # cp -> meta
    samples = []  # list of {cp, role, cat, max, h, w, area}
    by_cat: dict[str, list[float]] = defaultdict(list)

    for role, sub in (("target", "TargetImage"), ("style", "StyleImage")):
        d = root / split / sub / stem
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            # stem+uXXXX.png
            name = p.name
            if "+" not in name:
                continue
            cp = name.split("+", 1)[1].removesuffix(".png")
            meta = catalog.get(cp)
            if not meta:
                continue
            m = ink_of(p)
            if m is None:
                continue
            samples.append({
                "cp": cp, "ch": meta["ch"], "role": role, "cat": meta["cat"],
                "max": m["max"], "h": m["h"], "w": m["w"], "area": m["area"], "empty": m["empty"],
            })
            by_cat[meta["cat"]].append(m["max"])

    def med(vals: list[float]) -> float | None:
        if not vals:
            return None
        return float(np.median(vals))

    yong = next((s for s in samples if s["ch"] == "永"), None)
    all_max = [s["max"] for s in samples]
    return {
        "stem": stem,
        "split": split,
        "n": len(samples),
        "n_empty": sum(1 for s in samples if s["empty"]),
        "yong_max": yong["max"] if yong else None,
        "yong_h": yong["h"] if yong else None,
        "median_max": med(all_max),
        "p5_max": float(np.percentile(all_max, 5)) if all_max else None,
        "p95_max": float(np.percentile(all_max, 95)) if all_max else None,
        "mean_max": float(np.mean(all_max)) if all_max else None,
        "cat_median": {k: med(v) for k, v in sorted(by_cat.items())},
        "samples": samples,
    }


def bin_of(v: float) -> str:
    for key, lo, hi, _ in BINS:
        if lo <= v < hi:
            return key
    return "full"


def build_report(font_rows: list[dict], catalog: dict) -> dict:
    # global glyph pool for bin examples (across fonts)
    bin_hits: dict[str, list] = defaultdict(list)
    cat_vals: dict[str, list[float]] = defaultdict(list)
    all_vals: list[float] = []
    font_summaries = []

    for fr in font_rows:
        for s in fr["samples"]:
            all_vals.append(s["max"])
            cat_vals[s["cat"]].append(s["max"])
            bin_hits[bin_of(s["max"])].append({
                "stem": fr["stem"], "split": fr["split"],
                "ch": s["ch"], "cp": s["cp"], "role": s["role"], "cat": s["cat"],
                "max": round(s["max"], 4), "h": round(s["h"], 4), "area": round(s["area"], 4),
            })
        font_summaries.append({
            "stem": fr["stem"],
            "split": fr["split"],
            "n": fr["n"],
            "n_empty": fr["n_empty"],
            "yong_max": fr["yong_max"],
            "yong_h": fr["yong_h"],
            "median_max": fr["median_max"],
            "p5_max": fr["p5_max"],
            "p95_max": fr["p95_max"],
            "mean_max": fr["mean_max"],
            "cat_median": fr["cat_median"],
            # keep top/bottom few glyphs for this font
            "smallest": sorted(fr["samples"], key=lambda x: x["max"])[:6],
            "largest": sorted(fr["samples"], key=lambda x: -x["max"])[:6],
        })

    # sort fonts: smallest median ink first (problem fonts first)
    font_summaries.sort(key=lambda f: (f["median_max"] if f["median_max"] is not None else 9, f["stem"]))

    n = len(all_vals)
    arr = np.array(all_vals) if all_vals else np.array([0.0])

    def pct(p: float) -> float:
        return float(np.percentile(arr, p)) if n else 0.0

    bins_out = []
    for key, lo, hi, label in BINS:
        items = bin_hits[key]
        # diversify examples: prefer different stems/cats
        items_sorted = sorted(items, key=lambda x: x["max"])
        examples = []
        seen_stem, seen_cat = set(), set()
        for it in items_sorted + items_sorted[::-1]:
            if len(examples) >= 12:
                break
            tag = (it["stem"], it["cp"])
            if tag in {(e["stem"], e["cp"]) for e in examples}:
                continue
            # prefer new stem/cat
            score = (it["stem"] in seen_stem) + (it["cat"] in seen_cat)
            if score >= 2 and len(examples) < 8:
                continue
            examples.append(it)
            seen_stem.add(it["stem"])
            seen_cat.add(it["cat"])
        bins_out.append({
            "id": key, "label": label, "lo": lo, "hi": hi,
            "n": len(items), "pct": round(100 * len(items) / max(n, 1), 2),
            "examples": examples[:12],
        })

    cat_out = []
    for cat, vals in sorted(cat_vals.items(), key=lambda kv: np.median(kv[1])):
        a = np.array(vals)
        cat_out.append({
            "cat": cat,
            "n": len(vals),
            "median": float(np.median(a)),
            "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
            "mean": float(np.mean(a)),
        })

    return {
        "dataset": DS.name,
        "proto": "A",
        "metric": "ink_max_dim / 96 (max of ink bbox height,width)",
        "n_glyphs": n,
        "n_fonts": len(font_summaries),
        "global": {
            "min": float(arr.min()),
            "p5": pct(5), "p25": pct(25), "median": pct(50),
            "p75": pct(75), "p95": pct(95), "max": float(arr.max()),
            "mean": float(arr.mean()),
            "n_empty": int(np.sum(arr < 0.02)),
        },
        "bins": bins_out,
        "by_category": cat_out,
        "fonts": font_summaries,
        "probes": {
            "style_han": list("永和书风骨韵天地"),
            "ascii_letters": list("AaGgWwMm"),
            "ascii_digits": list("0123456789"),
            "bopomofo": list("ㄅㄆㄇㄈㄉㄊ"),
            "hiragana": list("あいうえおか"),
            "katakana": list("アイウエオカ"),
        },
    }


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>协议 A · 墨量分布预览</title>
<style>
:root{--bg:#eef1f5;--panel:#fff;--line:#d5dbe3;--muted:#5c6570;--accent:#1f4a6f}
*{box-sizing:border-box} html,body{height:100%;margin:0}
body{font:13px/1.4 system-ui,sans-serif;background:var(--bg);color:#1a1a1a;display:flex;flex-direction:column}
header{flex:0 0 auto;background:#fff;border-bottom:1px solid var(--line);padding:10px 14px}
h1{margin:0;font-size:1.1rem} .meta{color:var(--muted);font-size:12px;margin-top:2px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
input,select,button{padding:5px 9px;border:1px solid var(--line);border-radius:4px;font:inherit;background:#fff}
.layout{flex:1;min-height:0;display:grid;grid-template-columns:320px 1fr}
aside{background:var(--panel);border-right:1px solid var(--line);overflow:auto}
#fontList{list-style:none;margin:0;padding:0}
#fontList li{padding:8px 10px;border-bottom:1px solid #eef1f5;cursor:pointer}
#fontList li:hover{background:#f5f7fa} #fontList li.active{background:#e8eef5}
.stem{font-weight:600;font-family:ui-monospace,monospace;font-size:12px}
.sub{color:var(--muted);font-size:11px;margin-top:2px}
main{overflow:auto;padding:12px 16px 48px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px;margin-bottom:12px}
.bar{display:flex;height:18px;border-radius:3px;overflow:hidden;border:1px solid var(--line);margin:8px 0}
.bar span{display:block;height:100%;font-size:10px;color:#fff;text-align:center;line-height:18px}
.bin-empty{background:#9aa3ad}.bin-tiny{background:#c0392b}.bin-small{background:#d68910}
.bin-mid{background:#2980b9}.bin-large{background:#27ae60}.bin-full{background:#1a7a4c}
.table{width:100%;border-collapse:collapse;font-size:12px}
.table th,.table td{border:1px solid var(--line);padding:4px 6px;text-align:left}
.strip{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.strip figure{margin:0;width:64px;text-align:center}
.strip img{width:64px;height:64px;image-rendering:pixelated;border:1px solid #e0e4ea;background:#fff;display:block}
.strip figcaption{font-size:10px;color:#444;margin-top:1px}
.badge{display:inline-block;font-size:10px;padding:0 5px;border-radius:3px;background:#eef1f5;margin-right:4px}
.sec h3{margin:0 0 6px;font-size:13px}
</style>
</head>
<body>
<header>
  <h1>协议 A · 墨量分布与预览</h1>
  <div class="meta">墨量 = max(墨迹高,墨迹宽)/96 · 字体按 median 墨量升序 · 图片直接读 A 数据集 PNG</div>
  <div class="toolbar">
    <input type="search" id="q" placeholder="搜 stem" style="min-width:160px"/>
    <select id="split"><option value="all">全部 split</option><option value="train">train</option><option value="val">val</option><option value="test">test</option></select>
    <select id="sort">
      <option value="median_asc">median 墨量 ↑</option>
      <option value="median_desc">median 墨量 ↓</option>
      <option value="yong_asc">永字墨量 ↑</option>
      <option value="yong_desc">永字墨量 ↓</option>
    </select>
    <span id="progress"></span>
  </div>
</header>
<div class="layout">
  <aside><ul id="fontList"></ul></aside>
  <main id="main">加载中…</main>
</div>
<script>
const pct=v=>v==null?'—':Math.round(v*100)+'%';
const f2=v=>v==null?'—':Number(v).toFixed(2);
let DATA=null, filtered=[], cur=-1;
const BIN_COLOR={empty:'bin-empty',tiny:'bin-tiny',small:'bin-small',mid:'bin-mid',large:'bin-large',full:'bin-full'};

function imgUrl(f, role, cp){
  const root='../../fontdiffuser-p253-t295-s338-cn2west-v2/'+f.split+'/';
  const dir=role==='style'?'StyleImage':'TargetImage';
  return root+dir+'/'+f.stem+'/'+f.stem+'+'+cp+'.png';
}
function cp(ch){return 'u'+ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0');}

function overviewHTML(){
  const g=DATA.global, bins=DATA.bins;
  const bar=bins.map(b=>{
    const w=Math.max(b.pct,0.4);
    return `<span class="${BIN_COLOR[b.id]}" style="width:${w}%" title="${b.label}: ${b.n}">${b.pct>=4?b.pct+'%':''}</span>`;
  }).join('');
  const binRows=bins.map(b=>`
    <tr><td><b>${b.label}</b></td><td>${b.n.toLocaleString()}</td><td>${b.pct}%</td>
    <td><div class="strip">${b.examples.map(e=>
      `<figure><img loading="lazy" src="${imgUrl(e,e.role,e.cp)}" alt=""/><figcaption>${e.ch}<br/>${pct(e.max)}</figcaption></figure>`
    ).join('')}</div></td></tr>`).join('');
  const catRows=DATA.by_category.map(c=>
    `<tr><td>${c.cat}</td><td>${c.n.toLocaleString()}</td><td>${pct(c.median)}</td><td>${pct(c.p5)}</td><td>${pct(c.p95)}</td></tr>`
  ).join('');
  return `<div class="card">
    <h2 style="margin:0 0 6px">全局墨量分布（${DATA.n_glyphs.toLocaleString()} 张 / ${DATA.n_fonts} 字体）</h2>
    <div class="meta">min ${pct(g.min)} · p5 ${pct(g.p5)} · median ${pct(g.median)} · p95 ${pct(g.p95)} · max ${pct(g.max)} · mean ${pct(g.mean)}</div>
    <div class="bar">${bar}</div>
    <table class="table"><thead><tr><th>分档</th><th>数量</th><th>占比</th><th>示例字（跨字体）</th></tr></thead>
    <tbody>${binRows}</tbody></table>
  </div>
  <div class="card">
    <h2 style="margin:0 0 6px">按语种/类别 median 墨量</h2>
    <table class="table"><thead><tr><th>类别</th><th>样本量</th><th>median</th><th>p5</th><th>p95</th></tr></thead>
    <tbody>${catRows}</tbody></table>
  </div>
  <div class="card"><b>点左侧字体</b>查看该字体探针与最小/最大墨量字（已按墨量排序）。</div>`;
}

function applyFilter(){
  const q=document.getElementById('q').value.trim().toLowerCase();
  const split=document.getElementById('split').value;
  const sort=document.getElementById('sort').value;
  filtered=DATA.fonts.filter(f=>{
    if(split!=='all' && f.split!==split) return false;
    if(q && !f.stem.toLowerCase().includes(q)) return false;
    return true;
  });
  const key=sort.startsWith('yong')?'yong_max':'median_max';
  const desc=sort.endsWith('desc');
  filtered.sort((a,b)=>{
    const av=a[key], bv=b[key];
    if(av==null&&bv==null) return a.stem.localeCompare(b.stem);
    if(av==null) return 1; if(bv==null) return -1;
    return desc? bv-av : av-bv;
  });
  renderList();
  document.getElementById('progress').textContent=`显示 ${filtered.length}/${DATA.fonts.length}`;
  if(cur<0) document.getElementById('main').innerHTML=overviewHTML();
}
function renderList(){
  document.getElementById('fontList').innerHTML=filtered.map((f,i)=>`
    <li data-i="${i}" class="${cur===i?'active':''}">
      <div><span class="stem">${f.stem}</span></div>
      <div class="sub">${f.split} · med ${pct(f.median_max)} · 永 ${pct(f.yong_max)} · p5–p95 ${pct(f.p5_max)}–${pct(f.p95_max)}</div>
    </li>`).join('');
  document.querySelectorAll('#fontList li').forEach(li=>li.onclick=()=>selectFont(+li.dataset.i));
}
function selectFont(i){
  if(i<0||i>=filtered.length) return;
  cur=i; renderList(); renderDetail();
  document.getElementById('main').scrollTop=0;
}
function strip(f, items, title){
  return `<section class="sec card"><h3>${title}</h3><div class="strip">${items.map(s=>{
    const role=s.role||(DATA.probes.style_han.includes(s.ch)?'style':'target');
    const c=s.cp||cp(s.ch);
    const lab=s.max!=null?pct(s.max):(s.ch||'');
    return `<figure><img loading="lazy" src="${imgUrl(f,role,c)}" alt=""/><figcaption>${s.ch||''}<br/>${lab}</figcaption></figure>`;
  }).join('')}</div></section>`;
}
function renderDetail(){
  const f=filtered[cur];
  const probes=[];
  for(const [cat,chars] of Object.entries(DATA.probes)){
    probes.push(strip(f, chars.map(ch=>({ch, role:cat==='style_han'?'style':'target'})), `探针 · ${cat}`));
  }
  const small=f.smallest.map(s=>({...s}));
  const large=f.largest.map(s=>({...s}));
  document.getElementById('main').innerHTML=`
    <div class="card">
      <h2 style="margin:0 0 4px">${f.stem} <span class="badge">${f.split}</span></h2>
      <div class="meta">median ${pct(f.median_max)} · 永 ${pct(f.yong_max)} (h=${pct(f.yong_h)}) · p5 ${pct(f.p5_max)} · p95 ${pct(f.p95_max)} · empty ${f.n_empty}/${f.n}</div>
      <div class="meta" style="margin-top:4px">类别 median：${Object.entries(f.cat_median||{}).map(([k,v])=>k+' '+pct(v)).join(' · ')}</div>
      <div style="margin-top:8px"><button id="btnPrev">←</button> <button id="btnNext">→</button>
      <button id="btnOverview" style="margin-left:8px">总览</button></div>
    </div>
    ${strip(f, small, '本字体墨量最小')}
    ${strip(f, large, '本字体墨量最大')}
    ${probes.join('')}`;
  document.getElementById('btnPrev').onclick=()=>selectFont(cur-1);
  document.getElementById('btnNext').onclick=()=>selectFont(cur+1);
  document.getElementById('btnOverview').onclick=()=>{cur=-1;renderList();document.getElementById('main').innerHTML=overviewHTML();};
}
document.getElementById('q').oninput=()=>{cur=-1;applyFilter();};
document.getElementById('split').onchange=()=>{cur=-1;applyFilter();};
document.getElementById('sort').onchange=()=>{cur=-1;applyFilter();};
document.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
  if(e.key==='j'||e.key==='J') selectFont(cur+1);
  if(e.key==='k'||e.key==='K') selectFont(cur-1);
});
fetch('ink_report.json').then(r=>r.json()).then(d=>{DATA=d;applyFilter();});
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()

    summary = json.loads((DS / "summary.json").read_text(encoding="utf-8"))
    catalog = char_catalog()
    # make catalog picklable for workers
    jobs = [{"stem": f["stem"], "split": f["split"], "root": str(DS), "catalog": catalog} for f in summary["fonts"]]

    print(f"Scanning A · {len(jobs)} fonts · workers={args.workers}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_font, j): j["stem"] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 20 == 0 or done <= 3:
                print(f"  {done}/{len(jobs)}", flush=True)

    report = build_report(rows, catalog)
    # slim fonts for JSON: drop huge sample dumps already removed
    for f in report["fonts"]:
        for key in ("smallest", "largest"):
            f[key] = [
                {"ch": s["ch"], "cp": s["cp"], "role": s["role"], "cat": s["cat"], "max": round(s["max"], 4)}
                for s in f[key]
            ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ink_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "index.html").write_text(HTML, encoding="utf-8")

    g = report["global"]
    print(json.dumps({
        "n_glyphs": report["n_glyphs"],
        "n_fonts": report["n_fonts"],
        "median": round(g["median"], 4),
        "p5": round(g["p5"], 4),
        "p95": round(g["p95"], 4),
        "bins": {b["id"]: {"n": b["n"], "pct": b["pct"]} for b in report["bins"]},
        "out": str(OUT),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
