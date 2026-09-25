#!/usr/bin/env python3
"""Eye board: F2@40k vs F2-VEC raster preds on test16.

What you see is the UNet raster head (DPM++20). The VecHead ellipse occupancy
was only used as a training loss; vec_head.pth / SVG dumps are not in the repo,
so vector renders cannot be shown without the V100 checkpoint.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PRED = ROOT / "reports/f03_test16_strat/preds"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
OUT = ROOT / "reports/f03_test16_strat"
ITEMS = OUT / "metrics_items.json"
MASK = ROOT / "manifests/v0913_clean/pairs_eval47.tsv"
SEED = 3407

ARMS = [
    {"id": "F2", "label": "F2@40k", "mid": "F2_40000", "tone": "f2"},
    {"id": "VECfs", "label": "F2-VEC few", "mid": "F2VEC_40000", "tone": "vec"},
    {"id": "VECs1", "label": "F2-VEC s1", "mid": "F2VEC_40000_s1", "tone": "vec1"},
]
FOCUS = list("AQaRgあのIl1B7")
LEAD = [
    "FZMaWDBSJW",
    "FZCuanBZBKSJW",
    "FZLingFKSJW-B",
    "FZFengYKSJ",
    "FZDeSHJW_515H",
    "FZFeiHHJW-H",
    "FZDouNTJW_Te",
    "FZJianLTJW_Te",
    "FZChuangHJW_DB",
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def load_mask() -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    if not MASK.is_file():
        return out
    for line in MASK.read_text(encoding="utf-8").splitlines()[1:]:
        split, font, ch, *_rest, keep = line.split("\t")
        if split != "test":
            continue
        out[(font, ch)] = keep == "1"
    return out


def main() -> None:
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    mask = load_mask()
    by: dict[str, dict[tuple[str, str], dict]] = {a["mid"]: {} for a in ARMS}
    for r in items:
        mid = r.get("method")
        if mid in by:
            by[mid][(r["font"], r["char"])] = r

    fonts = sorted({k[0] for mid in by.values() for k in mid})
    ordered = [f for f in LEAD if f in fonts] + [f for f in fonts if f not in LEAD]
    chars = []
    seen = set()
    for mid in by.values():
        for (_f, ch), r in mid.items():
            if ch in seen:
                continue
            seen.add(ch)
            chars.append({"ch": ch, "cp": cp_of(ch), "bucket": r.get("bucket") or "other"})
    chars.sort(key=lambda c: (c["bucket"], c["ch"]))

    # per-font paired Δ L1 (VEC few - F2)
    font_delta = []
    f2m, vecm = by["F2_40000"], by["F2VEC_40000"]
    for f in ordered:
        ds = []
        for (font, ch), a in f2m.items():
            if font != f:
                continue
            b = vecm.get((font, ch))
            if not b:
                continue
            ds.append(b["L1"] - a["L1"])
        if ds:
            font_delta.append(
                {
                    "font": f,
                    "n": len(ds),
                    "dL1": sum(ds) / len(ds),
                    "vec_win": sum(1 for x in ds if x < 0) / len(ds),
                }
            )

    overall = {}
    for a in ARMS:
        rows = list(by[a["mid"]].values())
        clean = [r for r in rows if mask.get((r["font"], r["char"]), True)]

        def mean(xs, k):
            vals = [r[k] for r in xs if r.get(k) is not None]
            return sum(vals) / len(vals) if vals else None

        overall[a["mid"]] = {
            "n752": len(rows),
            "L1_752": mean(rows, "L1"),
            "SSIM_752": mean(rows, "SSIM"),
            "LPIPS_752": mean(rows, "LPIPS"),
            "n704": len(clean),
            "L1_704": mean(clean, "L1"),
            "SSIM_704": mean(clean, "SSIM"),
            "LPIPS_704": mean(clean, "LPIPS"),
        }

    payload = {
        "generated_at": utc(),
        "seed": SEED,
        "note": (
            "人眼看的是 UNet 栅格头（DPM++20）。VecHead 只在训练时出 16 个椭圆 occupancy 当辅助损失；"
            "仓库无 vec_head.pth / SVG，无法单独看矢量渲染。"
        ),
        "arms": ARMS,
        "fonts": ordered,
        "lead": [f for f in LEAD if f in fonts],
        "chars": chars,
        "focus_chars": [c for c in FOCUS if c in seen],
        "font_delta": font_delta,
        "metrics": overall,
        "paths": {
            "pred": f"preds/{{mid}}/test/{{stem}}/test__{{stem}}__{{cp}}__s{SEED}.png",
            "gt": "refs/gt/{stem}+{cp}.png",
            "style": "refs/style/{stem}.png",
            "content": "refs/content/{cp}.png",
        },
    }

    (OUT / "f2vec_eye_board.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    data_js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    (OUT / "f2vec_eye_board.html").write_text(HTML.replace("__DATA__", data_js), encoding="utf-8")
    print("wrote", OUT / "f2vec_eye_board.html")


HTML = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>F2-VEC 人眼板 · test16</title>
<style>
:root {
  --bg:#14120f; --card:#1c1914; --line:#3a342c; --fg:#e4dcc8;
  --muted:#8d8473; --acc:#b7c98a; --warn:#c8a24b; --head:#221e18;
  --gt:#c8a24b; --f2:#6aa8a0; --vec:#d4a574; --vec1:#9b8fd4;
}
* { box-sizing:border-box; }
html, body { height:100%; }
body {
  margin:0; display:flex; flex-direction:column; overflow:hidden;
  font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--fg);
}
header { flex:0 0 auto; background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px 8px; }
h1 { font-size:1.05rem; margin:0 0 4px; font-weight:650; }
.meta { color:var(--muted); font-size:12px; max-width:140ch; line-height:1.45; }
.warnbox {
  margin-top:8px; padding:8px 10px; border:1px solid #5a4020; background:#2a2010;
  color:#f0d078; font-size:12px; line-height:1.45; max-width:140ch;
}
.bar { display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; margin-top:8px; font-size:13px; }
.bar label { cursor:pointer; }
.bar.cols { gap:6px 10px; padding-top:6px; border-top:1px dashed var(--line); margin-top:10px; }
.bar .grp { color:var(--muted); font-size:12px; margin-right:2px; }
select, button, input[type=range] {
  background:var(--head); color:var(--fg); border:1px solid var(--line); padding:4px 8px; font:inherit;
}
button.preset { cursor:pointer; }
button.preset:hover { border-color:var(--acc); color:var(--acc); }
button.preset.on { border-color:var(--acc); background:#243018; color:var(--acc); }
label.chip {
  border:1px solid var(--line); padding:2px 8px; border-radius:999px; background:var(--head);
  user-select:none;
}
label.chip input { vertical-align:middle; margin-right:4px; }
label.chip.off { opacity:.45; }
main { flex:1; min-height:0; display:flex; flex-direction:column; padding:8px 16px 10px; gap:8px; }
.top { flex:0 0 auto; display:grid; grid-template-columns:1.2fr .8fr; gap:10px; }
@media (max-width:1100px){ .top { grid-template-columns:1fr; } }
.panel { border:1px solid var(--line); background:#17140f; padding:8px; overflow:auto; max-height:160px; }
.panel table { border-collapse:collapse; font-size:12px; width:100%; }
.panel th, .panel td { border:1px solid var(--line); padding:3px 6px; text-align:right; }
.panel th:first-child, .panel td:first-child { text-align:left; }
.win { color:#b7e4de; } .lose { color:#e8a0a0; }
.scroll { flex:1; min-height:0; overflow:auto; border:1px solid var(--line); background:#17140f; }
table.grid { border-collapse:separate; border-spacing:0; font-size:11px; }
.grid th, .grid td {
  border-right:1px solid var(--line); border-bottom:1px solid var(--line);
  padding:3px 4px; text-align:center; vertical-align:middle;
}
.grid thead th { position:sticky; top:0; z-index:6; background:var(--head); white-space:nowrap; }
.grid th.stub, .grid td.stub {
  position:sticky; left:0; z-index:8; background:var(--card); text-align:left;
  min-width:7em; padding-left:8px; box-shadow:1px 0 0 var(--line); font-weight:700;
}
.grid thead th.stub { z-index:10; }
.grid td.chr { font-size:18px; min-width:2.2em; }
img.g { width:var(--gs,72px); height:var(--gs,72px); image-rendering:pixelated; background:#fff; display:block; cursor:zoom-in; }
.tone-gt { background:#2f2614; }
thead th.tone-gt { color:#f0d078; box-shadow:inset 0 -3px 0 var(--gt); }
.tone-f2 { background:#141c1b; }
thead th.tone-f2 { color:#b7e4de; box-shadow:inset 0 -3px 0 var(--f2); }
.tone-vec { background:#241c14; }
thead th.tone-vec { color:#f0d0a8; box-shadow:inset 0 -3px 0 var(--vec); }
.tone-vec1 { background:#1b1724; }
thead th.tone-vec1 { color:#d4c8ff; box-shadow:inset 0 -3px 0 var(--vec1); }
.miss { color:#555; }
#zoom {
  display:none; position:fixed; inset:0; z-index:40; background:rgba(0,0,0,.78);
  align-items:center; justify-content:center; padding:24px;
}
#zoom.on { display:flex; }
#zoom .box { background:var(--card); border:1px solid var(--line); padding:12px 16px; max-width:96vw; }
#zoom .row { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; }
#zoom figure { margin:0; text-align:center; }
#zoom img { width:160px; height:160px; image-rendering:pixelated; background:#fff; }
#zoom figcaption { color:var(--muted); font-size:12px; margin-top:4px; }
</style></head><body>
<header>
  <h1>F2-VEC 人眼板 · test16 栅格输出</h1>
  <p class="meta">勾选下方「展示列」选择要看的结果；快捷预设一键切换。点图放大（放大窗也只显示已勾选列）。</p>
  <div class="warnbox" id="note"></div>
  <div class="bar">
    <label>视图
      <select id="view">
        <option value="char">固定字体 · 逐字</option>
        <option value="font">固定字 · 逐字体</option>
      </select>
    </label>
    <label>字体 <select id="font"></select></label>
    <label>字 <select id="char"></select></label>
    <label>语种 <select id="bucket"><option value="">全部</option></select></label>
    <label class="chip"><input type="checkbox" id="focus"/> 探针字</label>
    <label class="chip"><input type="checkbox" id="leadOnly"/> 只看重点字体</label>
    <label>大小 <input id="size" type="range" min="48" max="112" value="72"/></label>
  </div>
  <div class="bar cols" id="colBar">
    <span class="grp">展示列</span>
    <label class="chip"><input type="checkbox" data-col="content" checked/> Content</label>
    <label class="chip"><input type="checkbox" data-col="style" checked/> Style永</label>
    <label class="chip"><input type="checkbox" data-col="gt" checked/> GT</label>
    <span id="armToggles"></span>
    <span class="grp">快捷</span>
    <button type="button" class="preset" data-preset="all">全开</button>
    <button type="button" class="preset" data-preset="f2vec">F2 vs VEC few</button>
    <button type="button" class="preset" data-preset="f2s1">F2 vs VEC s1</button>
    <button type="button" class="preset" data-preset="vecpair">VEC few vs s1</button>
    <button type="button" class="preset" data-preset="preds">只看结果</button>
    <button type="button" class="preset" data-preset="gtf2">GT + F2</button>
  </div>
</header>
<main>
  <div class="top">
    <div class="panel" id="metrics"></div>
    <div class="panel" id="fontrank"></div>
  </div>
  <div class="scroll"><table class="grid" id="grid"></table></div>
</main>
<div id="zoom"><div class="box"><div id="zoomCap" class="meta"></div><div class="row" id="zoomRow"></div></div></div>
<script>
const DATA = __DATA__;
const LS_KEY = "f2vec_eye_cols_v1";
const $ = id => document.getElementById(id);
const tone = {gt:"tone-gt", f2:"tone-f2", vec:"tone-vec", vec1:"tone-vec1"};
const PRESETS = {
  all:    {content:1, style:1, gt:1, arms:["F2","VECfs","VECs1"]},
  f2vec:  {content:0, style:0, gt:1, arms:["F2","VECfs"]},
  f2s1:   {content:0, style:0, gt:1, arms:["F2","VECs1"]},
  vecpair:{content:0, style:0, gt:1, arms:["VECfs","VECs1"]},
  preds:  {content:0, style:0, gt:0, arms:["F2","VECfs","VECs1"]},
  gtf2:   {content:0, style:0, gt:1, arms:["F2"]},
};
function fmt(x,n=4){ return (x==null||Number.isNaN(x))?"—":Number(x).toFixed(n); }
function path(tpl,o){
  return tpl.replaceAll("{mid}",o.mid||"").replaceAll("{stem}",o.stem||"").replaceAll("{cp}",o.cp||"");
}
function img(src){
  return `<img class="g" src="${src}" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('span'),{className:'miss',textContent:'·'}))"/>`;
}
function defaultVis(){
  return {content:true, style:true, gt:true, arms:Object.fromEntries(DATA.arms.map(a=>[a.id,true]))};
}
function loadVis(){
  try {
    const raw = JSON.parse(localStorage.getItem(LS_KEY)||"null");
    if (!raw || typeof raw!=="object") return defaultVis();
    const d = defaultVis();
    d.content = !!raw.content; d.style = !!raw.style; d.gt = !!raw.gt;
    DATA.arms.forEach(a => { d.arms[a.id] = raw.arms && a.id in raw.arms ? !!raw.arms[a.id] : true; });
    return d;
  } catch(_e){ return defaultVis(); }
}
let VIS = loadVis();
function saveVis(){ localStorage.setItem(LS_KEY, JSON.stringify(VIS)); }
function shownArms(){ return DATA.arms.filter(a => VIS.arms[a.id]); }
function syncColUI(){
  document.querySelectorAll("#colBar input[data-col]").forEach(inp=>{
    inp.checked = !!VIS[inp.dataset.col];
    inp.closest("label").classList.toggle("off", !inp.checked);
  });
  document.querySelectorAll("#armToggles input[data-arm]").forEach(inp=>{
    inp.checked = !!VIS.arms[inp.dataset.arm];
    inp.closest("label").classList.toggle("off", !inp.checked);
  });
  const active = Object.entries(PRESETS).find(([_k,p])=>{
    if (!!p.content!==!!VIS.content || !!p.style!==!!VIS.style || !!p.gt!==!!VIS.gt) return false;
    if (DATA.arms.some(a => !!VIS.arms[a.id] !== p.arms.includes(a.id))) return false;
    return true;
  });
  document.querySelectorAll("button.preset").forEach(b=>{
    b.classList.toggle("on", !!(active && active[0]===b.dataset.preset));
  });
}
function applyPreset(name){
  const p = PRESETS[name]; if (!p) return;
  VIS.content=!!p.content; VIS.style=!!p.style; VIS.gt=!!p.gt;
  DATA.arms.forEach(a => { VIS.arms[a.id] = p.arms.includes(a.id); });
  if (!shownArms().length && !VIS.content && !VIS.style && !VIS.gt) VIS.gt = true;
  saveVis(); syncColUI(); renderMetrics(); render();
}
function fill(){
  $("note").textContent = DATA.note;
  $("armToggles").innerHTML = DATA.arms.map(a =>
    `<label class="chip"><input type="checkbox" data-arm="${a.id}" checked/> ${a.label}</label>`
  ).join("");
  $("font").innerHTML = DATA.fonts.map(f => {
    const d = (DATA.font_delta.find(x=>x.font===f)||{});
    const tag = DATA.lead.includes(f) ? " · 重点" : "";
    const dl = d.dL1!=null ? ` · ΔL1 ${d.dL1>=0?"+":""}${d.dL1.toFixed(3)}` : "";
    return `<option value="${f}">${f}${tag}${dl}</option>`;
  }).join("");
  $("font").value = DATA.lead[0] || DATA.fonts[0];
  $("char").innerHTML = DATA.chars.map(c => `<option value="${c.ch}">${c.ch} · ${c.bucket}</option>`).join("");
  $("char").value = (DATA.focus_chars[0] || DATA.chars[0].ch);
  const buckets = [...new Set(DATA.chars.map(c=>c.bucket))];
  $("bucket").innerHTML = `<option value="">全部</option>` + buckets.map(b=>`<option value="${b}">${b}</option>`).join("");
  syncColUI();
}
function charsFiltered(){
  const b=$("bucket").value, focus=$("focus").checked;
  return DATA.chars.filter(c => (!b||c.bucket===b) && (!focus||DATA.focus_chars.includes(c.ch)));
}
function fontsFiltered(){
  return $("leadOnly").checked ? DATA.fonts.filter(f=>DATA.lead.includes(f)) : DATA.fonts;
}
function renderMetrics(){
  const arms = shownArms();
  let h = `<table><thead><tr><th>方法</th><th>n752</th><th>L1</th><th>SSIM</th><th>LPIPS</th><th>n704</th><th>L1<sub>704</sub></th></tr></thead><tbody>`;
  (arms.length?arms:DATA.arms).forEach(a=>{
    const m=DATA.metrics[a.mid]||{};
    h += `<tr><td>${a.label}</td><td>${m.n752||0}</td><td>${fmt(m.L1_752)}</td><td>${fmt(m.SSIM_752)}</td><td>${fmt(m.LPIPS_752)}</td><td>${m.n704||0}</td><td>${fmt(m.L1_704)}</td></tr>`;
  });
  h += `</tbody></table>`;
  $("metrics").innerHTML = h;
  const ranked = [...DATA.font_delta].sort((a,b)=>a.dL1-b.dL1);
  let f = `<table><thead><tr><th>字体</th><th>ΔL1 VEC−F2</th><th>VEC胜率</th></tr></thead><tbody>`;
  ranked.forEach(r=>{
    const cls = r.dL1<0 ? "win" : "lose";
    f += `<tr><td>${r.font}</td><td class="${cls}">${r.dL1>=0?"+":""}${r.dL1.toFixed(4)}</td><td>${(100*r.vec_win).toFixed(0)}%</td></tr>`;
  });
  f += `</tbody></table><div class="meta" style="margin-top:4px">负 Δ = VEC 像素更近 GT（常是更规整，不是更花体）</div>`;
  $("fontrank").innerHTML = f;
}
function cell(mid, stem, ch, cp, cls){
  const src = path(DATA.paths.pred, {mid, stem, cp});
  return `<td class="${cls}" data-stem="${stem}" data-ch="${ch}" data-cp="${cp}">${img(src)}</td>`;
}
function refCells(stem, cp){
  let s = "";
  if (VIS.content) s += `<td class="tone-gt">${img(path(DATA.paths.content,{cp}))}</td>`;
  if (VIS.style) s += `<td class="tone-gt">${img(path(DATA.paths.style,{stem}))}</td>`;
  if (VIS.gt) s += `<td class="tone-gt">${img(path(DATA.paths.gt,{stem,cp}))}</td>`;
  return s;
}
function render(){
  document.documentElement.style.setProperty("--gs", $("size").value+"px");
  const view=$("view").value;
  const arms = shownArms();
  let top = `<tr><th class="stub">${view==="char"?"字":"字体"}</th>`;
  if (VIS.content) top += `<th class="tone-gt">Content</th>`;
  if (VIS.style) top += `<th class="tone-gt">Style永</th>`;
  if (VIS.gt) top += `<th class="tone-gt">GT</th>`;
  arms.forEach(a => top += `<th class="${tone[a.tone]}">${a.label}</th>`);
  top += `</tr>`;
  let body="";
  if (view==="char"){
    const stem=$("font").value;
    charsFiltered().forEach(c=>{
      body += `<tr><td class="stub chr">${c.ch}<div class="meta">${c.bucket}</div></td>${refCells(stem,c.cp)}`;
      arms.forEach(a => { body += cell(a.mid, stem, c.ch, c.cp, tone[a.tone]); });
      body += `</tr>`;
    });
  } else {
    const ch=$("char").value;
    const rec = DATA.chars.find(c=>c.ch===ch) || DATA.chars[0];
    fontsFiltered().forEach(stem=>{
      body += `<tr><td class="stub"><code>${stem}</code></td>${refCells(stem,rec.cp)}`;
      arms.forEach(a => { body += cell(a.mid, stem, rec.ch, rec.cp, tone[a.tone]); });
      body += `</tr>`;
    });
  }
  $("grid").innerHTML = `<thead>${top}</thead><tbody>${body}</tbody>`;
}
function zoom(td){
  const stem=td.dataset.stem, ch=td.dataset.ch, cp=td.dataset.cp;
  $("zoomCap").textContent = `${stem} · ${ch}`;
  let html = "";
  if (VIS.content) html += `<figure><figcaption>Content</figcaption>${img(path(DATA.paths.content,{cp}))}</figure>`;
  if (VIS.style) html += `<figure><figcaption>Style永</figcaption>${img(path(DATA.paths.style,{stem}))}</figure>`;
  if (VIS.gt) html += `<figure><figcaption>GT</figcaption>${img(path(DATA.paths.gt,{stem,cp}))}</figure>`;
  shownArms().forEach(a=>{
    html += `<figure><figcaption>${a.label}</figcaption>${img(path(DATA.paths.pred,{mid:a.mid,stem,cp}))}</figure>`;
  });
  $("zoomRow").innerHTML = html || `<div class="meta">未勾选任何列</div>`;
  $("zoom").classList.add("on");
}
function bind(){
  ["view","font","char","bucket","focus","leadOnly","size"].forEach(id=>{
    $(id).addEventListener("change", render);
  });
  $("size").addEventListener("input", render);
  $("view").addEventListener("change", ()=>{
    $("font").disabled = $("view").value!=="char";
    $("char").disabled = $("view").value!=="font";
  });
  $("colBar").addEventListener("change", e=>{
    const t=e.target;
    if (t.dataset.col){
      VIS[t.dataset.col] = t.checked;
    } else if (t.dataset.arm){
      VIS.arms[t.dataset.arm] = t.checked;
      if (!shownArms().length){ VIS.arms[t.dataset.arm]=true; t.checked=true; }
    } else return;
    saveVis(); syncColUI(); renderMetrics(); render();
  });
  $("colBar").addEventListener("click", e=>{
    const b=e.target.closest("button.preset");
    if (b) applyPreset(b.dataset.preset);
  });
  $("grid").addEventListener("click", e=>{
    const td=e.target.closest("td[data-stem]");
    if (td) zoom(td);
  });
  $("zoom").addEventListener("click", ()=>$("zoom").classList.remove("on"));
  document.addEventListener("keydown", e=>{ if(e.key==="Escape") $("zoom").classList.remove("on"); });
}
fill(); renderMetrics();
$("font").disabled=false; $("char").disabled=true;
bind(); render();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
