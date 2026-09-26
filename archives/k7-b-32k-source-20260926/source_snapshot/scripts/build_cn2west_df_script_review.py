#!/usr/bin/env python3
"""Build D vs F per-script probe review page for manual check."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "data/cn2west_v2_abc_review/proto_DF_script_review"
THUMBS = OUT / "thumbs"
FONTS_JSON = ROOT / "data/cn2west_v2_abc_review/fonts.json"
CHARSET = json.loads((ROOT / "manifests/charset_cn2west_v2_planned.json").read_text(encoding="utf-8"))

FONT_DIRS = [
    Path("/root/projects/font_crosslingual/data/suiti_fonts_probe/随体字体集"),
    Path("/root/projects/font_crosslingual/data/lffont_cn2latin/ttf_train"),
    Path("/root/data/font_50"),
]

PROBES = [
    ("style_han", "Style 汉字", "永和书风骨韵天地繁慕慧健"),
    ("ascii_letters", "ASCII 字母", "AaBbGgQqWwMm"),
    ("ascii_digits", "ASCII 数字", "0123456789"),
    ("latin_ext_letters", "拉丁扩展", "àéêüāēǎǐǒǔǖǘ"),
    ("hiragana", "平假名", "あいうえおかがきぎぱぽ"),
    ("katakana", "片假名", "アイウエオカガキギジヴ"),
    ("bopomofo", "注音", "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐ"),
]

CELL = 64


def resolve_ttf(stem: str) -> Path | None:
    for d in FONT_DIRS:
        if not d.is_dir():
            continue
        for ext in (".TTF", ".ttf", ".OTF", ".otf", ".TTC", ".ttc"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
        # case-insensitive stem match
        for p in d.iterdir():
            if p.suffix.lower() in {".ttf", ".otf", ".ttc"} and p.stem.lower() == stem.lower():
                return p
    return None


def render_D(ttf: Path, ch: str, size: int = 128) -> tuple[Image.Image, int]:
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    lo, hi, best_fs, best = 8, size, None, None
    for _ in range(14):
        fs = (lo + hi) // 2
        f = ImageFont.truetype(str(ttf), fs)
        bbox = draw.textbbox((0, 0), ch, font=f)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= size - 8 and h <= size - 8:
            best, best_fs = (f, bbox), fs
            lo = fs + 1
        else:
            hi = fs - 1
    if best is None:
        best_fs = max(10, size // 2)
        f = ImageFont.truetype(str(ttf), best_fs)
        bbox = draw.textbbox((0, 0), ch, font=f)
    else:
        f, bbox = best
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - w) // 2 - bbox[0], (size - h) // 2 - bbox[1]), ch, font=f, fill=0)
    rgb = Image.merge("RGB", (img, img, img)).resize((CELL, CELL), Image.BILINEAR)
    return rgb, int(best_fs)


def render_F(ttf: Path, ch: str, canvas: int = 96) -> tuple[Image.Image, int]:
    target = canvas * 0.08
    best_im, best_score, best_fs = None, 1e9, None
    for fs in range(int(canvas * 0.50), int(canvas * 0.92), 2):
        im = Image.new("RGB", (canvas, canvas), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(str(ttf), fs)
        bbox = dr.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            continue
        x = (canvas - w) // 2 - bbox[0]
        y = (canvas - h) // 2 - bbox[1]
        dr.text((x, y), ch, fill=(0, 0, 0), font=font)
        a = np.asarray(im.convert("L"))
        ys, xs = np.where(a < 250)
        if len(ys) == 0:
            continue
        m = min(int(xs.min()), int(ys.min()), canvas - 1 - int(xs.max()), canvas - 1 - int(ys.max()))
        score = abs(m - target)
        if score < best_score:
            best_im, best_score, best_fs = im, score, fs
    if best_im is None:
        best_im = Image.new("RGB", (canvas, canvas), (230, 230, 230))
        best_fs = 0
    return best_im.resize((CELL, CELL), Image.NEAREST), int(best_fs or 0)


def make_font_sheet(ttf: Path, stem: str, name: str, sev: str) -> Path:
    try:
        ui = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14)
        ui_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        ui = ui_s = ImageFont.load_default()

    sections = []
    for cat_id, label, sample in PROBES:
        chars = list(sample)
        # D row + F row
        row_h = CELL + 18
        row_w = 36 + len(chars) * (CELL + 3)
        sec_h = 28 + row_h * 2 + 8
        sec = Image.new("RGB", (max(row_w, 400), sec_h), (255, 255, 255))
        dr = ImageDraw.Draw(sec)
        dr.text((4, 4), f"{label} · 示意 {len(chars)} 字", fill=(30, 30, 30), font=ui)
        for ri, (tag, renderer) in enumerate((("D", render_D), ("F", render_F))):
            y = 26 + ri * row_h
            dr.text((4, y + CELL // 2 - 6), tag, fill=(80, 80, 80), font=ui)
            x = 36
            for ch in chars:
                try:
                    im, fs = renderer(ttf, ch)
                except Exception:
                    im = Image.new("RGB", (CELL, CELL), (230, 230, 230))
                    fs = -1
                sec.paste(im, (x, y))
                dr.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=(210, 210, 210))
                dr.text((x + 2, y + CELL + 1), ch, fill=(40, 40, 40), font=ui_s)
                x += CELL + 3
        sections.append(sec)

    width = max(s.width for s in sections) + 16
    height = 48 + sum(s.height + 10 for s in sections)
    canvas = Image.new("RGB", (width, height), (240, 242, 245))
    dr = ImageDraw.Draw(canvas)
    dr.text((8, 8), f"{stem} · {name} · {sev} · 上D(逐字最大@128→96) 下F(墨迹边距≈8%@96)", fill=(20, 20, 20), font=ui)
    y = 40
    for sec in sections:
        canvas.paste(sec, (8, y))
        y += sec.height + 10
    out = THUMBS / f"{stem}.png"
    canvas.save(out)
    return out


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>D vs F · 各语种示意检查</title>
<style>
:root{--bg:#eef1f5;--panel:#fff;--line:#d5dbe3;--muted:#5c6570;--drop:#8b1e1e;--review:#8a5a00;--ok:#2f5d3a;--accent:#1f4a6f}
*{box-sizing:border-box} html,body{height:100%;margin:0}
body{font:13px/1.4 system-ui,sans-serif;background:var(--bg);color:#1a1a1a;display:flex;flex-direction:column}
header{flex:0 0 auto;background:#fff;border-bottom:1px solid var(--line);padding:10px 14px}
h1{margin:0;font-size:1.05rem} .meta{color:var(--muted);font-size:12px;margin-top:2px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
input,select,button,.chip{padding:5px 9px;border:1px solid var(--line);border-radius:4px;font:inherit;background:#fff;cursor:pointer}
.chip.on{background:#e8eef5;border-color:#9db4cc}
.layout{flex:1;min-height:0;display:grid;grid-template-columns:300px 1fr}
aside{background:var(--panel);border-right:1px solid var(--line);overflow:auto}
#fontList{list-style:none;margin:0;padding:0}
#fontList li{padding:8px 10px;border-bottom:1px solid #eef1f5;cursor:pointer}
#fontList li:hover{background:#f5f7fa} #fontList li.active{background:#e8eef5}
.stem{font-weight:600;font-family:ui-monospace,monospace;font-size:12px}
.sub{color:var(--muted);font-size:11px;margin-top:2px}
.badge{display:inline-block;font-size:10px;padding:0 6px;border-radius:3px;margin-right:4px}
.badge.drop{background:#fde8e8;color:var(--drop)} .badge.review{background:#fff3d6;color:var(--review)} .badge.ok{background:#e8f2ea;color:var(--ok)}
main{overflow:auto;padding:12px 16px 40px}
main img{max-width:100%;height:auto;border:1px solid var(--line);background:#fff;image-rendering:auto}
.miss{color:#a30;padding:20px}
</style>
</head>
<body>
<header>
  <h1>D vs F · 各语种示意检查</h1>
  <div class="meta">D=逐字逻辑框最大@128→显示96 · F=逐字墨迹边距≈8%@96 · 探针非全字 · 排序沿用 B 盘自动筛查 · J/K 换字体</div>
  <div class="toolbar">
    <input type="search" id="q" placeholder="搜 stem / 名" style="min-width:160px"/>
    <select id="split"><option value="all">全部 split</option><option value="train">train</option><option value="val">val</option><option value="test">test</option></select>
    <button type="button" class="chip on" data-sev="drop">建议丢</button>
    <button type="button" class="chip on" data-sev="review">再看</button>
    <button type="button" class="chip on" data-sev="ok">正常</button>
    <span id="progress"></span>
  </div>
</header>
<div class="layout">
  <aside><ul id="fontList"></ul></aside>
  <main id="main">从左侧选字体</main>
</div>
<script>
let DATA=null, filtered=[], cur=-1;
const sevOn=new Set(['drop','review','ok']);
const pct=v=>v==null?'—':Math.round(v*100)+'%';
const f2=v=>v==null?'—':Number(v).toFixed(2);

function applyFilter(){
  const q=document.getElementById('q').value.trim().toLowerCase();
  const split=document.getElementById('split').value;
  filtered=DATA.fonts.filter(f=>{
    if(!sevOn.has(f.severity)) return false;
    if(split!=='all' && f.split!==split) return false;
    if(q && !(f.stem.toLowerCase().includes(q)||(f.name||'').toLowerCase().includes(q))) return false;
    if(f.missing_ttf) return true; // still list
    return true;
  });
  renderList();
  document.getElementById('progress').textContent=`显示 ${filtered.length}/${DATA.fonts.length} · 已渲 ${DATA.n_rendered}`;
}
function renderList(){
  document.getElementById('fontList').innerHTML=filtered.map((f,i)=>`
    <li data-i="${i}" class="${cur===i?'active':''}">
      <div><span class="badge ${f.severity}">${f.severity}</span><span class="stem">${f.stem}</span>${f.missing_ttf?' ⚠无TTF':''}</div>
      <div class="sub">${f.split} · ${f.name||''} · 永${pct(f.yong_h)} · 色散${f2(f.disp)}</div>
    </li>`).join('');
  document.querySelectorAll('#fontList li').forEach(li=>li.onclick=()=>selectFont(+li.dataset.i));
}
function selectFont(i){
  if(i<0||i>=filtered.length) return;
  cur=i; const f=filtered[i];
  renderList();
  const main=document.getElementById('main');
  if(f.missing_ttf){ main.innerHTML=`<div class="miss">找不到字体文件：${f.stem}</div>`; return; }
  main.innerHTML=`<div style="margin-bottom:8px"><b>${f.stem}</b> ${f.name||''} · ${f.split} · ${f.severity}
    <div style="color:#5c6570;font-size:12px">${(f.reasons||[]).join(' · ')||'未打标'} · 永H(B盘)=${pct(f.yong_h)} · 色散=${f2(f.disp)}</div>
    <div style="margin-top:6px"><button id="btnPrev">← K</button> <button id="btnNext">J →</button></div>
  </div>
  <img src="thumbs/${f.stem}.png" alt="${f.stem}" loading="eager"/>`;
  document.getElementById('btnPrev').onclick=()=>selectFont(cur-1);
  document.getElementById('btnNext').onclick=()=>selectFont(cur+1);
  main.scrollTop=0;
}
document.querySelectorAll('.chip').forEach(ch=>ch.onclick=()=>{
  const s=ch.dataset.sev;
  if(sevOn.has(s)){sevOn.delete(s);ch.classList.remove('on');}else{sevOn.add(s);ch.classList.add('on');}
  cur=-1; applyFilter();
});
document.getElementById('q').oninput=()=>{cur=-1;applyFilter();};
document.getElementById('split').onchange=()=>{cur=-1;applyFilter();};
document.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
  if(e.key==='j'||e.key==='J'){e.preventDefault();selectFont(cur+1);}
  if(e.key==='k'||e.key==='K'){e.preventDefault();selectFont(cur-1);}
});
fetch('fonts.json').then(r=>r.json()).then(d=>{DATA=d;applyFilter();if(filtered.length)selectFont(0);});
</script>
</body>
</html>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    THUMBS.mkdir(parents=True, exist_ok=True)
    base = json.loads(FONTS_JSON.read_text(encoding="utf-8"))
    fonts = base["fonts"]
    # prioritize drop/review first for early viewing, but render all
    order = sorted(range(len(fonts)), key=lambda i: ({"drop": 0, "review": 1, "ok": 2}.get(fonts[i]["severity"], 9), fonts[i]["stem"]))

    rendered = 0
    missing = []
    out_fonts = []
    for idx, i in enumerate(order):
        f = fonts[i]
        stem = f["stem"]
        ttf = resolve_ttf(stem)
        rec = dict(f)
        if ttf is None:
            rec["missing_ttf"] = True
            missing.append(stem)
            out_fonts.append(rec)
            print(f"[{idx+1}/{len(order)}] MISS {stem}", flush=True)
            continue
        rec["missing_ttf"] = False
        rec["ttf"] = str(ttf)
        make_font_sheet(ttf, stem, f.get("name") or "", f.get("severity") or "")
        rendered += 1
        out_fonts.append(rec)
        if (idx + 1) % 20 == 0 or f["severity"] != "ok":
            print(f"[{idx+1}/{len(order)}] {f['severity']} {stem}", flush=True)

    # keep screen sort for UI (drop first)
    out_fonts.sort(key=lambda r: ({"drop": 0, "review": 1, "ok": 2}.get(r["severity"], 9), -(r.get("disp") or 0), r["stem"]))
    payload = {
        "title": "D vs F script probe review",
        "n_rendered": rendered,
        "n_missing_ttf": len(missing),
        "missing": missing,
        "probes": [{"id": a, "label": b, "sample": c} for a, b, c in PROBES],
        "note": "D=per-glyph max textbbox@128→BILINEAR96 display; F=per-glyph ink margin≈8%@96",
        "fonts": out_fonts,
    }
    (OUT / "fonts.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "index.html").write_text(HTML, encoding="utf-8")
    print(json.dumps({"out": str(OUT), "rendered": rendered, "missing": len(missing)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
