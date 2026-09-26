#!/usr/bin/env python3
"""Visualize Δ retrieval: CJK query vs neighbor CJK, then neighbor Latin vs GT.

CPU-only. Writes reports/f03_test16_strat/delta_retrieve/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/root/projects/hrfont")
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
OUT = ROOT / "reports/f03_test16_strat/delta_retrieve"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
TEST_STEMS = ROOT / "manifests/pipeline_v3_test_stems.txt"
F3_EVAL = ROOT / "reports/f03_test16_strat"
REF8 = list("永和书风骨韵天地")
LATIN = list("08AGQaeg")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def fonts() -> list[str]:
    return [l.strip() for l in TEST_STEMS.read_text().splitlines() if l.strip()]


def save96(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        Image.new("RGB", (96, 96), (235, 235, 235)).save(dst)
        return False
    Image.open(src).convert("RGB").resize((96, 96)).save(dst)
    return True


def glyph(split: str, stem: str, ch: str, kind: str) -> Path:
    folder = "StyleImage" if kind == "style" else "TargetImage"
    return DATA / split / folder / stem / f"{stem}+{cp_of(ch)}.png"


def pred(mid: str, stem: str, ch: str) -> Path:
    return F3_EVAL / "preds" / mid / "test" / stem / f"test__{stem}__{cp_of(ch)}__s3407.png"


def main() -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"))
    os.chdir(ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser")
    import torch
    import train as T
    from scripts.hrfont_delta_v2 import DeltaConfig, compute_alpha
    from scripts.hrfont_feature_cache import EsCache

    train_fonts = sorted(json.loads(SPLIT.read_text())["stems"]["train"])
    es = EsCache(ROOT / "artifacts/f0/es_spatial_f0")
    library = T._LibraryEs(es, train_fonts, T._style_chars_from_cache(es))
    cfg = DeltaConfig(tau=0.07, eps_alpha=0.01, k_max=10, k_top=10, mode="topk", rng_seed=3407)
    device = torch.device("cpu")
    ref_cp = [cp_of(c) for c in REF8]

    payload = {
        "library_n": len(train_fonts),
        "ref8": REF8,
        "latin": LATIN,
        "k": 10,
        "note": "检索用汉字 Es；取回的是这些训练字体的同一个西文字。",
        "fonts": {},
    }
    for i, query in enumerate(fonts()):
        print(f"[{i+1}/16] {query}", flush=True)
        samples = {
            "split": ["test"],
            "font_stem": [query],
            "char_cp": ["u0051"],
            "ref_chars": [ref_cp],
        }
        _, queries, *_ = T._style_conditions(es, samples, device)
        prototypes = library.prototypes(ref_cp, device)
        exclude = library.font_index.get(query)
        idx, weights, meta = compute_alpha(queries[0].to(device), prototypes, exclude, cfg)
        neighbors = [
            {"font": train_fonts[j], "weight": round(float(w), 4)}
            for j, w in zip(idx, weights.tolist())
        ]
        qdir = OUT / "img" / query
        for ch in REF8:
            save96(glyph("test", query, ch, "style"), qdir / f"q_cjk_{cp_of(ch)}.png")
        for ch in LATIN:
            save96(glyph("test", query, ch, "target"), qdir / f"q_gt_{cp_of(ch)}.png")
            for mid, tag in (("F0_100k", "f0"), ("F3_80k", "f3")):
                save96(pred(mid, query, ch), qdir / f"q_{tag}_{cp_of(ch)}.png")
        for n, nb in enumerate(neighbors, 1):
            stem = nb["font"]
            ndir = qdir / f"n{n:02d}"
            for ch in REF8:
                save96(glyph("train", stem, ch, "style"), ndir / f"cjk_{cp_of(ch)}.png")
            for ch in LATIN:
                save96(glyph("train", stem, ch, "target"), ndir / f"lat_{cp_of(ch)}.png")
        payload["fonts"][query] = {
            "exclude": exclude,
            "max_cosine": round(float(meta["max_cosine"]), 4),
            "entropy": round(float(meta["entropy"]), 3),
            "neighbors": neighbors,
        }

    (OUT / "data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_html(payload)
    print("wrote", OUT / "index.html")
    return 0


def write_html(payload: dict) -> None:
    html = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Δ 检索诊断</title>
<style>
:root{--ink:#1a1a1a;--muted:#5c6570;--line:#cfd6de;--accent:#1f4a6f;--q:#e8eef4;--out:#f3eee6}
*{box-sizing:border-box}
html,body{margin:0;background:#fff;color:var(--ink);font:12px/1.35 system-ui,"Noto Sans SC",sans-serif}
.bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:6px 8px;border-bottom:1px solid var(--line)}
h1{margin:0;font-size:14px;font-weight:650;white-space:nowrap}
select{padding:3px 6px;border:1px solid var(--line);font-size:12px}
.meta{color:var(--muted)}
.hint{padding:4px 8px;color:var(--muted);font-size:11px}
table{border-collapse:collapse;width:max-content}
th,td{border:1px solid var(--line);padding:0;text-align:center;vertical-align:middle;font-size:10px;color:var(--muted);background:#fff}
th.grp{background:#eef2f6;color:var(--ink);font-weight:650;letter-spacing:.04em}
th.ch{font:700 12px/1.2 ui-serif,serif;color:var(--ink);padding:2px 0}
td.name{text-align:left;padding:2px 6px;font:11px/1.25 ui-monospace,monospace;color:var(--ink);width:118px;max-width:118px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
img{width:44px;height:44px;image-rendering:pixelated;display:block;background:#fff}
tr.query td{background:var(--q)}
tr.out td{background:var(--out)}
.wt{color:var(--accent);font-variant-numeric:tabular-nums}
a{color:var(--accent);text-decoration:none}
.split{border-left:2px solid var(--accent) !important}
</style></head><body>
<div class="bar">
  <h1>Δ 检索：左汉字依据 · 右取回的西文</h1>
  <select id="font"></select>
  <span class="meta" id="stats"></span>
  <span class="meta"><a href="../">评测</a> · <a href="../timeline.html">过程</a></span>
</div>
<div class="hint" id="hint"></div>
<div id="board"></div>
<script>
const DATA = __PAYLOAD__;
const fontSel = document.getElementById('font');
Object.keys(DATA.fonts).forEach(f => {
  const o=document.createElement('option'); o.value=f; o.textContent=f; fontSel.appendChild(o);
});
fontSel.value = DATA.fonts['FZHuoYYJW-T'] ? 'FZHuoYYJW-T' : Object.keys(DATA.fonts)[0];
function cp(ch){ return 'u'+ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0'); }
function im(src){ return '<img src="'+src+'"/>'; }
function render(){
  const q=fontSel.value, rec=DATA.fonts[q], nbs=rec.neighbors;
  document.getElementById('stats').textContent = 'cos '+rec.max_cosine+' · H '+rec.entropy+' · 库'+DATA.library_n;
  document.getElementById('hint').textContent =
    '左 8 字只说明汉字风格像；右 8 字才是灌进 RSI 的拉丁。汉字细、西文方 → 变方来自邻居拉丁。F0 不检索，F3 用这 10 个西文。';
  const n = (i)=>'n'+String(i+1).padStart(2,'0');
  let h='<table><thead>';
  h+='<tr><th></th><th class="grp" colspan="'+DATA.ref8.length+'">检索用汉字</th><th class="grp split" colspan="'+DATA.latin.length+'">取回的西文 → RSI</th></tr><tr><th></th>';
  DATA.ref8.forEach(c=> h+='<th class="ch">'+c+'</th>');
  DATA.latin.forEach((c,i)=> h+='<th class="ch'+(i===0?' split':'')+'">'+c+'</th>');
  h+='</tr></thead><tbody>';
  h+='<tr class="query"><td class="name" title="'+q+'"><b>查询 GT</b> '+q+'</td>';
  DATA.ref8.forEach(c=> h+='<td>'+im('img/'+q+'/q_cjk_'+cp(c)+'.png')+'</td>');
  DATA.latin.forEach((c,i)=> h+='<td'+(i===0?' class="split"':'')+'>'+im('img/'+q+'/q_gt_'+cp(c)+'.png')+'</td>');
  h+='</tr>';
  nbs.forEach((nb,i)=>{
    h+='<tr><td class="name" title="'+nb.font+'">#'+(i+1)+' <span class="wt">'+nb.weight.toFixed(3)+'</span> '+nb.font.replace(/^FZ/,'')+'</td>';
    DATA.ref8.forEach(c=> h+='<td>'+im('img/'+q+'/'+n(i)+'/cjk_'+cp(c)+'.png')+'</td>');
    DATA.latin.forEach((c,j)=> h+='<td'+(j===0?' class="split"':'')+'>'+im('img/'+q+'/'+n(i)+'/lat_'+cp(c)+'.png')+'</td>');
    h+='</tr>';
  });
  [['F0@100k 不检索','f0'],['F3@80k 用上表西文','f3']].forEach(([lab,tag])=>{
    h+='<tr class="out"><td class="name">'+lab+'</td>';
    DATA.ref8.forEach(()=> h+='<td></td>');
    DATA.latin.forEach((c,j)=> h+='<td'+(j===0?' class="split"':'')+'>'+im('img/'+q+'/q_'+tag+'_'+cp(c)+'.png')+'</td>');
    h+='</tr>';
  });
  document.getElementById('board').innerHTML=h+'</tbody></table>';
}
fontSel.onchange=render; render();
</script>
</body></html>
"""
    html = html.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    (OUT / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
