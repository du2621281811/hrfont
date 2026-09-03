#!/usr/bin/env python3
"""Build an offline interactive human-review page for paired Stage A outputs."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image

from hrfont_stagea_mvp_gallery import (
    error_tile,
    glyph_path,
    gt_path,
    prediction_path,
    tile,
)


def save_tile(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((128, 128), Image.Resampling.NEAREST).save(path, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-metrics", type=Path, required=True)
    parser.add_argument("--delta-metrics", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    control = json.loads(args.control_metrics.read_text(encoding="utf-8"))
    delta = json.loads(args.delta_metrics.read_text(encoding="utf-8"))
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    c_rows = {(row["font"], row["char"]): row["l1"] for row in control["rows"]}
    d_rows = {(row["font"], row["char"]): row["l1"] for row in delta["rows"]}

    assets = args.output_dir / "assets"
    records = []
    for index, (face, char) in enumerate(sorted(set(c_rows) & set(d_rows))):
        control_path = prediction_path(args.control_dir, face, char)
        delta_path = prediction_path(args.delta_dir, face, char)
        if not control_path.exists() or not delta_path.exists():
            continue
        prefix = f"{index:03d}_{face}_u{ord(char):04X}"
        source_images = {
            "content": tile(glyph_path("val", "ContentImage", "", char)),
            "style": tile(glyph_path("val", "StyleImage", face, "永")),
            "gt": tile(gt_path(face, char)),
            "control": tile(control_path),
            "delta": tile(delta_path),
        }
        source_images["error_control"] = error_tile(
            source_images["control"], source_images["gt"]
        )
        source_images["error_delta"] = error_tile(
            source_images["delta"], source_images["gt"]
        )
        image_paths = {}
        for label, image in source_images.items():
            filename = f"{prefix}_{label}.png"
            save_tile(image, assets / filename)
            image_paths[label] = f"assets/{filename}"
        difference = d_rows[(face, char)] - c_rows[(face, char)]
        records.append(
            {
                "key": f"{face}|u{ord(char):04X}",
                "font": face,
                "char": char,
                "code": f"U+{ord(char):04X}",
                "control_l1": c_rows[(face, char)],
                "delta_l1": d_rows[(face, char)],
                "difference": difference,
                "outcome": "improved" if difference < 0 else "regressed",
                "images": image_paths,
            }
        )

    fonts = sorted({record["font"] for record in records})
    full_by_font = [
        {"font": face, **values} for face, values in comparison["by_font"].items()
    ]
    data_json = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    font_json = json.dumps(full_by_font, ensure_ascii=False).replace("</", "<\\/")

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage A MVP 人眼分析</title>
<style>
:root{{--bg:#f4f5f7;--panel:#fff;--text:#17191d;--muted:#646b76;--line:#d9dde3;--good:#147d45;--bad:#b42318;--accent:#2952cc}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}
header{{background:var(--panel);border-bottom:1px solid var(--line);padding:20px 28px}}
h1{{font-size:23px;margin:0 0 5px}} h2{{font-size:17px;margin:0 0 12px}} .muted{{color:var(--muted)}}
.summary{{display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:10px;margin-top:16px}}
.metric{{border-left:3px solid var(--accent);padding:7px 10px;background:var(--bg)}} .metric b{{display:block;font-size:20px}}
.notice{{margin:14px 0;padding:10px 12px;border:1px solid var(--line);background:#fff8e6}}
main{{padding:18px 28px 60px}} .font-summary{{background:var(--panel);border:1px solid var(--line);padding:15px;margin-bottom:16px}}
.font-row{{display:grid;grid-template-columns:170px 1fr 80px;gap:10px;align-items:center;margin:6px 0}}
.bar{{height:9px;background:#e9ebef;position:relative}} .bar i{{position:absolute;height:9px;left:50%}} .bar:after{{content:"";position:absolute;left:50%;top:-3px;height:15px;border-left:1px solid #8b919a}}
.controls{{position:sticky;top:0;z-index:5;background:var(--panel);border:1px solid var(--line);padding:12px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
select,input,button{{font:inherit;border:1px solid #b8bec7;background:white;padding:7px 9px;border-radius:3px}} button{{cursor:pointer}} button.primary{{background:var(--accent);color:white;border-color:var(--accent)}}
#grid{{display:grid;gap:12px;margin-top:14px}} .case{{background:var(--panel);border:1px solid var(--line);padding:13px}}
.case-head{{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px}} .delta.good{{color:var(--good)}} .delta.bad{{color:var(--bad)}}
.images{{display:grid;grid-template-columns:repeat(7,minmax(92px,1fr));gap:8px}} figure{{margin:0}} figure img{{width:100%;max-width:128px;image-rendering:auto;border:1px solid var(--line);background:white}} figcaption{{font-size:12px;color:var(--muted);margin-bottom:4px}}
.review{{display:flex;gap:8px;margin-top:10px}} .review input{{flex:1}} .count{{margin-left:auto;color:var(--muted)}}
@media(max-width:900px){{.summary{{grid-template-columns:1fr 1fr}}.images{{grid-template-columns:repeat(4,1fr)}}.font-row{{grid-template-columns:130px 1fr 65px}}}}
</style>
</head>
<body>
<header>
  <h1>Stage A MVP · 扩展人眼分析页</h1>
  <div class="muted">168组配对难例，严格相同采样seed；误差图为绝对像素误差×3。页面完全离线。</div>
  <div class="summary">
    <div class="metric"><span>全量 Control L1</span><b>{comparison["control_L1"]:.5f}</b></div>
    <div class="metric"><span>全量 Delta L1</span><b>{comparison["delta_L1"]:.5f}</b></div>
    <div class="metric"><span>全量改善</span><b>{comparison["control_minus_delta"]:.5f}</b></div>
    <div class="metric"><span>Delta胜率</span><b>{comparison["delta_win_rate"]:.1%}</b></div>
  </div>
  <div class="notice">范围说明：这是10k最终模型的扩展定性集，不是2.5k/5k/7.5k中间checkpoint；这些中间权重此前被覆盖，无法补造。</div>
</header>
<main>
  <section class="font-summary"><h2>全量976对的字体差异</h2><div id="fontBars"></div></section>
  <div class="controls">
    <select id="font"><option value="">全部字体</option>{''.join(f'<option>{html.escape(face)}</option>' for face in fonts)}</select>
    <select id="outcome"><option value="">全部结果</option><option value="improved">Delta改善</option><option value="regressed">Delta退化</option></select>
    <select id="sort"><option value="regressed">退化最大优先</option><option value="improved">改善最大优先</option><option value="absolute">差异最大优先</option><option value="font">字体/字符顺序</option></select>
    <input id="query" placeholder="字符或U+编码">
    <button id="export" class="primary">导出人工标注JSON</button>
    <span id="count" class="count"></span>
  </div>
  <div id="grid"></div>
</main>
<script>
const cases={data_json};
const fontStats={font_json};
const labels={{content:"Content",style:"Style：永",gt:"Ground truth",control:"Control",delta:"Delta",error_control:"|Err Control| ×3",error_delta:"|Err Delta| ×3"}};
const saved=JSON.parse(localStorage.getItem("stageA-human-review")||"{{}}");
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
function drawFontBars(){{
 const max=Math.max(...fontStats.map(x=>Math.abs(x.delta_minus_control)));
 document.getElementById("fontBars").innerHTML=fontStats.map(x=>{{
  const v=x.delta_minus_control,w=Math.abs(v)/max*50,left=v<0?50-w:50;
  return `<div class="font-row"><span>${{esc(x.font)}}</span><div class="bar"><i style="left:${{left}}%;width:${{w}}%;background:${{v<0?"var(--good)":"var(--bad)"}}"></i></div><b class="${{v<0?"delta good":"delta bad"}}">${{v>0?"+":""}}${{v.toFixed(4)}}</b></div>`;
 }}).join("");
}}
function reviewValue(key,field){{return saved[key]?.[field]||""}}
function render(){{
 const face=document.getElementById("font").value,outcome=document.getElementById("outcome").value,q=document.getElementById("query").value.trim().toLowerCase(),sort=document.getElementById("sort").value;
 let rows=cases.filter(x=>(!face||x.font===face)&&(!outcome||x.outcome===outcome)&&(!q||x.char.toLowerCase().includes(q)||x.code.toLowerCase().includes(q)));
 rows.sort((a,b)=>sort==="improved"?a.difference-b.difference:sort==="regressed"?b.difference-a.difference:sort==="absolute"?Math.abs(b.difference)-Math.abs(a.difference):(a.font+a.code).localeCompare(b.font+b.code));
 const marked=Object.values(saved).filter(x=>x.verdict||x.note).length;
 document.getElementById("count").textContent=`显示 ${{rows.length}} / ${{cases.length}} · 已标注 ${{marked}}`;
 document.getElementById("grid").innerHTML=rows.map(x=>`<article class="case" data-key="${{esc(x.key)}}">
  <div class="case-head"><div><b>${{esc(x.font)}}</b> · 字符 <b>${{esc(x.char)}}</b> · ${{x.code}}</div><div>Control ${{x.control_l1.toFixed(4)}} · Delta ${{x.delta_l1.toFixed(4)}} · <b class="delta ${{x.difference<0?"good":"bad"}}">Δ-C ${{x.difference>0?"+":""}}${{x.difference.toFixed(4)}}</b></div></div>
  <div class="images">${{Object.entries(x.images).map(([k,src])=>`<figure><figcaption>${{labels[k]}}</figcaption><img loading="lazy" src="${{src}}"></figure>`).join("")}}</div>
  <div class="review"><select class="verdict"><option value="">未标记</option>${{["Delta更好","Control更好","差不多","两者都差"].map(v=>`<option ${{reviewValue(x.key,"verdict")===v?"selected":""}}>${{v}}</option>`).join("")}}</select><input class="note" value="${{esc(reviewValue(x.key,"note"))}}" placeholder="记录结构、粗细、风格或可读性问题"></div>
 </article>`).join("");
 document.querySelectorAll(".case").forEach(card=>{{
  const key=card.dataset.key,save=()=>{{saved[key]={{verdict:card.querySelector(".verdict").value,note:card.querySelector(".note").value}};localStorage.setItem("stageA-human-review",JSON.stringify(saved));}};
  card.querySelector(".verdict").addEventListener("change",save);card.querySelector(".note").addEventListener("change",save);
 }});
}}
["font","outcome","sort","query"].forEach(id=>document.getElementById(id).addEventListener(id==="query"?"input":"change",render));
document.getElementById("export").addEventListener("click",()=>{{const blob=new Blob([JSON.stringify({{exported_at:new Date().toISOString(),annotations:saved}},null,2)],{{type:"application/json"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="stageA_human_annotations.json";a.click();URL.revokeObjectURL(a.href)}});
drawFontBars();render();
</script>
</body></html>"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {args.output_dir / 'index.html'} with {len(records)} paired cases")


if __name__ == "__main__":
    main()
