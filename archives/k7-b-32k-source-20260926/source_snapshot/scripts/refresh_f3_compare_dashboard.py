#!/usr/bin/env python3
"""Rebuild the F3 dashboard as Official / F0 / F3 comparison.

Assumes F3 preds already exist under reports/f3_ckpt_dashboard/preds/step*.
Samples Official and F0@100k with the same image DPM protocol (content + 1 style,
DPM++ 20, CFG 7.5, seed 3407) on GPU --device. Does not touch F3's training GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/f3_ckpt_dashboard"
PY = "/root/miniforge3/envs/boogu/bin/python"
PREVIEW = ROOT / "scripts/preview_ckpt_grid.py"
OFFICIAL_VARIANT = ROOT / "code/official/FontDiffuser"
OFFICIAL_CKPT = ROOT / "code/official/FontDiffuser/ckpt"
F0_VARIANT = ROOT / "code/variants/cn2west_f0_rsifree/FontDiffuser"
F0_CKPT = ROOT / "runs/F0-RSIFREE-FT-A-S3407/global_step_100000"
TEST_STEMS = ROOT / "manifests/pipeline_v3_test_stems.txt"
CHARS = list("AaG0e")
F3_COLS = [
    ("step5000", "F3@5k"),
    ("step10000", "F3@10k val-best"),
    ("step20000", "F3@20k"),
    ("step23000_last", "F3 last~24k"),
]


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def load_96(path: Path) -> Image.Image:
    if path.is_file():
        return Image.open(path).convert("RGB").resize((96, 96))
    return Image.new("RGB", (96, 96), (230, 230, 230))


def sample_image_dpm(label: str, variant: Path, ckpt: Path, dest: Path, device: str, n_fonts: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(PREVIEW),
        "--variant", str(variant),
        "--ckpt", str(ckpt),
        "--label", label,
        "--out", str(dest),
        "--split", "test",
        "--fonts", str(n_fonts),
        "--chars", "".join(CHARS),
        "--seed", "3407",
        "--device", device,
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def combined_sheet(fonts: list[str]) -> Image.Image:
    cols = ["GT", "Official", "F0@100k"] + [lab for _, lab in F3_COLS]
    srcs = {
        "GT": lambda font, ch: OUT / "refs/gt" / f"{font}+{cp_of(ch)}.png",
        "Official": lambda font, ch: OUT / "preds/official" / f"{font}+{cp_of(ch)}.png",
        "F0@100k": lambda font, ch: OUT / "preds/f0_100k" / f"{font}+{cp_of(ch)}.png",
    }
    for tag, lab in F3_COLS:
        srcs[lab] = lambda font, ch, t=tag: OUT / "preds" / t / f"{font}+{cp_of(ch)}.png"
    cell, pad, head, left = 96, 5, 28, 150
    n_row = len(fonts) * len(CHARS)
    w = left + len(cols) * (cell + pad) + pad
    h = head + n_row * (cell + pad) + pad + 18
    sheet = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), "Official vs F0@100k vs F3  test  DPM++20 CFG7.5 seed3407", fill="black")
    for j, name in enumerate(cols):
        d.text((left + j * (cell + pad) + 4, 10), name[:12], fill="black")
    i = 0
    for font in fonts:
        for ch in CHARS:
            y = head + i * (cell + pad)
            d.text((6, y + 40), f"{font[:14]} {ch}", fill="black")
            for j, name in enumerate(cols):
                im = load_96(srcs[name](font, ch))
                sheet.paste(im, (left + j * (cell + pad), y))
            i += 1
    return sheet


def write_html(fonts: list[str], meta: dict) -> None:
    f3_js = json.dumps([{"tag": t, "label": lab} for t, lab in F3_COLS], ensure_ascii=False)
    font_opts = "".join(f'<option value="{f}">{f}</option>' for f in fonts)
    char_opts = "".join(f'<option value="{c}">{c}</option>' for c in CHARS)
    tables = []
    for font in fonts:
        rows = []
        for ch in CHARS:
            c = cp_of(ch)
            tds = [
                f'<td class="ch">{ch}</td>',
                f'<td><img src="refs/content/{c}.png" alt="content"/></td>',
                f'<td><img src="refs/gt/{font}+{c}.png" alt="gt"/></td>',
                f'<td><img src="preds/official/{font}+{c}.png" alt="official"/></td>',
                f'<td><img src="preds/f0_100k/{font}+{c}.png" alt="f0"/></td>',
            ]
            for tag, lab in F3_COLS:
                tds.append(f'<td><img src="preds/{tag}/{font}+{c}.png" alt="{lab}"/></td>')
            rows.append("<tr>" + "".join(tds) + "</tr>")
        tables.append(
            f'<h3>{font}</h3><div class="scroll"><table class="grid">'
            "<thead><tr><th>字</th><th>Content</th><th>GT</th><th>官方</th><th>F0@100k</th>"
            + "".join(f"<th>{lab}</th>" for _, lab in F3_COLS)
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>官方 / F0 / F3 对照看板</title>
<style>
:root{{--bg:#eef1f5;--card:#fff;--line:#d5dbe3;--muted:#5c6570;--ink:#1a1a1a}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}}
header{{background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}}
h1{{margin:0;font-size:1.25rem}} h3{{margin:16px 0 8px;font-size:1rem}}
.meta{{color:var(--muted);font-size:12px;margin-top:4px}}
.note{{background:#fff8e8;border:1px solid #e6d7a8;padding:8px 10px;margin:10px 0;border-radius:4px;font-size:12px}}
main{{max-width:1280px;margin:16px auto;padding:0 14px 48px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px;margin:14px 0}}
.scroll{{overflow-x:auto}}
table.grid{{border-collapse:collapse}}
table.grid th,table.grid td{{border:1px solid var(--line);padding:4px;text-align:center;vertical-align:bottom;font-size:11px;color:var(--muted)}}
table.grid td.ch{{font:700 16px/1.2 ui-serif,serif;color:var(--ink);width:28px}}
table.grid img{{width:96px;height:96px;image-rendering:pixelated;display:block;background:#fff}}
img.loss{{width:100%;height:auto}}
.refs,.timeline{{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end}}
.timeline-wrap{{overflow-x:auto;border:1px solid var(--line);background:#f7f9fb;padding:10px}}
figure{{margin:0;text-align:center}} figcaption{{font-size:11px;color:var(--muted);margin-top:3px;font-family:ui-monospace,monospace}}
img.g{{width:96px;height:96px;image-rendering:pixelated;border:1px solid var(--line);background:#fff;display:block}}
select{{padding:6px 8px;border:1px solid var(--line);background:#fff}}
</style></head><body>
<header>
  <h1>官方 · F0 · F3 对照</h1>
  <div class="meta">test 4 字体 × A a G 0 e · DPM++ 20 / CFG 7.5 / seed 3407 · {meta.get("generated_at","")}</div>
</header>
<main>
<div class="note">
<b>怎么比：</b>同一 Content、同一张 Style（优先「永」）、同一采样器。<br/>
<b>官方</b>：预训练 FontDiffuser，图像条件 + 官方 RSI（Ec(style) 做结构）。<br/>
<b>F0@100k</b>：RSI-free 在本数据集上训满，同样图像条件（无 RSI）。<br/>
<b>F3</b>：在 F0@100k 上接 identity-safe Δ+Support，推理走训练 val 条件（cache Δ + 同字体 Support×8，无 drop），<i>不是</i>再跑一遍图像 RSI。<br/>
看点：官方是否已经能抄风格；F0 相对官方是否更贴 GT；F3 相对 F0 是否把难字体（尤其 FZChuangHJW_DB）拉回来。
</div>
<section class="card">
  <h2>对照表（按字体）</h2>
  {"".join(tables)}
</section>
<section class="card">
  <h2>总 contact sheet</h2>
  <p class="meta">列：GT · 官方 · F0 · F3@5k · F3@10k（trainer val best）· F3@20k · F3 last</p>
  <img class="loss" src="compare_sheet.png" alt="compare sheet"/>
</section>
<section class="card">
  <h2>单字放大</h2>
  <p><label>字体 <select id="font">{font_opts}</select></label>
     <label>字 <select id="ch">{char_opts}</select></label></p>
  <div class="refs" id="refs"></div>
  <div class="timeline-wrap"><div class="timeline" id="timeline"></div></div>
</section>
</main>
<script>
const F3 = {f3_js};
function cp(ch){{return 'u'+ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0');}}
function fig(src, cap){{
  return `<figure><img class="g" src="${{src}}" onerror="this.style.opacity=0.15"/><figcaption>${{cap}}</figcaption></figure>`;
}}
function render(){{
  const font=document.getElementById('font').value;
  const ch=document.getElementById('ch').value;
  const c=cp(ch);
  document.getElementById('refs').innerHTML =
    fig(`refs/content/${{c}}.png`,'Content')+
    fig(`refs/style/${{font}}.png`,'Style')+
    fig(`refs/gt/${{font}}+${{c}}.png`,'GT');
  document.getElementById('timeline').innerHTML =
    fig(`preds/official/${{font}}+${{c}}.png`,'官方')+
    fig(`preds/f0_100k/${{font}}+${{c}}.png`,'F0@100k')+
    F3.map(t=>fig(`preds/${{t.tag}}/${{font}}+${{c}}.png`, t.label)).join('');
}}
document.getElementById('font').onchange=render;
document.getElementById('ch').onchange=render;
render();
</script></body></html>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fonts", type=int, default=4)
    ap.add_argument("--skip-sample", action="store_true")
    args = ap.parse_args()
    fonts = [l.strip() for l in TEST_STEMS.read_text().splitlines() if l.strip()][: args.fonts]
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_sample:
        sample_image_dpm("Official", OFFICIAL_VARIANT, OFFICIAL_CKPT, OUT / "preds/official", args.device, args.fonts)
        sample_image_dpm("F0@100k", F0_VARIANT, F0_CKPT, OUT / "preds/f0_100k", args.device, args.fonts)
    sheet = combined_sheet(fonts)
    sheet.save(OUT / "compare_sheet.png")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fonts": fonts,
        "chars": CHARS,
        "official_ckpt": str(OFFICIAL_CKPT),
        "f0_ckpt": str(F0_CKPT),
        "protocol": {
            "official_f0": "image DPM: ContentImage + StyleImage(永), dpmsolver++20 CFG7.5 seed3407",
            "f3": "cache Δ + Support×8, custom CFG, same sampler/seed",
        },
    }
    snap_path = OUT / "SNAPSHOT.json"
    snap = json.loads(snap_path.read_text()) if snap_path.is_file() else {}
    snap["compare"] = meta
    snap_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_html(fonts, meta)
    (OUT / "README.md").write_text(
        "# 官方 / F0 / F3 对照看板\n\n"
        "- http://127.0.0.1:8777/f3_ckpt_dashboard/\n"
        "- 官方与 F0：图像 DPM（Content + Style「永」）\n"
        "- F3：训练时 Δ + Support 条件\n",
        encoding="utf-8",
    )
    print("wrote", OUT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
