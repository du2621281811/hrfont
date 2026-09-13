#!/usr/bin/env python3
"""Bake one contact sheet per font: 2–3 probe glyphs per script.

Does not write the old A disk. Preview HTML is for :19003 only.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "data/fontdiffuser-p649-t295-s338-cn2west-v2a-r0"
REVIEW = ROOT / "data/p649_v2a_review"
LAYERS = ROOT / "data/p649_v2a_layers"
PREVIEW = REVIEW / "sheet_preview"
PREVIEW_COPY = LAYERS / "sheet_preview"
CARDS = REVIEW / "font_check_cards"
ZIP_NAME = "p649_font_check_649.zip"
NOTO = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

# 7 scripts × 3 chars. Chosen to show structure / dakuten / accents / ㄦ, not the full table.
SHEET_PROBES = [
    {"id": "style_han", "label": "汉字", "chars": list("永风骨")},
    {"id": "ascii_digits", "label": "数字", "chars": list("081")},
    {"id": "ascii_letters", "label": "拉丁", "chars": list("Aag")},
    {"id": "latin_ext_letters", "label": "扩拉", "chars": list("àēǚ")},
    {"id": "hiragana", "label": "平假", "chars": list("あがん")},
    {"id": "katakana", "label": "片假", "chars": list("アガン")},
    {"id": "bopomofo", "label": "注音", "chars": list("ㄅㄚㄦ")},
]


def glyph_folder(ch: str) -> str:
    return "StyleImage" if "\u4e00" <= ch <= "\u9fff" else "TargetImage"


def cp_name(ch: str) -> str:
    return "u" + hex(ord(ch))[2:].upper().zfill(4)


def glyph_path(split: str, stem: str, ch: str) -> Path:
    return OUT / split / glyph_folder(ch) / stem / f"{stem}+{cp_name(ch)}.png"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if NOTO.is_file():
        return ImageFont.truetype(str(NOTO), size=size, index=0)
    return ImageFont.load_default()


def cell(ch: str, split: str, stem: str, g: int) -> Image.Image:
    path = glyph_path(split, stem, ch)
    miss = Image.new("RGB", (g, g), (255, 236, 236))
    if not path.is_file():
        d = ImageDraw.Draw(miss)
        d.rectangle((0, 0, g - 1, g - 1), outline=(180, 40, 40))
        d.text((4, g // 2 - 6), "缺", fill=(180, 40, 40), font=load_font(max(12, g // 4)))
        return miss
    im = Image.open(path).convert("RGB")
    if im.size != (g, g):
        im = im.resize((g, g), Image.Resampling.NEAREST)
    return im


def _paste_group(img, draw, grp: dict, split: str, stem: str, x: int, y: int, g: int, font_l) -> None:
    gap = 4
    draw.text((x, y), grp["label"], fill=(70, 80, 90), font=font_l)
    gx = x
    gy = y + 18
    for ch in grp["chars"]:
        tile = cell(ch, split, stem, g)
        img.paste(tile, (gx, gy))
        draw.rectangle((gx, gy, gx + g - 1, gy + g - 1), outline=(220, 224, 228))
        gx += g + gap


def render_sheet(split: str, stem: str, *, g: int = 72) -> Image.Image:
    """Compact 2×4 card for sending one PNG per font. Not a wide strip."""
    gap = 4
    col_w = 3 * g + 2 * gap
    col_gap = 18
    row_h = 18 + g + 12
    pad = 14
    title_h = 28
    cols, rows_n = 2, 4
    w = pad + cols * col_w + (cols - 1) * col_gap + pad
    h = pad + title_h + rows_n * row_h + pad - 8
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w - 1, h - 1), outline=(210, 214, 218))
    font_t = load_font(16)
    font_l = load_font(13)
    draw.text((pad, pad), stem, fill=(20, 20, 20), font=font_t)
    groups = SHEET_PROBES
    for i, grp in enumerate(groups):
        r, c = divmod(i, cols)
        x = pad + c * (col_w + col_gap)
        y = pad + title_h + r * row_h
        _paste_group(img, draw, grp, split, stem, x, y, g, font_l)
    return img


def pick_preview_fonts(n: int = 5) -> list[dict]:
    rank = json.loads((REVIEW / "ink_ratio_rank.json").read_text(encoding="utf-8"))
    dec = (json.loads((REVIEW / "decisions.json").read_text(encoding="utf-8")).get("decisions") or {})
    fonts = rank["fonts"]
    picked: list[dict] = []
    want = [
        ("overlap260", "pass"),
        ("overlap260", "drop"),
        ("new389", "pass"),
        ("overlap260", "drop"),
        ("overlap260", "drop"),
    ]
    used = set()
    for subset, decision in want:
        for f in fonts:
            if f["stem"] in used:
                continue
            rec = dec.get(f["stem"]) or {}
            if rec.get("review_decision") != decision:
                continue
            if subset == "new389" and f.get("subset") == "overlap260":
                continue
            if subset == "overlap260" and f.get("subset") != "overlap260":
                continue
            picked.append({**f, "l1": decision})
            used.add(f["stem"])
            break
    for f in fonts:
        if len(picked) >= n:
            break
        if f["stem"] not in used:
            rec = dec.get(f["stem"]) or {}
            picked.append({**f, "l1": rec.get("review_decision") or ""})
            used.add(f["stem"])
    return picked[:n]


def write_preview_html(rows: list[dict], g: int) -> None:
    items = []
    for f in rows:
        stem = f["stem"]
        name = f"{stem}_g{g}.png"
        l1 = f.get("l1") or "—"
        items.append(
            f"""<article class="card">
  <header>L1 {l1} · {f.get('subset')} · {f.get('split')}</header>
  <img src="{name}" alt="{stem}"/>
</article>"""
        )
    probes = "　".join(f"{g['label']} {' '.join(g['chars'])}" for g in SHEET_PROBES)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>p649 语种拼图示例（内部）</title>
<style>
body{{font:14px/1.5 system-ui;margin:24px;max-width:1100px;background:#eef1f5;color:#1a1a1a}}
h1{{font-size:1.2rem;margin:0 0 8px}}
.note{{color:#5c6570;font-size:13px}}
.plan{{background:#fff;border:1px solid #d5dbe3;padding:12px 16px;margin:12px 0;border-radius:4px}}
.plan li{{margin:4px 0}}
.grid{{display:flex;flex-wrap:wrap;gap:14px}}
.card{{background:#fff;border:1px solid #d5dbe3;padding:8px}}
.card header{{font-size:12px;color:#3a4450;margin-bottom:6px}}
.card img{{image-rendering:pixelated;background:#fff;display:block;width:auto;height:auto}}
code{{font-family:ui-monospace,monospace}}
.warn{{background:#fff6e8;border:1px solid #e6c48a;padding:8px 10px;font-size:13px}}
</style></head>
<body>
<h1>字库检查图示例 · 一套一张卡片</h1>
<p class="warn">内部预览。导师首页没有入口。发给导师的是下面这种 PNG，不是网页。</p>
<div class="plan">
<p>每套一张约 500×400 的卡片：字体名写在图上，7 个语种各 3 个字，两列四行。微信/邮件直接发图即可。</p>
<p class="note">探针：{probes}</p>
</div>
<div class="grid">
{"".join(items)}
</div>
</body></html>
"""
    (PREVIEW / "index.html").write_text(html, encoding="utf-8")


def bake_cards(rows: list[dict], dest: Path, g: int) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(rows, 1):
        img = render_sheet(f["split"], f["stem"], g=g)
        img.save(dest / f"{f['stem']}.png", optimize=True)
        if i % 50 == 0 or i == len(rows):
            print(f"baked {i}/{len(rows)}", flush=True)
    probes = "\n".join(f"  {g['label']}: {' '.join(g['chars'])}" for g in SHEET_PROBES)
    readme = f"""p649 字库检查图
baked_at: {datetime.now(timezone.utc).isoformat()}
n: {len(rows)}
size: ~500x450 PNG, one file per font (stem.png)
probes:
{probes}

汉字 = StyleImage；其余 = TargetImage。缺文件格子为浅红。
发给导师时直接发这些 PNG 即可。
"""
    (dest / "README.txt").write_text(readme, encoding="utf-8")
    return {"n": len(rows)}


def write_zip(src: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(src.iterdir()):
            if p.is_file():
                zf.write(p, arcname=f"p649_font_check_649/{p.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--g", type=int, default=72)
    ap.add_argument("--stems", nargs="*")
    args = ap.parse_args()
    rank = json.loads((REVIEW / "ink_ratio_rank.json").read_text(encoding="utf-8"))
    by = {f["stem"]: f for f in rank["fonts"]}
    if args.all:
        rows = rank["fonts"]
        bake_cards(rows, CARDS, args.g)
        zip_path = REVIEW / ZIP_NAME
        write_zip(CARDS, zip_path)
        print("zip", zip_path, zip_path.stat().st_size)
        return
    PREVIEW.mkdir(parents=True, exist_ok=True)
    if args.preview:
        rows = pick_preview_fonts(5)
    else:
        rows = [by[s] for s in (args.stems or [])]
        if not rows:
            raise SystemExit("need --preview, --all, or --stems")
    for f in rows:
        img = render_sheet(f["split"], f["stem"], g=args.g)
        dest = PREVIEW / f"{f['stem']}_g{args.g}.png"
        img.save(dest, optimize=True)
        print("wrote", f["stem"], img.size)
    if args.preview:
        write_preview_html(rows, args.g)
        PREVIEW_COPY.mkdir(parents=True, exist_ok=True)
        for p in PREVIEW.iterdir():
            (PREVIEW_COPY / p.name).write_bytes(p.read_bytes())
        print("preview", PREVIEW / "index.html")


if __name__ == "__main__":
    main()
