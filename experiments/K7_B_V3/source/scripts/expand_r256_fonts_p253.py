#!/usr/bin/env python3
"""Render missing p253-train fonts into style256_meta (meta-aligned), skip val."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
R1 = ROOT / "data/retrain_v2/renders_r1"
REP = ROOT / "reports/retrain_v2/native256_walkaway/data_ablation"
LOG = ROOT / "logs_retrain_v2/data_ablation_render_p253.log"
FONT50 = Path("/root/data/font_50")
SUITI = ROOT / "data/suiti_fonts_probe/随体字体集"
FONT_DIRS = [FONT50, SUITI]
P253 = ROOT / "data/fontdiffuser_cn2cn_p253/train/TargetImage"
SCALE = 256 / 96
VAL_FONTS = {
    "FZChuangHJW_DB",
    "FZCuanBZBKSJW",
    "FZDouNTJW_Te",
    "FZFengYKSJ",
    "FZHanWZKJW",
    "FZHuoYYJW-T",
    "FZJingYLLTJW",
    "FZLingFKSJW-B",
}


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def resolve_ttf(stem: str) -> Path | None:
    for d in FONT_DIRS:
        if not d.exists():
            continue
        for ext in (".TTF", ".ttf", ".OTF", ".otf"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
        hits = [h for h in d.glob(stem + ".*") if h.suffix.lower() in {".ttf", ".otf"}]
        if hits:
            return hits[0]
    return None


def render_xy(ttf, ch, size, canvas, dx=0, dy=0) -> Image.Image:
    font = ImageFont.truetype(str(ttf), size=int(size))
    im = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - w) // 2 - bbox[0] + int(dx)
    y = (canvas - h) // 2 - bbox[1] + int(dy)
    draw.text((x, y), ch, fill=(0, 0, 0), font=font)
    return im


def main():
    REP.mkdir(parents=True, exist_ok=True)
    style_out = R1 / "style256_meta"
    content = R1 / "content_256_meta"
    style_out.mkdir(parents=True, exist_ok=True)

    # reuse 96 params if present
    sp96 = R1 / "meta" / "style_sizes_96.json"
    sizes = json.loads(sp96.read_text()) if sp96.exists() else {}

    chars = []
    for p in sorted(content.glob("u*.png")):
        try:
            chars.append(chr(int(p.stem[1:], 16)))
        except Exception:
            pass
    log(f"[chars] {len(chars)} from content_256_meta")

    train_fonts = sorted([p.name for p in P253.iterdir() if p.is_dir() and p.name not in VAL_FONTS])
    (REP / "fonts_f253.txt").write_text("\n".join(train_fonts) + "\n", encoding="utf-8")
    allow35 = ROOT / "reports/retrain_v2/native256_walkaway/fd256_a_train_fonts.txt"
    f35 = [ln.strip() for ln in allow35.read_text().splitlines() if ln.strip()]
    (REP / "fonts_f35.txt").write_text("\n".join(f35) + "\n", encoding="utf-8")

    yong_name = f"u{ord('永'):04X}.png"
    nchars = max(len(chars), 1)
    need = []
    for f in train_fonts:
        d = style_out / f
        n = len(list(d.glob('u*.png'))) if d.exists() else 0
        if (not d.exists()) or (not (d / yong_name).exists()) or n < nchars * 0.95:
            need.append(f)
    log(f"[plan] train_fonts={len(train_fonts)} need_render={len(need)}")

    done = 0
    miss_ttf = []
    for stem in need:
        ttf = resolve_ttf(stem)
        if ttf is None:
            miss_ttf.append(stem)
            continue
        # default size from 96 meta or search
        info = sizes.get(stem, 72)
        if isinstance(info, dict):
            size96 = int(info.get("size", 72))
            dx96 = float(info.get("dx", 0))
            dy96 = float(info.get("dy", 0))
        else:
            size96 = int(info)
            dx96 = 0.0
            dy96 = 0.0
        size = max(8, int(round(size96 * SCALE)))
        dx = dx96 * SCALE
        dy = dy96 * SCALE
        out_dir = style_out / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for ch in chars:
            fp = out_dir / f"u{ord(ch):04X}.png"
            if fp.exists():
                continue
            im = render_xy(ttf, ch, size, 256, dx=dx, dy=dy)
            im.save(fp)
        # style ref 永
        yong = out_dir / f"u{ord('永'):04X}.png"
        if not yong.exists():
            render_xy(ttf, "永", size, 256, dx=dx, dy=dy).save(yong)
        done += 1
        if done % 10 == 0 or done == len(need):
            log(f"[progress] rendered_fonts={done}/{len(need)} last={stem}")

    # coverage check
    ok = []
    bad = []
    yong = f"u{ord('永'):04X}.png"
    for stem in train_fonts:
        d = style_out / stem
        n = len(list(d.glob("u*.png"))) if d.exists() else 0
        if d.exists() and (d / yong).exists() and n >= len(chars) * 0.95:
            ok.append(stem)
        else:
            bad.append((stem, n))
    (REP / "fonts_f253_ready.txt").write_text("\n".join(ok) + "\n", encoding="utf-8")
    status = {
        "train_fonts": len(train_fonts),
        "ready": len(ok),
        "bad": len(bad),
        "miss_ttf": miss_ttf,
        "bad_sample": bad[:20],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (REP / "render_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[done] ready={len(ok)} bad={len(bad)} miss_ttf={len(miss_ttf)}")
    if len(ok) < 200:
        log(f"[warn] ready={len(ok)} < 200")


if __name__ == "__main__":
    main()
