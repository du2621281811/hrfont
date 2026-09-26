#!/usr/bin/env python3
"""Build a separate, out-of-distribution Chinese-font challenge set for K6.

This asset is deliberately not a replacement for K_TEST/K_VAL192.  It uses
publicly released fonts, verifies that their family names do not occur in the
frozen V2 test manifest, renders native 96x96 RGB PNGs with the same
per-glyph-fit renderer, and records provenance plus SHA256 hashes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[6]
FONT_DIR = REPO / "work/external_font_challenge/fonts"
OUT = REPO / "work/external_font_challenge/rendered"
AUDIT = REPO / "work/external_font_challenge/external_font_manifest.json"
K_TEST = REPO / "repo/experiments/K/K_TEST.json"

FONTS = [
    {
        "stem": "MaShanZheng",
        "path": "MaShanZheng-Regular.ttf",
        "source": "https://github.com/googlefonts/mashanzheng",
        "license": "OFL-1.1",
        "license_file": "licenses/OFL-GoogleFonts.txt",
        "role": "heavy brush display",
    },
    {
        "stem": "LiuJianMaoCao",
        "path": "LiuJianMaoCao-Regular.ttf",
        "source": "https://github.com/googlefonts/liujianmaocao",
        "license": "OFL-1.1",
        "license_file": "licenses/OFL-GoogleFonts.txt",
        "role": "expressive cursive",
    },
    {
        "stem": "ZhiMangXing",
        "path": "ZhiMangXing-Regular.ttf",
        "source": "https://fonts.google.com/specimen/Zhi+Mang+Xing",
        "license": "OFL-1.1",
        "license_file": "licenses/OFL-GoogleFonts.txt",
        "role": "running brush script",
    },
    {
        "stem": "LongCang",
        "path": "LongCang-Regular.ttf",
        "source": "https://fonts.google.com/specimen/Long+Cang",
        "license": "OFL-1.1",
        "license_file": "licenses/OFL-GoogleFonts.txt",
        "role": "long-stroke brush display",
    },
    {
        "stem": "Iansui",
        "path": "Iansui-Regular.ttf",
        "source": "https://github.com/ButTaiwan/iansui",
        "license": "OFL-1.1",
        "license_file": "licenses/OFL-Iansui.txt",
        "role": "traditional handwritten / Klee-derived",
    },
    {
        "stem": "NotoSerifSC",
        "path": "NotoSerifSC-Regular.ttf",
        "source": "https://fonts.google.com/specimen/Noto+Serif+SC",
        "license": "OFL-1.1",
        "license_file": "licenses/OFL-NotoSerifSC.txt",
        "role": "serif control",
    },
]

# Use eight Chinese reference glyphs from the style side, while keeping the
# target side inside the frozen V2 target alphabet so the standard content
# encoder/cache and Delta donor bank remain valid.
REF_PREF = list("永和骨天地山水人")
TARGET_PREF = list("0123ABCDWXYabgmz")
CANVAS = 96
TARGET_MARGIN = round(CANVAS * 0.08)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cp(ch: str) -> str:
    return f"u{ord(ch):04X}"


def cmap(path: Path) -> set[str]:
    font = TTFont(path, lazy=True)
    try:
        return {chr(x) for x in font.getBestCmap()}
    finally:
        font.close()


def render_fit(path: Path, ch: str) -> tuple[Image.Image, int, int]:
    """Match the V2 F renderer: fit ink to an approximately 8px margin."""
    def draw_at(fs: int) -> tuple[int | None, Image.Image]:
        im = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(str(path), fs)
        box = dr.textbbox((0, 0), ch, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
        if w <= 0 or h <= 0:
            return None, im
        dr.text(((CANVAS - w) // 2 - box[0], (CANVAS - h) // 2 - box[1]), ch,
                font=font, fill=(0, 0, 0))
        a = np.asarray(im.convert("L"))
        ys, xs = np.where(a < 250)
        if len(ys) == 0:
            return None, im
        margin = min(int(xs.min()), int(ys.min()), CANVAS - 1 - int(xs.max()),
                     CANVAS - 1 - int(ys.max()))
        return margin, im

    lo, hi, best = 8, 800, None
    while lo <= hi:
        fs = (lo + hi) // 2
        margin, _ = draw_at(fs)
        if margin is not None and margin >= TARGET_MARGIN:
            best = fs
            lo = fs + 1
        else:
            hi = fs - 1
    if best is None:
        best = 48
    candidates = []
    for fs in range(max(8, best - 2), best + 3):
        margin, im = draw_at(fs)
        if margin is not None:
            candidates.append((abs(margin - TARGET_MARGIN), -fs, fs, margin, im))
    if not candidates:
        raise RuntimeError(f"no visible glyph for {path} {ch!r}")
    _, _, fs, margin, im = sorted(candidates)[0]
    return im, fs, margin


def main() -> None:
    current = json.loads(K_TEST.read_text(encoding="utf-8"))
    current_fonts = sorted({j["font"] for j in current["jobs"]})
    cmap_by_stem = {}
    rows = []
    for item in FONTS:
        path = FONT_DIR / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        chars = cmap(path)
        cmap_by_stem[item["stem"]] = chars
        row = dict(item)
        row.update(
            path=str(path), sha256=sha256(path), glyph_count=len(chars),
            current_manifest_overlap=item["stem"] in current_fonts,
        )
        if row["current_manifest_overlap"]:
            raise RuntimeError(f"external stem overlaps K_TEST: {item['stem']}")
        rows.append(row)

    common = set.intersection(*(cmap_by_stem[x["stem"]] for x in FONTS))
    refs = [ch for ch in REF_PREF if ch in common]
    targets = [ch for ch in TARGET_PREF if ch in common and ch not in refs]
    if len(refs) != len(REF_PREF):
        raise RuntimeError(f"reference intersection incomplete: {refs}")
    if len(targets) < 16:
        raise RuntimeError(f"target intersection too small: {targets}")
    targets = targets[:16]

    if OUT.exists():
        shutil.rmtree(OUT)
    for item in FONTS:
        font_path = FONT_DIR / item["path"]
        stem = item["stem"]
        style_dir = OUT / "StyleImage" / stem
        target_dir = OUT / "TargetImage" / stem
        style_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        for ch in refs:
            im, _, _ = render_fit(font_path, ch)
            im.save(style_dir / f"{stem}+{cp(ch)}.png", optimize=True)
        for ch in targets:
            im, _, _ = render_fit(font_path, ch)
            im.save(target_dir / f"{stem}+{cp(ch)}.png", optimize=True)

    manifest = {
        "kind": "hrfont_k6_external_font_ood_challenge",
        "version": "20260921-v1",
        "not_main_metric": True,
        "renderer": "V2-F-per-glyph-ink-margin-fit96",
        "protocol": "K-original-CFG1-DPM20-v1",
        "seed": 3407,
        "shot_schedule": [1, 2, 4, 8],
        "refs": refs,
        "targets": targets,
        "fonts": rows,
        "current_k_test_fonts": current_fonts,
        "current_k_test_manifest_sha256": sha256(K_TEST),
        "source_font_dir": str(FONT_DIR),
        "rendered_dir": str(OUT),
    }
    AUDIT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fonts": [x["stem"] for x in rows], "refs": refs,
                      "targets": targets, "manifest": str(AUDIT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
