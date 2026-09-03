#!/usr/bin/env python3
"""Build FontDiffuser layout for retrain_v2 with CN style refs.

Layout (PROTOCOL.md):
  TargetImage/<font>/<font>+uXXXX.jpg  — P1∪P2 GT (training target)
  StyleImage/<font>/<font>+uXXXX.jpg   — same-font Chinese pool (random style)
  ContentImage/uXXXX.jpg               — content glyphs

If StyleImage exists, FontDataset samples style from it (CN).
Without StyleImage, falls back to old random-from-TargetImage behavior.

Style pool = all PNGs under font/train/chinese/<font>/ (expand via
expand_retrain_v2_cn_style_pool.py); not limited to ref8.
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path("/root/projects/hrfont")
SRC = ROOT / "data/retrain_v2/font"
DEFAULT_DST = ROOT / "data/fontdiffuser"
META = json.loads((ROOT / "data/unified_v1/meta.json").read_text(encoding="utf-8"))
REF8 = list(META.get("ref8") or "永和书风骨韵天地")
DEFAULT_FONT_LIST = ROOT / "reports/retrain_v2/suiti_font_screen/pipeline_v2_train_stems.txt"


def char_key_from_fname(name: str) -> str:
    stem = Path(name).stem
    if stem.endswith("+") and len(stem) == 2 and stem[0].isupper():
        return f"u{ord(stem[0]):04X}"
    if stem.startswith("u") and len(stem) == 5:
        return stem
    if len(stem) == 1:
        return f"u{ord(stem):04X}"
    return stem


def png_to_jpg(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst, quality=95)


def load_stems(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    stems = {
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    return stems


def build_split(dst: Path, phase_src: str, phase_dst: str, stems: set[str] | None):
    eng = SRC / phase_src / "english"
    cn = SRC / phase_src / "chinese"
    content_src = SRC / phase_src / "source"
    out_target = dst / phase_dst / "TargetImage"
    out_style = dst / phase_dst / "StyleImage"
    out_content = dst / phase_dst / "ContentImage"
    out_target.mkdir(parents=True, exist_ok=True)
    out_style.mkdir(parents=True, exist_ok=True)
    out_content.mkdir(parents=True, exist_ok=True)

    jobs = []
    for p in content_src.glob("*.png"):
        key = char_key_from_fname(p.name)
        jobs.append((p, out_content / f"{key}.jpg"))

    font_dirs = sorted(d for d in eng.iterdir() if d.is_dir())
    if stems is not None:
        font_dirs = [d for d in font_dirs if d.name in stems]

    n_tgt = 0
    n_style = 0
    miss_style = []
    for font_dir in font_dirs:
        style = font_dir.name
        for p in font_dir.glob("*.png"):
            key = char_key_from_fname(p.name)
            jobs.append((p, out_target / style / f"{style}+{key}.jpg"))
            n_tgt += 1
        cn_dir = cn / style
        if not cn_dir.is_dir():
            miss_style.append(style)
            continue
        cn_pngs = sorted(cn_dir.glob("*.png"))
        if not cn_pngs:
            miss_style.append(style)
            continue
        for src in cn_pngs:
            key = char_key_from_fname(src.name)
            jobs.append((src, out_style / style / f"{style}+{key}.jpg"))
            n_style += 1

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda pair: png_to_jpg(*pair), jobs))

    style_per_font = []
    for font_dir in font_dirs:
        d = out_style / font_dir.name
        style_per_font.append(len(list(d.glob("*.jpg"))) if d.is_dir() else 0)

    return {
        "content": len(list(out_content.glob("*.jpg"))),
        "target": n_tgt,
        "style_cn": n_style,
        "fonts": len(font_dirs),
        "style_per_font_min": min(style_per_font) if style_per_font else 0,
        "style_per_font_max": max(style_per_font) if style_per_font else 0,
        "style_source": "all chinese/*.png (same-font random pool)",
        "ref8_eval_hint": REF8,
        "miss_style_n": len(miss_style),
        "miss_style_head": miss_style[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--font-list",
        type=Path,
        default=None,
        help="Optional stem list (e.g. pipeline_v2_train_stems.txt). "
        "Default=None keeps all english dirs under train/ (legacy 42+).",
    )
    ap.add_argument(
        "--use-p253-list",
        action="store_true",
        help=f"Shortcut: use {DEFAULT_FONT_LIST}",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_DST,
        help="Output root (default data/fontdiffuser)",
    )
    args = ap.parse_args()
    dst: Path = args.dst

    font_list = DEFAULT_FONT_LIST if args.use_p253_list else args.font_list
    stems = load_stems(font_list)

    if dst.exists():
        shutil.rmtree(dst)
    summary = {
        "train": build_split(dst, "train", "train", stems),
        "val": build_split(dst, "test_unknown_style", "val", None),  # Demo-8 always
        "font_list": str(font_list) if font_list else None,
        "protocol": "CN StyleImage (ref8) + West TargetImage; see PROTOCOL.md",
    }
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
