#!/usr/bin/env python3
"""Build E12 cache from protocol-A StyleImage PNGs (train228 main-data path).

Layout matches scripts/eval_framework/data.py build_cache():
  {cache}/{font_id}/uXXXXXX.png + manifest with records[].
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path("/root/projects/hrfont")
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
CHARSET = ROOT / "manifests/charset_cn2west_v2_planned.json"
STYLE = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2/train/StyleImage"
TARGET = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2/train/TargetImage"
OUT = ROOT / "artifacts/e12/cache_train228_v51"


def cp4(ch: str) -> str:
    return f"u{ord(ch):04X}"


def cp6(ch: str) -> str:
    return f"u{ord(ch):06X}"


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    cs = json.loads(CHARSET.read_text(encoding="utf-8"))
    stems = list(split["stems"]["train"])

    cn = list(cs["style_ref8_subset"]) + list(cs["style_han_338"])[:96]
    seen: set[str] = set()
    cn_chars: list[str] = []
    for c in cn:
        if c not in seen:
            seen.add(c)
            cn_chars.append(c)
    west: list[str] = []
    for chars in cs["target"].values():
        west.extend(chars)
    chars = cn_chars + [c for c in west if c not in seen]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    font_rows = []
    records = []
    n_ok = n_miss = 0
    for stem in stems:
        src = STYLE / stem
        if not src.is_dir():
            continue
        fid = f"{stem}-{hashlib.sha256(stem.encode()).hexdigest()[:10]}"
        fam_dir = OUT / fid
        fam_dir.mkdir(parents=True, exist_ok=True)
        wrote = 0
        for ch in chars:
            # CN refs live under StyleImage; western targets under TargetImage.
            src_png = src / f"{stem}+{cp4(ch)}.png"
            if not src_png.is_file():
                src_png = TARGET / stem / f"{stem}+{cp4(ch)}.png"
            if not src_png.is_file():
                n_miss += 1
                continue
            im = Image.open(src_png).convert("L")
            if im.size != (96, 96):
                im = im.resize((96, 96))
            rel = Path(fid) / f"{cp6(ch)}.png"
            im.save(OUT / rel)
            records.append(
                {
                    "font_id": fid,
                    "family": stem,
                    "group": stem,
                    "stem": stem,
                    "char": ch,
                    "path": str(rel),
                }
            )
            wrote += 1
            n_ok += 1
        if wrote == 0:
            fam_dir.rmdir()
            continue
        font_rows.append(
            {
                "path": str(src),
                "stem": stem,
                "family": stem,
                "group": stem,
                "sha256": hashlib.sha256(stem.encode()).hexdigest(),
                "id": fid,
                "n_glyphs": wrote,
            }
        )

    manifest = {
        "version": 1,
        "canvas": 96,
        "strategy": "png_import_protocol_a",
        "source": str(STYLE),
        "chars": chars,
        "chinese_chars": cn_chars,
        "fonts": font_rows,
        "records": records,
        "n_fonts": len(font_rows),
        "n_png_ok": n_ok,
        "n_png_miss": n_miss,
        "cache_sha256": hashlib.sha256(
            json.dumps({"fonts": [f["id"] for f in font_rows], "chars": chars}, sort_keys=True).encode()
        ).hexdigest(),
        "note": "E12 v5.1 PI: fit on main train228 StyleImage; narrative = same-domain protocol scorer.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "fonts": len(font_rows), "png_ok": n_ok, "png_miss": n_miss, "chars": len(chars)}, indent=2))


if __name__ == "__main__":
    main()
