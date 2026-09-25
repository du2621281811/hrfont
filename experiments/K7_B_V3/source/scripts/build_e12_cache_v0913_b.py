#!/usr/bin/env python3
"""Build E12-b cache from protocol-A PNGs filtered by v0913_clean usability.

- Train fonts: usable only (exclude dropped).
- Per font: only chars allowed by that font's script bucket (from pairs_train +
  StyleImage CN refs always kept when present).
Output: artifacts/e12/cache_v0913_b/
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path("/root/projects/hrfont")
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
CHARSET = ROOT / "manifests/charset_cn2west_v2_planned.json"
FONTS_TSV = ROOT / "manifests/v0913_clean/fonts.tsv"
PAIRS_TRAIN = ROOT / "manifests/v0913_clean/pairs_train.tsv"
STYLE = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2/train/StyleImage"
TARGET = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2/train/TargetImage"
OUT = ROOT / "artifacts/e12/cache_v0913_b"


def cp4(ch: str) -> str:
    return f"u{ord(ch):04X}"


def cp6(ch: str) -> str:
    return f"u{ord(ch):06X}"


def load_usable_train_fonts() -> dict[str, str]:
    """stem -> bucket for train usable fonts."""
    out: dict[str, str] = {}
    for line in FONTS_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        stem, split, bucket, usable = line.split("\t")[:4]
        if split == "train" and usable == "1":
            out[stem] = bucket
    return out


def load_allowed_chars() -> dict[str, set[str]]:
    """font -> set of target chars allowed by clean train pairs."""
    allowed: dict[str, set[str]] = defaultdict(set)
    for line in PAIRS_TRAIN.read_text(encoding="utf-8").splitlines()[1:]:
        _split, font, ch, *_rest = line.split("\t")
        allowed[font].add(ch)
    return allowed


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    cs = json.loads(CHARSET.read_text(encoding="utf-8"))
    usable = load_usable_train_fonts()
    allowed = load_allowed_chars()
    stems = [s for s in split["stems"]["train"] if s in usable]
    if len(stems) != 223:
        raise SystemExit(f"expected 223 usable train fonts, got {len(stems)}")

    cn = list(cs["style_ref8_subset"]) + list(cs["style_han_338"])[:96]
    seen: set[str] = set()
    cn_chars: list[str] = []
    for c in cn:
        if c not in seen:
            seen.add(c)
            cn_chars.append(c)

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
        want_chars: list[str] = list(cn_chars)
        for ch in sorted(allowed.get(stem, ())):
            if ch not in cn_chars:
                want_chars.append(ch)

        for ch in want_chars:
            src_png = src / f"{stem}+{cp4(ch)}.png"
            if not src_png.is_file():
                src_png = TARGET / stem / f"{stem}+{cp4(ch)}.png"
            if not src_png.is_file():
                n_miss += 1
                continue
            if ch not in cn_chars and ch not in allowed.get(stem, ()):
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
                    "bucket": usable[stem],
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
                "bucket": usable[stem],
                "sha256": hashlib.sha256(stem.encode()).hexdigest(),
                "id": fid,
                "n_glyphs": wrote,
            }
        )

    all_chars = sorted({r["char"] for r in records})
    manifest = {
        "version": 1,
        "canvas": 96,
        "strategy": "png_import_v0913_clean",
        "dataset_id": "v0913_clean",
        "experiment": "E12-B",
        "source_style": str(STYLE),
        "source_target": str(TARGET),
        "chars": all_chars,
        "chinese_chars": cn_chars,
        "fonts": font_rows,
        "records": records,
        "n_fonts": len(font_rows),
        "n_png_ok": n_ok,
        "n_png_miss": n_miss,
        "cache_sha256": hashlib.sha256(
            json.dumps({"fonts": [f["id"] for f in font_rows], "n": n_ok}, sort_keys=True).encode()
        ).hexdigest(),
        "note": "E12-b same-domain scorer on v0913_clean usability; does not overwrite cache_train228_v51.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "fonts": len(font_rows),
                "png_ok": n_ok,
                "png_miss": n_miss,
                "chars": len(all_chars),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
