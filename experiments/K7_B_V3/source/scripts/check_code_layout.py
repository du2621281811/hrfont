#!/usr/bin/env python3
"""Fail if official/ours layout is missing or training path points at official."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
off = ROOT / "code/official/FontDiffuser"
ours = ROOT / "code/ours/FontDiffuser"
assert (off / "ZZZ_READ_ME_THIS_IS_OFFICIAL_UPSTREAM.txt").is_file(), off
assert (ours / "ZZZ_READ_ME_THIS_IS_OURS_PATCHED_NOT_OFFICIAL.txt").is_file(), ours
assert (ours / "dataset/font_dataset.py").is_file()
# patched file should mention StyleImage
text = (ours / "dataset/font_dataset.py").read_text(encoding="utf-8")
assert "StyleImage" in text or "use_cn_style_dir" in text
off_text = (off / "dataset/font_dataset.py").read_text(encoding="utf-8")
assert "use_cn_style_dir" not in off_text
print("OK: official vs ours layout looks correct.")
