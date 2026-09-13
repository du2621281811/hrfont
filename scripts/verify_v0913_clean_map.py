#!/usr/bin/env python3
"""Verify frozen v0913_clean maps; optionally symlink a clean tree over dirty PNGs.

Does not copy pixels. Other machines: read manifests/v0913_clean/INDEX.json,
point --png-root at the existing protocol-A disk, then either train with the
pair lists or --materialize-to a symlink root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
MAP = ROOT / "manifests/v0913_clean"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def require_file(root: Path, rel: str) -> Path:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_hashes(index: dict) -> None:
    for name, digest in index["files"].items():
        path = MAP / name
        live = sha256_file(path)
        if live != digest:
            raise SystemExit(f"hash mismatch {name}: {live} != {digest}")


def iter_needed_paths(png_root: Path, index: dict) -> list[Path]:
    needed: set[Path] = set()
    templates = index["path_templates"]
    for split in ("train", "val", "test"):
        for row in load_tsv(MAP / f"pairs_{split}.tsv"):
            needed.add(Path(templates["target"].format(png_root=png_root, **row)))
            needed.add(Path(templates["content"].format(png_root=png_root, **row)))
    fonts = load_tsv(MAP / "fonts.tsv")
    charset = json.loads((ROOT / "manifests/charset_cn2west_v2_planned.json").read_text(encoding="utf-8"))
    style_cps = [f"u{ord(ch):04X}" for ch in charset["style_han_338"]]
    for rec in fonts:
        if rec["usable"] != "1":
            continue
        for cp in style_cps:
            needed.add(
                Path(
                    templates["style"].format(
                        png_root=png_root,
                        split=rec["split"],
                        font=rec["stem"],
                        cp=cp,
                    )
                )
            )
    return sorted(needed)


def materialize(png_root: Path, dest: Path, index: dict) -> int:
    dest = dest.resolve()
    png_root = png_root.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in iter_needed_paths(png_root, index):
        rel = src.relative_to(png_root)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_symlink() or out.exists():
            if out.resolve() != src.resolve():
                raise SystemExit(f"refusing to replace {out}")
            continue
        out.symlink_to(src)
        n += 1
    (dest / "V0913_CLEAN_ROOT.json").write_text(
        json.dumps(
            {
                "dataset_id": "v0913_clean",
                "dirty_png_root": str(png_root),
                "index_sha256": sha256_file(MAP / "INDEX.json"),
                "note": "Symlink view. Train weights are in manifests/v0913_clean/sample_weights.json.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-root", default="data/fontdiffuser-p253-t295-s338-cn2west-v2")
    ap.add_argument("--materialize-to", default=None)
    ap.add_argument("--skip-png", action="store_true")
    args = ap.parse_args()
    index = json.loads((MAP / "INDEX.json").read_text(encoding="utf-8"))
    verify_hashes(index)
    if not args.skip_png:
        png_root = Path(args.png_root)
        png_root = png_root if png_root.is_absolute() else ROOT / png_root
        missing = [str(p) for p in iter_needed_paths(png_root, index) if not p.is_file()]
        if missing:
            preview = "\n".join(missing[:8])
            raise SystemExit(f"missing {len(missing)} dirty PNGs, first:\n{preview}")
        if args.materialize_to:
            dest = Path(args.materialize_to)
            dest = dest if dest.is_absolute() else ROOT / dest
            n = materialize(png_root, dest, index)
            print(f"materialized {n} new symlinks → {dest}")
    print(
        "v0913_clean OK  "
        f"train={index['counts']['train']['all']} "
        f"val={index['counts']['val']['all']} "
        f"test={index['counts']['test']['all']} "
        f"eval47={index['counts']['eval47']['keep']}/752"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
