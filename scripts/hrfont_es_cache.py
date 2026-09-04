#!/usr/bin/env python3
"""Build the E2 per-(font, character) Es cache.

Real mode imports ``StyleEncoder`` from the Stage-A variant and loads an E1
``style_encoder.pth`` (or a checkpoint directory containing it). ``--dummy``
uses a deterministic stub so the cache/data contract can be tested locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_stage_a/FontDiffuser"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image(path: Path) -> torch.Tensor:
    import numpy as np
    image = Image.open(path).convert("RGB")
    if image.size != (96, 96):
        raise ValueError(f"expected native 96x96: {path} ({image.size})")
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 127.5 - 1


class DummyEs(nn.Module):
    def forward(self, images):
        pooled = F.adaptive_avg_pool2d(images, 1).flatten(1)
        return images, pooled, [images]


def load_es(checkpoint: Path | None, dummy: bool):
    if dummy:
        return DummyEs(), None
    if checkpoint is None:
        raise SystemExit("--es-ckpt is required unless --dummy is used")
    ckpt = checkpoint / "style_encoder.pth" if checkpoint.is_dir() else checkpoint
    sys.path.insert(0, str(VARIANT))
    from src.modules.style_encoder import StyleEncoder
    model = StyleEncoder(G_ch=64, resolution=96)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model, sha256(ckpt)


def char_map(directory: Path, font: str) -> dict[str, Path]:
    prefix = f"{font}+"
    return {
        path.stem[len(prefix):]: path
        for path in sorted(directory.glob("*.png"))
        if path.stem.startswith(prefix)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--es-ckpt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    stems = sorted(manifest["stems"]["train"])
    style_root = args.data_root / "train" / "StyleImage"
    physical = {path.name for path in style_root.iterdir() if path.is_dir()}
    if physical != set(stems):
        raise RuntimeError(f"train split mismatch: missing={sorted(set(stems)-physical)} "
                           f"extra={sorted(physical-set(stems))}")

    es, checkpoint_sha = load_es(args.es_ckpt, args.dummy)
    es = es.to(args.device).eval().requires_grad_(False)
    cache = {}
    expected_chars = None
    with torch.no_grad():
        for font_index, font in enumerate(stems, 1):
            images = char_map(style_root / font, font)
            chars = sorted(images)
            if len(chars) != 338:
                raise RuntimeError(f"expected 338 style chars for {font}, got {len(chars)}")
            if expected_chars is None:
                expected_chars = chars
            elif chars != expected_chars:
                raise RuntimeError(f"style character pool mismatch for {font}")
            for cp in chars:
                _, pooled, _ = es(load_image(images[cp]).unsqueeze(0).to(args.device))
                cache[(font, cp)] = F.normalize(pooled.float(), dim=1).squeeze(0).half().cpu()
            if font_index % 50 == 0 or font_index == len(stems):
                print(f"Es cache: {font_index}/{len(stems)} fonts", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    output_manifest = args.manifest_output or args.output.with_suffix(".manifest.json")
    output_manifest.write_text(json.dumps({
        "stems": stems,
        "chars": expected_chars,
        "es_checkpoint_sha256": checkpoint_sha,
        "cache_sha256": sha256(args.output),
        "dtype": "float16",
        "entries": len(cache),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
