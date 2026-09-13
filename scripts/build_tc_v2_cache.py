#!/usr/bin/env python3
"""Build the TC-v2 VGG appearance cache.

The target train rows alone fit the standardizer; val/test rows are only
descriptors for evaluation/inference.  The script does not construct Ec/Es,
and refuses to write into a non-empty cache directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
sys.path.insert(0, str(VARIANT))
from src.criterion import VGG16  # noqa: E402
from src.tc_v2 import (TC_APPEARANCE_DIM, AppearanceStandardizer,
                       appearance_stats)  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_clean_map(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("pairs_train.tsv", "pairs_val.tsv", "pairs_test.tsv", "sample_weights.json"):
        file = path / name
        if not file.is_file():
            raise FileNotFoundError(file)
        digest.update(name.encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def char_map(directory: Path, font: str) -> dict[str, Path]:
    prefix = f"{font}+"
    return {p.stem[len(prefix):]: p for p in sorted(directory.glob("*.png"))
            if p.stem.startswith(prefix)}


def collect_jobs(data_root: Path, split: dict, clean_map: Path | None = None) -> list[tuple[str, str, str, str, Path]]:
    allowed = {}
    if clean_map is not None:
        import csv
        for part in ("train", "val", "test"):
            pair_file = clean_map / f"pairs_{part}.tsv"
            with pair_file.open(encoding="utf-8", newline="") as handle:
                allowed[part] = {(row["font"], row["cp"])
                                 for row in csv.DictReader(handle, delimiter="\t")}
    jobs = []
    for part in ("train", "val", "test"):
        for font in sorted(split[part]):
            style = char_map(data_root / part / "StyleImage" / font, font)
            target = char_map(data_root / part / "TargetImage" / font, font)
            if not style or not target:
                raise RuntimeError(f"missing glyphs for {part}/{font}")
            if clean_map is not None and part != "test":
                missing = sorted((font, cp) for (font2, cp) in allowed[part]
                                 if font2 == font and cp not in target)
                if missing:
                    raise RuntimeError(f"clean-map pairs missing target files: {missing[:5]}")
            # Test references are allowed for inference; test targets are never
            # cached because they must not enter teacher/statistics selection.
            jobs.extend(("ref", part, font, cp, path) for cp, path in style.items())
            if part != "test":
                jobs.extend(("target", part, font, cp, path)
                            for cp, path in target.items()
                            if clean_map is None or (font, cp) in allowed[part])
    if not jobs:
        raise RuntimeError("no TC jobs")
    return jobs


def load_rgb(path: Path, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (resolution, resolution):
        raise ValueError(f"expected {resolution}x{resolution}: {path} ({image.size})")
    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0


def encode(model, paths: list[Path], device: str, resolution: int) -> np.ndarray:
    images = torch.stack([load_rgb(path, resolution) for path in paths]).to(device)
    # VGG16 is used by the existing perceptual loss after ImageNet normalization
    # on [0,1] images.  The diffusion dataset's [-1,1] protocol is not reused.
    mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    images = (images - mean) / std
    with torch.no_grad():
        return appearance_stats(model(images)).cpu().numpy().astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clean-map", type=Path, required=True,
                    help="v0913_clean directory; target rows follow pairs_{split}.tsv")
    ap.add_argument("--vgg-weights", type=Path, default=None,
                    help="local torchvision VGG16 state_dict; omit to use torchvision pretrained weights")
    ap.add_argument("--resolution", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"refusing non-empty cache: {args.out}; choose a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split = payload.get("stems", payload)
    split = {part: sorted(split[part]) for part in ("train", "val", "test")}
    jobs = collect_jobs(args.data_root, split, args.clean_map)
    keys = [f"tc|{role}|{part}|{font}|{cp}" for role, part, font, cp, _ in jobs]
    (args.out / "keys.txt").write_text("\n".join(keys) + "\n", encoding="utf-8")
    arr = np.memmap(args.out / "appearance.dat", dtype=np.float16, mode="w+",
                    shape=(len(jobs), TC_APPEARANCE_DIM))
    model = VGG16(weights_path=args.vgg_weights).to(args.device).eval().requires_grad_(False)
    bs = max(1, args.batch_size)
    for start in range(0, len(jobs), bs):
        chunk = jobs[start:start + bs]
        arr[start:start + len(chunk)] = encode(model, [item[-1] for item in chunk],
                                               args.device, args.resolution)
        if (start + len(chunk)) % (bs * 16) == 0 or start + len(chunk) == len(jobs):
            arr.flush()
            print(f"tc cache {start + len(chunk)}/{len(jobs)}", flush=True)
    arr.flush()

    train_rows = [i for i, (role, part, *_rest) in enumerate(jobs)
                  if role == "target" and part == "train"]
    train_raw = torch.from_numpy(np.array(arr[train_rows], copy=True)).float()
    standardizer = AppearanceStandardizer.fit(train_raw)
    train_stats = standardizer.state_dict_json()
    state_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        state_digest.update(name.encode("utf-8"))
        state_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    vgg_digest = sha256_file(args.vgg_weights) if args.vgg_weights else state_digest.hexdigest()
    manifest = {
        "kind": "tc_v2_appearance_cache",
        "descriptor": "VGG16 enc_1/2/3 mean+std population",
        "appearance_dim": TC_APPEARANCE_DIM,
        "dtype": "float16",
        "resolution": args.resolution,
        "input_normalize": "RGB / 255 then ImageNet mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)",
        "vgg_weights": str(args.vgg_weights.resolve()) if args.vgg_weights else None,
        "vgg_weights_sha256": vgg_digest,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": digest_json(args.split_manifest),
        "clean_map": str(args.clean_map.resolve()) if args.clean_map else None,
        "clean_map_sha256": digest_clean_map(args.clean_map),
        "entries": len(jobs),
        "train_target_entries": len(train_rows),
        "train_target_stats": train_stats,
        "roles": ["ref", "target"],
        "splits": ["train", "val", "test"],
        "target_splits": ["train", "val"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    (args.out / "progress.json").write_text(json.dumps({"done": len(jobs), "total": len(jobs)}) + "\n",
                                                        encoding="utf-8")
    print(f"wrote {args.out} entries={len(jobs)} train_targets={len(train_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
