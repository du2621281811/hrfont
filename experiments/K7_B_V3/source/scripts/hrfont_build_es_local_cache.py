#!/usr/bin/env python3
"""Build cache-only Es local tokens for F2-RL128.

Hook F0 StyleEncoder.blocks[2][0] (third DBlock, after shortcut+downsample),
adaptive-avg-pool to 4×4, store [16,256] fp16 per (split, font, style char).

Reuse the key order of artifacts/f0/es_spatial_f0. Never trains; cache-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_LOAD_EXEC = ThreadPoolExecutor(max_workers=8)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/root/projects/hrfont")
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
sys.path.insert(0, str(ROOT / "scripts"))
from hrfont_feature_cache import ES_LOCAL, MemmapTable, key_es, sha256_file  # noqa: E402

HOOK_PATH = "StyleEncoder.blocks[2][0]"
POOL = 4


def load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (96, 96):
        raise ValueError(f"expected native 96x96: {path} ({image.size})")
    array = np.array(image, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1


def char_map(directory: Path, font: str) -> dict[str, Path]:
    prefix = f"{font}+"
    return {
        path.stem[len(prefix):]: path
        for path in sorted(directory.glob("*.png"))
        if path.stem.startswith(prefix)
    }


def collect_jobs(data_root: Path, split: dict) -> list[tuple[str, Path]]:
    jobs = []
    expected = None
    for part, fonts in split.items():
        for font in fonts:
            mapping = char_map(data_root / part / "StyleImage" / font, font)
            chars = sorted(mapping)
            if len(chars) != 338:
                raise RuntimeError(f"{part}/{font} style chars={len(chars)} expected 338")
            if expected is None:
                expected = chars
            elif chars != expected:
                raise RuntimeError(f"style charset mismatch {part}/{font}")
            for cp in chars:
                jobs.append((key_es(part, font, cp), mapping[cp]))
    return jobs


def load_encoder(ckpt: Path, device: str):
    sys.path.insert(0, str(VARIANT))
    from src.modules.style_encoder import StyleEncoder

    model = StyleEncoder(G_ch=64, resolution=96)
    weights = ckpt / "style_encoder.pth"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, sha256_file(weights)


def encode_batch(model, paths: list[Path], device: str) -> np.ndarray:
    if len(paths) <= 2:
        images = torch.stack([load_rgb(p) for p in paths]).to(device)
    else:
        images = torch.stack(list(_LOAD_EXEC.map(load_rgb, paths))).to(device)
    captured: dict[str, torch.Tensor] = {}

    def hook(_mod, _inp, out):
        captured["h"] = out

    handle = model.blocks[2][0].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(images)
    finally:
        handle.remove()
    h = captured["h"]
    if tuple(h.shape[1:]) != (256, 12, 12):
        raise RuntimeError(f"block2 shape {tuple(h.shape)} expected [B,256,12,12]")
    pooled = F.adaptive_avg_pool2d(h.float(), POOL)
    tokens = pooled.permute(0, 2, 3, 1).reshape(h.shape[0], POOL * POOL, 256)
    return tokens.half().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs/F0-RSIFREE-FT-A-S3407/best")
    ap.add_argument("--data-root", type=Path,
                    default=ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2")
    ap.add_argument("--split", type=Path, default=ROOT / "manifests/split_v3_228_16_16.json")
    ap.add_argument("--spatial-cache", type=Path, default=ROOT / "artifacts/f0/es_spatial_f0")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/f0/es_local_f0_block2_pool4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--verify", type=int, default=16)
    args = ap.parse_args()

    payload = json.loads(args.split.read_text(encoding="utf-8"))
    stems = payload.get("stems", payload)
    split = {k: sorted(stems[k]) for k in ("train", "val", "test")}
    jobs = collect_jobs(args.data_root, split)
    spatial_keys = [ln for ln in (args.spatial_cache / "keys.txt").read_text(encoding="utf-8").splitlines() if ln]
    job_keys = [k for k, _ in jobs]
    if job_keys != spatial_keys:
        raise RuntimeError("local-cache key order must match es_spatial_f0/keys.txt")

    spatial_man = json.loads((args.spatial_cache / "manifest.json").read_text(encoding="utf-8"))
    model, enc_sha = load_encoder(args.ckpt, args.device)
    if enc_sha != spatial_man.get("es_checkpoint_sha256"):
        raise RuntimeError("F0 style_encoder SHA != es_spatial_f0 binding")

    n = len(jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    table = MemmapTable(args.out, create=True)
    table.set_keys(job_keys)
    arr = table.open_array("local", (n, *ES_LOCAL), "w+")
    start = table.load_progress()
    if start and start < n:
        print(f"resume from {start}/{n}", flush=True)
    index = start
    bs = args.batch_size
    t0 = time.time()
    while index < n:
        chunk = jobs[index:index + bs]
        try:
            tokens = encode_batch(model, [p for _, p in chunk], args.device)
            arr[index:index + len(chunk)] = tokens
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and bs > 1:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                print(f"OOM, retry batch={bs}", flush=True)
                continue
            raise
        index += len(chunk)
        if index % 1024 == 0 or index == n:
            table.flush()
            table.save_progress(index, n)
            rate = index / max(1e-6, time.time() - t0)
            print(f"local: {index}/{n} ({100.0 * index / n:.1f}%) {rate:.1f}/s", flush=True)
    table.flush()
    table.save_progress(n, n)
    digest = hashlib.sha256()
    with (args.out / "local.dat").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    payload_sha = digest.hexdigest()
    man = {
        "kind": "es_local",
        "role": "es_local_f0_block2_pool4",
        "entries": n,
        "hook_path": HOOK_PATH,
        "pool": POOL,
        "raw_shape": [256, 12, 12],
        "saved_shape": [n, *ES_LOCAL],
        "dtype": "float16",
        "normalize": "RGB ToTensor Normalize(0.5)",
        "resolution": 96,
        "encoder_sha256": enc_sha,
        "es_checkpoint_sha256": enc_sha,
        "es_spatial_cache": str(args.spatial_cache),
        "es_spatial_es_checkpoint_sha256": spatial_man.get("es_checkpoint_sha256"),
        "ckpt_dir": str(args.ckpt.resolve()),
        "data_root": str(args.data_root),
        "split_manifest": str(args.split),
        "payload_sha256": payload_sha,
        "created_unix": int(time.time()),
    }
    table.save_manifest(man)
    if args.verify:
        rng = np.random.default_rng(3407)
        picks = rng.choice(n, size=min(args.verify, n), replace=False)
        for idx in picks:
            key, path = jobs[int(idx)]
            live = encode_batch(model, [path], args.device)[0].astype(np.float32)
            got = np.array(arr[int(idx)], copy=True).astype(np.float32)
            if not np.allclose(live, got, atol=2e-3, rtol=2e-3):
                raise RuntimeError(f"local verify mismatch {key}")
        print(f"verify PASS n={len(picks)}", flush=True)
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
