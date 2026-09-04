#!/usr/bin/env python3
"""Build resume-safe Es spatial + Ec multi-scale caches from E1@100k.

Usage:
  python scripts/hrfont_build_e1_caches.py --which es --device cuda --gpu 2
  python scripts/hrfont_build_e1_caches.py --which ec --device cuda --gpu 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_stage_a/FontDiffuser"
sys.path.insert(0, str(ROOT / "scripts"))
from hrfont_feature_cache import (  # noqa: E402
    EC_SHAPES, ES_POOLED, ES_SPATIAL, MemmapTable, key_ec, key_es, sha256_file,
)


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


def load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stems = payload.get("stems", payload)
    return {key: sorted(stems[key]) for key in ("train", "val", "test")}


def load_encoder(kind: str, ckpt_dir: Path, device: str):
    sys.path.insert(0, str(VARIANT))
    if kind == "es":
        from src.modules.style_encoder import StyleEncoder
        model = StyleEncoder(G_ch=64, resolution=96)
        weights = ckpt_dir / "style_encoder.pth"
    else:
        from src.modules.content_encoder import ContentEncoder
        model = ContentEncoder(G_ch=64, resolution=96)
        weights = ckpt_dir / "content_encoder.pth"
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, sha256_file(weights)


def collect_es_jobs(data_root: Path, split: dict) -> list[tuple[str, Path]]:
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


def collect_ec_jobs(data_root: Path, split: dict) -> list[tuple[str, Path]]:
    jobs = []
    train_font = split["train"][0]
    targets = sorted(char_map(data_root / "train" / "TargetImage" / train_font, train_font))
    if len(targets) != 295:
        raise RuntimeError(f"target charset {len(targets)} expected 295")
    content_dir = data_root / "train" / "ContentImage"
    for cp in targets:
        path = content_dir / f"{cp}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        jobs.append((key_ec("content", "", cp), path))
    for font in split["train"]:
        mapping = char_map(data_root / "train" / "TargetImage" / font, font)
        if sorted(mapping) != targets:
            raise RuntimeError(f"target charset mismatch train/{font}")
        for cp in targets:
            jobs.append((key_ec("target", font, cp), mapping[cp]))
    for part, fonts in split.items():
        for font in fonts:
            mapping = char_map(data_root / part / "StyleImage" / font, font)
            for cp, path in mapping.items():
                jobs.append((key_ec("style", font, cp), path))
    return jobs


def encode_es_batch(model, paths: list[Path], device: str) -> tuple[np.ndarray, np.ndarray]:
    images = torch.stack([load_rgb(path) for path in paths]).to(device)
    with torch.no_grad():
        spatial, pooled, _ = model(images)
        pooled = F.normalize(pooled.float(), dim=1)
    return spatial.detach().half().cpu().numpy(), pooled.detach().half().cpu().numpy()


def encode_ec_batch(model, paths: list[Path], device: str) -> list[np.ndarray]:
    images = torch.stack([load_rgb(path) for path in paths]).to(device)
    with torch.no_grad():
        final, residuals = model(images)
        scales = list(residuals) + [final]
    if len(scales) != len(EC_SHAPES):
        raise RuntimeError(f"Ec scale count {len(scales)} != {len(EC_SHAPES)}")
    out = []
    for tensor, shape in zip(scales, EC_SHAPES):
        if tuple(tensor.shape[1:]) != shape:
            raise RuntimeError(f"Ec shape {tuple(tensor.shape[1:])} != {shape}")
        out.append(tensor.detach().half().cpu().numpy())
    return out


def run_loop(kind: str, jobs: list[tuple[str, Path]], model, device: str, out_dir: Path,
             batch_size: int, encoder_sha: str, extra_manifest: dict) -> None:
    table = MemmapTable(out_dir)
    keys = [key for key, _ in jobs]
    if table.keys_path.exists():
        existing = table.load_keys()
        if existing != keys:
            raise RuntimeError(f"{out_dir} keys mismatch; delete directory to rebuild")
    else:
        table.set_keys(keys)
    n = len(keys)
    mode = "r+" if (out_dir / ("spatial.dat" if kind == "es" else "s0.dat")).exists() else "w+"
    if kind == "es":
        spatial = table.open_array("spatial", (n, *ES_SPATIAL), mode)
        pooled = table.open_array("pooled", (n, *ES_POOLED), mode)
        arrays = None
    else:
        arrays = [table.open_array(f"s{i}", (n, *shape), mode) for i, shape in enumerate(EC_SHAPES)]
        spatial = pooled = None
    start = table.load_progress()
    print(f"{kind}: resume {start}/{n} device={device} batch={batch_size}", flush=True)
    index = start
    current_bs = batch_size
    while index < n:
        chunk = jobs[index:index + current_bs]
        paths = [path for _, path in chunk]
        try:
            if kind == "es":
                spat, pool = encode_es_batch(model, paths, device)
                spatial[index:index + len(chunk)] = spat
                pooled[index:index + len(chunk)] = pool
            else:
                scales = encode_ec_batch(model, paths, device)
                for array, values in zip(arrays, scales):
                    array[index:index + len(chunk)] = values
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and current_bs > 1:
                torch.cuda.empty_cache()
                current_bs = max(1, current_bs // 2)
                print(f"{kind}: OOM, retry batch={current_bs}", flush=True)
                continue
            raise
        index += len(chunk)
        if index % 1024 == 0 or index == n:
            table.flush()
            table.save_progress(index, n)
            print(f"{kind}: {index}/{n} ({100.0 * index / n:.1f}%)", flush=True)
    table.flush()
    table.save_progress(n, n)
    payload = {
        "kind": kind,
        "entries": n,
        "encoder_sha256": encoder_sha,
        "dtype": "float16",
        "created_unix": int(time.time()),
        **extra_manifest,
    }
    if kind == "es":
        payload["spatial_shape"] = [n, *ES_SPATIAL]
        payload["pooled_shape"] = [n, *ES_POOLED]
        payload["es_checkpoint_sha256"] = encoder_sha
    else:
        payload["scale_shapes"] = [[n, *shape] for shape in EC_SHAPES]
        payload["ec_checkpoint_sha256"] = encoder_sha
    table.save_manifest(payload)
    print(f"{kind}: wrote {out_dir}", flush=True)


def verify_es(model, jobs, cache_dir: Path, device: str, n_check: int, seed: int = 3407) -> None:
    from hrfont_feature_cache import EsCache
    cache = EsCache(cache_dir)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(jobs), size=min(n_check, len(jobs)), replace=False)
    for idx in picks:
        key, path = jobs[int(idx)]
        _, split, font, cp = key.split("|")
        spat, pool = encode_es_batch(model, [path], device)
        got_s = cache.spatial_tensor(split, font, cp).numpy()
        got_p = cache.pooled_tensor(split, font, cp).numpy()
        if not np.allclose(spat[0].astype(np.float32), got_s, atol=2e-3, rtol=2e-3):
            raise RuntimeError(f"Es spatial mismatch {key}")
        if not np.allclose(pool[0].astype(np.float32), got_p, atol=2e-3, rtol=2e-3):
            raise RuntimeError(f"Es pooled mismatch {key}")
    print(f"es verify: PASS n={len(picks)}", flush=True)


def verify_ec(model, jobs, cache_dir: Path, device: str, n_check: int, seed: int = 3407) -> None:
    from hrfont_feature_cache import EcCache
    cache = EcCache(cache_dir)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(jobs), size=min(n_check, len(jobs)), replace=False)
    for idx in picks:
        key, path = jobs[int(idx)]
        _, role, font, cp = key.split("|")
        scales = encode_ec_batch(model, [path], device)
        cached = cache.features(role, font, cp)
        for a, b in zip(scales, cached):
            if not np.allclose(a[0].astype(np.float32), b.squeeze(0).numpy(), atol=2e-3, rtol=2e-3):
                raise RuntimeError(f"Ec mismatch {key} scale {tuple(b.shape)}")
    print(f"ec verify: PASS n={len(picks)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=("es", "ec", "both"), default="both")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2")
    parser.add_argument("--split", type=Path, default=ROOT / "manifests/split_v3_228_16_16.json")
    parser.add_argument("--ckpt-dir", type=Path,
                        default=ROOT / "runs/E1-FTV2-A-S3407/global_step_100000")
    parser.add_argument("--es-out", type=Path, default=ROOT / "artifacts/e2/es_spatial_e1_100k")
    parser.add_argument("--ec-out", type=Path, default=ROOT / "artifacts/e2/ec_multiscale_e1_100k")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verify-n", type=int, default=8)
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.device = "cuda"
    split = load_split(args.split)
    extra = {
        "data_root": str(args.data_root),
        "split_manifest": str(args.split),
        "ckpt_dir": str(args.ckpt_dir),
        "split_counts": {k: len(v) for k, v in split.items()},
    }
    kinds = ("es", "ec") if args.which == "both" else (args.which,)
    for kind in kinds:
        jobs = collect_es_jobs(args.data_root, split) if kind == "es" else collect_ec_jobs(args.data_root, split)
        out = args.es_out if kind == "es" else args.ec_out
        attempts = 0
        while True:
            attempts += 1
            try:
                model, sha = load_encoder(kind, args.ckpt_dir, args.device)
                run_loop(kind, jobs, model, args.device, out, args.batch_size, sha, extra)
                if args.verify_n:
                    (verify_es if kind == "es" else verify_ec)(
                        model, jobs, out, args.device, args.verify_n)
                break
            except Exception:
                traceback.print_exc()
                if attempts >= 5:
                    raise
                print(f"{kind}: retry {attempts}/5 in 15s", flush=True)
                time.sleep(15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
