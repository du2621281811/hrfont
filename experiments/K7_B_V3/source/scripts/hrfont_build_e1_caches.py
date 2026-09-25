#!/usr/bin/env python3
"""Build resume-safe Es spatial + Ec multi-scale caches.

Usage:
  python scripts/hrfont_build_e1_caches.py --which es --device cuda --gpu 2
  python scripts/hrfont_build_e1_caches.py --which ec --device cuda --gpu 3
  # G0 8-GPU path (orchestrated by rebuild_g0_caches.py):
  #   --mode init | shard --begin i --end j | finalize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_LOAD_EXEC = ThreadPoolExecutor(max_workers=8)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_stage_a/FontDiffuser"
sys.path.insert(0, str(ROOT / "scripts"))
from hrfont_feature_cache import (  # noqa: E402
    COSINE_DTYPE, EC_SHAPES, ES_POOLED, ES_SPATIAL, EsCache, MemmapTable,
    key_ec, key_es, sha256_file,
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


def load_rgb_batch(paths: list[Path]) -> torch.Tensor:
    if len(paths) <= 2:
        return torch.stack([load_rgb(path) for path in paths])
    return torch.stack(list(_LOAD_EXEC.map(load_rgb, paths)))


def encode_es_batch(model, paths: list[Path], device: str) -> tuple[np.ndarray, np.ndarray]:
    images = load_rgb_batch(paths).to(device)
    with torch.no_grad():
        spatial, pooled, _ = model(images)
        pooled = F.normalize(pooled.float(), dim=1)
    return spatial.detach().half().cpu().numpy(), pooled.detach().half().cpu().numpy()


def encode_ec_batch(model, paths: list[Path], device: str) -> list[np.ndarray]:
    images = load_rgb_batch(paths).to(device)
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


def _marker(kind: str, out_dir: Path) -> Path:
    return out_dir / ("spatial.dat" if kind == "es" else "s0.dat")


def shard_progress_path(out_dir: Path, begin: int, end: int) -> Path:
    return out_dir / f"progress_shard_{begin}_{end}.json"


def init_tables(kind: str, jobs: list[tuple[str, Path]], out_dir: Path) -> int:
    table = MemmapTable(out_dir)
    keys = [key for key, _ in jobs]
    if table.keys_path.exists():
        existing = table.load_keys()
        if existing != keys:
            raise RuntimeError(f"{out_dir} keys mismatch; delete directory to rebuild")
    else:
        table.set_keys(keys)
    n = len(keys)
    if _marker(kind, out_dir).exists():
        print(f"{kind}: init skip, arrays exist n={n}", flush=True)
        return n
    if kind == "es":
        table.open_array("spatial", (n, *ES_SPATIAL), "w+")
        table.open_array("pooled", (n, *ES_POOLED), "w+")
    else:
        for i, shape in enumerate(EC_SHAPES):
            table.open_array(f"s{i}", (n, *shape), "w+")
    table.flush()
    table.save_progress(0, n)
    print(f"{kind}: init {out_dir} n={n}", flush=True)
    return n


def write_kind_manifest(kind: str, table: MemmapTable, n: int, encoder_sha: str, extra_manifest: dict) -> None:
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


def run_loop(kind: str, jobs: list[tuple[str, Path]], model, device: str, out_dir: Path,
             batch_size: int, encoder_sha: str, extra_manifest: dict,
             begin: int = 0, end: int = 0, write_manifest: bool = True) -> None:
    table = MemmapTable(out_dir)
    keys = [key for key, _ in jobs]
    if table.keys_path.exists():
        existing = table.load_keys()
        if existing != keys:
            raise RuntimeError(f"{out_dir} keys mismatch; delete directory to rebuild")
    else:
        table.set_keys(keys)
    n = len(keys)
    stop = end if end > 0 else n
    if begin < 0 or stop > n or begin > stop:
        raise RuntimeError(f"bad range begin={begin} end={stop} n={n}")
    is_shard = begin > 0 or stop < n or not write_manifest
    mode = "r+" if _marker(kind, out_dir).exists() else "w+"
    if is_shard and mode != "r+":
        raise RuntimeError(f"{kind} shard requires init first: {out_dir}")
    if kind == "es":
        spatial = table.open_array("spatial", (n, *ES_SPATIAL), mode)
        pooled = table.open_array("pooled", (n, *ES_POOLED), mode)
        arrays = None
    else:
        arrays = [table.open_array(f"s{i}", (n, *shape), mode) for i, shape in enumerate(EC_SHAPES)]
        spatial = pooled = None
    shard_path = shard_progress_path(out_dir, begin, stop)
    if is_shard:
        start = begin
        if shard_path.exists():
            done = int(json.loads(shard_path.read_text(encoding="utf-8")).get("done", begin))
            if begin <= done <= stop:
                start = done
    else:
        start = table.load_progress()
        if start < begin:
            start = begin
    print(
        f"{kind}: range {start}:{stop}/{n} device={device} batch={batch_size} shard={is_shard}",
        flush=True,
    )
    index = start
    current_bs = batch_size
    t0 = time.time()
    while index < stop:
        chunk = jobs[index:min(index + current_bs, stop)]
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
        if index % 1024 == 0 or index == stop:
            table.flush()
            if is_shard:
                shard_path.write_text(
                    json.dumps({"done": index, "begin": begin, "end": stop, "total": n}) + "\n",
                    encoding="utf-8",
                )
            else:
                table.save_progress(index, n)
            rate = (index - start) / max(1e-6, time.time() - t0)
            print(
                f"{kind}: {index}/{stop} of {n} ({100.0 * index / n:.1f}%) {rate:.1f}/s",
                flush=True,
            )
    table.flush()
    if is_shard:
        shard_path.write_text(
            json.dumps({"done": stop, "begin": begin, "end": stop, "total": n}) + "\n",
            encoding="utf-8",
        )
    else:
        table.save_progress(n, n)
    if write_manifest:
        write_kind_manifest(kind, table, n, encoder_sha, extra_manifest)
        print(f"{kind}: wrote {out_dir}", flush=True)
    else:
        print(f"{kind}: shard done {begin}:{stop} -> {out_dir}", flush=True)


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


def build_cosine_table(es_dir: Path, split: dict, out_path: Path) -> None:
    cache = EsCache(es_dir)
    manifest_sha = cache.manifest.get("es_checkpoint_sha256")
    if not manifest_sha:
        raise RuntimeError("Es manifest has no recorded es_checkpoint_sha256")
    ckpt = Path(cache.manifest.get("ckpt_dir", "")) / "style_encoder.pth"
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.is_file() or sha256_file(ckpt) != manifest_sha:
        raise RuntimeError("Es cache recorded SHA does not match its style_encoder checkpoint")
    fonts = sorted(split["train"])
    first = fonts[0]
    chars = [key.split("|")[3] for key in cache.table.keys if key.startswith(f"es|train|{first}|")]
    if len(fonts) != 228 or len(chars) != 338:
        raise RuntimeError(f"cosine dimensions must be 228x338, got {len(fonts)}x{len(chars)}")
    indices = np.asarray([[cache._row("train", font, cp) for cp in chars] for font in fonts])
    pooled = np.asarray(cache.pooled[indices], dtype=np.float32)
    pooled /= np.linalg.norm(pooled, axis=2, keepdims=True).clip(min=1e-12)
    shape = (len(fonts), len(chars), len(fonts))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = np.memmap(out_path, dtype=COSINE_DTYPE, mode="w+", shape=shape)
    max_error = 0.0
    for ci in range(len(chars)):
        cosine = pooled[:, ci] @ pooled[:, ci].T
        stored = cosine.astype(COSINE_DTYPE)
        table[:, ci, :] = stored
        max_error = max(max_error, float(np.max(np.abs(cosine - stored.astype(np.float32)))))
    table.flush()
    meta = {
        "version": 1, "dtype": "float16", "shape": list(shape),
        "layout": "C[query_font,style_char,library_font]",
        "fonts": fonts, "library_fonts": fonts, "chars": chars,
        "font_to_row": {font: i for i, font in enumerate(fonts)},
        "char_to_col": {cp: i for i, cp in enumerate(chars)},
        "es_cache_sha256": sha256_file(cache.pooled_path),
        "es_checkpoint_sha256": manifest_sha,
        "build_params": {"normalize_dtype": "float32", "matmul_dtype": "float32"},
        "fp32_to_fp16_max_abs_error": max_error,
        "created_unix": int(time.time()),
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"cosine: wrote {out_path} shape={shape} max_abs_error={max_error:.8g}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=("es", "ec", "both", "cosine"), default="both")
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2")
    parser.add_argument("--split", type=Path, default=ROOT / "manifests/split_v3_228_16_16.json")
    parser.add_argument("--ckpt-dir", type=Path,
                        default=ROOT / "runs/E1-FTV2-A-S3407/global_step_100000")
    parser.add_argument("--es-out", type=Path, default=ROOT / "artifacts/e2/es_spatial_e1_100k")
    parser.add_argument("--ec-out", type=Path, default=ROOT / "artifacts/e2/ec_multiscale_e1_100k")
    parser.add_argument("--cosine-out", type=Path,
                        default=ROOT / "artifacts/e2/es_cosine_e1_100k.f16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verify-n", type=int, default=8)
    parser.add_argument("--mode", choices=("full", "init", "shard", "finalize"), default="full")
    parser.add_argument("--begin", type=int, default=0, help="inclusive start for shard mode")
    parser.add_argument("--end", type=int, default=0, help="exclusive end; 0 = all remaining")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.device = "cuda"
    split = load_split(args.split)
    if args.which == "cosine":
        build_cosine_table(args.es_out, split, args.cosine_out)
        return 0
    extra = {
        "data_root": str(args.data_root),
        "split_manifest": str(args.split),
        "ckpt_dir": str(args.ckpt_dir),
        "split_counts": {k: len(v) for k, v in split.items()},
    }
    kinds = ("es", "ec") if args.which == "both" else (args.which,)
    if args.mode == "shard" and args.which == "both":
        raise SystemExit("shard mode requires --which es or --which ec")
    for kind in kinds:
        jobs = collect_es_jobs(args.data_root, split) if kind == "es" else collect_ec_jobs(args.data_root, split)
        out = args.es_out if kind == "es" else args.ec_out
        if args.mode == "init":
            init_tables(kind, jobs, out)
            continue
        attempts = 0
        while True:
            attempts += 1
            try:
                if args.mode == "finalize":
                    table = MemmapTable(out)
                    table.load_keys()
                    if not args.ckpt_dir.is_dir():
                        raise RuntimeError(f"missing ckpt {args.ckpt_dir}")
                    sha = sha256_file(
                        args.ckpt_dir / ("style_encoder.pth" if kind == "es" else "content_encoder.pth")
                    )
                    table.save_progress(len(jobs), len(jobs))
                    write_kind_manifest(kind, table, len(jobs), sha, extra)
                    if args.verify_n:
                        model, enc_sha = load_encoder(kind, args.ckpt_dir, args.device)
                        if enc_sha != sha:
                            raise RuntimeError("finalize encoder sha mismatch")
                        (verify_es if kind == "es" else verify_ec)(
                            model, jobs, out, args.device, args.verify_n)
                    print(f"{kind}: finalize {out}", flush=True)
                    break
                model, sha = load_encoder(kind, args.ckpt_dir, args.device)
                run_loop(
                    kind, jobs, model, args.device, out, args.batch_size, sha, extra,
                    begin=args.begin, end=args.end,
                    write_manifest=(args.mode == "full"),
                )
                if args.mode == "full" and args.verify_n:
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
