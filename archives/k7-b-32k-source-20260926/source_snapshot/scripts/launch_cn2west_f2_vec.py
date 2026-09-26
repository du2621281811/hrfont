#!/usr/bin/env python3
"""Launch F2-VEC-MT: F2 mean-Δ + differentiable vector head, fp16, single GPU.

Design: reports/F2_VEC_MULTITASK_DESIGN_20260913.md
Does not overwrite F2-DELTARSI-A-S3407.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
VARIANT = ROOT / "code/variants/cn2west_f2_vec/FontDiffuser"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
RUNS = ROOT / "runs"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
PARENT = ROOT / "runs/F0-RSIFREE-FT-A-S3407/best"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=40_000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--run_id", default="F2-VEC-MT-A-S3407")
    ap.add_argument("--parent", default=str(PARENT))
    ap.add_argument("--es_cache", default="artifacts/f0/es_spatial_f0")
    ap.add_argument("--ec_cache", default="artifacts/f0/ec_multiscale_f0")
    ap.add_argument("--ckpt_interval", type=int, default=5000)
    ap.add_argument("--state_interval", type=int, default=1000)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    run_id = args.run_id
    if args.smoke:
        args.max_steps = 20
        args.ckpt_interval = 20
        args.state_interval = 20
        run_id = f"smoke-F2VEC-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not args.smoke and not args.yes:
        print("Refusing full train without --yes (use --smoke first, or pass --yes).", file=sys.stderr)
        return 2

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else ROOT / p

    parent, es_cache, ec_cache = resolve(args.parent), resolve(args.es_cache), resolve(args.ec_cache)
    for name in ("unet.pth", "style_encoder.pth", "content_encoder.pth"):
        if not (parent / name).is_file():
            print(f"missing parent: {parent / name}", file=sys.stderr)
            return 2
    for cache in (es_cache, ec_cache):
        if not (cache / "manifest.json").is_file():
            print(f"missing cache: {cache / 'manifest.json'}", file=sys.stderr)
            return 2

    out_dir = RUNS / run_id
    if out_dir.exists():
        print(f"output exists, refuse overwrite: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=False)
    meta = {
        "run_id": run_id,
        "arm": "F2VEC",
        "variant": "cn2west_f2_vec",
        "paper": False,
        "side_study": True,
        "do_not_overwrite": "F2-DELTARSI-A-S3407",
        "parent": str(parent),
        "es_cache": str(es_cache),
        "ec_cache": str(ec_cache),
        "seed": args.seed,
        "gpu": args.gpu,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "mixed_precision": "fp16",
        "design": "reports/F2_VEC_MULTITASK_DESIGN_20260913.md",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "launch_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "NOT_FOR_PAPER.txt").write_text(
        "F2-VEC-MT side study. Do not cite as F2. Do not use as parent.\n", encoding="utf-8"
    )

    cmd = [
        PY, str(VARIANT / "train.py"),
        "--arm", "F2VEC",
        "--rsi_source", "delta",
        "--no-support",
        "--source_drop", "0.25",
        "--seed", str(args.seed),
        "--experience_name", run_id,
        "--output_dir", str(out_dir),
        "--data_root", str(DATA),
        "--split_manifest", str(SPLIT),
        "--es_cache_path", str(es_cache),
        "--ec_cache_path", str(ec_cache),
        "--phase_1_ckpt_dir", str(parent),
        "--resolution", "96",
        "--style_image_size", "96",
        "--content_image_size", "96",
        "--train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", "1",
        "--max_train_steps", str(args.max_steps),
        "--learning_rate", str(args.lr),
        "--lr_scheduler", "linear",
        "--lr_warmup_steps", str(args.warmup),
        "--mixed_precision", "fp16",
        "--ckpt_interval", str(args.ckpt_interval),
        "--state_interval", str(args.state_interval),
        "--log_interval", str(args.log_interval),
        "--drop_prob", "0.1",
        "--perceptual_coefficient", "0.01",
        "--offset_coefficient", "0.5",
        "--vec_warmup_steps", "5000",
        "--vec_unet_scale", "0.15",
        "--vec_consist", "0.1",
        "--parity_check",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    print("META", json.dumps(meta, ensure_ascii=False), flush=True)
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(VARIANT), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
