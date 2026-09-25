#!/usr/bin/env python3
"""Launch F0 RSI-free FT (cn2west_f0_rsifree) from official P1 on protocol-A data.

Matches DESIGN / JOINT_PREP (2026-09-05):
  StyleUpBlockNoRSI · P1 init drop RSI/DCN · offset_coefficient=0
  seed=3407 · A/train228 · bs=8 accum=1 · steps=100k · lr=1e-5 · warmup=5k
  fp16 · SCR off · CFG drop=0.1 · no Resize

Example:
  python scripts/launch_cn2west_f0_rsifree.py --smoke
  python scripts/launch_cn2west_f0_rsifree.py --yes
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
VARIANT = ROOT / "code/variants/cn2west_f0_rsifree/FontDiffuser"
OFFICIAL_CKPT = ROOT / "code/official/FontDiffuser/ckpt"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
RUNS = ROOT / "runs"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2, help="Physical GPU (default 2: most free under zombie VRAM)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--ckpt_interval", type=int, default=5000)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--run_id", type=str, default="F0-RSIFREE-FT-A-S3407")
    ap.add_argument("--dataset_id", type=str, default="",
                    help="Use v0913_clean to filter dirty PNGs via manifests/v0913_clean.")
    ap.add_argument("--v0913_clean_map", type=Path,
                    default=ROOT / "manifests/v0913_clean")
    ap.add_argument("--data_root", type=Path, default=DATA)
    ap.add_argument("--phase_1_ckpt_dir", type=Path, default=OFFICIAL_CKPT)
    ap.add_argument("--smoke", action="store_true", help="20-step sanity run into runs/smoke_*")
    ap.add_argument("--yes", action="store_true", help="Required for full (non-smoke) training")
    args = ap.parse_args()

    if args.smoke:
        args.max_steps = 20
        args.ckpt_interval = 20
        args.run_id = f"smoke-F0-RSIFREE-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    if not args.smoke and not args.yes:
        print("Refusing full train without --yes (use --smoke first, or pass --yes).", file=sys.stderr)
        return 2

    for name in ("unet.pth", "style_encoder.pth", "content_encoder.pth"):
        p = args.phase_1_ckpt_dir / name
        if not p.is_file():
            print(f"missing P1 weight: {p}", file=sys.stderr)
            return 2
    for sub in ("train/TargetImage", "train/StyleImage", "train/ContentImage"):
        if not (args.data_root / sub).is_dir():
            print(f"missing data: {args.data_root / sub}", file=sys.stderr)
            return 2
    if not SPLIT.is_file():
        print(f"missing split manifest: {SPLIT}", file=sys.stderr)
        return 2
    clean_map = None
    if args.dataset_id == "v0913_clean":
        clean_map = args.v0913_clean_map
        if not clean_map.is_absolute():
            clean_map = ROOT / clean_map
        index = clean_map / "INDEX.json"
        weights = clean_map / "sample_weights.json"
        pairs = clean_map / "pairs_train.tsv"
        if not index.is_file() or not weights.is_file() or not pairs.is_file():
            print(f"missing v0913_clean map: {clean_map}", file=sys.stderr)
            return 2
        if args.run_id == "F0-RSIFREE-FT-A-S3407" and not args.smoke:
            print("refuse to reuse dirty F0 run_id on v0913_clean", file=sys.stderr)
            return 2

    out_dir = RUNS / args.run_id
    if out_dir.exists():
        print(f"output exists, refuse overwrite: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "run_id": args.run_id,
        "variant": "cn2west_f0_rsifree",
        "experiment": "F0-RSIFREE-FT",
        "seed": args.seed,
        "gpu": args.gpu,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch": args.batch_size * args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "warmup": args.warmup,
        "mixed_precision": "fp16",
        "scr": False,
        "offset_coefficient": 0.0,
        "up_block": "StyleUpBlockNoRSI",
        "data_root": str(args.data_root),
        "split": str(SPLIT),
        "dataset_id": args.dataset_id or "dirty_protocol_A",
        "v0913_clean_map": str(clean_map) if clean_map else "",
        "phase_1_ckpt_dir": str(args.phase_1_ckpt_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Joint mainline F0; reuse protocol-A renders; drop RSI/DCN from P1.",
    }
    (out_dir / "launch_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    cmd = [
        PY, str(VARIANT / "train.py"),
        "--seed", str(args.seed),
        "--experience_name", args.run_id,
        "--output_dir", str(out_dir),
        "--data_root", str(args.data_root),
    ]
    if clean_map:
        cmd += ["--v0913_clean_map", str(clean_map)]
    cmd += [
        "--resolution", "96",
        "--style_image_size", "96",
        "--content_image_size", "96",
        "--train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--max_train_steps", str(args.max_steps),
        "--learning_rate", str(args.lr),
        "--lr_scheduler", "linear",
        "--lr_warmup_steps", str(args.warmup),
        "--phase_1_ckpt_dir", str(args.phase_1_ckpt_dir),
        "--mixed_precision", "fp16",
        "--ckpt_interval", str(args.ckpt_interval),
        "--log_interval", str(args.log_interval),
        "--drop_prob", "0.1",
        "--perceptual_coefficient", "0.01",
        "--offset_coefficient", "0.0",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    print("META", json.dumps(meta, ensure_ascii=False), flush=True)
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(VARIANT), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
