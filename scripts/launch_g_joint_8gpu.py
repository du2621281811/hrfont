#!/usr/bin/env python3
"""8-GPU accelerate launch for G short pilots / TC joint runs.

Keeps absolute lr (1e-5 UNet; optional local_lr / tc_lr). Global batch = 8×8=64
(same as G2 kids). New run_id only; no overwrite of existing G2/G2-RL/pilot.
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
ACCEL = "/root/miniforge3/envs/boogu/bin/accelerate"
TRAIN = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser/train.py"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
CLEAN = ROOT / "manifests/v0913_clean"
CACHE = ROOT / "artifacts/g0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--arm", required=True, choices=["F2", "F2RL", "F1"])
    ap.add_argument("--parent", required=True, type=Path)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--port", default="29540")
    ap.add_argument("--tc", action="store_true", help="enable TC-v2")
    ap.add_argument("--no-tc", action="store_true", dest="no_tc")
    ap.add_argument("--tc_cache", type=Path, default=ROOT / "artifacts/tc_v2_cache")
    ap.add_argument("--tc_head", type=Path, default=ROOT / "artifacts/tc_v2_head/tc_head.pth")
    ap.add_argument("--local_lr", type=float, default=0.0, help=">0 sets --local_learning_rate")
    ap.add_argument("--rsi", default="", help="override rsi_source; default by arm")
    args = ap.parse_args()
    if not args.yes:
        print("pass --yes", file=sys.stderr)
        return 2
    if args.tc and args.no_tc:
        print("choose --tc or --no-tc", file=sys.stderr)
        return 2
    if not args.tc and not args.no_tc:
        args.no_tc = True
    if not TRAIN.is_file():
        print(f"missing {TRAIN}", file=sys.stderr)
        return 2
    if not (args.parent / "unet.pth").is_file() and not (args.parent / "trainer_state.pt").is_file():
        print(f"bad parent {args.parent}", file=sys.stderr)
        return 2
    out = ROOT / "runs" / args.run_id
    if out.exists() and any(out.iterdir()):
        if (out / "DONE.json").is_file():
            print(f"already done: {out}")
            return 0
        print(f"output exists, refuse overwrite: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    rsi = args.rsi or ("official" if args.arm == "F1" else "delta")
    meta = {
        "run_id": args.run_id,
        "arm": args.arm,
        "parent": str(args.parent),
        "max_steps": args.max_steps,
        "num_processes": 8,
        "train_batch_size_per_device": 8,
        "global_batch": 64,
        "learning_rate": 1e-5,
        "local_learning_rate": args.local_lr or None,
        "tc_enabled": bool(args.tc),
        "lr_rule": "absolute; do not scale with batch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "8-GPU short run per user request after G queue + v0913 board",
    }
    (out / "launch_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    cmd = [
        ACCEL, "launch",
        "--num_processes", "8",
        "--num_machines", "1",
        "--mixed_precision", "fp16",
        "--multi_gpu",
        "--main_process_port", str(args.port),
        str(TRAIN),
        "--arm", args.arm,
        "--rsi_source", rsi,
        "--no-support",
        "--freeze_encoders",
        "--encoder_runtime", "cache_only",
        "--warm_start_from", str(args.parent),
        "--no-parity_check",
        "--data_root", str(DATA),
        "--split_manifest", str(SPLIT),
        "--v0913_clean_map", str(CLEAN),
        "--es_cache_path", str(CACHE / "es_spatial"),
        "--ec_cache_path", str(CACHE / "ec_multiscale"),
        "--es_local_cache_path", str(CACHE / "es_local"),
        "--resolution", "96", "--style_image_size", "96", "--content_image_size", "96",
        "--nshot_min", "1", "--nshot_max", "8",
        "--source_drop", "0.25", "--drop_prob", "0.1",
        "--train_batch_size", "8", "--gradient_accumulation_steps", "1",
        "--mixed_precision", "fp16",
        "--learning_rate", "1e-5",
        "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "200",
        "--max_train_steps", str(args.max_steps),
        # Disk tight on V100: fewer weight dumps (still keep best via interval).
        "--ckpt_interval", "2500", "--state_interval", "2500", "--best_min_step", "1000",
        "--perceptual_coefficient", "0.01", "--offset_coefficient", "0.5",
        "--max_grad_norm", "1.0",
        "--seed", "3407", "--log_interval", "100",
        "--experience_name", args.run_id,
        "--output_dir", str(out),
    ]
    if args.local_lr and args.local_lr > 0:
        cmd += ["--local_learning_rate", f"{args.local_lr:.8g}"]
    if args.tc:
        if not args.tc_cache.is_dir() or not args.tc_head.is_file():
            print(f"TC assets missing: cache={args.tc_cache} head={args.tc_head}", file=sys.stderr)
            return 3
        cmd += [
            "--tc_enabled",
            "--tc_cache_path", str(args.tc_cache),
            "--tc_head_ckpt", str(args.tc_head),
            "--tc_loss_coefficient", "0.01",
            "--tc_learning_rate", "1e-4",
        ]
    else:
        cmd += ["--no-tc_enabled"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env["PYTHONUNBUFFERED"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    print("META", json.dumps(meta, ensure_ascii=False), flush=True)
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(TRAIN.parent), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
