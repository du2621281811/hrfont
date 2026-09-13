#!/usr/bin/env python3
"""G0: v0913_clean F0, 8-GPU DDP, global batch 256.

8×32=256. LR linear-scales from 1e-5 @ bs=8 → 3.2e-4.
Does not overwrite F0-CLEAN-V0913-* .
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
CLEAN_MAP = ROOT / "manifests/v0913_clean"
GPUS = "0,1,2,3,4,5,6,7"
NPROC = 8
PER_DEVICE = 32
BASE_BS = 8
BASE_LR = 1e-5
RUN_ID = "G0-F0-V0913-BS256-A-S3407"
ACCEL_PORT = "29521"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=10_000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ckpt_interval", type=int, default=2500)
    ap.add_argument("--state_interval", type=int, default=500)
    ap.add_argument("--run_id", type=str, default=RUN_ID)
    ap.add_argument("--gpus", type=str, default=GPUS)
    ap.add_argument("--resume_from", type=str, default="")
    args = ap.parse_args()

    nproc = len([x for x in args.gpus.split(",") if x.strip() != ""])
    if nproc != NPROC:
        print(f"expected {NPROC} GPUs, got {nproc}: {args.gpus}", file=sys.stderr)
        return 2

    run_id = args.run_id
    max_steps = args.max_steps
    ckpt_interval = args.ckpt_interval
    if args.smoke:
        max_steps = 20
        ckpt_interval = 20
        run_id = f"smoke-G0-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    elif not args.yes:
        print("Refusing full train without --yes (use --smoke first, or pass --yes).", file=sys.stderr)
        return 2

    global_bs = PER_DEVICE * nproc
    lr = BASE_LR * (global_bs / BASE_BS)
    out_dir = RUNS / run_id
    resume_from = args.resume_from.strip()
    if out_dir.exists() and not args.resume and not resume_from:
        print(f"output exists, refuse overwrite: {out_dir}", file=sys.stderr)
        return 2
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=False)
    if args.resume and not resume_from:
        last = out_dir / "last_state"
        if (last / "trainer_state.pt").is_file():
            resume_from = str(last)
        else:
            print(f"resume requested but no last_state: {last}", file=sys.stderr)
            return 2

    accel = "/root/miniforge3/envs/boogu/bin/accelerate"
    if not Path(accel).is_file():
        launch = [PY, "-m", "accelerate.commands.launch"]
    else:
        launch = [accel, "launch"]

    meta = {
        "run_id": run_id,
        "variant": "cn2west_f0_rsifree",
        "experiment": "G0",
        "group": "G",
        "seed": 3407,
        "gpus": args.gpus,
        "num_processes": nproc,
        "train_batch_size_per_device": PER_DEVICE,
        "gradient_accumulation_steps": 1,
        "global_batch": global_bs,
        "lr": lr,
        "lr_rule": "linear_scale from 1e-5 @ bs=8",
        "warmup": args.warmup,
        "max_steps": max_steps,
        "mixed_precision": "fp16",
        "dataset_id": "v0913_clean",
        "v0913_clean_map": str(CLEAN_MAP),
        "phase_1_ckpt_dir": str(OFFICIAL_CKPT),
        "resume_from": resume_from or None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "8×32=256. Does not overwrite F0-CLEAN-V0913-*. Next: G2, then G1+G2-PRL.",
    }
    (out_dir / "launch_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    cmd = launch + [
        "--num_processes", str(nproc),
        "--num_machines", "1",
        "--mixed_precision", "fp16",
        "--multi_gpu",
        "--main_process_port", ACCEL_PORT,
        str(VARIANT / "train.py"),
        "--seed", "3407",
        "--experience_name", run_id,
        "--output_dir", str(out_dir),
        "--data_root", str(DATA),
        "--v0913_clean_map", str(CLEAN_MAP),
        "--resolution", "96",
        "--style_image_size", "96",
        "--content_image_size", "96",
        "--train_batch_size", str(PER_DEVICE),
        "--gradient_accumulation_steps", "1",
        "--max_train_steps", str(max_steps),
        "--learning_rate", f"{lr:.8g}",
        "--lr_scheduler", "linear",
        "--lr_warmup_steps", str(args.warmup),
        "--phase_1_ckpt_dir", str(OFFICIAL_CKPT),
        "--mixed_precision", "fp16",
        "--ckpt_interval", str(ckpt_interval),
        "--state_interval", str(args.state_interval),
        "--log_interval", "50",
        "--drop_prob", "0.1",
        "--perceptual_coefficient", "0.01",
        "--offset_coefficient", "0.0",
    ]
    if resume_from:
        cmd += ["--resume_from", resume_from]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["PYTHONUNBUFFERED"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    print("META", json.dumps(meta, ensure_ascii=False), flush=True)
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(VARIANT), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
