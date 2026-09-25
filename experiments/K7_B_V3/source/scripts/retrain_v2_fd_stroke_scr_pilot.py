#!/usr/bin/env python3
"""Launch FD stroke-SCR A/B pilot (ctrl vs stroke InfoNCE).

See reports/retrain_v2/FD_STROKE_SCR_PLAN.md
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
REPO = ROOT / "code/ours/FontDiffuser"
OUT = ROOT / "runs_retrain_v2/fontdiffuser"
DATA = ROOT / "data/fontdiffuser"
CKPT25 = OUT / "ft_cnstyle" / "global_step_25000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--arm", choices=["ctrl", "stroke"], required=True)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_neg", type=int, default=8)
    ap.add_argument("--sc_coefficient", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ckpt_interval", type=int, default=1000)
    args = ap.parse_args()

    if not (CKPT25 / "unet.pth").exists():
        raise SystemExit(f"missing {CKPT25}")
    if not (DATA / "train/StyleImage").exists():
        raise SystemExit(f"missing CN StyleImage under {DATA}")

    run_name = "fd_scr_ctrl" if args.arm == "ctrl" else "fd_scr_stroke"
    out_dir = OUT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY,
        str(REPO / "train.py"),
        "--experience_name",
        run_name,
        "--output_dir",
        str(out_dir),
        "--data_root",
        str(DATA),
        "--resolution",
        "96",
        "--style_image_size",
        "96",
        "--content_image_size",
        "96",
        "--train_batch_size",
        str(args.batch_size),
        "--max_train_steps",
        str(args.max_steps),
        "--learning_rate",
        str(args.lr),
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "200",
        "--phase_1_ckpt_dir",
        str(CKPT25),
        "--mixed_precision",
        "no",
        "--ckpt_interval",
        str(args.ckpt_interval),
        "--log_interval",
        "50",
        "--perceptual_coefficient",
        "0.01",
        "--offset_coefficient",
        "0.5",
    ]
    if args.arm == "stroke":
        cmd += [
            "--stroke_scr",
            "--stroke_scr_mode",
            "infonce",
            "--sc_coefficient",
            str(args.sc_coefficient),
            "--temperature",
            "0.07",
            "--num_neg",
            str(args.num_neg),
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
