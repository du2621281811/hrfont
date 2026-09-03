#!/usr/bin/env python3
"""FontDiffuser Phase-B finetune (no SCR; CN-style protocol via StyleImage/).

Examples:
  # rebuild CN-style data (42 fonts under train/) then ft 30k from official
  python scripts/build_retrain_v2_fontdiffuser_data.py
  python scripts/retrain_v2_finetune_fontdiffuser.py --gpu 0 --max_steps 30000 --run_name ft_cnstyle

  # 253 fonts, 60k steps (fast scale-up)
  python scripts/build_retrain_v2_fontdiffuser_data.py --use-p253-list --dst data/fontdiffuser_p253
  python scripts/retrain_v2_finetune_fontdiffuser.py --gpu 0 --max_steps 60000 \\
      --data_root data/fontdiffuser_p253 --run_name ft_p253_cnstyle
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
REPO = ROOT / "code/FontDiffuser"
OUT = ROOT / "runs_retrain_v2/fontdiffuser"
CKPT = REPO / "ckpt"
DATA_DEFAULT = ROOT / "data/fontdiffuser"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--run_name", type=str, default="ft_cnstyle")
    ap.add_argument("--data_root", type=Path, default=DATA_DEFAULT)
    ap.add_argument(
        "--phase_1_ckpt_dir",
        type=Path,
        default=CKPT,
        help="Init weights. Official ckpt (default) or a prior global_step_* dir.",
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Keep effective batch ≈ batch_size * accum (scale_lr is off by default)")
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--lr_warmup_steps", type=int, default=10000)
    ap.add_argument("--ckpt_interval", type=int, default=5000)
    ap.add_argument("--rebuild_data", action="store_true", help="Rebuild data_root before train")
    ap.add_argument("--use-p253-list", action="store_true", help="With --rebuild_data, filter to 253 stems")
    args = ap.parse_args()

    data = args.data_root.resolve()
    if args.rebuild_data or not (data / "train/TargetImage").exists() or not (data / "train/StyleImage").exists():
        build_cmd = [PY, str(ROOT / "scripts/build_retrain_v2_fontdiffuser_data.py"), "--dst", str(data)]
        if args.use_p253_list:
            build_cmd.append("--use-p253-list")
        print("BUILD", " ".join(build_cmd), flush=True)
        rc = subprocess.call(build_cmd)
        if rc != 0:
            return rc
    if not (data / "train/StyleImage").exists():
        print("ERROR: missing StyleImage/ — rebuild with updated builder (CN style protocol)", file=sys.stderr)
        return 2
    if not (args.phase_1_ckpt_dir / "unet.pth").exists():
        print("missing", args.phase_1_ckpt_dir / "unet.pth", file=sys.stderr)
        return 2

    out_dir = OUT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(REPO / "train.py"),
        "--experience_name", args.run_name,
        "--output_dir", str(out_dir),
        "--data_root", str(data),
        "--resolution", "96",
        "--style_image_size", "96",
        "--content_image_size", "96",
        "--train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--max_train_steps", str(args.max_steps),
        "--learning_rate", str(args.learning_rate),
        "--lr_warmup_steps", str(args.lr_warmup_steps),
        "--phase_1_ckpt_dir", str(args.phase_1_ckpt_dir),
        "--mixed_precision", "no",
        "--ckpt_interval", str(args.ckpt_interval),
        "--log_interval", "200",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
