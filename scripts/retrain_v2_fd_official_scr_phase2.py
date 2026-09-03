#!/usr/bin/env python3
"""P1b: official SCR Phase-2 A/B launcher (from E10★).

See reports/retrain_v2/FD_OFFICIAL_SCR_PHASE2_PLAN.md

Examples:
  # 50-step smoke (SCR arm only — validates sc_loss path)
  python scripts/retrain_v2_fd_official_scr_phase2.py --arm scr --smoke --gpu 3

  # full first-cut (12k); run both arms on separate GPUs
  python scripts/retrain_v2_fd_official_scr_phase2.py --arm scr  --gpu 0 --max_steps 12000
  python scripts/retrain_v2_fd_official_scr_phase2.py --arm ctrl --gpu 1 --max_steps 12000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
REPO = ROOT / "code/FontDiffuser"
OUT = ROOT / "runs_retrain_v2/fontdiffuser"
DATA = ROOT / "data/fontdiffuser_cn2cn_p253"
E10_STAR = OUT / "ft_cn2cn_p253_cont12000" / "global_step_6000"  # abs step 18000
SCR_CKPT = REPO / "ckpt" / "scr_210000.pth"

# Plan defaults (official train_phase_2.sh aligned)
DEFAULTS = {
    "lr": 1e-5,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 1000,
    "sc_coefficient": 0.01,
    "num_neg": 16,
    "perceptual_coefficient": 0.01,
    "offset_coefficient": 0.5,
    "drop_prob": 0.1,
    "resolution": 96,
    "batch_size": 4,
    "grad_accum": 4,  # eff_bs ≈ 16
    "max_steps": 12000,
    "ckpt_interval": 2000,
    "log_interval": 50,
}


def sha256_file(p: Path, nbytes: int = 0) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        if nbytes:
            h.update(f.read(nbytes))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def write_protocol(run_dir: Path, arm: str, args: argparse.Namespace, smoke: bool) -> None:
    proto = {
        "experiment": (
            "P1b_official_SCR_phase2_sc05_exploratory"
            if (getattr(args, "run_suffix", "") == "sc05" or args.sc_coefficient != DEFAULTS["sc_coefficient"])
            and arm == "scr"
            else "P1b_official_SCR_phase2"
        ),
        "arm": arm,
        "note": (
            "探索臂：更高 sc；不替代官方 sc=0.01 主对照"
            if arm == "scr" and args.sc_coefficient != DEFAULTS["sc_coefficient"]
            else None
        ),
        "smoke": smoke,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "init_ckpt": str(E10_STAR),
        "init_abs_step": 18000,
        "init_local_step": 6000,
        "init_abs_formula": "ft_cn2cn_p253@12000 + cont12000 local 6000 = 18000",
        "init_note": "E10★ abs 18000; folder name global_step_6000 is LOCAL to cont12000 — NOT abs 6000",
        "data_root": str(DATA),
        "scr_ckpt": str(SCR_CKPT) if arm == "scr" else None,
        "scr_sha256": sha256_file(SCR_CKPT) if arm == "scr" and SCR_CKPT.exists() else None,
        "scr_bytes": SCR_CKPT.stat().st_size if arm == "scr" and SCR_CKPT.exists() else None,
        "phase_2": arm == "scr",
        "hyperparams": {
            "learning_rate": args.learning_rate,
            "lr_scheduler": DEFAULTS["lr_scheduler"],
            "lr_warmup_steps": args.lr_warmup_steps,
            "sc_coefficient": args.sc_coefficient if arm == "scr" else None,
            "num_neg": args.num_neg if arm == "scr" else None,
            "perceptual_coefficient": DEFAULTS["perceptual_coefficient"],
            "offset_coefficient": DEFAULTS["offset_coefficient"],
            "drop_prob": DEFAULTS["drop_prob"],
            "train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "eff_batch": args.batch_size * args.grad_accum,
            "max_train_steps": args.max_steps,
            "ckpt_interval": args.ckpt_interval,
            "resolution": 96,
        },
        "plan": "reports/retrain_v2/FD_OFFICIAL_SCR_PHASE2_PLAN.md",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "protocol.json").write_text(json.dumps(proto, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cmd(arm: str, run_name: str, out_dir: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        PY,
        str(REPO / "train.py"),
        "--seed",
        "123",
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
        "--content_encoder_downsample_size",
        "3",
        "--content_start_channel",
        "64",
        "--style_start_channel",
        "64",
        "--train_batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_accum),
        "--max_train_steps",
        str(args.max_steps),
        "--ckpt_interval",
        str(args.ckpt_interval),
        "--log_interval",
        str(args.log_interval),
        "--learning_rate",
        str(args.learning_rate),
        "--lr_scheduler",
        DEFAULTS["lr_scheduler"],
        "--lr_warmup_steps",
        str(args.lr_warmup_steps),
        "--perceptual_coefficient",
        str(DEFAULTS["perceptual_coefficient"]),
        "--offset_coefficient",
        str(DEFAULTS["offset_coefficient"]),
        "--drop_prob",
        str(DEFAULTS["drop_prob"]),
        "--phase_1_ckpt_dir",
        str(E10_STAR),
        "--mixed_precision",
        "no",
        "--report_to",
        "tensorboard",
    ]
    if arm == "scr":
        cmd += [
            "--phase_2",
            "--scr_ckpt_path",
            str(SCR_CKPT),
            "--sc_coefficient",
            str(args.sc_coefficient),
            "--num_neg",
            str(args.num_neg),
            "--nce_layers",
            "0,1,2,3",
            "--temperature",
            "0.07",
        ]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="P1b official SCR Phase-2 launcher")
    ap.add_argument("--arm", choices=["scr", "ctrl"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="50-step path check (SCR arm recommended)")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--grad_accum", type=int, default=DEFAULTS["grad_accum"])
    ap.add_argument("--num_neg", type=int, default=DEFAULTS["num_neg"])
    ap.add_argument("--sc_coefficient", type=float, default=DEFAULTS["sc_coefficient"])
    ap.add_argument("--learning_rate", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--lr_warmup_steps", type=int, default=DEFAULTS["lr_warmup_steps"])
    ap.add_argument("--ckpt_interval", type=int, default=None)
    ap.add_argument("--log_interval", type=int, default=None)
    ap.add_argument("--run_suffix", type=str, default="", help="optional tag, e.g. smoke")
    args = ap.parse_args()

    if not (E10_STAR / "unet.pth").exists():
        print(f"ERROR missing E10★ {E10_STAR}", file=sys.stderr)
        return 2
    if not (DATA / "train/StyleImage").exists():
        print(f"ERROR missing data {DATA}", file=sys.stderr)
        return 2
    if args.arm == "scr" and not SCR_CKPT.exists():
        print(f"ERROR missing SCR {SCR_CKPT}", file=sys.stderr)
        return 2

    if args.smoke:
        args.max_steps = args.max_steps or 50
        args.ckpt_interval = args.ckpt_interval or 50
        args.log_interval = args.log_interval or 10
        args.lr_warmup_steps = min(args.lr_warmup_steps, 10)  # don't sit in warmup for smoke
        suffix = args.run_suffix or "smoke"
        run_name = f"fd_p2_{args.arm}_{suffix}"
    else:
        args.max_steps = args.max_steps or DEFAULTS["max_steps"]
        args.ckpt_interval = args.ckpt_interval or DEFAULTS["ckpt_interval"]
        args.log_interval = args.log_interval or DEFAULTS["log_interval"]
        suffix = args.run_suffix
        run_name = f"fd_p2_{args.arm}" + (f"_{suffix}" if suffix else "")

    out_dir = OUT / run_name
    write_protocol(out_dir, args.arm, args, smoke=bool(args.smoke))
    cmd = build_cmd(args.arm, run_name, out_dir, args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"

    print("PROTO", out_dir / "protocol.json", flush=True)
    print("CMD", " ".join(cmd), flush=True)
    print(f"GPU={args.gpu} arm={args.arm} steps={args.max_steps} "
          f"bs={args.batch_size}x{args.grad_accum} smoke={args.smoke}", flush=True)
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
