#!/usr/bin/env python3
"""Launch G-RL-pilot (stage 1a): warm-start F2RL from clean G2, +5k, local_lr=1e-4.

Spec: reports/G_STYLE_COMPLETION_PLAN_20260914.md §3/§4
Handoff: reports/TC_V2_EXECUTION_HANDOFF_20260914.md §4 (G-RL-pilot bullet)
Does not enable TC. Does not overwrite G2 / G0b.
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
TRAIN = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser/train.py"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
CLEAN = ROOT / "manifests/v0913_clean"
CACHE = ROOT / "artifacts/g0"
G2 = ROOT / "runs/G2-F2-V0913-A-S3407"
RUN_ID = "G-RL-pilot-V0913-A-S3407"
MAX_STEPS = 5000


def pick_g2_ckpt() -> Path:
    for name in ("global_step_10000", "best", "last_state"):
        d = G2 / name
        if (d / "unet.pth").is_file() or (d / "trainer_state.pt").is_file():
            return d
    steps = sorted(G2.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]), reverse=True)
    for d in steps:
        if (d / "unet.pth").is_file() or (d / "trainer_state.pt").is_file():
            return d
    raise FileNotFoundError(f"no G2 checkpoint under {G2}")


def main() -> int:
    ap = argparse.ArgumentParser(description="G-RL-pilot: F2RL warm-start from G2, +2k/+5k")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2-step smoke only")
    ap.add_argument("--max_steps", type=int, default=MAX_STEPS)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--parent", type=str, default="")
    ap.add_argument("--run_id", type=str, default=RUN_ID)
    args = ap.parse_args()

    if not args.yes and not args.smoke:
        print("pass --yes or --smoke", file=sys.stderr)
        return 2
    if not TRAIN.is_file():
        print(f"missing {TRAIN}", file=sys.stderr)
        return 2
    # require warm_start support in train.py
    text = TRAIN.read_text(encoding="utf-8", errors="replace")
    if "warm_start_from" not in text or "local_learning_rate" not in text:
        print("train.py missing warm_start_from / local_learning_rate; wait for git sync", file=sys.stderr)
        return 3

    parent = Path(args.parent) if args.parent else pick_g2_ckpt()
    max_steps = 2 if args.smoke else args.max_steps
    run_id = args.run_id if not args.smoke else f"smoke-{RUN_ID}"
    out = ROOT / "runs" / run_id
    if out.exists() and any(out.iterdir()) and not args.smoke:
        # allow resume of same pilot via last_state only if STOP not set — refuse overwrite of foreign content
        if (out / "DONE.json").is_file():
            print(f"already done: {out}", file=sys.stderr)
            return 0
        if not (out / "last_state").exists() and not (out / "heartbeat.json").exists():
            print(f"output exists, refuse overwrite: {out}", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "alias": "G-RL-pilot",
        "arm": "F2RL",
        "parent": str(parent),
        "max_steps": max_steps,
        "learning_rate": 1e-5,
        "local_learning_rate": 1e-4,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 200,
        "tc_enabled": False,
        "warm_start_from": str(parent),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan": "reports/G_STYLE_COMPLETION_PLAN_20260914.md",
        "handoff": "reports/TC_V2_EXECUTION_HANDOFF_20260914.md §4",
    }
    (out / "launch_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cmd = [
        PY, str(TRAIN),
        "--arm", "F2RL", "--rsi_source", "delta", "--no-support",
        "--freeze_encoders", "--encoder_runtime", "cache_only",
        "--warm_start_from", str(parent), "--no-parity_check",
        "--no-tc_enabled",
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
        "--local_learning_rate", "1e-4",
        "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "200",
        "--max_train_steps", str(max_steps),
        "--ckpt_interval", "1000", "--state_interval", "1000", "--best_min_step", "1000",
        "--perceptual_coefficient", "0.01", "--offset_coefficient", "0.5",
        "--max_grad_norm", "1.0",
        "--seed", "3407", "--log_interval", "100",
        "--experience_name", run_id,
        "--output_dir", str(out),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"
    print("META", json.dumps(meta, ensure_ascii=False), flush=True)
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(TRAIN.parent), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
