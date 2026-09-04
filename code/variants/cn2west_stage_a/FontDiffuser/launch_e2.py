#!/usr/bin/env python3
"""Lean YAML launcher for E2/E2b Stage A."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/e2_stage_a_s3407.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        print("Refusing full training without --yes; run scripts/e2_smoke_test.py first.", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = cfg["experiment"]["id"]
    data, model, train = cfg["data"], cfg["model"], cfg["train"]
    delta, nshot = model["delta"], model["nshot"]
    init_dir = ROOT / cfg["init"]["checkpoint"]
    cache = ROOT / data["es_cache_path"]
    for path in (init_dir / "unet.pth", init_dir / "style_encoder.pth",
                 init_dir / "content_encoder.pth", cache):
        if not path.exists():
            print(f"missing required artifact: {path}", file=sys.stderr)
            return 2
    output = ROOT / "runs" / run_id
    if output.exists():
        print(f"output exists, refuse overwrite: {output}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable, str(Path(__file__).with_name("train.py")),
        "--config_path", str(args.config),
        "--experience_name", run_id, "--output_dir", str(output),
        "--data_root", str(ROOT / data["root"]),
        "--split_manifest", str(ROOT / data["split_manifest"]),
        "--excluded", *data["excluded"],
        "--es_cache_path", str(cache), "--phase_1_ckpt_dir", str(init_dir),
        "--rsi_source", model["rsi_source"],
        "--delta_tau", str(delta["tau"]), "--delta_eps_alpha", str(delta["eps_alpha"]),
        "--delta_k_max", str(delta["k_max"]), "--delta_mode", delta["mode"],
        "--delta_drop", str(delta["drop"]),
        "--nshot_min", str(nshot["min_n"]), "--nshot_max", str(nshot["max_n"]),
        "--eval_refs", *nshot["eval_refs"],
        "--seed", str(train["seed"]), "--max_train_steps", str(train["steps"]),
        "--train_batch_size", str(train["batch_size"]),
        "--gradient_accumulation_steps", str(train["accumulation"]),
        "--learning_rate", str(train["lr"]), "--lr_scheduler", train["scheduler"],
        "--lr_warmup_steps", str(train["warmup_steps"]), "--mixed_precision", "fp16",
        "--drop_prob", str(train["cfg_joint_drop"]),
        "--perceptual_coefficient", str(cfg["loss"]["perceptual"]),
        "--offset_coefficient", str(cfg["loss"]["offset"]),
        "--ckpt_interval", str(cfg["checkpoint"]["every_steps"]),
    ]
    cmd.append("--delta_enabled" if delta["enabled"] else "--no-delta_enabled")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=Path(__file__).parent, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
