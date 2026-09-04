#!/usr/bin/env python3
"""YAML launcher for E1c/E2/E2b Stage A (cache-only)."""
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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e2_stage_a_s3407.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if not args.yes:
        print("Refusing full training without --yes; run scripts/e2_smoke_test.py first.", file=sys.stderr)
        return 2

    cfg_path = args.config.resolve() if args.config.is_absolute() else (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run_id = cfg["experiment"]["id"]
    data, model, train = cfg["data"], cfg["model"], cfg["train"]
    delta, nshot = model["delta"], model["nshot"]
    init_dir = ROOT / cfg["init"]["checkpoint"]
    es_cache = ROOT / data["es_cache_path"]
    ec_cache = ROOT / data["ec_cache_path"]
    required = [
        init_dir / "unet.pth", init_dir / "style_encoder.pth", init_dir / "content_encoder.pth",
        es_cache / "manifest.json", es_cache / "progress.json",
        ec_cache / "manifest.json", ec_cache / "progress.json",
    ]
    for path in required:
        if not path.exists():
            print(f"missing required artifact: {path}", file=sys.stderr)
            return 2
    for progress_path in (es_cache / "progress.json", ec_cache / "progress.json"):
        payload = yaml.safe_load(progress_path.read_text(encoding="utf-8"))
        if payload.get("done") != payload.get("total"):
            print(f"incomplete cache: {progress_path} {payload}", file=sys.stderr)
            return 2
    output = Path(args.output_dir) if args.output_dir else ROOT / "runs" / run_id
    last_state = output / "last_state"
    if (output / "DONE.json").exists():
        print(f"output already completed: {output}", file=sys.stderr)
        return 0
    has_ckpt = last_state.exists() or any(output.glob("global_step_*")) if output.exists() else False
    ignore = {"logs", "watchdog.log", "watchdog_train.log"}
    if output.exists() and not (output / "STOP").exists() and not has_ckpt:
        leftover = [p.name for p in output.iterdir() if p.name not in ignore]
        if leftover:
            print(f"output exists without checkpoint (failed smoke?); remove it first: {output}", file=sys.stderr)
            return 2

    eval_refs = nshot.get("eval_refs") or ["永"]
    steps = args.max_steps or train["steps"]
    cmd = [
        sys.executable, str(Path(__file__).with_name("train.py")),
        "--config_path", str(cfg_path),
        "--experience_name", run_id, "--output_dir", str(output),
        "--data_root", str(ROOT / data["root"]),
        "--split_manifest", str(ROOT / data["split_manifest"]),
        "--excluded", *data["excluded"],
        "--es_cache_path", str(es_cache), "--ec_cache_path", str(ec_cache),
        "--phase_1_ckpt_dir", str(init_dir),
        "--rsi_source", model["rsi_source"],
        "--encoder_runtime", model.get("encoder_runtime", "cache_only"),
        "--delta_tau", str(delta["tau"]), "--delta_eps_alpha", str(delta["eps_alpha"]),
        "--delta_k_max", str(delta["k_max"]), "--delta_k_top", str(delta["k_top"]),
        "--delta_mode", delta["mode"], "--delta_drop", str(delta["drop"]),
        "--nshot_min", str(nshot["min_n"]), "--nshot_max", str(nshot["max_n"]),
        "--eval_refs", *eval_refs,
        "--seed", str(train["seed"]), "--max_train_steps", str(steps),
        "--train_batch_size", str(train["batch_size"]),
        "--gradient_accumulation_steps", str(train["accumulation"]),
        "--learning_rate", str(train["lr"]), "--lr_scheduler", train["scheduler"],
        "--lr_warmup_steps", str(train["warmup_steps"]), "--mixed_precision", "fp16",
        "--drop_prob", str(train["cfg_joint_drop"]),
        "--perceptual_coefficient", str(cfg["loss"]["perceptual"]),
        "--offset_coefficient", str(cfg["loss"]["offset"]),
        "--ckpt_interval", str(cfg["checkpoint"]["every_steps"]),
        "--state_interval", "1000",
    ]
    if last_state.exists() and (last_state / "trainer_state.pt").exists():
        cmd.extend(["--resume_from", str(last_state)])
        print(f"resuming from {last_state}", flush=True)
    cmd.append("--delta_enabled" if delta["enabled"] else "--no-delta_enabled")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=Path(__file__).parent, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
