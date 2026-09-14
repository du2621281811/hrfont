#!/usr/bin/env python3
"""Launch F1 / F2 / F3 matched arms from F0 on protocol-A data.

The three arms share one code path; `--arm` picks the single factor that changes:

  F1  rsi_source=official  support=off
  F2  rsi_source=delta     support=off   -> F1 vs F2 isolates the Delta source
  F3  rsi_source=delta     support=on    -> F2 vs F3 isolates Support

Everything else (parent ckpt, Es/Ec caches, source_drop, CFG drop, batch order, RNG,
steps, lr, schedule) is identical, so the arms cannot drift apart by configuration.

Preconditions (all fail closed):
  - F0 finished and a milestone selected  -> --parent
  - Es/Ec caches rebuilt from that F0 milestone (E1 caches are NOT reusable)
  - F3 additionally needs a support bank  -> --support_bank

Example:
  python scripts/launch_cn2west_f123.py --arm F2 --parent runs/F0-RSIFREE-FT-A-S3407/best --smoke
  python scripts/launch_cn2west_f123.py --arm F2 --parent runs/F0-RSIFREE-FT-A-S3407/best --yes
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
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
RUNS = ROOT / "runs"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"

ARMS = {
    "F1": {"rsi_source": "official", "support": False, "run_id": "F1-OFFRSI-A-S3407"},
    "F2": {"rsi_source": "delta", "support": False, "run_id": "F2-DELTARSI-A-S3407"},
    "F3": {"rsi_source": "delta", "support": True, "run_id": "F3-JOINT-DS-A-S3407"},
    "F3b": {
        "rsi_source": "delta",
        "support": True,
        "run_id": "f3b_joint_crossbank_s3407",
        "support_bank": "artifacts/f0/support_bank_f3b_stroke.json",
    },
    "F2P": {"rsi_source": "delta", "support": False, "run_id": "f2_pattn_s3407"},
    "F3bP": {
        "rsi_source": "delta",
        "support": True,
        "run_id": "f3b_pattn_s3407",
        "support_bank": "artifacts/f0/support_bank_f3b_topology.json",
    },
    "F2RL": {"rsi_source": "delta", "support": False, "run_id": "F2-RL128-A-S3407"},
    "F2PRL": {"rsi_source": "delta", "support": False, "run_id": "F2-PRL-A-S3407"},
}

CLEAN_RUN = {
    "F1": "F1-CLEAN-V0913-A-S3407",
    "F2": "F2-CLEAN-V0913-A-S3407",
    "F2RL": "F2-RL128-CLEAN-V0913-A-S3407",
    "F2PRL": "F2-PRL-CLEAN-V0913-A-S3407",
}

G_RUN = {
    "F1": "G1-F1-V0913-A-S3407",
    "F2": "G2-F2-V0913-A-S3407",
    "F2RL": "G2-RL-V0913-A-S3407",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--parent", required=True, help="F0 milestone dir (unet/style/content .pth)")
    ap.add_argument("--es_cache", default="artifacts/f0/es_spatial_f0")
    ap.add_argument("--es_local_cache", default="artifacts/f0/es_local_f0_block2_pool4")
    ap.add_argument("--ec_cache", default="artifacts/f0/ec_multiscale_f0")
    ap.add_argument("--support_bank", default="artifacts/f0/support_bank.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=40_000,
                    help="PI 2026-09-12: new arms train/eval at 40k (was 80k).")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument(
        "--lr_scheduler", default="",
        help="empty: constant_with_warmup if max_steps<=10000 else linear. Do not linear-scale AdamW lr with batch.",
    )
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--allow_high_lr", action="store_true",
                    help="Permit lr > 8e-5. Death line on V100 fp16 + per-device 32 was ~1.17e-4.")
    ap.add_argument("--seed", type=int, default=3407, help="PI freeze: single seed only")
    ap.add_argument("--source_drop", type=float, default=0.25)
    ap.add_argument("--support_drop", type=float, default=0.20)
    ap.add_argument("--support_k", type=int, default=8)
    ap.add_argument("--offset_coefficient", type=float, default=0.5)
    ap.add_argument("--ckpt_interval", type=int, default=5000)
    ap.add_argument("--state_interval", type=int, default=1000)
    ap.add_argument("--log_interval", type=int, default=100)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--dataset_id", default="", help="v0913_clean filters dirty PNGs via the frozen pair map.")
    ap.add_argument("--v0913_clean_map", default="manifests/v0913_clean")
    ap.add_argument("--smoke", action="store_true", help="20-step sanity run into runs/smoke_*")
    ap.add_argument("--yes", action="store_true", help="Required for full (non-smoke) training")
    args = ap.parse_args()

    spec = ARMS[args.arm]
    if args.dataset_id == "v0913_clean" and not args.run_id and not args.smoke:
        if args.arm not in CLEAN_RUN:
            print(f"v0913_clean has no run id for arm {args.arm}", file=sys.stderr)
            return 2
        run_id = CLEAN_RUN[args.arm]
    else:
        run_id = args.run_id or spec["run_id"]
    if args.dataset_id == "v0913_clean" and run_id == spec["run_id"] and not args.smoke:
        print(f"refuse to reuse dirty run_id {run_id} on v0913_clean", file=sys.stderr)
        return 2
    if args.smoke:
        args.max_steps = 20
        args.ckpt_interval = 20
        args.state_interval = 20
        run_id = f"smoke-{args.arm}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not args.smoke and not args.yes:
        print("Refusing full train without --yes (use --smoke first, or pass --yes).", file=sys.stderr)
        return 2
    if args.lr > 8e-5 and not args.allow_high_lr:
        print(
            f"refuse lr={args.lr:g} > 8e-5 (G0 3.2e-4 and F0-c-128 1.6e-4 both NaN'd; "
            f"death ~1.17e-4). Pass --allow_high_lr only with a written reason.",
            file=sys.stderr,
        )
        return 2
    sched = args.lr_scheduler or ("constant_with_warmup" if args.max_steps <= 10_000 else "linear")
    if args.max_steps <= 10_000 and not args.warmup:
        args.warmup = 500
    elif args.max_steps <= 10_000 and args.warmup == 5000:
        args.warmup = 500

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else ROOT / p

    parent = resolve(args.parent)
    es_cache = resolve(args.es_cache)
    ec_cache = resolve(args.ec_cache)
    clean_map = None
    if args.dataset_id == "v0913_clean":
        clean_map = resolve(args.v0913_clean_map)
        if not (clean_map / "INDEX.json").is_file():
            print(f"missing v0913_clean map: {clean_map}", file=sys.stderr)
            return 2

    for name in ("unet.pth", "style_encoder.pth", "content_encoder.pth"):
        if not (parent / name).is_file():
            print(f"missing parent weight: {parent / name}", file=sys.stderr)
            return 2
    for cache in (es_cache, ec_cache):
        if not (cache / "manifest.json").is_file():
            print(f"missing cache manifest: {cache / 'manifest.json'}\n"
                  f"Rebuild Es/Ec from the F0 milestone; E1 caches are bound to E1 encoders.",
                  file=sys.stderr)
            return 2
    es_local = resolve(args.es_local_cache)
    if args.arm in ("F2RL", "F2PRL") and not (es_local / "manifest.json").is_file():
        print(f"arm {args.arm} needs local Es cache: {es_local}", file=sys.stderr)
        return 2
    if spec["support"] and not resolve(spec.get("support_bank") or args.support_bank).is_file():
        print(f"arm {args.arm} needs a support bank: {resolve(spec.get('support_bank') or args.support_bank)}", file=sys.stderr)
        return 2
    for sub in ("train/TargetImage", "train/StyleImage", "train/ContentImage"):
        if not (DATA / sub).is_dir():
            print(f"missing data: {DATA / sub}", file=sys.stderr)
            return 2

    out_dir = RUNS / run_id
    if out_dir.exists():
        print(f"output exists, refuse overwrite: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "run_id": run_id,
        "arm": args.arm,
        "variant": "cn2west_f123_rsi",
        "rsi_block": "StyleRSIUpBlockIdentitySafe",
        "rsi_source": spec["rsi_source"],
        "support": spec["support"],
        "parent": str(parent),
        "es_cache": str(es_cache),
        "ec_cache": str(ec_cache),
        "seed": args.seed,
        "gpu": args.gpu,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch": args.batch_size * args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "lr_scheduler": sched,
        "warmup": args.warmup,
        "source_drop": args.source_drop,
        "support_drop": args.support_drop,
        "offset_coefficient": args.offset_coefficient,
        "mixed_precision": "fp16",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": args.dataset_id or "dirty_protocol_A",
        "v0913_clean_map": str(clean_map) if clean_map else "",
    }
    (out_dir / "launch_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    cmd = [
        PY, str(VARIANT / "train.py"),
        "--arm", args.arm,
        "--rsi_source", spec["rsi_source"],
        "--support" if spec["support"] else "--no-support",
        "--support_drop", str(args.support_drop),
        "--support_k", str(args.support_k),
        "--source_drop", str(args.source_drop),
        "--seed", str(args.seed),
        "--experience_name", run_id,
        "--output_dir", str(out_dir),
        "--data_root", str(DATA),
        "--split_manifest", str(SPLIT),
    ]
    if clean_map:
        cmd += ["--v0913_clean_map", str(clean_map)]
    cmd += [
        "--es_cache_path", str(es_cache),
        "--es_local_cache_path", str(es_local),
        "--ec_cache_path", str(ec_cache),
        "--phase_1_ckpt_dir", str(parent),
        "--resolution", "96",
        "--style_image_size", "96",
        "--content_image_size", "96",
        "--train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--max_train_steps", str(args.max_steps),
        "--learning_rate", str(args.lr),
        "--lr_scheduler", sched,
        "--lr_warmup_steps", str(args.warmup),
        "--mixed_precision", "fp16",
        "--ckpt_interval", str(args.ckpt_interval),
        "--state_interval", str(args.state_interval),
        "--log_interval", str(args.log_interval),
        "--drop_prob", "0.1",
        "--perceptual_coefficient", "0.01",
        "--offset_coefficient", str(args.offset_coefficient),
        "--parity_check",
    ]
    if spec["support"]:
        bank_path = spec.get("support_bank") or args.support_bank
        cmd += ["--support_bank", str(resolve(bank_path))]
        meta["support_bank"] = str(resolve(bank_path))
        if args.arm in ("F3b", "F3bP"):
            meta["support_mode"] = "f3b_preset_topology_ownfont" if "topology" in str(bank_path) else "f3b_preset_stroke_ownfont"
        else:
            meta["support_mode"] = "f3_fixed_ref8"
        if args.arm in ("F2P", "F3bP"):
            meta["style_pattn"] = True
            meta["style_token"] = "per_ref_pooled_h"
    if args.arm in ("F2P", "F3bP"):
        meta["style_pattn"] = True
        meta["style_token"] = "per_ref_pooled_h"
    if args.arm == "F2RL":
        meta["style_token"] = "global9_plus_es_block2_pool4_l128"
        meta["es_local_cache"] = str(es_local)
        meta["local_proj"] = "Linear(256,1024)"
    if args.arm == "F2PRL":
        meta["style_pattn"] = True
        meta["style_token"] = "per_ref_pooled_h_plus_es_block2_pool4_l128"
        meta["es_local_cache"] = str(es_local)
        meta["local_proj"] = "Linear(256,1024)"
        meta["note"] = "Drop mean global9 on up-path; keep down-path mean 3x3. Delta still same-char."

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    print("META", json.dumps(meta, ensure_ascii=False), flush=True)
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(VARIANT), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
