#!/usr/bin/env python3
"""Orchestrate matched F0/F2 @ v0913_clean (does not overwrite dirty runs).

Subcommands:
  check      static correctness checks
  status     refresh STATUS.md
  smoke-f0   20-step F0 smoke
  train-f0   F0-CLEAN 100k
  scan-f0    val16 milestone pick + best symlink
  cache      Es/Ec under artifacts/f0_clean_v0913
  smoke-f2   20-step F2 smoke
  train-f2   F2-CLEAN 80k from clean F0 best + clean caches

Contract: reports/EXPERIMENT_F0F2_CLEAN_V0913_20260914.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
CLEAN_MAP = ROOT / "manifests/v0913_clean"
F0_ID = "F0-CLEAN-V0913-A-S3407"
F2_ID = "F2-CLEAN-V0913-A-S3407"
F0_RUN = ROOT / "runs" / F0_ID
F2_RUN = ROOT / "runs" / F2_ID
CACHE = ROOT / "artifacts/f0_clean_v0913"
DIRTY_F0 = ROOT / "runs/F0-RSIFREE-FT-A-S3407"
DIRTY_F2 = ROOT / "runs/F2-DELTARSI-A-S3407"
DIRTY_ES = ROOT / "artifacts/f0/es_spatial_f0"
DIRTY_EC = ROOT / "artifacts/f0/ec_multiscale_f0"


def run(cmd: list[str]) -> int:
    print("CMD", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def refresh_status() -> None:
    run([PY, str(ROOT / "scripts/status_f0f2_clean_v0913.py")])


def cmd_check() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for p in (
        CLEAN_MAP / "INDEX.json",
        CLEAN_MAP / "pairs_train.tsv",
        CLEAN_MAP / "sample_weights.json",
        CLEAN_MAP / "donor_train.json",
        ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2/train/TargetImage",
        ROOT / "code/official/FontDiffuser/ckpt/unet.pth",
        DIRTY_F0 / "best" / "unet.pth",
        DIRTY_F2 / "global_step_40000" / "unet.pth",
        DIRTY_ES / "manifest.json",
        DIRTY_EC / "manifest.json",
    ):
        if not p.exists():
            errors.append(f"missing required: {p}")

    # dataset load smoke
    code = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/root/projects/hrfont')
sys.path.insert(0,str(ROOT/'code/variants/cn2west_f0_rsifree/FontDiffuser'))
from dataset.font_dataset import FontDataset
args=SimpleNamespace(
  data_root=str(ROOT/'data/fontdiffuser-p253-t295-s338-cn2west-v2'),
  resolution=96,
  v0913_clean_map=str(ROOT/'manifests/v0913_clean'),
)
ds=FontDataset(args=args, phase='train', transforms=None, scr=False)
assert len(ds)==56429, len(ds)
assert len(ds.style_to_images)==223, len(ds.style_to_images)
assert ds.sample_weights is not None and abs(sum(ds.sample_weights)-1)<1e-6
print('dataset_ok', len(ds), len(ds.style_to_images))
"""
    r = subprocess.run([PY, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        errors.append(f"dataset smoke failed: {r.stderr[-500:]}")
    else:
        print(r.stdout.strip(), flush=True)

    # donor excludes
    donors = json.loads((CLEAN_MAP / "donor_train.json").read_text(encoding="utf-8"))
    exc = {"FZBenMWYJW", "FZCHYJW", "FZGuangHTJW-H", "FZHeJYSXZJW", "FZZhuoQHJW"}
    all_d = set(donors["latin"]) | set(donors["kana"]) | set(donors["bopomofo"])
    if exc & all_d:
        errors.append(f"exclude fonts still in donors: {sorted(exc & all_d)}")
    if len(donors["latin"]) != 223:
        warnings.append(f"donor latin n={len(donors['latin'])} expected 223")

    # overwrite guards
    for p in (DIRTY_F0, DIRTY_F2, DIRTY_ES, DIRTY_EC):
        if not p.exists():
            warnings.append(f"dirty baseline missing (compare later will fail): {p}")

    # known footguns
    warnings.append(
        "scan_f0_val_loss default val = FULL dirty val16 (same surface as dirty F0 pick). "
        "Pass --v0913_clean_map only if you intentionally select on clean val."
    )
    warnings.append(
        "launch_cn2west_f123.py default max_steps=40000; train-f2 here forces 80000 to match dirty."
    )
    warnings.append(
        "rebuild_g0_caches.py may DELETE dirty Ec — never use it for this experiment; "
        "use rebuild_f0_clean_v0913_caches.py."
    )
    warnings.append(
        "eval_f03_test16_strat.py still hardcodes dirty Es/Ec; wire F2-CLEAN methods to "
        "artifacts/f0_clean_v0913 before generating compare preds."
    )

    print("ERRORS", len(errors), flush=True)
    for e in errors:
        print("  ERR", e, flush=True)
    print("WARNINGS", len(warnings), flush=True)
    for w in warnings:
        print("  WARN", w, flush=True)
    refresh_status()
    return 1 if errors else 0


def cmd_smoke_f0(gpu: int) -> int:
    return run([
        PY, str(ROOT / "scripts/launch_cn2west_f0_rsifree.py"),
        "--smoke", "--dataset_id", "v0913_clean", "--gpu", str(gpu),
    ])


def cmd_train_f0(gpu: int) -> int:
    if F0_RUN.exists():
        print(f"refuse overwrite: {F0_RUN}", file=sys.stderr)
        return 2
    rc = run([
        PY, str(ROOT / "scripts/launch_cn2west_f0_rsifree.py"),
        "--yes", "--dataset_id", "v0913_clean",
        "--run_id", F0_ID,
        "--max_steps", "100000", "--ckpt_interval", "5000", "--warmup", "5000",
        "--batch_size", "8", "--lr", "1e-5", "--seed", "3407",
        "--gpu", str(gpu),
    ])
    refresh_status()
    return rc


def cmd_scan_f0(gpu: int) -> int:
    if not F0_RUN.is_dir():
        print(f"missing {F0_RUN}", file=sys.stderr)
        return 2
    out = ROOT / "reports/training_logs" / F0_ID
    rc = run([
        PY, str(ROOT / "scripts/scan_f0_val_loss.py"),
        "--run", str(F0_RUN),
        "--out", str(out),
        "--device", f"cuda:{gpu}",
    ])
    refresh_status()
    return rc


def cmd_cache(gpu: int) -> int:
    return run([
        PY, str(ROOT / "scripts/rebuild_f0_clean_v0913_caches.py"),
        "--run", str(F0_RUN), "--gpu", str(gpu), "--yes",
    ])


def cmd_smoke_f2(gpu: int) -> int:
    parent = F0_RUN / "best"
    es, ec = CACHE / "es_spatial", CACHE / "ec_multiscale"
    for p in (parent / "unet.pth", es / "manifest.json", ec / "manifest.json"):
        if not p.exists():
            print(f"missing precondition: {p}", file=sys.stderr)
            return 2
    return run([
        PY, str(ROOT / "scripts/launch_cn2west_f123.py"),
        "--arm", "F2", "--smoke", "--dataset_id", "v0913_clean",
        "--parent", str(parent),
        "--es_cache", str(es), "--ec_cache", str(ec),
        "--gpu", str(gpu),
    ])


def cmd_train_f2(gpu: int) -> int:
    if F2_RUN.exists():
        print(f"refuse overwrite: {F2_RUN}", file=sys.stderr)
        return 2
    parent = F0_RUN / "best"
    es, ec = CACHE / "es_spatial", CACHE / "ec_multiscale"
    for p in (parent / "unet.pth", es / "manifest.json", ec / "manifest.json"):
        if not p.exists():
            print(f"missing precondition: {p}", file=sys.stderr)
            return 2
    # Refuse dirty parent / dirty caches.
    if "F0-RSIFREE-FT-A-S3407" in str(parent.resolve()):
        print("refuse dirty F0 parent for clean F2", file=sys.stderr)
        return 2
    if es.resolve() == DIRTY_ES.resolve() or ec.resolve() == DIRTY_EC.resolve():
        print("refuse dirty Es/Ec for clean F2", file=sys.stderr)
        return 2
    rc = run([
        PY, str(ROOT / "scripts/launch_cn2west_f123.py"),
        "--arm", "F2", "--yes", "--dataset_id", "v0913_clean",
        "--run_id", F2_ID,
        "--parent", str(parent),
        "--es_cache", str(es), "--ec_cache", str(ec),
        "--max_steps", "80000", "--ckpt_interval", "5000", "--warmup", "5000",
        "--batch_size", "8", "--lr", "1e-5", "--seed", "3407",
        "--source_drop", "0.25", "--offset_coefficient", "0.5",
        "--gpu", str(gpu),
    ])
    refresh_status()
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=["check", "status", "smoke-f0", "train-f0", "scan-f0", "cache", "smoke-f2", "train-f2"],
    )
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "check":
        return cmd_check()
    if args.cmd == "status":
        refresh_status()
        return 0
    if args.cmd == "smoke-f0":
        return cmd_smoke_f0(args.gpu)
    if args.cmd == "train-f0":
        return cmd_train_f0(args.gpu)
    if args.cmd == "scan-f0":
        return cmd_scan_f0(args.gpu)
    if args.cmd == "cache":
        return cmd_cache(args.gpu)
    if args.cmd == "smoke-f2":
        return cmd_smoke_f2(args.gpu)
    if args.cmd == "train-f2":
        return cmd_train_f2(args.gpu)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
