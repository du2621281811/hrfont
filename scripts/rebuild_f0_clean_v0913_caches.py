#!/usr/bin/env python3
"""Build Es/Ec from F0-CLEAN best. Never deletes dirty artifacts/f0 caches.

Output (fixed):
  artifacts/f0_clean_v0913/es_spatial/
  artifacts/f0_clean_v0913/ec_multiscale/
  artifacts/f0_clean_v0913/READY.json
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
DEFAULT_RUN = ROOT / "runs/F0-CLEAN-V0913-A-S3407"
OUT = ROOT / "artifacts/f0_clean_v0913"
DIRTY_ES = ROOT / "artifacts/f0/es_spatial_f0"
DIRTY_EC = ROOT / "artifacts/f0/ec_multiscale_f0"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"


def cache_done(directory: Path) -> bool:
    prog = directory / "progress.json"
    man = directory / "manifest.json"
    if not prog.is_file() or not man.is_file():
        return False
    payload = json.loads(prog.read_text(encoding="utf-8"))
    return payload.get("done") == payload.get("total") and int(payload.get("total", 0)) > 0


def pick_ckpt(run: Path) -> Path:
    best = run / "best"
    if best.is_dir() and (best / "style_encoder.pth").is_file():
        return best.resolve() if best.is_symlink() else best
    done = run / "DONE.json"
    if done.is_file():
        step = int(json.loads(done.read_text(encoding="utf-8")).get("global_step", 0))
        cand = run / f"global_step_{step}"
        if (cand / "style_encoder.pth").is_file():
            return cand
    mile = ROOT / "reports/training_logs" / run.name / "F0_MILESTONE.json"
    if mile.is_file():
        step = int(json.loads(mile.read_text(encoding="utf-8"))["selected_step"])
        cand = run / f"global_step_{step}"
        if (cand / "style_encoder.pth").is_file():
            return cand
    raise SystemExit(
        f"no F0-CLEAN best/milestone under {run}; run scan_f0_val_loss.py first"
    )


def free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    if not args.yes:
        print("Refusing without --yes (will write ~95G under artifacts/f0_clean_v0913).", file=sys.stderr)
        return 2
    if not run.is_dir():
        print(f"missing run: {run}", file=sys.stderr)
        return 2
    # Hard safety: never touch dirty caches.
    for dirty in (DIRTY_ES, DIRTY_EC):
        if not dirty.exists():
            print(f"note: dirty cache absent (ok): {dirty}", flush=True)
    free = free_gb(ROOT)
    if free < 110 and not (cache_done(OUT / "es_spatial") and cache_done(OUT / "ec_multiscale")):
        print(f"need ~110G free for new Ec; have {free:.1f}G", file=sys.stderr)
        return 2

    ckpt = pick_ckpt(run)
    OUT.mkdir(parents=True, exist_ok=True)
    es = OUT / "es_spatial"
    ec = OUT / "ec_multiscale"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"

    def run_cmd(cmd: list[str]) -> None:
        print("CMD", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    if not cache_done(es):
        run_cmd([
            PY, str(ROOT / "scripts/hrfont_build_e1_caches.py"),
            "--which", "es", "--gpu", "0",
            "--ckpt-dir", str(ckpt),
            "--data-root", str(DATA),
            "--es-out", str(es),
            "--batch-size", "16",
        ])
    else:
        print(f"skip es (done): {es}", flush=True)
    if not cache_done(ec):
        run_cmd([
            PY, str(ROOT / "scripts/hrfont_build_e1_caches.py"),
            "--which", "ec", "--gpu", "0",
            "--ckpt-dir", str(ckpt),
            "--data-root", str(DATA),
            "--ec-out", str(ec),
            "--batch-size", "8",
        ])
    else:
        print(f"skip ec (done): {ec}", flush=True)

    man_es = json.loads((es / "manifest.json").read_text(encoding="utf-8"))
    man_ec = json.loads((ec / "manifest.json").read_text(encoding="utf-8"))
    if Path(man_es.get("ckpt_dir", "")).resolve() != ckpt.resolve() and str(man_es.get("ckpt_dir")) not in str(ckpt):
        # accept relative ckpt_dir
        resolved = (ROOT / man_es["ckpt_dir"]).resolve() if not Path(man_es["ckpt_dir"]).is_absolute() else Path(man_es["ckpt_dir"]).resolve()
        if resolved != ckpt.resolve():
            print(f"WARNING: es ckpt_dir={man_es.get('ckpt_dir')} != {ckpt}", flush=True)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_run": str(run),
        "ckpt": str(ckpt),
        "es": str(es),
        "ec": str(ec),
        "es_ckpt_dir": man_es.get("ckpt_dir"),
        "ec_ckpt_dir": man_ec.get("ckpt_dir"),
        "never_deletes": [str(DIRTY_ES), str(DIRTY_EC)],
        "free_gb_after": free_gb(ROOT),
    }
    (OUT / "READY.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    subprocess.run([PY, str(ROOT / "scripts/status_f0f2_clean_v0913.py")], cwd=str(ROOT), check=False)
    print("READY", json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
