#!/usr/bin/env python3
"""Rebuild Es/Ec/es_local from G0. Do not reuse dirty or F0-clean caches."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
G0 = ROOT / "runs/G0-F0-V0913-BS256-A-S3407"
OUT = ROOT / "artifacts/g0"
DIRTY_EC = ROOT / "artifacts/f0/ec_multiscale_f0"
MIN_STEP = 10000


def pick_ckpt() -> Path:
    done = G0 / "DONE.json"
    if done.is_file():
        payload = json.loads(done.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or int(payload.get("global_step", 0)) < MIN_STEP:
            raise SystemExit(f"G0 not complete: {payload}")
    for name in (f"global_step_{MIN_STEP}", "last_state"):
        d = G0 / name
        if (d / "unet.pth").is_file() and (d / "style_encoder.pth").is_file():
            return d
    raise SystemExit(f"G0 {MIN_STEP} weights missing")


def cache_done(directory: Path) -> bool:
    prog = directory / "progress.json"
    man = directory / "manifest.json"
    if not prog.is_file() or not man.is_file():
        return False
    p = json.loads(prog.read_text(encoding="utf-8"))
    return p.get("done") == p.get("total") and p.get("total", 0) > 0


def free_bytes(path: Path) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def maybe_free_dirty_ec() -> str:
    need = 110 * 1024 ** 3
    free = free_bytes(ROOT)
    if free >= need:
        return f"disk_ok free_gb={free / 1024**3:.1f}"
    if DIRTY_EC.is_dir():
        shutil.rmtree(DIRTY_EC)
        return f"deleted_dirty_ec free_gb={free_bytes(ROOT) / 1024**3:.1f}"
    raise SystemExit(f"not enough disk for new Ec: free={free / 1024**3:.1f}G")


def run(cmd: list[str], env: dict) -> None:
    print("CMD", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    ckpt = pick_ckpt()
    note = maybe_free_dirty_ec()
    print(f"ckpt={ckpt} {note}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    es = OUT / "es_spatial"
    ec = OUT / "ec_multiscale"
    local = OUT / "es_local"
    data = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
    if not cache_done(es):
        run([
            PY, str(ROOT / "scripts/hrfont_build_e1_caches.py"),
            "--which", "es", "--gpu", "0",
            "--ckpt-dir", str(ckpt),
            "--data-root", str(data),
            "--es-out", str(es),
            "--batch-size", "16",
        ], env)
    if not cache_done(ec):
        run([
            PY, str(ROOT / "scripts/hrfont_build_e1_caches.py"),
            "--which", "ec", "--gpu", "0",
            "--ckpt-dir", str(ckpt),
            "--data-root", str(data),
            "--ec-out", str(ec),
            "--batch-size", "8",
        ], env)
    if not cache_done(local):
        run([
            PY, str(ROOT / "scripts/hrfont_build_es_local_cache.py"),
            "--ckpt", str(ckpt),
            "--spatial-cache", str(es),
            "--out", str(local),
            "--device", "cuda:0",
            "--batch-size", "32",
        ], env)
    meta = {
        "parent": str(ckpt),
        "es": str(es),
        "ec": str(ec),
        "es_local": str(local),
        "disk_note": note,
        "group": "G",
    }
    (OUT / "READY.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("READY", json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
