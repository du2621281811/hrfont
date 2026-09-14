#!/usr/bin/env python3
"""Rebuild Es/Ec/es_local from G0. Do not reuse dirty or F0-clean caches.

After G0, all 8 GPUs are free. Default path:
  1. init memmaps
  2. Es 8-way shards (batch 64)
  3. finalize Es
  4. Ec 7-way shards (batch 16) ∥ es_local on GPU 7 (batch 64)
  5. finalize Ec
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
G0 = ROOT / "runs/G0b-F0-V0913-BS256-A-S3407"
OUT = ROOT / "artifacts/g0"
DIRTY_EC = ROOT / "artifacts/f0/ec_multiscale_f0"
MIN_STEP = 10000
BUILDER = ROOT / "scripts/hrfont_build_e1_caches.py"
LOCAL_BUILDER = ROOT / "scripts/hrfont_build_es_local_cache.py"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"

sys.path.insert(0, str(ROOT / "scripts"))
from hrfont_build_e1_caches import collect_ec_jobs, collect_es_jobs, load_split  # noqa: E402


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


def shard_ranges(n: int, workers: int) -> list[tuple[int, int]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    workers = min(workers, n)
    base, rem = divmod(n, workers)
    out = []
    start = 0
    for i in range(workers):
        size = base + (1 if i < rem else 0)
        if size:
            out.append((start, start + size))
            start += size
    return out


def common_args(ckpt: Path, es: Path, ec: Path) -> list[str]:
    return [
        PY, str(BUILDER),
        "--ckpt-dir", str(ckpt),
        "--data-root", str(DATA),
        "--es-out", str(es),
        "--ec-out", str(ec),
    ]


def spawn(cmd: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("CMD", " ".join(cmd), ">", log.name, flush=True)
    handle = log.open("a", encoding="utf-8")
    handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    handle.write("CMD " + " ".join(cmd) + "\n")
    handle.flush()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )


def wait_all(jobs: list[tuple[str, subprocess.Popen]]) -> None:
    failed = []
    for name, proc in jobs:
        rc = proc.wait()
        print(f"DONE {name} rc={rc}", flush=True)
        if rc != 0:
            failed.append((name, rc))
    if failed:
        raise SystemExit(f"cache workers failed: {failed}")


def run_shards(which: str, ckpt: Path, es: Path, ec: Path, n: int,
               gpus: list[int], batch: int, log_dir: Path) -> None:
    ranges = shard_ranges(n, len(gpus))
    jobs = []
    for gpu, (begin, end) in zip(gpus, ranges):
        cmd = common_args(ckpt, es, ec) + [
            "--which", which, "--mode", "shard",
            "--begin", str(begin), "--end", str(end),
            "--gpu", str(gpu),
            "--batch-size", str(batch),
            "--verify-n", "0",
        ]
        proc = spawn(cmd, log_dir / f"{which}_gpu{gpu}_{begin}_{end}.log")
        jobs.append((f"{which}:gpu{gpu}:{begin}:{end}", proc))
    wait_all(jobs)


def finalize(which: str, ckpt: Path, es: Path, ec: Path, gpu: int, log_dir: Path) -> None:
    cmd = common_args(ckpt, es, ec) + [
        "--which", which, "--mode", "finalize",
        "--gpu", str(gpu), "--verify-n", "8",
    ]
    proc = spawn(cmd, log_dir / f"{which}_finalize.log")
    wait_all([(f"{which}:finalize", proc)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=-1, help="unused when --workers>1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--es-batch", type=int, default=64)
    ap.add_argument("--ec-batch", type=int, default=16)
    ap.add_argument("--local-batch", type=int, default=64)
    args = ap.parse_args()
    ckpt = pick_ckpt()
    note = maybe_free_dirty_ec()
    print(f"ckpt={ckpt} {note} workers={args.workers}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    es = OUT / "es_spatial"
    ec = OUT / "ec_multiscale"
    local = OUT / "es_local"
    split = load_split(ROOT / "manifests/split_v3_228_16_16.json")
    n_es = len(collect_es_jobs(DATA, split))
    n_ec = len(collect_ec_jobs(DATA, split))
    workers = max(1, args.workers)
    gpus = list(range(workers))
    t0 = time.time()

    if not cache_done(es):
        init = spawn(
            common_args(ckpt, es, ec) + ["--which", "es", "--mode", "init", "--verify-n", "0"],
            log_dir / "es_init.log",
        )
        wait_all([("es:init", init)])
        run_shards("es", ckpt, es, ec, n_es, gpus, args.es_batch, log_dir)
        finalize("es", ckpt, es, ec, gpus[0], log_dir)
        print(f"es done in {time.time() - t0:.0f}s", flush=True)
    else:
        print("es already complete", flush=True)

    pending = []
    need_ec = not cache_done(ec)
    need_local = not cache_done(local)
    local_gpu = gpus[-1] if (need_ec and need_local and workers > 1) else gpus[0]
    if need_ec:
        init = spawn(
            common_args(ckpt, es, ec) + ["--which", "ec", "--mode", "init", "--verify-n", "0"],
            log_dir / "ec_init.log",
        )
        wait_all([("ec:init", init)])
        ec_gpus = [g for g in gpus if not (need_local and g == local_gpu)] or gpus
        ranges = shard_ranges(n_ec, len(ec_gpus))
        for gpu, (begin, end) in zip(ec_gpus, ranges):
            cmd = common_args(ckpt, es, ec) + [
                "--which", "ec", "--mode", "shard",
                "--begin", str(begin), "--end", str(end),
                "--gpu", str(gpu),
                "--batch-size", str(args.ec_batch),
                "--verify-n", "0",
            ]
            pending.append((
                f"ec:gpu{gpu}:{begin}:{end}",
                spawn(cmd, log_dir / f"ec_gpu{gpu}_{begin}_{end}.log"),
            ))

    if need_local:
        cmd = [
            PY, str(LOCAL_BUILDER),
            "--ckpt", str(ckpt),
            "--spatial-cache", str(es),
            "--out", str(local),
            "--device", "cuda:0",
            "--batch-size", str(args.local_batch),
        ]
        env_gpu = os.environ.copy()
        env_gpu["PYTHONUNBUFFERED"] = "1"
        env_gpu["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
        log = log_dir / f"es_local_gpu{local_gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        handle.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        handle.write("CMD " + " ".join(cmd) + "\n")
        handle.flush()
        print("CMD", " ".join(cmd), f"CUDA_VISIBLE_DEVICES={local_gpu}", flush=True)
        pending.append((
            f"es_local:gpu{local_gpu}",
            subprocess.Popen(
                cmd, cwd=str(ROOT), env=env_gpu,
                stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            ),
        ))

    if pending:
        wait_all(pending)
    if not cache_done(ec):
        finalize("ec", ckpt, es, ec, gpus[0], log_dir)

    if not cache_done(es) or not cache_done(ec) or not cache_done(local):
        raise SystemExit(
            f"cache incomplete es={cache_done(es)} ec={cache_done(ec)} local={cache_done(local)}"
        )

    meta = {
        "parent": str(ckpt),
        "es": str(es),
        "ec": str(ec),
        "es_local": str(local),
        "disk_note": note,
        "group": "G",
        "workers": workers,
        "es_jobs": n_es,
        "ec_jobs": n_ec,
        "elapsed_s": round(time.time() - t0, 1),
        "accel": "8-gpu shard Es; 7-gpu shard Ec ∥ es_local; threaded PIL; larger batch",
    }
    (OUT / "READY.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("READY", json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
