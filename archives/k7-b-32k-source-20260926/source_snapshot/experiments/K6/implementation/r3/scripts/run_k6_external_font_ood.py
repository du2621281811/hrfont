#!/usr/bin/env python3
"""Resumable three-arm external-font OOD queue."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

CODE = Path("/root/projects/hrfont_k6_20260920_r3")
OUT_ROOT = Path("/root/data2/hrfont_k6_20260920/external_font_challenge_20260921")
EVAL = CODE / "scripts/eval_k6_external_ood.py"
PY = "/root/miniforge3/envs/boogu/bin/torchrun"
WORLD = "8"
CUDA = "0,1,2,3,4,5,6,7"
EXPECTED = 6 * 16 * 4
JOBS = [
    ("K6-A", "/root/data2/hrfont_k6_20260920/checkpoints_ext20k/K6-A-R2-V2-K0-S3407-EXT20K-DATA2/global_step_20000"),
    ("K6-B-RSI-NOOFFSETLOSS", "/root/data2/hrfont_k6_20260920/checkpoints_ext20k/K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407-EXT20K-DATA2/global_step_20000"),
    ("K6-C-RSI-AUX-NOOFFSETLOSS", "/root/data2/hrfont_k6_20260920/checkpoints_c10k/K6-C-RSI-AUX-NOOFFSETLOSS-K0-S3407-10K/global_step_10000"),
]


def write_status(payload: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    temp = OUT_ROOT / "queue_status.json.tmp"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temp.replace(OUT_ROOT / "queue_status.json")


def complete(out: Path) -> bool:
    try:
        done = json.loads((out / "DONE.json").read_text())
        metrics = out / "metrics.json"
        return done.get("status") == "completed" and done.get("images") == EXPECTED and metrics.is_file()
    except (OSError, ValueError):
        return False


def main() -> None:
    status = dict(status="running", started_at=time.time(), expected_images=EXPECTED, jobs=[])
    write_status(status)
    for arm, checkpoint in JOBS:
        out = OUT_ROOT / arm
        record = dict(arm=arm, checkpoint=checkpoint, out=str(out), expected_images=EXPECTED,
                      status="running", started_at=time.time())
        status["jobs"].append(record)
        write_status(status)
        if complete(out):
            record.update(status="already_complete", finished_at=time.time())
            write_status(status)
            continue
        out.mkdir(parents=True, exist_ok=True)
        log = OUT_ROOT / f"{arm.replace('/', '_')}.log"
        command = [PY, "--standalone", "--nproc_per_node=" + WORLD, str(EVAL),
                   "--arm", arm, "--checkpoint", checkpoint, "--out", str(out)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=CUDA, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, finished_at=time.time(),
                      status="completed" if result.returncode == 0 and complete(out) else "failed",
                      log=str(log))
        write_status(status)
        if record["status"] != "completed":
            status.update(status="failed", finished_at=time.time())
            write_status(status)
            raise SystemExit(f"external OOD failed: {arm}; see {log}")
    status.update(status="completed", finished_at=time.time())
    write_status(status)


if __name__ == "__main__":
    main()
