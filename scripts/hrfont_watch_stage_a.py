#!/usr/bin/env python3
"""Supervise Stage-A launches: restart on crash, stop on DONE/STOP."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "code/variants/cn2west_stage_a/FontDiffuser/launch_e2.py"


def log(path: Path, message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--sleep", type=int, default=30)
    parser.add_argument("--max-restarts", type=int, default=50)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    watch_log = output / "watchdog.log"
    python = "/root/miniforge3/envs/boogu/bin/python"
    restarts = 0
    while True:
        if (output / "DONE.json").exists():
            log(watch_log, "DONE.json present; supervisor exit 0")
            return 0
        if (output / "STOP").exists() and (output / "stopped_step").exists():
            log(watch_log, "STOP honored; supervisor exit 0")
            return 0
        cmd = [
            python, "-u", str(LAUNCHER),
            "--config", str(args.config.resolve()),
            "--yes", "--gpu", str(args.gpu),
            "--output-dir", str(output),
        ]
        if args.max_steps:
            cmd.extend(["--max-steps", str(args.max_steps)])
        log(watch_log, f"launch gpu={args.gpu} restart={restarts} cmd={' '.join(cmd)}")
        train_log = output / "watchdog_train.log"
        with train_log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== restart {restarts} {datetime.now().isoformat()} =====\n")
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
        if (output / "DONE.json").exists():
            log(watch_log, "training completed")
            return 0
        if proc.returncode == 0:
            log(watch_log, "launcher returned 0 without DONE; stop")
            return 1
        tail = train_log.read_text(encoding="utf-8", errors="replace")[-8000:]
        oom = "out of memory" in tail.lower() or "CUDA out of memory" in tail
        restarts += 1
        log(watch_log, f"crash rc={proc.returncode} oom={oom}; sleep {args.sleep}s ({restarts}/{args.max_restarts})")
        if oom and restarts >= 3:
            log(watch_log, "CUDA OOM repeated; supervisor stop")
            return 1
        if restarts > args.max_restarts:
            log(watch_log, "too many restarts")
            return 1
        time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
