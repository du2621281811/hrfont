#!/usr/bin/env python3
"""Watchdog for E1-FTV2-A-S3407: keep training alive until DONE.json.

- Never lowers batch/lr/steps/seed (no experimental degradation).
- On crash/OOM: log incident, free our GPU process if needed, resume from last_state.
- Network blips ignored (local training).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
VARIANT = ROOT / "code/variants/cn2west_ft_v2/FontDiffuser"
OFFICIAL_CKPT = ROOT / "code/official/FontDiffuser/ckpt"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
RUN_ID = "E1-FTV2-A-S3407"
OUT = ROOT / "runs" / RUN_ID
INCIDENTS = OUT / "incidents.jsonl"
STATUS = OUT / "watchdog_status.json"
LOG = OUT / "watchdog_train.log"
PIDFILE = OUT / "train.pid"

# Frozen E1 hyperparams — do not change on recovery
GPU = 3
BATCH = 8
ACCUM = 1
STEPS = 100_000
LR = 1e-5
WARMUP = 5000
SEED = 3407
CKPT_INTERVAL = 5000
STATE_INTERVAL = 1000


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_incident(kind: str, detail: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utcnow(), "kind": kind, **detail}
    with INCIDENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INCIDENT] {kind}: {detail}", flush=True)


def write_status(**kwargs) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"ts": utcnow(), **kwargs}
    STATUS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def done() -> bool:
    p = OUT / "DONE.json"
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return int(d.get("global_step", 0)) >= STEPS and d.get("status") == "completed"
    except Exception:
        return False


def latest_step() -> int:
    last = OUT / "last_state" / "trainer_state.pt"
    if last.is_file():
        try:
            import torch
            st = torch.load(str(last), map_location="cpu", weights_only=False)
            return int(st.get("global_step", 0))
        except Exception:
            pass
    steps = []
    for p in OUT.glob("global_step_*"):
        try:
            steps.append(int(p.name.split("_")[-1]))
        except ValueError:
            pass
    return max(steps) if steps else 0


def build_cmd(resume: bool) -> list[str]:
    cmd = [
        PY, str(VARIANT / "train.py"),
        "--seed", str(SEED),
        "--experience_name", RUN_ID,
        "--output_dir", str(OUT),
        "--data_root", str(DATA),
        "--resolution", "96",
        "--style_image_size", "96",
        "--content_image_size", "96",
        "--train_batch_size", str(BATCH),
        "--gradient_accumulation_steps", str(ACCUM),
        "--max_train_steps", str(STEPS),
        "--learning_rate", str(LR),
        "--lr_scheduler", "linear",
        "--lr_warmup_steps", str(WARMUP),
        "--phase_1_ckpt_dir", str(OFFICIAL_CKPT),
        "--mixed_precision", "fp16",
        "--ckpt_interval", str(CKPT_INTERVAL),
        "--state_interval", str(STATE_INTERVAL),
        "--log_interval", "100",
        "--drop_prob", "0.1",
        "--perceptual_coefficient", "0.01",
        "--offset_coefficient", "0.5",
    ]
    if resume:
        cmd.extend(["--resume_from", "auto"])
    return cmd


def kill_stale_on_gpu() -> None:
    """Only kill processes that are our previous train workers on CUDA_VISIBLE=GPU."""
    if not PIDFILE.is_file():
        return
    try:
        old = int(PIDFILE.read_text().strip())
    except Exception:
        return
    try:
        os.kill(old, 0)
    except OSError:
        return
    log_incident("kill_stale_pid", {"pid": old})
    try:
        os.kill(old, signal.SIGTERM)
        time.sleep(5)
        os.kill(old, signal.SIGKILL)
    except OSError:
        pass


def run_once(resume: bool) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(resume=resume)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    write_status(phase="starting", resume=resume, step=latest_step(), cmd=cmd)
    with LOG.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== START {utcnow()} resume={resume} =====\n")
        logf.write("CMD " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(VARIANT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        PIDFILE.write_text(str(proc.pid) + "\n", encoding="utf-8")
        write_status(phase="running", pid=proc.pid, resume=resume, step=latest_step())
        rc = proc.wait()
        logf.write(f"\n===== EXIT {utcnow()} rc={rc} =====\n")
    return rc


def classify_failure(rc: int) -> str:
    text = ""
    if LOG.is_file():
        text = LOG.read_text(encoding="utf-8", errors="replace")[-8000:].lower()
    if "out of memory" in text or "cuda out of memory" in text:
        return "oom"
    if rc < 0:
        return f"signal_{-rc}"
    if "nans" in text or "nan" in text and "loss" in text:
        return "nan"
    return f"exit_{rc}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": RUN_ID,
        "gpu": GPU,
        "batch": BATCH,
        "accum": ACCUM,
        "steps": STEPS,
        "lr": LR,
        "warmup": WARMUP,
        "seed": SEED,
        "started_at": utcnow(),
        "policy": "no_degradation_resume_until_done",
    }
    (OUT / "watchdog_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log_incident("watchdog_start", meta)

    attempt = 0
    while not done():
        attempt += 1
        resume = (OUT / "last_state" / "trainer_state.pt").is_file() or any(OUT.glob("global_step_*"))
        # first attempt: fresh if no state
        if attempt == 1 and not resume:
            resume = False
        write_status(phase="attempt", attempt=attempt, resume=resume, step=latest_step())
        try:
            kill_stale_on_gpu()
            rc = run_once(resume=resume)
        except Exception as e:
            log_incident("watchdog_exception", {"attempt": attempt, "error": str(e), "tb": traceback.format_exc()})
            time.sleep(30)
            continue

        if done():
            break

        kind = classify_failure(rc)
        step = latest_step()
        log_incident("train_crash", {"attempt": attempt, "rc": rc, "kind": kind, "step": step})
        # OOM: do NOT lower batch — wait and retry after brief cooldown (maybe transient)
        if kind == "oom":
            log_incident("oom_retry_same_batch", {"batch": BATCH, "gpu": GPU, "wait_s": 60})
            time.sleep(60)
        else:
            time.sleep(15)
        # Always resume next loop if any state exists
        if step <= 0 and attempt > 3:
            log_incident("fatal_no_progress", {"attempt": attempt})
            # still keep trying — user forbade abandoning
            time.sleep(60)

    write_status(phase="completed", step=latest_step(), attempts=attempt)
    log_incident("watchdog_complete", {"step": latest_step(), "attempts": attempt})
    print("WATCHDOG: experiment completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
