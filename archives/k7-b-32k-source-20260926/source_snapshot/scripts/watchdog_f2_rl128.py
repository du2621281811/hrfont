#!/usr/bin/env python3
"""Session-independent watchdog for F2-RL128-A-S3407.

Policy:
  - Never degrade batch/lr/steps/seed; never write STOP.
  - Honor an existing STOP (do not relaunch; do not delete it).
  - Adopt the live train PID; relaunch only if the process is dead and DONE is missing.
  - Resume from the higher of last_state / stopped_step (optimizer+RNG).
  - GPU0 only. Survive Cursor/SSH disconnect (caller: setsid/nohup).
  - Do not SIGTERM a live train process (val at 5k milestones can stall heartbeat).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
PARENT = ROOT / "runs/F0-RSIFREE-FT-A-S3407/best"
RUN = ROOT / "runs/F2-RL128-A-S3407"
WD = ROOT / "reports/watchdog_f2_rl128"
INCIDENTS = WD / "incidents.jsonl"
STATUS = WD / "status.json"
PIDFILE = WD / "watchdog.pid"
TRAIN_LOG = ROOT / "reports/training_logs/F2-RL128-A-S3407/launch.stdout"
ES = ROOT / "artifacts/f0/es_spatial_f0"
ES_LOCAL = ROOT / "artifacts/f0/es_local_f0_block2_pool4"
EC = ROOT / "artifacts/f0/ec_multiscale_f0"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"

GPU = 0
MAX_STEPS = 40_000
POLL_S = 60
STALL_WARN_S = 30 * 60


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_incident(kind: str, detail: dict) -> None:
    WD.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utcnow(), "kind": kind, **detail}
    with INCIDENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INCIDENT] {kind} {detail}", flush=True)


def write_status(payload: dict) -> None:
    WD.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"ts": utcnow(), **payload}, ensure_ascii=False, indent=2) + "\n")


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _cmdline_args(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [x.decode("utf-8", errors="replace") for x in raw.split(b"\x00") if x]


def find_train_pid() -> int | None:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        args = _cmdline_args(pid)
        if len(args) < 2:
            continue
        if not Path(args[0]).name.startswith("python"):
            continue
        if not any(a.endswith("FontDiffuser/train.py") for a in args[1:3]):
            continue
        if "--arm" in args and "F2RL" in args and "F2-RL128-A-S3407" in args:
            if alive(pid):
                return pid
    return None


def done() -> bool:
    p = RUN / "DONE.json"
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return int(d.get("global_step", 0)) >= MAX_STEPS and d.get("status") == "completed"
    except Exception:
        return False


def heartbeat() -> dict:
    p = RUN / "heartbeat.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def heartbeat_age_s() -> float | None:
    p = RUN / "heartbeat.json"
    if not p.is_file():
        return None
    return time.time() - p.stat().st_mtime


def read_step() -> int:
    best = 0
    hb = heartbeat()
    try:
        best = max(best, int(hb.get("step") or 0))
    except (TypeError, ValueError):
        pass
    for p in RUN.glob("global_step_*"):
        try:
            best = max(best, int(p.name.split("_")[-1]))
        except ValueError:
            pass
    return best


def resume_dir() -> Path | None:
    candidates: list[Path] = []
    for name in ("stopped_step", "last_state"):
        d = RUN / name
        if (d / "trainer_state.pt").is_file():
            candidates.append(d)
    if not candidates:
        return None
    scored = []
    for d in candidates:
        step = 0
        try:
            import torch
            st = torch.load(str(d / "trainer_state.pt"), map_location="cpu", weights_only=False)
            step = int(st.get("step", st.get("global_step", 0)))
        except Exception:
            step = 0
        scored.append((step, (d / "trainer_state.pt").stat().st_mtime, d))
    scored.sort(reverse=True)
    return scored[0][2]


def launch_resume(rd: Path) -> int:
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(VARIANT / "train.py"),
        "--arm", "F2RL", "--rsi_source", "delta", "--no-support",
        "--support_drop", "0.2", "--support_k", "8", "--source_drop", "0.25",
        "--seed", "3407",
        "--experience_name", "F2-RL128-A-S3407",
        "--output_dir", str(RUN),
        "--data_root", str(DATA),
        "--split_manifest", str(SPLIT),
        "--es_cache_path", str(ES),
        "--es_local_cache_path", str(ES_LOCAL),
        "--ec_cache_path", str(EC),
        "--phase_1_ckpt_dir", str(PARENT),
        "--resolution", "96", "--style_image_size", "96", "--content_image_size", "96",
        "--train_batch_size", "8", "--gradient_accumulation_steps", "1",
        "--max_train_steps", str(MAX_STEPS),
        "--learning_rate", "1e-05", "--lr_scheduler", "linear",
        "--lr_warmup_steps", "5000",
        "--mixed_precision", "fp16",
        "--ckpt_interval", "5000", "--state_interval", "1000", "--log_interval", "100",
        "--drop_prob", "0.1",
        "--perceptual_coefficient", "0.01", "--offset_coefficient", "0.5",
        "--parity_check",
        "--resume_from", str(rd),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU)
    env["PYTHONUNBUFFERED"] = "1"
    with TRAIN_LOG.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== WATCHDOG RESUME {utcnow()} =====\n")
        logf.write("CMD " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(VARIANT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return proc.pid


def ensure() -> dict:
    if done():
        return {"job": "F2RL", "state": "done", "step": read_step()}
    if (RUN / "STOP").is_file():
        pid = find_train_pid()
        return {
            "job": "F2RL",
            "state": "stopped_by_user",
            "step": read_step(),
            "pid": pid,
            "note": "STOP present; watchdog will not relaunch",
        }
    pid = find_train_pid()
    step = read_step()
    age = heartbeat_age_s()
    if pid and alive(pid):
        snap = {"job": "F2RL", "state": "running", "pid": pid, "step": step, "hb_age_s": age}
        if age is not None and age > STALL_WARN_S:
            snap["state"] = "running_stall_warn"
            log_incident("stall_warn_alive", {"pid": pid, "step": step, "hb_age_s": round(age)})
        return snap
    rd = resume_dir()
    if rd is None:
        log_incident("no_resume", {"step": step})
        return {"job": "F2RL", "state": "blocked_no_ckpt", "step": step}
    new_pid = launch_resume(rd)
    log_incident("relaunch", {"pid": new_pid, "resume": str(rd), "step": step})
    return {"job": "F2RL", "state": "relaunched", "pid": new_pid, "resume": str(rd), "step": step}


def main() -> int:
    WD.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()) + "\n")
    log_incident("watchdog_start", {"pid": os.getpid(), "poll_s": POLL_S, "gpu": GPU, "max_steps": MAX_STEPS})
    while not done():
        try:
            snap = ensure()
            write_status({"jobs": {"F2RL": snap}})
            if snap.get("state") == "done":
                break
        except Exception as e:
            log_incident("watchdog_exception", {"error": str(e), "tb": traceback.format_exc()})
        time.sleep(POLL_S)
    write_status({"phase": "completed", "jobs": {"F2RL": ensure()}})
    log_incident("watchdog_complete", {"step": read_step()})
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(main())
