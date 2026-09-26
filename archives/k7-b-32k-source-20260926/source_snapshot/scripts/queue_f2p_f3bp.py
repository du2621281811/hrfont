#!/usr/bin/env python3
"""Wait for peer inference to finish, then train F2-P → F3b-P serially with resume watchdog.

Avoids overlapping I/O with eval_f03 generate jobs. On crash/stall, relaunches from last_state.
Does not use the network beyond local disk/GPU.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
PARENT = ROOT / "runs/F0-RSIFREE-FT-A-S3407/best"
LAUNCH = ROOT / "scripts/launch_cn2west_f123.py"
OUT = ROOT / "reports/watchdog_f2p_f3bp"
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / "status.json"
INCIDENTS = OUT / "incidents.jsonl"
POLL_S = 60
STALL_S = 30 * 60  # no log/heartbeat progress for 30min → relaunch
# After host reboot, train_log.jsonl can sit ahead of last_state while tqdm
# is still catching up. Treat a freshly written train log as progress so the
# watchdog does not SIGTERM a live resume.
MAX_STEPS = 40_000
WARMUP = 2_000
# GPU0 still holds ~11GiB ghost CUDA contexts after the 2026-09-11 host reboot
# (nvidia-smi PIDs [Not Found]; gpu-reset refused). GPU1 has ~20GiB free.
GPU = 1


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_incident(kind: str, **payload) -> None:
    row = {"ts": utcnow(), "kind": kind, **payload}
    with INCIDENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("INCIDENT", json.dumps(row, ensure_ascii=False), flush=True)


def write_status(payload: dict) -> None:
    payload = {"ts": utcnow(), **payload}
    STATUS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def alive(pid: int) -> bool:
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


def _cmdline(pid: int) -> str:
    return " ".join(_cmdline_args(pid))


def is_python_script(pid: int, script_suffix: str) -> bool:
    """True iff argv is `python .../<script_suffix> ...` (not a bash -c that merely mentions it)."""
    args = _cmdline_args(pid)
    if len(args) < 2:
        return False
    exe = Path(args[0]).name
    if not exe.startswith("python"):
        return False
    return any(a.endswith(script_suffix) for a in args[1:3])


def inference_busy() -> list[dict]:
    """Any eval_f03 generate / FontDiffuser sample occupying GPUs."""
    busy = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if not alive(pid):
            continue
        args = _cmdline_args(pid)
        joined = " ".join(args)
        if is_python_script(pid, "eval_f03_test16_strat.py") and "generate" in args:
            busy.append({"pid": pid, "pattern": "eval_generate"})
        elif is_python_script(pid, "sample.py") and "FontDiffuser" in joined:
            busy.append({"pid": pid, "pattern": "sample"})
    return busy


def _tqdm_step_from_log(run_id: str) -> int:
    """Latest `29012/40000` tqdm counter from the live train log (resume catch-up)."""
    log = OUT / f"{run_id}.log"
    if not log.is_file():
        return 0
    try:
        tail = log.read_bytes()[-8192:].decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return 0
    best = 0
    needle = f"/{MAX_STEPS}"
    for line in tail.splitlines():
        if needle not in line:
            continue
        # tqdm: 73%|...| 29012/40000 [02:03<...]
        for tok in line.replace("|", " ").split():
            if tok.endswith(needle) and tok.split("/", 1)[0].isdigit():
                best = max(best, int(tok.split("/", 1)[0]))
    return best


def log_fresh(run_id: str, max_age_s: float = 15 * 60) -> bool:
    log = OUT / f"{run_id}.log"
    try:
        return log.is_file() and (time.time() - log.stat().st_mtime) < max_age_s
    except OSError:
        return False


def read_step(run: Path) -> int:
    best = 0
    hb = run / "heartbeat.json"
    if hb.is_file():
        try:
            d = json.loads(hb.read_text(encoding="utf-8"))
            # ignore loading_* heartbeats that reset step to 0 during resume
            if str(d.get("status") or "") in {"running", "completed"}:
                best = max(best, int(d.get("step") or 0))
        except Exception:
            pass
    # train_log is updated every log_interval; more reliable than stale heartbeat mid-interval
    tl = run / "train_log.jsonl"
    if tl.is_file():
        try:
            lines = tl.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                best = max(best, int(json.loads(lines[-1]).get("step") or 0))
        except Exception:
            pass
    for p in run.glob("global_step_*"):
        try:
            best = max(best, int(p.name.split("_")[-1]))
        except ValueError:
            pass
    best = max(best, _tqdm_step_from_log(run.name))
    last = run / "last_state" / "trainer_state.pt"
    if last.is_file():
        # mtime alone is not a step; prefer heartbeat/log
        pass
    return best


def resume_dir(run: Path) -> Path | None:
    last = run / "last_state"
    if (last / "trainer_state.pt").is_file() and (last / "unet.pth").is_file():
        return last
    best_step, best = -1, None
    for p in run.glob("global_step_*"):
        try:
            s = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if (p / "unet.pth").is_file() and s > best_step:
            best_step, best = s, p
    return best


def train_pid(run_id: str) -> int | None:
    """Find the real FontDiffuser train.py for this run_id (not launch wrappers / shells)."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if not is_python_script(pid, "train.py"):
            continue
        args = _cmdline_args(pid)
        joined = " ".join(args)
        if "FontDiffuser/train.py" not in joined and not any(
            a.endswith("FontDiffuser/train.py") for a in args
        ):
            continue
        if run_id not in joined:
            continue
        if alive(pid):
            return pid
    return None


def any_train_on_gpu_slot() -> list[int]:
    """Any FontDiffuser train.py still alive (guards against duplicate launches)."""
    out = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if not is_python_script(pid, "train.py"):
            continue
        args = _cmdline_args(pid)
        if not any(a.endswith("FontDiffuser/train.py") for a in args):
            continue
        if alive(pid):
            out.append(pid)
    return out


def launch_fresh(arm: str, run_id: str, gpu: int) -> int:
    log = OUT / f"{run_id}.log"
    cmd = [
        PY, str(LAUNCH),
        "--arm", arm,
        "--parent", str(PARENT),
        "--gpu", str(gpu),
        "--max_steps", str(MAX_STEPS),
        "--warmup", str(WARMUP),
        "--run_id", run_id,
        "--yes",
    ]
    if arm == "F3bP":
        cmd += ["--support_bank", "artifacts/f0/support_bank_f3b_topology.json"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== FRESH {utcnow()} =====\nCMD {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env,
            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
        )
    log_incident("launch_fresh", arm=arm, run_id=run_id, pid=proc.pid, log=str(log))
    return proc.pid


def launch_resume(arm: str, run: Path, gpu: int) -> int | None:
    rd = resume_dir(run)
    if rd is None:
        log_incident("no_resume", arm=arm, run=str(run))
        return None
    stop = run / "STOP"
    if stop.exists():
        stop.unlink()
        log_incident("removed_stop", arm=arm)
    log = OUT / f"{run.name}.log"
    cmd = [
        PY, str(VARIANT / "train.py"),
        "--arm", arm,
        "--rsi_source", "delta",
        "--support" if arm == "F3bP" else "--no-support",
        "--support_drop", "0.2", "--support_k", "8", "--source_drop", "0.25",
        "--seed", "3407",
        "--experience_name", run.name,
        "--output_dir", str(run),
        "--data_root", str(ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"),
        "--split_manifest", str(ROOT / "manifests/split_v3_228_16_16.json"),
        "--es_cache_path", str(ROOT / "artifacts/f0/es_spatial_f0"),
        "--ec_cache_path", str(ROOT / "artifacts/f0/ec_multiscale_f0"),
        "--phase_1_ckpt_dir", str(PARENT),
        "--resolution", "96", "--style_image_size", "96", "--content_image_size", "96",
        "--train_batch_size", "8", "--gradient_accumulation_steps", "1",
        "--max_train_steps", str(MAX_STEPS),
        "--learning_rate", "1e-5", "--lr_scheduler", "linear",
        "--lr_warmup_steps", str(WARMUP),
        "--mixed_precision", "fp16",
        "--ckpt_interval", "5000", "--state_interval", "1000", "--log_interval", "100",
        "--drop_prob", "0.1",
        "--perceptual_coefficient", "0.01", "--offset_coefficient", "0.5",
        "--parity_check",
        "--resume_from", str(rd),
    ]
    if arm == "F3bP":
        cmd += ["--support_bank", str(ROOT / "artifacts/f0/support_bank_f3b_topology.json")]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== RESUME {utcnow()} from {rd} =====\nCMD {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(VARIANT), env=env,
            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
        )
    log_incident("launch_resume", arm=arm, pid=proc.pid, resume=str(rd), step=read_step(run))
    return proc.pid


def wait_inference_clear(timeout_s: int = 6 * 3600) -> None:
    t0 = time.time()
    while True:
        busy = inference_busy()
        write_status({"phase": "wait_inference", "busy": busy, "elapsed_s": int(time.time() - t0)})
        if not busy:
            log_incident("inference_clear")
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"timeout waiting for inference: {busy}")
        time.sleep(30)


def run_arm(arm: str, run_id: str, gpu: int) -> None:
    run = ROOT / "runs" / run_id
    t0 = time.time()
    last_step = -1
    last_progress_t = time.time()

    if read_step(run) >= MAX_STEPS and (run / f"global_step_{MAX_STEPS}").is_dir():
        log_incident("already_done", arm=arm, step=read_step(run))
        return

    while True:
        step = read_step(run) if run.exists() else 0
        if step >= MAX_STEPS and (run / f"global_step_{MAX_STEPS}").is_dir():
            log_incident("arm_done", arm=arm, step=step, elapsed_s=int(time.time() - t0))
            write_status({"phase": "arm_done", "arm": arm, "step": step})
            return

        pid = train_pid(run_id)
        if pid and alive(pid):
            if step > last_step or log_fresh(run_id):
                if step > last_step:
                    last_step = step
                last_progress_t = time.time()
            elif time.time() - last_progress_t > STALL_S:
                log_incident("stall_kill", arm=arm, pid=pid, step=step)
                try:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(10)
                    if alive(pid):
                        os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                pid = None
            if pid and alive(pid):
                write_status({
                    "phase": "training", "arm": arm, "pid": pid, "step": step,
                    "elapsed_s": int(time.time() - t0),
                })
                time.sleep(POLL_S)
                continue

        # not running — never launch if any FontDiffuser train is alive
        existing = any_train_on_gpu_slot()
        if existing:
            write_status({
                "phase": "wait_existing_train", "arm": arm, "pids": existing, "step": step,
                "elapsed_s": int(time.time() - t0),
            })
            time.sleep(POLL_S)
            continue

        rd = resume_dir(run) if run.exists() else None
        if rd is not None:
            launch_resume(arm, run, gpu)
            last_progress_t = time.time()
            time.sleep(45)
            continue

        if not run.exists():
            busy = inference_busy()
            if busy:
                write_status({"phase": "wait_inference_before_launch", "arm": arm, "busy": busy})
                time.sleep(30)
                continue
            launch_fresh(arm, run_id, gpu)
            last_progress_t = time.time()
            time.sleep(45)
            continue

        # Run dir exists, no resume artifact yet, no live train:
        # either still booting, crashed before first last_state, or pid detection lag.
        if time.time() - last_progress_t > STALL_S and step == 0:
            log_incident("boot_stall_relaunch", arm=arm, run=str(run))
            # move aside incomplete run so fresh launch can proceed
            bak = run.with_name(run.name + f".broken_{int(time.time())}")
            run.rename(bak)
            launch_fresh(arm, run_id, gpu)
            last_progress_t = time.time()
            time.sleep(45)
            continue

        log_incident("blocked", arm=arm, run=str(run), step=step)
        write_status({
            "phase": "blocked_wait", "arm": arm, "step": step,
            "elapsed_s": int(time.time() - t0),
        })
        time.sleep(POLL_S)


def preflight() -> None:
    for p in (PARENT / "unet.pth", PARENT / "style_encoder.pth", PARENT / "content_encoder.pth",
              ROOT / "artifacts/f0/es_spatial_f0/manifest.json",
              ROOT / "artifacts/f0/ec_multiscale_f0/manifest.json",
              ROOT / "artifacts/f0/support_bank_f3b_topology.json",
              LAUNCH):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    # import smoke: style_pattn path
    sys.path.insert(0, str(VARIANT))
    import train as T  # noqa: F401
    assert "F2P" in T.ARM_SPEC and "F3bP" in T.ARM_SPEC
    log_incident("preflight_ok", git=subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True).strip())


def main() -> int:
    log_incident("watchdog_start", max_steps=MAX_STEPS, warmup=WARMUP, gpu=GPU)
    preflight()
    f2_run = ROOT / "runs/f2_pattn_s3407"
    f2_step = read_step(f2_run) if f2_run.exists() else 0
    if f2_step > 0 or train_pid("f2_pattn_s3407") or any_train_on_gpu_slot():
        log_incident("attach_mode", f2_step=f2_step, train_pids=any_train_on_gpu_slot())
    else:
        wait_inference_clear()
    write_status({"phase": "start_f2p"})
    run_arm("F2P", "f2_pattn_s3407", GPU)
    write_status({"phase": "start_f3bp"})
    run_arm("F3bP", "f3b_pattn_s3407", GPU)
    write_status({"phase": "completed", "f2p_step": read_step(ROOT / "runs/f2_pattn_s3407"),
                  "f3bp_step": read_step(ROOT / "runs/f3b_pattn_s3407")})
    log_incident("watchdog_complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log_incident("fatal", error=str(e))
        write_status({"phase": "fatal", "error": str(e)})
        raise
