"""Run K7-B from the verified 20K state through 50K with recovery.

The supervisor is detached from the user's shell. It waits for the already
running 20K final protocol to finish, pauses the old per-checkpoint queue, then
launches a separate 50K continuation run. Every restart and decision is
recorded under reports/k7_b_50k_20260923/.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path("/root/projects/hrfont")
CODE = Path("/root/projects/hrfont_k7_v3_20260922")
PYTHON = "/root/miniforge3/envs/boogu/bin/python"
TORCHRUN = "/root/miniforge3/envs/boogu/bin/torchrun"
REPORT = ROOT / "reports/k7_b_50k_20260923"
LOCK_PATH = REPORT / "supervisor.lock"
OLD_RUN = ROOT / "runs/K7-B-V3-K0-S3407-20K-R2"
OLD_STATE = ROOT / (
    "data/v3_v2_plus_v0921_20260922/checkpoint_store/"
    "K7-B-V3-K0-S3407-20K-R2/state_step_20000"
)
RUN = ROOT / "runs/K7-B-V3-K0-S3407-50K-R1"
CHECKPOINT_ROOT = ROOT / (
    "data/v3_v2_plus_v0921_20260922/checkpoint_store/"
    "K7-B-V3-K0-S3407-50K-R1"
)
QUEUE_STATUS = ROOT / "reports/k7_b_val_checkpoint_20260922/queue_status.json"
FINAL_ROOT = ROOT / "data/v3_v2_plus_v0921_20260922/inference_final/K7-B"
WRAPPER = ROOT / "scripts/train_k7_b_50k_overlay.py"
TRAIN = CODE / "experiments/K6/implementation/r3/scripts/train_k6.py"
AUTH = CODE / "experiments/K6/AUTHORIZATION_K7_B_50K.json"
MIN_FREE = 1.25 * 1024**3
MAX_RESTARTS = 50
HEARTBEAT_TIMEOUT = 15 * 60

# Loading the eight model replicas and the 32K optimizer state can take
# substantially longer than a normal training heartbeat on this full disk.
# Keep startup recovery bounded, but do not kill ranks that are still making
# progress through model/state initialization.
STARTUP_TIMEOUT = 45 * 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(event: str, **payload: object) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    row = {"time": time.time(), "event": event, **payload}
    with (REPORT / "DECISIONS.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(REPORT / "status.json", row)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def process_ids(pattern: str) -> list[int]:
    try:
        output = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [int(line) for line in output.split() if line.isdigit()]


def terminate_pattern(pattern: str) -> None:
    pids = process_ids(pattern)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pids:
        time.sleep(8)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if pids:
        record("processes_stopped", pattern=pattern, pids=pids)


def final_protocol_done() -> bool:
    queue = read_json(QUEUE_STATUS)
    final = queue.get("final_inference", {})
    expected = {"train", "val", "test"}
    queue_done = expected.issubset(final) and all(
        final[name] in {"completed", "already_complete"} for name in expected
    )
    disk_done = all((FINAL_ROOT / name / "DONE.json").is_file() for name in expected)
    return queue_done and disk_done


def pause_old_inference() -> None:
    # The final 20K train/val/test protocol is retained; old per-ckpt VAL is
    # deliberately paused so it cannot occupy all GPUs during the 50K run.
    terminate_pattern(r"[w]atch_k7_inference.sh")
    terminate_pattern(r"[r]un_k7_b_val_checkpoint_board.py")
    terminate_pattern(r"[e]val_k7_protocol.py")
    queue = read_json(QUEUE_STATUS)
    queue.update(
        {
            "status": "paused_for_k7_b_50k_training",
            "pause_reason": "prioritize approved 20K->50K continuation",
            "paused_at": time.time(),
        }
    )
    write_json(QUEUE_STATUS, queue)
    record(
        "decision_pause_old_checkpoint_queue",
        retained_final_protocol=True,
        reason="The 20K final Train/Val/Test outputs are complete; per-ckpt VAL is deferred.",
    )


def wait_for_final_protocol() -> None:
    while not final_protocol_done():
        queue = read_json(QUEUE_STATUS)
        final = queue.get("final_inference", {})
        record(
            "waiting_for_20k_final_protocol",
            status=queue.get("status"),
            final=final,
            test_done=(FINAL_ROOT / "test" / "DONE.json").is_file(),
        )
        time.sleep(30)
    record("20k_final_protocol_complete")


def prepare_continuation() -> Path:
    resume_state = OLD_STATE
    if not (resume_state / "trainer.pt").is_file():
        resume_state = RUN / "last_state"
        checkpoint = read_json(resume_state / "checkpoint.json")
        if not (resume_state / "trainer.pt").is_file() or not checkpoint.get("complete"):
            raise RuntimeError(
                f"missing verified resume state: {OLD_STATE}; "
                f"existing continuation state is incomplete: {resume_state}"
            )
        record(
            "recovered_existing_continuation_state",
            resume=str(resume_state),
            step=checkpoint.get("step"),
            reason="The prior supervisor stopped after a complete checkpoint; continue without rollback.",
        )
    if not (OLD_RUN / "config.json").is_file():
        raise RuntimeError(f"missing source config: {OLD_RUN / 'config.json'}")
    RUN.mkdir(parents=True, exist_ok=True)
    config = read_json(OLD_RUN / "config.json")
    config["authorized_updates"] = 50000
    if (RUN / "config.json").is_file() and read_json(RUN / "config.json") != config:
        raise RuntimeError("50K continuation config already exists but differs")
    write_json(RUN / "config.json", config)
    manifest = {
        "arm": "K7-B",
        "from_step": read_json(resume_state / "checkpoint.json").get("step", 20000),
        "to_step": 50000,
        "source_run": str(OLD_RUN),
        "source_state": str(resume_state),
        "continuation_run": str(RUN),
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "trainer_sha256": sha256(TRAIN),
        "overlay_sha256": sha256(WRAPPER),
        "authorization_sha256": sha256(AUTH),
        "disk_free_at_prepare": shutil.disk_usage(ROOT).free,
    }
    write_json(REPORT / "CONTINUATION_MANIFEST.json", manifest)
    record("continuation_prepared", **manifest)
    return resume_state


def launch(resume: Path, restart: int) -> subprocess.Popen:
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE:
        record("disk_gate_blocked", free_bytes=free, minimum_bytes=MIN_FREE)
        raise RuntimeError(f"disk free {free} below safety floor {MIN_FREE}")
    cmd = [
        TORCHRUN,
        "--standalone",
        "--nproc_per_node=8",
        str(WRAPPER),
        "--arm",
        "K7-B",
        "--run-id",
        RUN.name,
        "--limit",
        "50000",
        "--state-interval",
        "1000",
        "--checkpoint-root",
        str(CHECKPOINT_ROOT),
        "--v0921-spec",
        str(ROOT / "data/v3_v2_plus_v0921_20260922/v0921_spec.json"),
        "--resume",
        str(resume),
    ]
    log_path = REPORT / f"training_attempt_{restart:02d}.log"
    stream = log_path.open("a", encoding="utf-8")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{CODE}:{CODE / 'scripts'}",
        K7_TRAINING_RUNTIME="1",
        K7_SKIP_MODEL_BROADCAST="1",
        K7_SEQUENTIAL_RESUME="1",
        NCCL_ASYNC_ERROR_HANDLING="1",
        TORCH_NCCL_ENABLE_MONITORING="1",
        TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="600",
    )
    write_json(
        RUN / "heartbeat.json",
        {
            "state": "INITIALIZING",
            "step": read_json(resume / "checkpoint.json").get("step", 0),
            "time": time.time(),
            "supervisor_attempt": restart,
        },
    )
    proc = subprocess.Popen(
        cmd,
        cwd=CODE,
        env=env,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    record("training_launched", restart=restart, pid=proc.pid, resume=str(resume), command=cmd)
    return proc


def monitor_training() -> None:
    resume = prepare_continuation()
    for restart in range(1, MAX_RESTARTS + 1):
        if (RUN / "DONE.json").is_file():
            break
        if restart > 1:
            resume = RUN / "last_state" if (RUN / "last_state").is_dir() else resume
        proc = launch(resume, restart)
        launched_at = time.time()
        while proc.poll() is None and not (RUN / "DONE.json").is_file():
            heartbeat = read_json(RUN / "heartbeat.json")
            free = shutil.disk_usage(ROOT).free
            record(
                "training_heartbeat",
                pid=proc.pid,
                step=heartbeat.get("step"),
                state=heartbeat.get("state"),
                free_bytes=free,
                attempt=restart,
            )
            heartbeat_time = float(heartbeat.get("time", 0))
            heartbeat_state = heartbeat.get("state")
            stale_seconds = time.time() - heartbeat_time if heartbeat_time else 0
            startup_stale = (
                heartbeat_state == "INITIALIZING"
                and time.time() - launched_at > STARTUP_TIMEOUT
            )
            heartbeat_stale = (
                heartbeat_state == "TRAINING" and stale_seconds > HEARTBEAT_TIMEOUT
            )
            if startup_stale or heartbeat_stale:
                record(
                    "training_stale_timeout",
                    pid=proc.pid,
                    step=heartbeat.get("step"),
                    state=heartbeat_state,
                    stale_seconds=stale_seconds,
                    startup_stale=startup_stale,
                    reason="Terminate a live-but-stalled DDP attempt so recovery can resume from last_state.",
                )
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=15)
                break
            time.sleep(60)
        returncode = proc.poll()
        if (RUN / "DONE.json").is_file():
            record("training_50k_complete", returncode=returncode)
            return
        resume = RUN / "last_state" if (RUN / "last_state").is_dir() else resume
        record(
            "training_restart",
            returncode=returncode,
            resume=str(resume),
            reason="worker/process exited before DONE.json",
        )
        time.sleep(min(300, restart * 15))
    raise RuntimeError("K7-B 50K recovery budget exhausted")


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        record(
            "duplicate_supervisor_blocked",
            reason="A K7-B 50K supervisor already owns the exclusive recovery lock.",
        )
        return
    record("supervisor_start", arm="K7-B", target_step=50000)
    if os.environ.get("K7_SKIP_20K_INFERENCE") == "1":
        record(
            "decision_skip_20k_inference",
            reason="User explicitly requested immediate 20K->50K training.",
            retained_partial_outputs=True,
        )
    else:
        wait_for_final_protocol()
    pause_old_inference()
    monitor_training()
    write_json(
        REPORT / "DONE_50K.json",
        {
            "status": "completed",
            "arm": "K7-B",
            "step": 50000,
            "run": str(RUN),
            "checkpoint": str(CHECKPOINT_ROOT / "global_step_50000"),
            "finished_at": time.time(),
        },
    )
    record("supervisor_complete", step=50000)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record("supervisor_failed", error=repr(exc))
        raise
