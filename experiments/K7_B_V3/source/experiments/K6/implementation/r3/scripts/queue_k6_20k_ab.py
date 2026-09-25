"""Persistent serial supervisor for authorized K6-A/B 20k extensions."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


CODE = Path("/root/projects/hrfont_k6_20260920_r3")
ROOT = Path("/root/projects/hrfont")
RUNS = ROOT / "runs"
STORE = Path("/root/data2/hrfont_k6_20260920")
CHECKPOINT_ROOT = STORE / "checkpoints_ext20k"
CTRL = STORE / "control_20k_ab"
BOARD_STATUS = ROOT / "reports/k6_checkpoint_20260920/queue_status.json"
TORCHRUN = "/root/miniforge3/envs/boogu/bin/torchrun"
SCRIPT = CODE / "scripts/train_k6_ab_extend.py"
ENV = dict(os.environ, PYTHONPATH=str(CODE), OMP_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
A_SOURCE = RUNS / "K6-A-R2-V2-K0-S3407"
B_SOURCE = RUNS / "K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407"
A_EXT = "K6-A-R2-V2-K0-S3407-EXT20K-DATA2"
B_EXT = "K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407-EXT20K-DATA2"


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def status(value):
    atomic(CTRL / "status.json", dict(time=time.time(), **value))


def board_ready():
    if not BOARD_STATUS.is_file():
        return False, "missing"
    try:
        d = json.loads(BOARD_STATUS.read_text())
    except ValueError:
        return False, "unreadable"
    if d.get("status") == "completed":
        return True, "completed"
    if d.get("status") == "failed":
        jobs = d.get("jobs", [])
        all_done = bool(jobs) and all(j.get("status") in {"completed", "already_complete"} for j in jobs)
        return all_done, "failed_all_inference_done" if all_done else "failed"
    return False, d.get("status")


def prepare(run_id, source):
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    config = out / "config.json"
    if not config.exists():
        shutil.copy2(source / "config.json", config)
    elif config.read_bytes() != (source / "config.json").read_bytes():
        raise RuntimeError(f"extension config drift: {config}")
    return out


def complete(run_id):
    p = RUNS / run_id / "DONE.json"
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
    except ValueError:
        return False
    return d.get("status") == "completed" and int(d.get("step", -1)) == 20000


def run_arm(label, arm, run_id, source):
    out = prepare(run_id, source)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    attempts = []
    for retry in range(1, 4):
        if complete(run_id):
            status(dict(stage=label + "_ALREADY_COMPLETE", run_id=run_id, attempts=attempts))
            return
        resume = out / "last_state" if (out / "last_state").exists() else source / "last_state"
        if not resume.exists():
            raise RuntimeError(f"missing complete resume state: {resume}")
        log = CTRL / f"{label.lower()}_attempt{retry}.log"
        command = [TORCHRUN, "--standalone", "--nproc_per_node=8", str(SCRIPT),
                   "--arm", arm, "--run-id", run_id, "--limit", "20000",
                   "--state-interval", "200", "--checkpoint-root", str(CHECKPOINT_ROOT),
                   "--extend", "--resume", str(resume)]
        status(dict(stage=label + "_RUNNING", arm=arm, run_id=run_id, retry=retry,
                    resume=str(resume), log=str(log)))
        with log.open("a") as stream:
            result = subprocess.run(command, cwd=CODE, env=ENV, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT)
        attempts.append(dict(retry=retry, returncode=result.returncode, log=str(log), time=time.time()))
        if result.returncode == 0 and complete(run_id):
            status(dict(stage=label + "_COMPLETED", arm=arm, run_id=run_id,
                        step=20000, attempts=attempts))
            return
        status(dict(stage=label + "_RETRYING", arm=arm, run_id=run_id, attempts=attempts))
        time.sleep(20)
    raise RuntimeError(f"{label} failed after three resumable attempts")


def main():
    CTRL.mkdir(parents=True, exist_ok=True)
    status(dict(stage="WAIT_CHECKPOINT_BOARD", board_status=str(BOARD_STATUS)))
    while True:
        ready, board_state = board_ready()
        if ready:
            break
        if board_state == "failed":
            raise RuntimeError("checkpoint inference failed before A/B extension")
        time.sleep(30)
    status(dict(stage="BOARD_READY", board_status=board_state))
    run_arm("K6_A_EXT20K", "K6-A", A_EXT, A_SOURCE)
    run_arm("K6_B_EXT20K", "K6-B-RSI-NOOFFSETLOSS", B_EXT, B_SOURCE)
    status(dict(stage="ALL_COMPLETED", A_run=A_EXT, B_run=B_EXT, board_status=board_state))
    atomic(CTRL / "DONE.json", dict(status="completed", time=time.time(), A_run=A_EXT, B_run=B_EXT))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status(dict(stage="FAILED", error=repr(exc)))
        raise
