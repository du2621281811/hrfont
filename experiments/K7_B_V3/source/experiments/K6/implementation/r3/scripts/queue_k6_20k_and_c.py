"""Persistent serial supervisor for K6 A/B 20k extensions followed by K6-C.

The supervisor waits for the already-running checkpoint review queue, then
resumes A and B from their complete 10k states into separate data2 storage.
K6-C is the literal 20-successful-update combined validation requested by the
user: K6-A's pure-noise generation ranking plus K6-B's zero offset loss.
"""
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
C_CHECKPOINT_ROOT = STORE / "checkpoints_c20"
CTRL = STORE / "control_20k"
BOARD_STATUS = ROOT / "reports/k6_checkpoint_20260920/queue_status.json"
PY = "/root/miniforge3/envs/boogu/bin/python"
TORCHRUN = "/root/miniforge3/envs/boogu/bin/torchrun"
SCRIPT = CODE / "scripts/train_k6_extension.py"
ENV = dict(
    os.environ,
    PYTHONPATH=str(CODE),
    OMP_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
)

A_SOURCE = RUNS / "K6-A-R2-V2-K0-S3407"
B_SOURCE = RUNS / "K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407"
A_EXT = "K6-A-R2-V2-K0-S3407-EXT20K-DATA2"
B_EXT = "K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407-EXT20K-DATA2"
C_RUN = "K6-C-RSI-AUX-NOOFFSETLOSS-S3407-20STEP"


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def status(value):
    atomic(CTRL / "status.json", dict(time=time.time(), **value))


def board_finished():
    if not BOARD_STATUS.is_file():
        return False, "missing"
    try:
        data = json.loads(BOARD_STATUS.read_text())
    except ValueError:
        return False, "unreadable"
    if data.get("status") == "completed":
        return True, "completed"
    if data.get("status") == "failed":
        jobs = data.get("jobs", [])
        all_done = bool(jobs) and all(j.get("status") in {"completed", "already_complete"} for j in jobs)
        return all_done, "failed_all_inference_done" if all_done else "failed"
    return False, data.get("status")


def prepare_extension(run_id, source):
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    config = out / "config.json"
    source_config = source / "config.json"
    if not config.exists():
        shutil.copy2(source_config, config)
    elif config.read_bytes() != source_config.read_bytes():
        raise RuntimeError(f"extension config drift: {config}")
    return out


def completed(run_id, step):
    done = RUNS / run_id / "DONE.json"
    if not done.is_file():
        return False
    try:
        payload = json.loads(done.read_text())
    except ValueError:
        return False
    return payload.get("status") == "completed" and int(payload.get("step", -1)) == step


def launch_training(label, arm, run_id, source, limit, checkpoint_root, extra):
    out = prepare_extension(run_id, source) if source else RUNS / run_id
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    attempts = []
    for retry in range(1, 4):
        if completed(run_id, limit):
            status(dict(stage=label + "_ALREADY_COMPLETE", run_id=run_id, step=limit, attempts=attempts))
            return
        if source:
            resume = out / "last_state" if (out / "last_state").exists() else source / "last_state"
            if not resume.exists():
                raise RuntimeError(f"no complete resume state for {label}: {resume}")
        else:
            resume = None
        command = [
            TORCHRUN,
            "--standalone",
            "--nproc_per_node=8",
            str(SCRIPT),
            "--arm",
            arm,
            "--run-id",
            run_id,
            "--limit",
            str(limit),
            "--state-interval",
            "200" if limit > 1000 else "2",
            "--checkpoint-root",
            str(checkpoint_root),
        ] + list(extra)
        if resume:
            command += ["--resume", str(resume)]
        log = CTRL / f"{label.lower()}_attempt{retry}.log"
        status(dict(stage=label + "_RUNNING", arm=arm, run_id=run_id, limit=limit,
                    retry=retry, resume=str(resume) if resume else None, log=str(log)))
        with log.open("a") as stream:
            result = subprocess.run(command, cwd=CODE, env=ENV, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT)
        attempts.append(dict(retry=retry, returncode=result.returncode, log=str(log), time=time.time()))
        if result.returncode == 0 and completed(run_id, limit):
            status(dict(stage=label + "_COMPLETED", arm=arm, run_id=run_id, step=limit,
                        attempts=attempts))
            return
        status(dict(stage=label + "_RETRYING", arm=arm, run_id=run_id, limit=limit,
                    attempts=attempts))
        time.sleep(20)
    raise RuntimeError(f"{label} did not complete after three resumable attempts")


def main():
    CTRL.mkdir(parents=True, exist_ok=True)
    status(dict(stage="WAIT_CHECKPOINT_BOARD", board_status=str(BOARD_STATUS)))
    while True:
        finished, board_state = board_finished()
        if finished:
            break
        if board_state == "failed":
            status(dict(stage="BOARD_FAILED_BUT_NO_TRAINING_STARTED", board_status=board_state))
            raise RuntimeError("checkpoint board failed; inspect it before training extensions")
        time.sleep(30)

    status(dict(stage="BOARD_READY", board_status=board_state))
    launch_training("K6_A_EXT20K", "K6-A", A_EXT, A_SOURCE, 20000, CHECKPOINT_ROOT, ["--extend"])
    launch_training("K6_B_EXT20K", "K6-B-RSI-NOOFFSETLOSS", B_EXT, B_SOURCE, 20000, CHECKPOINT_ROOT, ["--extend"])
    launch_training("K6_C_20STEP", "K6-C-RSI-AUX-NOOFFSETLOSS", C_RUN, None, 20,
                    C_CHECKPOINT_ROOT, ["--c-short"])
    status(dict(stage="ALL_COMPLETED", arms={"K6-A": A_EXT, "K6-B": B_EXT, "K6-C": C_RUN},
                board_status="completed"))
    atomic(CTRL / "DONE.json", dict(status="completed", time=time.time(),
                                     A_run=A_EXT, B_run=B_EXT, C_run=C_RUN,
                                     C_semantics="K6-A pure-noise ranking + K6-B offset loss=0",
                                     C_updates=20))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status(dict(stage="FAILED", error=repr(exc)))
        raise
