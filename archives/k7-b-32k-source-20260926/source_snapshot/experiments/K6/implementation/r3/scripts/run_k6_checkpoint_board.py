"""Detached, resumable fixed-VAL192 inference queue for K6 checkpoints."""
import json
import os
import subprocess
import time
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
ROOT = Path("/root/projects/hrfont")
REPORT = ROOT / "reports/k6_checkpoint_20260920"
PYTHON = "/root/miniforge3/envs/boogu/bin/python"
TORCHRUN = "/root/miniforge3/envs/boogu/bin/torchrun"
CUDA = "0,1,2,3,4,5,6,7"
MANIFEST = ROOT / "experiments/K/K_VAL192.json"
EXPECTED = 192
STEPS = [2000, 4000, 5000, 6000, 8000, 10000]
JOBS = [
    (
        "K6-A",
        Path("/root/data1/hrfont_k6_20260920/checkpoints/K6-A-R2-V2-K0-S3407"),
        Path("/root/data1/hrfont_k6_20260920/inference_checkpoint_board/K6-A"),
    ),
    (
        "K6-B-RSI-NOOFFSETLOSS",
        Path("/root/data2/hrfont_k6_20260920/checkpoints/K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407"),
        Path("/root/data2/hrfont_k6_20260920/inference_checkpoint_board/K6-B"),
    ),
]


def write_status(payload):
    REPORT.mkdir(parents=True, exist_ok=True)
    tmp = REPORT / "queue_status.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(REPORT / "queue_status.json")


def complete(out):
    done = out / "DONE.json"
    metrics = out / "metrics.json"
    if not (done.is_file() and metrics.is_file()):
        return False
    try:
        payload = json.loads(done.read_text())
        return payload.get("status") == "completed" and payload.get("images") == EXPECTED
    except (OSError, ValueError):
        return False


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    queue = {
        "status": "running",
        "started_at": time.time(),
        "code": str(CODE),
        "protocol": "K-original-CFG1-DPM20-v1",
        "split": "val",
        "manifest": str(MANIFEST),
        "steps": STEPS,
        "jobs": [],
    }
    write_status(queue)
    for arm, checkpoint_root, output_root in JOBS:
        for step in STEPS:
            checkpoint = checkpoint_root / f"global_step_{step}"
            out = output_root / f"step_{step}"
            record = {
                "arm": arm,
                "step": step,
                "checkpoint": str(checkpoint),
                "out": str(out),
                "status": "pending",
            }
            queue["jobs"].append(record)
            write_status(queue)
            if not (checkpoint / "ema.pth").is_file():
                record["status"] = "missing_checkpoint"
                write_status(queue)
                queue["status"] = "failed"
                queue["finished_at"] = time.time()
                write_status(queue)
                raise SystemExit(f"missing checkpoint: {checkpoint}")
            if complete(out):
                record["status"] = "already_complete"
                record["finished_at"] = time.time()
                write_status(queue)
                continue
            out.mkdir(parents=True, exist_ok=True)
            log = REPORT / f"{arm.replace('/', '_')}_step_{step}.log"
            record["status"] = "running"
            record["started_at"] = time.time()
            record["log"] = str(log)
            write_status(queue)
            command = [
                TORCHRUN,
                "--standalone",
                "--nproc_per_node=8",
                "scripts/eval_k6_protocol.py",
                "--arm",
                arm,
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(MANIFEST),
                "--out",
                str(out),
                "--step",
                str(step),
            ]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=CUDA,
                PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="1",
            )
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=CODE, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record["finished_at"] = time.time()
            record["returncode"] = result.returncode
            record["status"] = "completed" if result.returncode == 0 and complete(out) else "failed"
            write_status(queue)
            if record["status"] != "completed":
                queue["status"] = "failed"
                queue["finished_at"] = time.time()
                write_status(queue)
                raise SystemExit(f"K6 checkpoint inference failed: {arm} step {step}; see {log}")
    builder = [PYTHON, "scripts/build_k6_checkpoint_board.py"]
    build_log = REPORT / "build_board.log"
    with build_log.open("w") as stream:
        result = subprocess.run(builder, cwd=CODE, stdout=stream, stderr=subprocess.STDOUT)
    queue["builder"] = str(build_log)
    queue["status"] = "completed" if result.returncode == 0 else "failed"
    queue["finished_at"] = time.time()
    write_status(queue)
    if result.returncode:
        raise SystemExit(f"checkpoint board build failed; see {build_log}")


if __name__ == "__main__":
    main()
