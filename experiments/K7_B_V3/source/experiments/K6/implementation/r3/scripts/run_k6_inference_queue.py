"""Sequential, resumable K6 normal-protocol inference queue."""
import json
import os
import subprocess
import time
from pathlib import Path


CODE = Path(__file__).resolve().parents[1]
ROOT = Path("/root/projects/hrfont")
MANIFEST_ROOT = ROOT / "experiments/K"
STATUS_DIR = ROOT / "reports/k6_protocol_20260920"
PYTHON = "/root/miniforge3/envs/boogu/bin/python"
TORCHRUN = "/root/miniforge3/envs/boogu/bin/torchrun"
CUDA = "0,1,2,3,4,5,6,7"

JOBS = [
    (
        "K6-A",
        "/root/data1/hrfont_k6_20260920/checkpoints/K6-A-R2-V2-K0-S3407/global_step_10000",
        "/root/data1/hrfont_k6_20260920/inference/K6-A",
    ),
    (
        "K6-B-RSI-NOOFFSETLOSS",
        "/root/data2/hrfont_k6_20260920/checkpoints/K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407/global_step_10000",
        "/root/data2/hrfont_k6_20260920/inference/K6-B",
    ),
]
SPLITS = [("test", "K_TEST.json"), ("val", "K_VAL192.json")]


def write_status(payload):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_DIR / "queue_status.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(STATUS_DIR / "queue_status.json")


def is_complete(out, expected):
    done = out / "DONE.json"
    metrics = out / "metrics.json"
    if not (done.is_file() and metrics.is_file()):
        return False
    try:
        payload = json.loads(done.read_text())
        return payload.get("status") == "completed" and payload.get("images") == expected
    except (OSError, ValueError):
        return False


def main():
    queue = {
        "status": "running",
        "started_at": time.time(),
        "code": str(CODE),
        "protocol": "K-original-CFG1-DPM20-v1",
        "jobs": [],
    }
    write_status(queue)
    for arm, checkpoint, output_root in JOBS:
        for split, manifest_name in SPLITS:
            manifest = MANIFEST_ROOT / manifest_name
            expected = 2816 if split == "test" else 192
            out = Path(output_root) / split
            record = {
                "arm": arm,
                "split": split,
                "checkpoint": checkpoint,
                "out": str(out),
                "status": "running",
                "started_at": time.time(),
            }
            queue["jobs"].append(record)
            write_status(queue)
            if is_complete(out, expected):
                record["status"] = "already_complete"
                record["finished_at"] = time.time()
                write_status(queue)
                continue
            out.mkdir(parents=True, exist_ok=True)
            log = STATUS_DIR / f"{arm.replace('/', '_')}_{split}.log"
            command = [
                TORCHRUN,
                "--standalone",
                "--nproc_per_node=8",
                "scripts/eval_k6_protocol.py",
                "--arm",
                arm,
                "--checkpoint",
                checkpoint,
                "--manifest",
                str(manifest),
                "--out",
                str(out),
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
            record["status"] = "completed" if result.returncode == 0 and is_complete(out, expected) else "failed"
            record["log"] = str(log)
            write_status(queue)
            if record["status"] != "completed":
                queue["status"] = "failed"
                queue["finished_at"] = time.time()
                write_status(queue)
                raise SystemExit(f"K6 inference failed: {arm} {split}; see {log}")
    queue["status"] = "completed"
    queue["finished_at"] = time.time()
    write_status(queue)


if __name__ == "__main__":
    main()
