"""Persistent, resumable normal-protocol inference for final K6 A/B/C models."""
import json
import os
import subprocess
import time
from pathlib import Path


CODE = Path('/root/projects/hrfont_k6_20260920_r3')
ROOT = Path('/root/projects/hrfont')
MANIFEST_ROOT = ROOT / 'experiments/K'
OUT_ROOT = Path('/root/data2/hrfont_k6_20260920/inference_final_20260921')
REPORT = ROOT / 'reports/k6_final_inference_20260921'
PY = '/root/miniforge3/envs/boogu/bin/torchrun'
EVAL = 'scripts/eval_k6_protocol_abc.py'
CUDA = '0,1,2,3,4,5,6,7'

JOBS = [
    ('K6-A', '/root/data2/hrfont_k6_20260920/checkpoints_ext20k/K6-A-R2-V2-K0-S3407-EXT20K-DATA2/global_step_20000'),
    ('K6-B-RSI-NOOFFSETLOSS', '/root/data2/hrfont_k6_20260920/checkpoints_ext20k/K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407-EXT20K-DATA2/global_step_20000'),
    ('K6-C-RSI-AUX-NOOFFSETLOSS', '/root/data2/hrfont_k6_20260920/checkpoints_c10k/K6-C-RSI-AUX-NOOFFSETLOSS-K0-S3407-10K/global_step_10000'),
]
# The established normal protocol is train/val/test × 1/2/4/8-shot.
# K6 uses the frozen K5 train panel (2304 rows) and the fixed K6 VAL192/TEST
# manifests; keeping the manifest names explicit prevents accidental split
# substitution when the queue is resumed after a network interruption.
SPLITS = [
    ('train', 'K_TRAIN_K5FIXED.json', 2304),
    ('val', 'K_VAL192.json', 192),
    ('test', 'K_TEST.json', 2816),
]


def write_status(payload):
    REPORT.mkdir(parents=True, exist_ok=True)
    tmp = REPORT / 'queue_status.json.tmp'
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(REPORT / 'queue_status.json')


def complete(out, expected):
    done, metrics = out / 'DONE.json', out / 'metrics.json'
    if not (done.is_file() and metrics.is_file()):
        return False
    try:
        data = json.loads(done.read_text())
        return data.get('status') == 'completed' and data.get('images') == expected
    except (OSError, ValueError):
        return False


def main():
    queue = dict(status='running', started_at=time.time(), code=str(CODE),
                 protocol='K-original-CFG1-DPM20-v1', weights='EMA', jobs=[])
    write_status(queue)
    for arm, checkpoint in JOBS:
        for split, manifest_name, expected in SPLITS:
            out = OUT_ROOT / arm / split
            record = dict(arm=arm, split=split, checkpoint=checkpoint, out=str(out),
                          status='running', started_at=time.time(), expected_images=expected)
            queue['jobs'].append(record)
            write_status(queue)
            if complete(out, expected):
                record.update(status='already_complete', finished_at=time.time())
                write_status(queue)
                continue
            out.mkdir(parents=True, exist_ok=True)
            log = REPORT / f'{arm.replace("/", "_")}_{split}.log'
            command = [PY, '--standalone', '--nproc_per_node=8', EVAL,
                       '--arm', arm, '--checkpoint', checkpoint,
                       '--manifest', str(MANIFEST_ROOT / manifest_name), '--out', str(out)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=CUDA, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                        stdout=stream, stderr=subprocess.STDOUT)
            record.update(finished_at=time.time(), returncode=result.returncode,
                          status='completed' if result.returncode == 0 and complete(out, expected) else 'failed',
                          log=str(log))
            write_status(queue)
            if record['status'] != 'completed':
                queue.update(status='failed', finished_at=time.time())
                write_status(queue)
                raise SystemExit(f'inference failed: {arm} {split}; see {log}')
    queue.update(status='completed', finished_at=time.time())
    write_status(queue)


if __name__ == '__main__':
    main()
