#!/usr/bin/env python3
"""Bounded 8-V100 continuation queue; dry-run by default, never kills jobs.

Waits for the incumbent evaluation process AND all CUDA compute clients to
finish, then runs two existing-recipe +5k controls sequentially. No Git overlay,
checkpoint removal, unbounded training, or automatic scientific selection.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('/root/projects/hrfont')
PY = '/root/miniforge3/envs/boogu/bin/python'
RUNS = (
    ('G-CONT-G2-8gpu-V0914-A-S3407', 'F2', 'G2-F2-V0913-A-S3407', '29551'),
    ('G-CONT-G2RL-8gpu-V0914-A-S3407', 'F2RL', 'G2-RL-V0913-A-S3407', '29552'),
)


def command(run):
    name, arm, parent, port = run
    return [PY, str(ROOT / 'scripts/launch_g_joint_8gpu.py'), '--yes',
            '--run_id', name, '--arm', arm, '--parent',
            str(ROOT / 'runs' / parent / 'global_step_10000'),
            '--max_steps', '5000', '--port', port, '--no-tc']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def capture(cmd):
    return subprocess.check_output(cmd, text=True, cwd=ROOT).strip()


def busy(wait_pid):
    # nvidia-smi failure is an error, never interpreted as an idle server.
    clients = capture(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    try:
        os.kill(wait_pid, 0)
        incumbent_alive = True
    except ProcessLookupError:
        incumbent_alive = False
    return incumbent_alive or bool(clients)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--wait-pid', type=int, default=0)
    ap.add_argument('--poll-seconds', type=int, default=30)
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps([command(run) for run in RUNS], indent=2))
        return 0
    if args.wait_pid <= 1 or args.poll_seconds < 10:
        ap.error('execution requires a verified incumbent --wait-pid > 1 and poll >= 10')
    out = ROOT / 'reports/g_matched_continue_20260914'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'queue.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another instance owns this queue')
    results = []

    def status(state, **details):
        payload = dict(ts=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                       state=state, completed=results, **details)
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(payload, indent=2) + '\n')
        tmp.replace(out / 'status.json')
        print(json.dumps(payload), flush=True)

    status('WAITING', incumbent_pid=args.wait_pid)
    for run in RUNS:
        name = run[0]
        run_dir = ROOT / 'runs' / name
        if (run_dir / 'DONE.json').is_file():
            done = json.loads((run_dir / 'DONE.json').read_text())
            if done.get('status') != 'completed' or done.get('global_step') != 5000:
                status('BLOCKED', reason='invalid existing DONE', run_id=name)
                return 2
            results.append(dict(run_id=name, state='ALREADY_DONE'))
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            status('BLOCKED', reason='existing incomplete output; manual resume needed', run_id=name)
            return 2
        idle_checks = 0
        while idle_checks < 2:
            if (out / 'STOP').exists():
                status('STOPPED_BEFORE_NEXT_JOB')
                return 0
            try:
                occupied = busy(args.wait_pid)
            except Exception as exc:
                status('WAITING_PROBE_ERROR', error=str(exc))
                time.sleep(args.poll_seconds)
                continue
            idle_checks = 0 if occupied else idle_checks + 1
            status('WAITING' if occupied else 'IDLE_CONFIRM', run_id=name,
                   idle_checks=idle_checks)
            if idle_checks < 2:
                time.sleep(args.poll_seconds)
        # Observed pilot outputs occupy ~4 GiB. Reserve >12 GiB at job start.
        free = shutil.disk_usage(ROOT).free / 2**30
        if free < 12:
            status('BLOCKED_DISK', free_gib=free, run_id=name)
            return 3
        cmd = command(run)
        parent = ROOT / 'runs' / run[2] / 'global_step_10000'
        provenance = {
            'command': cmd, 'git_head': capture(['git', 'rev-parse', 'HEAD']),
            'parent': str(parent), 'parent_unet_sha256': digest(parent / 'unet.pth'),
            'launcher_sha256': digest(ROOT / 'scripts/launch_g_joint_8gpu.py'),
            'train_sha256': digest(ROOT / 'code/variants/cn2west_f123_rsi/FontDiffuser/train.py'),
            'tracked_changes': capture(['git', 'diff', '--stat']),
            'free_gib': free,
        }
        (out / (name + '.launch.json')).write_text(json.dumps(provenance, indent=2) + '\n')
        status('RUNNING', run_id=name, **provenance)
        with (out / (name + '.log')).open('a') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                status('RUNNING', run_id=name, child_pid=child.pid,
                       heartbeat=str(run_dir / 'heartbeat.json'))
                time.sleep(args.poll_seconds)
        done_path = run_dir / 'DONE.json'
        done = json.loads(done_path.read_text()) if done_path.is_file() else {}
        ok = child.returncode == 0 and done.get('status') == 'completed' and done.get('global_step') == 5000
        results.append(dict(run_id=name, state='DONE' if ok else 'FAILED', rc=child.returncode))
        if not ok:
            status('FAILED', reason='inspect logs; no automatic retry or overwrite')
            return 4
    status('ALL_DONE_REVIEW_NEXT_STAGE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
