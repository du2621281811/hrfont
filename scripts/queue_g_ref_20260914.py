#!/usr/bin/env python3
"""Approved P0 + four matched R arms, isolated immutable code, all eight V100s.

Never kills incumbent jobs or deletes prior experiments. N and TC are decision
gated after P1 validation, not silently selected by diffusion loss alone.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

from queue_g_matched_continue_20260914 import busy, digest

ROOT = Path('/root/projects/hrfont')
CODE = Path(__file__).resolve().parents[1]
PY = '/root/miniforge3/envs/boogu/bin/python'
OUT = ROOT / 'reports/g_ref_20260914'
ARMS = [('A0', 'F2', False), ('A1', 'F2', True), ('B0', 'F2RL', False), ('B1', 'F2RL', True)]


def command(tag, arm, reader, smoke=False):
    parent = 'G2-F2-V0913-A-S3407' if arm == 'F2' else 'G2-RL-V0913-A-S3407'
    name = f'G-REF-{tag}-V0914-S3407'
    cmd = [PY, str(CODE / 'scripts/launch_g_joint_8gpu.py'), '--yes', '--run_id', name,
           '--arm', arm, '--parent', str(ROOT / 'runs' / parent / 'global_step_10000'),
           '--max_steps', '100' if smoke else '2500', '--port', '29561', '--no-tc', '--ref_recipe']
    if reader:
        cmd += ['--ref_aggregation']
    if smoke:
        cmd += ['--smoke']
    return name, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--wait-pid', type=int, required=True)
    ap.add_argument('--idle-background-pids', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    a = ap.parse_args()
    if not a.execute:
        print(json.dumps([command(*row) for row in ARMS], indent=2))
        return
    if a.wait_pid <= 1:
        raise ValueError('verified incumbent PID required')
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'status.json').exists():
        old = json.loads((OUT / 'status.json').read_text())
        if old.get('state') != 'NEEDS_ATTENTION':
            raise RuntimeError('existing queue is not in a failed state; inspect before restarting')
        archive = OUT / f'prior_attempt_{a.attempt}.status.json'
        if archive.exists():
            raise RuntimeError('attempt identifier already used')
        shutil.copy2(OUT / 'status.json', archive)
    backgrounds = [int(x) for x in a.idle_background_pids.split(',')]
    completed = []

    def status(state, **kw):
        payload = dict(state=state, time=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                       code_root=str(CODE), attempt=a.attempt, completed=completed, **kw)
        tmp = OUT / 'status.tmp'
        tmp.write_text(json.dumps(payload, indent=2) + '\n')
        tmp.replace(OUT / 'status.json')
        print(json.dumps(payload), flush=True)

    def idle():
        count = 0
        while count < 2:
            if (OUT / 'STOP').exists():
                raise RuntimeError('STOP requested before next job')
            count = 0 if busy(a.wait_pid, backgrounds) else count + 1
            status('WAITING_FOR_INCUMBENT' if count == 0 else 'IDLE_CONFIRM', checks=count)
            if count < 2:
                time.sleep(30)
        if shutil.disk_usage(ROOT).free < 8 * 2**30:
            raise RuntimeError('less than 8 GiB free; no automatic deletion')

    def run(tag, cmd):
        status('RUNNING', task=tag, command=cmd)
        with (OUT / (tag + '.log')).open('a') as log:
            child = subprocess.Popen(cmd, cwd=CODE,
                                     env=dict(os.environ, HRFONT_CODE_ROOT=str(CODE), OMP_NUM_THREADS='1'),
                                     stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                status('RUNNING', task=tag, child_pid=child.pid)
                time.sleep(30)
        if child.returncode:
            raise RuntimeError(f'{tag} failed rc={child.returncode}; inspect log, no automatic retry')
        completed.append(tag)

    try:
        idle()
        run('unit_tests', [PY, str(CODE / 'tests/test_g_ref_aggregation.py'), '-v'])
        run('tc_regressions', [PY, str(CODE / 'scripts/test_tc_v2_integration_review.py')])
        run('P0', [PY, str(CODE / 'scripts/eval_g_ref_panel.py'), '--out', str(OUT / 'P0')])
        for tag, arm in [('SMOKE-A1', 'F2'), ('SMOKE-B1', 'F2RL')]:
            idle()
            if a.attempt > 1:
                tag += f'-T{a.attempt}'
            name, cmd = command(tag, arm, True, smoke=True)
            run(tag, cmd)
            rows = [json.loads(x) for x in (ROOT / 'runs' / name / 'train_log.jsonl').read_text().splitlines()]
            if rows[-1]['step'] != 100 or not any(r['ref_grad_norm'] > 0 for r in rows):
                raise RuntimeError('smoke did not reach 100 updates with reader gradients')
            if abs(rows[-1]['lr'] - 5e-6) > 1e-10:
                raise RuntimeError('actual-update LR warmup mismatch')
        for tag, arm, reader in ARMS:
            idle()
            name, cmd = command(tag, arm, reader)
            run_dir = ROOT / 'runs' / name
            if run_dir.exists():
                raise RuntimeError(f'existing run: {name}; never overwrite')
            parent = Path(cmd[cmd.index('--parent') + 1])
            (OUT / (tag + '.provenance.json')).write_text(json.dumps(dict(
                command=cmd, code_root=str(CODE), parent_unet_sha256=digest(parent / 'unet.pth'),
                train_sha256=digest(CODE / 'code/variants/cn2west_f123_rsi/FontDiffuser/train.py'),
                reader_sha256=digest(CODE / 'scripts/hrfont_ref_aggregation.py')), indent=2) + '\n')
            run(tag, cmd)
            done = json.loads((run_dir / 'DONE.json').read_text())
            if done.get('status') != 'completed' or done.get('global_step') != 2500:
                raise RuntimeError(f'incomplete run {name}')
            idle()
            panel = [PY, str(CODE / 'scripts/eval_g_ref_panel.py'), '--checkpoint',
                     str(run_dir / 'global_step_2500'), '--out', str(OUT / (tag + '_panel'))]
            if arm == 'F2RL':
                panel += ['--rl']
            run(tag + '_panel', panel)
        status('P1_COMPLETE_SELECT_N_AND_TC', remaining='N conditional; TC matched pairs on selected parents, +2500 each')
    except Exception as exc:
        status('NEEDS_ATTENTION', reason=str(exc))
        raise


if __name__ == '__main__':
    main()
