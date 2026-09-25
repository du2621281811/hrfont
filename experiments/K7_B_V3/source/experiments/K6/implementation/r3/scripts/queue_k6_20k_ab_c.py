"""Persistent K6-A 20k -> K6-B 20k -> independent K6-C 10k supervisor."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


CODE = Path('/root/projects/hrfont_k6_20260920_r3')
ROOT = Path('/root/projects/hrfont')
RUNS = ROOT / 'runs'
STORE = Path('/root/data2/hrfont_k6_20260920')
CHECKPOINT_ROOT = STORE / 'checkpoints_ext20k'
C_CHECKPOINT_ROOT = STORE / 'checkpoints_c10k'
CTRL = STORE / 'control_20k_ab_c'
BOARD_STATUS = ROOT / 'reports/k6_checkpoint_20260920/queue_status.json'
TORCHRUN = '/root/miniforge3/envs/boogu/bin/torchrun'
A_SCRIPT = CODE / 'scripts/train_k6_a_extend.py'
BC_SCRIPT = CODE / 'scripts/train_k6_bc.py'
ENV = dict(os.environ, PYTHONPATH=str(CODE), OMP_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')

A_SOURCE = RUNS / 'K6-A-R2-V2-K0-S3407'
B_SOURCE = RUNS / 'K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407'
A_RUN = 'K6-A-R2-V2-K0-S3407-EXT20K-DATA2'
B_RUN = 'K6-B-RSI-NOOFFSETLOSS-R1-V2-K0-S3407-EXT20K-DATA2'
C_RUN = 'K6-C-RSI-AUX-NOOFFSETLOSS-K0-S3407-10K'


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def status(value):
    atomic(CTRL / 'status.json', dict(time=time.time(), **value))


def board_ready():
    if not BOARD_STATUS.is_file():
        return False, 'missing'
    try:
        data = json.loads(BOARD_STATUS.read_text())
    except ValueError:
        return False, 'unreadable'
    if data.get('status') == 'completed':
        return True, 'completed'
    if data.get('status') == 'failed':
        jobs = data.get('jobs', [])
        done = bool(jobs) and all(j.get('status') in {'completed', 'already_complete'} for j in jobs)
        return done, 'failed_all_inference_done' if done else 'failed'
    return False, data.get('status')


def copy_config(run_id, source):
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'config.json'
    if not target.exists():
        shutil.copy2(source / 'config.json', target)
    elif target.read_bytes() != (source / 'config.json').read_bytes():
        raise RuntimeError(f'config drift: {target}')
    return out


def done(run_id, step):
    p = RUNS / run_id / 'DONE.json'
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text())
    except ValueError:
        return False
    return data.get('status') == 'completed' and int(data.get('step', -1)) == step


def run_arm(label, arm, run_id, source, script, limit, checkpoint_root, extra):
    out = copy_config(run_id, source) if source else RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    attempts = []
    for retry in range(1, 4):
        if done(run_id, limit):
            status(dict(stage=label + '_ALREADY_COMPLETE', run_id=run_id, step=limit, attempts=attempts))
            return
        if source:
            resume = out / 'last_state' if (out / 'last_state').exists() else source / 'last_state'
        else:
            resume = out / 'last_state' if (out / 'last_state').exists() else None
            if resume is None and retry > 1 and any(out.iterdir()):
                quarantine = RUNS / f'{run_id}.failed_attempt{retry-1}'
                if not quarantine.exists():
                    out.rename(quarantine)
                out.mkdir(parents=True, exist_ok=True)
        if resume is not None and not resume.exists():
            raise RuntimeError(f'missing resume state: {resume}')
        log = CTRL / f'{label.lower()}_attempt{retry}.log'
        command = [TORCHRUN, '--standalone', '--nproc_per_node=8', str(script),
                   '--arm', arm, '--run-id', run_id, '--limit', str(limit),
                   '--state-interval', '200', '--checkpoint-root', str(checkpoint_root)] + list(extra)
        if resume is not None:
            command += ['--resume', str(resume)]
        status(dict(stage=label + '_RUNNING', arm=arm, run_id=run_id, limit=limit,
                    retry=retry, resume=str(resume) if resume else None, log=str(log)))
        with log.open('a') as stream:
            result = subprocess.run(command, cwd=CODE, env=ENV, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT)
        attempts.append(dict(retry=retry, returncode=result.returncode, log=str(log), time=time.time()))
        if result.returncode == 0 and done(run_id, limit):
            status(dict(stage=label + '_COMPLETED', run_id=run_id, step=limit, attempts=attempts))
            return
        status(dict(stage=label + '_RETRYING', run_id=run_id, limit=limit, attempts=attempts))
        time.sleep(20)
    raise RuntimeError(f'{label} failed after three resumable attempts')


def main():
    CTRL.mkdir(parents=True, exist_ok=True)
    status(dict(stage='WAIT_CHECKPOINT_BOARD', board_status=str(BOARD_STATUS)))
    while True:
        ready, board_state = board_ready()
        if ready:
            break
        if board_state == 'failed':
            raise RuntimeError('checkpoint board failed before K6 extensions')
        time.sleep(30)
    status(dict(stage='BOARD_READY', board_status=board_state))
    run_arm('K6_A_EXT20K', 'K6-A', A_RUN, A_SOURCE, A_SCRIPT, 20000, CHECKPOINT_ROOT, ['--extend'])
    run_arm('K6_B_EXT20K', 'K6-B-RSI-NOOFFSETLOSS', B_RUN, B_SOURCE, BC_SCRIPT, 20000, CHECKPOINT_ROOT, ['--extend'])
    run_arm('K6_C_10K', 'K6-C-RSI-AUX-NOOFFSETLOSS', C_RUN, None, BC_SCRIPT, 10000, C_CHECKPOINT_ROOT, ['--c-mode'])
    status(dict(stage='ALL_COMPLETED', arms={'K6-A': A_RUN, 'K6-B': B_RUN, 'K6-C': C_RUN}, board_status=board_state))
    atomic(CTRL / 'DONE.json', dict(status='completed', time=time.time(), A_run=A_RUN, B_run=B_RUN, C_run=C_RUN,
        C_semantics='independent K0; K6-B RSI architecture; offset loss=0; K6-A pure-noise ranking auxiliary', C_updates=10000))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status(dict(stage='FAILED', error=repr(exc)))
        raise
