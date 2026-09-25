"""Serial K6-B queue: wait for K6-A, run isolated preflight, then formal training."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

C = Path('/root/projects/hrfont_k6_20260920_r3')
RUN = Path('/root/projects/hrfont/runs')
STORE = Path('/root/data2/hrfont_k6_20260920')
CODE_STORE = Path('/root/data1/hrfont_k6_20260920')
CTRL = STORE / 'control_rsi_no_offset'
A_DONE = RUN / 'K6-A-R2-V2-K0-S3407' / 'DONE.json'
ARM = 'K6-B-RSI-NOOFFSETLOSS'
FORMAL = ARM + '-R1-V2-K0-S3407'
PREFLIGHT = ARM + '-R1-PREFLIGHT-RETRY2-20260920'

CTRL.mkdir(parents=True, exist_ok=True)
lock = (CTRL / 'QUEUE.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def event(action, **payload):
    row = dict(time=time.time(), action=action, **payload)
    with (CTRL / 'DECISIONS.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


ENV = dict(os.environ, PYTHONPATH=str(C), OMP_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
PY = str(Path(sys.executable).parent / 'python')
TORCHRUN = str(Path(sys.executable).parent / 'torchrun')
SCRIPT = C / 'scripts' / 'train_k6.py'
CHECKPOINTS = Path('/root/data2/hrfont_k6_20260920/checkpoints')
PREFLIGHT_STATES = Path('/root/data2/hrfont_k6_preflight_rsi_no_offset')


def launch(name, limit, smoke, resume=False):
    args = [TORCHRUN, '--standalone', '--nproc_per_node=8', str(SCRIPT),
            '--arm', ARM, '--run-id', name, '--limit', str(limit),
            '--state-interval', '2' if smoke else '200',
            '--checkpoint-root', str(PREFLIGHT_STATES if smoke else CHECKPOINTS)]
    if smoke:
        args.append('--smoke')
    if resume:
        args += ['--resume', str(RUN / name / 'last_state')]
    log = CTRL / (name + ('_resume' if resume else '') + '.log')
    write(CTRL / 'status.json', dict(stage='LAUNCH', name=name, limit=limit,
                                     smoke=smoke, resume=resume, time=time.time()))
    event('launch', arm=ARM, name=name, limit=limit, smoke=smoke, resume=resume)
    with log.open('a') as f:
        result = subprocess.run(args, cwd=C, env=ENV, stdout=f, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
    if result.returncode:
        event('failed', name=name, returncode=result.returncode)
        raise RuntimeError(f'{name} exited with {result.returncode}; inspect {log}')
    return json.loads((RUN / name / 'DONE.json').read_text())


def validate_preflight(name):
    rows = []
    for rank in range(8):
        path = RUN / name / f'rank{rank}.jsonl'
        assert path.is_file(), path
        rank_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rank_rows and rank_rows[-1]['step'] == 18, (rank, len(rank_rows))
        assert all(row['skips'] == 0 and row['ddp_spread'] < 1e-4 for row in rank_rows)
        assert all(row['k6_aux'] is False for row in rank_rows)
        assert all(row.get('offset_loss_weight') == 0.0 for row in rank_rows)
        assert all('offset_diagnostic' in row for row in rank_rows)
        rows.extend(rank_rows)
    config = json.loads((RUN / name / 'config.json').read_text())
    assert 'hrfont_k5' not in config['parent'] and 'hrfont_k6' not in config['parent']
    assert config['architecture'] == 'K5-B-dual-Ec-Es-RSI'
    assert config['pure_noise_auxiliary'] is False
    assert config['offset_loss_weight'] == 0.0
    identity = json.loads((C / 'K6_CODE_IDENTITY.json').read_text())
    result = dict(status='PASS', arm=ARM, identity=identity, time=time.time(),
                  checks=['8-rank forward/backward', '2-to-18 full-state resume',
                          'no auxiliary branch', 'offset coefficient zero',
                          'offset path diagnostic retained', 'DDP parity'],
                  records=len(rows), last_step=18,
                  code_sha256=hashlib.sha256((C / 'scripts' / 'train_k6.py').read_bytes()).hexdigest())
    write(CODE_STORE / 'control' / 'PREFLIGHT_PASSED_RSI_NO_OFFSET.json', result)
    event('preflight_passed', records=len(rows), last_step=18)


try:
    write(CTRL / 'status.json', dict(stage='WAIT_K6_A_DONE', a_done=str(A_DONE), time=time.time()))
    event('waiting_for_k6_a', a_done=str(A_DONE))
    while not A_DONE.is_file():
        time.sleep(30)
    done = json.loads(A_DONE.read_text())
    assert done['status'] == 'completed' and done['step'] == 10000 and not done['smoke']
    event('k6_a_completed', step=done['step'], attempt=done['attempt'])
    pre = launch(PREFLIGHT, 2, True)
    assert pre['step'] == 2
    pre = launch(PREFLIGHT, 18, True, True)
    assert pre['step'] == 18
    validate_preflight(PREFLIGHT)
    formal = launch(FORMAL, 10000, False)
    assert formal['status'] == 'completed' and formal['step'] == 10000 and not formal['smoke']
    event('training_completed', step=formal['step'], attempt=formal['attempt'])
    write(CTRL / 'status.json', dict(stage='TRAINING_COMPLETED_EVALUATION_PENDING',
                                     run_id=FORMAL, time=time.time()))
except Exception as exc:
    event('needs_diagnosis', error=repr(exc))
    write(CTRL / 'status.json', dict(stage='NEEDS_DIAGNOSIS', error=repr(exc), time=time.time()))
    raise
