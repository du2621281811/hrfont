"""One-shot approved resource recovery: I4 matched/archive, then unchanged I5 queue.

Deploy outside the sealed numerical snapshots. Never repeat I4 training/formal inference.
"""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/root/projects/hrfont')
CODE4 = Path('/root/projects/hrfont_i34_20260915.bJikwQ')
CODE5 = Path('/root/projects/hrfont_i56_20260916.xUFfHO')
REC = ROOT / 'reports/i45_recovery_20260916'
sys.path.insert(0, str(CODE4 / 'scripts'))
import queue_i34_20260916 as q4

spec = importlib.util.spec_from_file_location('sealed_i5_queue', CODE5 / 'scripts/queue_i56_20260916.py')
q5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q5)


def inspect():
    q4.guard()
    q5.guard()
    s4, f4 = q4.read(q4.OUT / 'state.json'), q4.read(q4.OUT / 'failure.json')
    s5, f5 = q5.read(q5.OUT / 'state.json'), q5.read(q5.OUT / 'failure.json')
    assert s4['state'] == 'I4_TRAIN' and 'New GPU context:' in f4['error']
    assert s5['state'] == 'REVIEW_REQUIRED' and 'I4 predecessor failed' in f5['error']
    for state in (s4, s5):
        for key in ('pid', 'child_pid'):
            pid = state.get(key)
            assert not pid or not Path(f'/proc/{pid}').exists(), ('Process still alive', pid)
    run = ROOT / 'runs/I4-V0916-S3407'
    d = q4.read(run / 'DONE.json')
    assert d['arm'] == 'I4' and d['step'] == d['attempt'] == 10000
    assert d['inference_complete'] and not d['smoke']
    formal = q4.read(run / 'eval_step_10000/DONE.json')
    assert formal['status'] == 'completed' and formal['images'] == 4096
    ck = q4.read(run / 'global_step_10000/checkpoint.json')
    assert ck['complete'] and ck['step'] == 10000 and ck['arm'] == 'I4'
    ex = q4.read(q4.OUT / 'I4_TRAIN_EXIT.json')
    assert ex['returncode'] == 0 and ex['log_sha256'] == q4.sha(q4.OUT / 'I4_TRAIN.log')
    absent = [q4.OUT / 'I4_K1248.log', run / 'K1248_DONE.json',
              ROOT / 'reports/experiments/I/I4', q4.OUT / 'I4_ARCHIVE_READY.json',
              ROOT / 'runs/I5-V0916-S3407', ROOT / 'runs/I5-PREFLIGHT-V0916-S3407',
              q5.OUT / 'CALIBRATE.log', q5.PREP / 'calibration.json', REC / 'history']
    absent += [ROOT / f'reports/g_v0913_shot_k1248/preds/I4_s{k}' for k in (1, 2, 4, 8)]
    assert not any(p.exists() for p in absent), 'Partial outputs need separate recovery review'
    for name in ('eval_i34_k1248.py', 'archive_i34_k1248.py', 'seal_i_eval_archive.py'):
        assert (CODE4 / 'scripts' / name).is_file(), name
    q4.gpu_admission()
    q5.gpu_admission()
    return dict(prior_i4_state=s4, prior_i4_failure=f4, prior_i5_state=s5,
                prior_i5_failure=f5, i4_done=d, i4_formal_done=formal,
                i4_ema_sha256=q4.sha(run / 'global_step_10000/ema.pth'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    REC.mkdir(parents=True, exist_ok=True)
    with (REC / 'recovery.lock').open('a+') as recovery, (q5.OUT / 'supervisor.lock').open('a+') as own, (ROOT / 'reports/i_20260915/queue.lock').open('a+') as lock:
        for handle in (recovery, own, lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = inspect()
        if not a.execute:
            print('RESOURCE_RECOVERY_DRY_RUN_PASSED', flush=True)
            return
        audit.update(time=time.time(), authorization='PI: 启动训练',
                     recovery_sha256=q4.sha(Path(__file__)),
                     note='I4 matched/archive only; unchanged I5 gates and no I6')
        history = REC / 'history'
        history.mkdir()
        q4.write(REC / 'AUTHORIZED_RECOVERY.json', audit)
        for label, module in (('i4', q4), ('i5', q5)):
            folder = history / label
            folder.mkdir()
            for name in ('state.json', 'failure.json'):
                shutil.copy2(module.OUT / name, folder / name)
                assert module.sha(module.OUT / name) == module.sha(folder / name)
        # Keep all original failures, but remove their active marker only after verified backup.
        (q4.OUT / 'failure.json').rename(history / 'i4/active_failure_original.json')
        try:
            q4.run('I4_K1248', ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
                              'scripts/eval_i34_k1248.py', '--arm', 'I4', '--run-id', 'I4-V0916-S3407'], lock)
            q4.archive_arm('I4', 'I4-V0916-S3407')
            q4.write(q4.OUT / 'state.json', dict(state='COMPUTE_COMPLETE_ARCHIVE_PENDING', time=time.time(), pid=os.getpid()))
            # No I5 GPU stage has run; re-arm the original queue after exact state comparison.
            for name in ('state.json', 'failure.json'):
                assert q5.sha(q5.OUT / name) == q5.sha(history / 'i5' / name)
                (q5.OUT / name).rename(history / 'i5' / ('active_' + name))
            q4.write(REC / 'HANDOFF.json', dict(time=time.time(), pid=os.getpid(), next='Unchanged I5 GPU preflight and main-only training'))
        except BaseException as exc:
            q4.write(q4.OUT / 'failure.json', dict(time=time.time(), error=repr(exc), recovery=str(REC)))
            q4.write(REC / 'failure.json', dict(time=time.time(), error=repr(exc)))
            raise
    # Release both queue locks before exec; the original I5 singleton/compute locks apply.
    os.execv(q5.PY, [q5.PY, str(CODE5 / 'scripts/queue_i56_20260916.py'), '--execute'])


if __name__ == '__main__':
    main()
