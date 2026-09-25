"""Approved I2 5146->10000 continuation, with exclusive lock and disk guard.

Dry run by default. Does not alter model/training implementations or reinitialize runs.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/root/projects/hrfont')
CODE = Path('/root/projects/hrfont_i_20260915.zvbB1H')
OUT = ROOT / 'reports/i_20260915'
RUN = ROOT / 'runs/I2-V0915-S3407'
STATE = RUN / 'state_step_5146'
AUDIT = OUT / 'resume_5146_20260915'
PY = '/root/miniforge3/envs/boogu/bin/python'
BOARD = ROOT / 'reports/g_v0913_shot_k1248'


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def png_ok(path):
    from PIL import Image
    with Image.open(path) as im:
        assert im.size == (96, 96), str(path)
        im.load()
    with Image.open(path) as im:
        im.verify()


def other_i_processes():
    rows = subprocess.check_output(['ps', '-eo', 'pid=,comm=,args='], text=True).splitlines()
    matches = []
    for row in rows:
        fields = row.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, comm, args = fields
        if int(pid) == os.getpid():
            continue
        if not (comm.startswith('python') or comm in ('pt_elastic', 'bash')):
            continue
        words = args.split()
        if any(w.endswith('.py') and w.rsplit('/', 1)[-1].startswith(
                ('train_i', 'eval_i', 'queue_i', 'resume_i_completion')) for w in words):
            matches.append(int(pid))
        elif any(w.endswith('/run_i01_k1248.sh') for w in words):
            matches.append(int(pid))
    return matches


def verify_board(arms):
    protocol = read(BOARD / 'PROTOCOL.json')
    assert protocol['sampling'] == dict(sampler='dpmsolver++', steps=20, cfg=7.5, seed=3407)
    assert protocol['ref8'] == '永和书风骨韵天地'
    pairs = [(split, font) for split, key in [('test', 'test'), ('train', 'train5'), ('val', 'val5')]
             for font in protocol['fonts'][key]]
    assert len(pairs) == 26 and len(protocol['chars']) == 47
    files = {}
    for arm in arms:
        for k in (1, 2, 4, 8):
            folder = BOARD / 'preds' / f'{arm}_s{k}'
            done = read(folder / 'DONE.json')
            assert done['status'] == 'completed' and done['images'] == done['expected'] == 1222
            expected = {folder / split / font / f'{split}__{font}__u{ord(ch):04X}__s3407.png'
                        for split, font in pairs for ch in protocol['chars']}
            assert set(folder.rglob('*.png')) == expected
            for p in sorted(expected):
                png_ok(p)
                files[str(p.relative_to(BOARD))] = sha(p)
    return dict(protocol_sha256=sha(BOARD / 'PROTOCOL.json'), counts={a: 4888 for a in arms}, files=files)


def preflight():
    import torch
    assert not AUDIT.exists(), 'Existing resume audit; inspect instead of rerunning'
    assert not (RUN / 'DONE.json').exists()
    assert read(OUT / 'k1248_infer_status.json')['state'] == 'COMPLETE'
    assert not other_i_processes(), 'Other I work is still active'
    assert shutil.disk_usage(ROOT).free >= 10 * 2**30
    stops = {}
    for path in (OUT / 'STOP', RUN / 'STOP'):
        s = path.lstat()
        assert not path.is_symlink() and s.st_size == 0 and int(s.st_mtime) == 1789466543, 'New or changed STOP'
        stops[str(path)] = dict(inode=s.st_ino, mtime_ns=s.st_mtime_ns, size=s.st_size)
    stopped = read(RUN / 'STOPPED.json')
    assert stopped == dict(step=5146, attempt=5146, status='safely_stopped')
    assert (RUN / 'last_state').resolve() == STATE
    meta = read(STATE / 'checkpoint.json')
    assert meta['complete'] and meta['step'] == meta['attempt'] == 5146 and meta['arm'] == 'I2'
    config = read(RUN / 'config.json')
    for name, expected in config['code_sha256'].items():
        assert sha(CODE / name) == expected, f'Training code drift: {name}'
    assert config['limit'] == 10000 and config['global_batch'] == 64 and config['per_gpu'] == 8
    parent = Path(config['parent'])
    assert sha(parent / 'unet.pth') == config['parent_unet_sha256']
    assert sha(ROOT / 'artifacts/h_20260915/manifest.json') == config['cache_manifest_sha256']
    cn = ROOT / 'artifacts/i_20260915/cn_content'
    manifest = read(cn / 'COMPLETE.json')
    assert manifest['n'] == 338 and len(list(cn.glob('*.png'))) == 338
    assert sha(cn / 'features.pt') == manifest['features_sha256']
    assert sha(parent / 'content_encoder.pth') == manifest['parent_ec_sha256']
    for key, expected in manifest['neutral_png_sha256'].items():
        p = cn / (key if key.endswith('.png') else key + '.png')
        assert sha(p) == expected, str(p)
    trainer = torch.load(STATE / 'trainer.pt', map_location='cpu', weights_only=False)
    assert trainer['step'] == trainer['attempt'] == 5146 and len(trainer['rngs']) == 8
    assert len(trainer['optimizer']['state']) == 910 and trainer['scaler']['scale'] == 4096
    for rng in trainer['rngs']:
        assert all(k in rng for k in ('python', 'numpy', 'torch', 'cuda'))
    summary = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   resume_step=5146, limit=10000, optimizer_entries=910, rng_ranks=8,
                   scaler=trainer['scaler'], code_sha256=config['code_sha256'], stops=stops,
                   checkpoint_sha256={name: sha(STATE / name) for name in ('model.pth', 'ema.pth', 'trainer.pt', 'checkpoint.json')},
                   cn_png_verified=338, cn_features_sha256=manifest['features_sha256'],
                   free_bytes=shutil.disk_usage(ROOT).free,
                   evaluator_sha256={name: sha(CODE / 'scripts' / name) for name in ('eval_i_k1248.py', 'eval_i2_k1248.py')})
    del trainer
    board = verify_board(('I0', 'I1'))
    assert not other_i_processes(), 'Concurrent I work appeared during verification'
    return summary, board


def status(state, **details):
    save(OUT / 'resume_status.json', dict(state=state, pid=os.getpid(), time=time.time(), **details))


def execute(tag, args, training=False):
    assert not (OUT / 'STOP').exists() and not (RUN / 'STOP').exists(), 'STOP blocks dispatch'
    assert shutil.disk_usage(ROOT).free >= 10 * 2**30, 'Disk below10GiB'
    command = [PY, '-m', 'torch.distributed.run', '--nproc_per_node=8', '--master_port=29583'] + args
    stopped = False
    with (OUT / f'{tag}.log').open('x') as log:
        child = subprocess.Popen(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                                          OMP_NUM_THREADS='1', NCCL_IB_DISABLE='1', PYTHONWARNINGS='ignore'))
        while child.poll() is None:
            free = shutil.disk_usage(ROOT).free
            if free < 10 * 2**30 or (OUT / 'STOP').exists() or (RUN / 'STOP').exists():
                stopped = True
                # Training observes this marker and saves safely. Inference is not killed.
                for p in (OUT / 'STOP', RUN / 'STOP'):
                    if not p.exists():
                        with p.open('x') as f:
                            f.write('Continuation guard: disk below10GiB or external STOP; review required\n')
            status('STOP_REQUESTED' if stopped else 'RUNNING', task=tag, child_pid=child.pid,
                   free_bytes=free, command=command, training=training)
            time.sleep(10)
    if stopped:
        raise RuntimeError(f'{tag}: STOP requested; do not dispatch next phase')
    if child.returncode:
        raise RuntimeError(f'{tag} failed rc={child.returncode}; inspect before retry')


def verify_formal(folder, expected=4096):
    assert read(folder / 'DONE.json')['images'] == expected
    rows = read(folder / 'metrics.json')['rows']
    assert len(rows) == expected and len({r['png'] for r in rows}) == expected
    for row in rows:
        png_ok(folder / row['png'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    lock = (OUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    summary, board = preflight()
    print(json.dumps(dict(preflight='PASS', **summary, board_counts=board['counts']), ensure_ascii=False), flush=True)
    if not args.execute:
        return
    AUDIT.mkdir()
    save(AUDIT / 'PREFLIGHT.json', summary)
    save(AUDIT / 'I01_K1248_VERIFIED.json', board)
    save(AUDIT / 'AUTHORIZATION.json', dict(user='等推理结束了，把I系列的模型都跑完，然后跑同款的推理。',
         source='reports/I_COMPLETION_QUEUE_20260915.md', resumed_from=5146, total_limit=10000,
         authorized_stop_mtime=1789466543, new_stops_not_authorized=True))
    for name in ('config.json', 'STOPPED.json'):
        shutil.copy2(RUN / name, AUDIT / name)
    for rank in range(8):
        shutil.copytree(RUN / f'provenance_rank{rank}', AUDIT / f'provenance_rank{rank}')
    for p, expected in summary['stops'].items():
        path = Path(p)
        s = path.lstat()
        assert dict(inode=s.st_ino, mtime_ns=s.st_mtime_ns, size=s.st_size) == expected
    for path, name in ((OUT / 'STOP', 'QUEUE_STOP'), (RUN / 'STOP', 'I2_STOP')):
        path.rename(AUDIT / name)
    try:
        execute('I2-resume-5146', ['scripts/train_i.py', '--arm', 'I2', '--run-id', RUN.name,
                '--limit', '10000', '--microbatch', '8', '--resume', str(STATE)], training=True)
        done = read(RUN / 'DONE.json')
        assert done['step'] == 10000 and done['inference_complete']
        verify_formal(RUN / 'eval_step_10000')
        for step in (6000, 8000):
            verify_formal(RUN / f'eval_step_{step}', 192)
        execute('I2-k1248-final', ['scripts/eval_i2_k1248.py', '--shots', '1,2,4,8'])
        save(AUDIT / 'I2_K1248_VERIFIED.json', verify_board(('I2',)))
        baseline = ROOT / 'reports/experiments/I/I0/step00010000'
        if not (baseline / 'DONE.json').exists():
            execute('I0-formal4096-final', ['scripts/eval_i_checkpoint.py', '--baseline-i0', '--out', str(baseline)])
        verify_formal(baseline)
        status('COMPUTE_COMPLETE_ARCHIVE_PENDING', i2_step=10000, board_images_per_arm=4888, formal_images_per_arm=4096)
    except BaseException as exc:
        status('NEEDS_ATTENTION', error=str(exc))
        raise


if __name__ == '__main__':
    main()
