"""Bounded recovery of the missing archive sealer; never re-trains I3."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
from PIL import Image
import queue_i34_20260916 as q


def verify_archive(path):
    m=q.read(path/'ARCHIVE.json')
    for name,digest in m['files_sha256'].items():
        p=path/name;assert q.sha(p)==digest, str(p)
        if p.suffix=='.png':
            with Image.open(p) as im:im.load();assert im.size==(96,96)
    return len(m['files_sha256'])


def finish_i3_archive():
    run=q.ROOT/'runs/I3-V0916-S3407';dest=q.ROOT/'reports/experiments/I/I3'
    done=q.read(run/'DONE.json');assert done['step']==10000 and done['attempt']==10000 and done['inference_complete'] and not done['smoke']
    matched=q.read(run/'K1248_DONE.json');assert matched['images']==4888
    assert matched['weight_sha256']==q.sha(run/'global_step_10000/ema.pth')
    for stage in ('I3_TRAIN','I3_K1248'):
        exit=q.read(q.OUT/(stage+'_EXIT.json'));assert exit['returncode']==0
        assert exit['log_sha256']==q.sha(q.OUT/(stage+'.log'))
    counts={'step00010000_k1248':verify_archive(dest/'step00010000_k1248')}
    assert q.read(dest/'step00010000_k1248/protocol.json')['weight_sha256']==matched['weight_sha256']
    for step in (2000,4000,6000,8000,10000):
        src=run/f'eval_step_{step}';target=dest/f'step{step:08d}'
        if target.exists():
            # Existing partial copy is accepted only when byte-identical to source.
            files={str(p.relative_to(src)):q.sha(p) for p in src.rglob('*') if p.is_file()}
            existing={str(p.relative_to(target)) for p in target.rglob('*') if p.is_file() and p.name!='ARCHIVE.json'}
            assert existing==set(files)
            assert all(q.sha(target/name)==digest for name,digest in files.items())
        else:shutil.copytree(src,target)
        if not (target/'ARCHIVE.json').exists():
            subprocess.run([q.PY,str(q.CODE/'scripts/seal_i_eval_archive.py'),str(target),
                '--run-id',run.name,'--step',str(step),'--ema-sha256',q.sha(run/f'global_step_{step}/ema.pth'),
                '--expected','4096' if step==10000 else '192'],check=True)
        counts[target.name]=verify_archive(target)
    for name in ('config.json','train_log.jsonl','DONE.json','K1248_DONE.json'):
        if (dest/name).exists():assert q.sha(dest/name)==q.sha(run/name)
        else:shutil.copy2(run/name,dest/name)
    q.write(q.OUT/'I3_ARCHIVE_READY.json',dict(directory=str(dest),time=time.time(),status='ready_for_git_sync',verified_files=counts))
    return counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    with (q.ROOT/'reports/i_20260915/queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q.guard();q.gpu_admission()
        old=q.read(q.OUT/'state.json');failure=q.read(q.OUT/'failure.json')
        assert old['state']=='I3_K1248'
        assert not Path(f"/proc/{old['pid']}").exists() and not Path(f"/proc/{old['child_pid']}").exists()
        assert 'seal_i_eval_archive.py' in failure['error'] and 'CalledProcessError(2' in failure['error']
        assert not (q.ROOT/'runs/E12C-V0916-S3407').exists() and not (q.ROOT/'runs/I4-V0916-S3407').exists()
        dependencies=('seal_i_eval_archive.py','archive_i34_k1248.py','train_e12c.py','train_i34.py','eval_i34_k1248.py')
        assert all((q.CODE/'scripts'/name).is_file() for name in dependencies)
        counts=finish_i3_archive()
        audit=dict(time=time.time(),prior_state=old,prior_failure=failure,verified_files=counts,
            recovery_sha256=q.sha(Path(__file__)),dependencies_sha256={n:q.sha(q.CODE/'scripts'/n) for n in dependencies},
            note='Missing deployment-only sealer added; frozen numerical code and checkpoints unchanged; no I3 rerun')
        q.write(q.OUT/'ARCHIVE_RECOVERY_READY.json',audit)
        if not a.execute:print('RECOVERY_DRY_RUN_PASSED',counts,flush=True);return
        assert not (q.OUT/'failure_before_archive_fix.json').exists()
        (q.OUT/'failure.json').rename(q.OUT/'failure_before_archive_fix.json')
        try:
            e12=q.ROOT/'runs/E12C-V0916-S3407'
            q.run('E12C_TRAIN',['scripts/train_e12c.py','--out',str(e12)],lock,{'CUDA_VISIBLE_DEVICES':'0'})
            done=q.read(e12/'DONE.json');assert not done['smoke'] and done['A_steps']==2000 and done['B_steps']==1000
            target=q.ROOT/'reports/experiments/E/E12c/V0916';target.mkdir(parents=True,exist_ok=False)
            for name in ('config.json','train_log.jsonl','internal_test.json','DONE.json'):shutil.copy2(e12/name,target/name)
            q.write(target/'ARCHIVE.json',dict(files_sha256={p.name:q.sha(p) for p in target.iterdir()},checkpoint=str(e12),manifest_sha256=done['manifest_sha256']))
            q.write(q.OUT/'E12C_ARCHIVE_READY.json',dict(directory=str(target),time=time.time()))
            q.run('I4_TRAIN',['-m','torch.distributed.run','--standalone','--nproc_per_node=8','scripts/train_i34.py',
                '--arm','I4','--run-id','I4-V0916-S3407','--difficulty-manifest',str(q.PREP/'difficulty_manifest.json')],lock)
            q.run('I4_K1248',['-m','torch.distributed.run','--standalone','--nproc_per_node=8','scripts/eval_i34_k1248.py',
                '--arm','I4','--run-id','I4-V0916-S3407'],lock)
            q.archive_arm('I4','I4-V0916-S3407')
            q.write(q.OUT/'state.json',dict(state='COMPUTE_COMPLETE_ARCHIVE_PENDING',time=time.time(),pid=os.getpid()))
        except BaseException as exc:
            q.write(q.OUT/'failure.json',dict(error=repr(exc),time=time.time()));raise


if __name__=='__main__':main()
