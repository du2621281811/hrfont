"""Single-owner, fail-closed I3 -> E12c -> I4 execution, followed by archive handoff."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path('/root/projects/hrfont');CODE=Path(__file__).resolve().parents[1]
PREP=ROOT/'artifacts/i34_20260915';OUT=ROOT/'reports/i34_20260916'
PY='/root/miniforge3/envs/boogu/bin/python'


def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def guard():
    assert not (OUT/'STOP').exists() and not (ROOT/'reports/i_20260915/STOP').exists(), 'STOP requested'
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/root/data1/hrfont_i34_20260915').free)>10*2**30
    ready=read(PREP/'PREFLIGHT_PASSED.json');assert ready['status']=='PASSED'
    assert sha(PREP/'difficulty_manifest.json')==ready['difficulty_sha256']
    assert sha(PREP/'e12c_manifest.json')==ready['e12c_manifest_sha256']
    assert read(PREP/'difficulty_manifest.json')['status']=='REVIEWED'
    for path,digest in ready['code_sha256'].items():assert sha(CODE/path)==digest, 'Code drift '+path
    assert read(OUT/'OLD_ARCHIVES_SYNCED.json')['git_commit']
    return ready


def gpu_admission():
    # Observed before/during/after smoke: idle host-namespace contexts, invisible in this container.
    # They are not ours to terminate. Reject new contexts or insufficient capacity instead.
    baseline={1397110,1397113,1397116,1397119,1397122,1397125,1397128,1397131}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    pids={int(s.strip()) for s in raw.splitlines() if s.strip()}
    assert pids<=baseline, 'New GPU context: '+str(pids-baseline)
    assert not any(Path(f'/proc/{pid}').exists() for pid in pids), 'Baseline namespace/ownership changed'
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True).splitlines()
    parsed=[[int(x.strip()) for x in row.split(',')] for row in rows]
    assert len(parsed)==8 and all(free>=12000 and util<=5 for _,free,util in parsed),parsed
    write(OUT/'gpu_admission.json',dict(time=time.time(),existing_host_pids=sorted(pids),
        index_free_mib_util=parsed,policy='Preserve observed idle external contexts; no termination'))


def run(name,args,lock,extra_env=None):
    guard();gpu_admission();log=OUT/(name+'.log')
    assert not log.exists(), 'Do not silently repeat an existing stage: '+name
    with log.open('x') as f:
        env=os.environ.copy();env.update(extra_env or {})
        proc=subprocess.Popen([PY,*args],cwd=CODE,stdout=f,stderr=subprocess.STDOUT,
                              env=env,pass_fds=(lock.fileno(),),start_new_session=True)
        started=time.time();stopping=None
        while proc.poll() is None:
            low=min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/root/data1/hrfont_i34_20260915').free)<10*2**30
            stop=(OUT/'STOP').exists() or (ROOT/'reports/i_20260915/STOP').exists() or low
            if stop:
                runid='E12C-V0916-S3407' if name.startswith('E12C') else name[:2]+'-V0916-S3407'
                folder=ROOT/'runs'/runid
                if folder.exists():(folder/'STOP').touch(exist_ok=True)
                if stopping is None:stopping=time.time()
                if time.time()-stopping>120:os.killpg(proc.pid,signal.SIGTERM)
            write(OUT/'state.json',dict(state=name,pid=os.getpid(),child_pid=proc.pid,started=started,
                checked=time.time(),stop_requested=stop,command=[PY,*args]))
            time.sleep(10)
        code=proc.wait()
    assert code==0 and stopping is None,(name,'stage stopped',code)
    write(OUT/(name+'_EXIT.json'),dict(returncode=code,time=time.time(),log_sha256=sha(log)))


def archive_arm(arm,runid):
    run=ROOT/'runs'/runid
    assert read(run/'DONE.json')['step']==10000 and read(run/'DONE.json')['inference_complete']
    assert read(run/'K1248_DONE.json')['images']==4888
    from archive_i34_k1248 import archive
    archive(arm,runid)
    dest=ROOT/'reports/experiments/I'/arm
    for step in (2000,4000,6000,8000,10000):
        src=run/f'eval_step_{step}';target=dest/f'step{step:08d}'
        assert not target.exists()
        shutil.copytree(src,target)
        subprocess.run([PY,str(CODE/'scripts/seal_i_eval_archive.py'),str(target),'--run-id',runid,
            '--step',str(step),'--ema-sha256',sha(run/f'global_step_{step}/ema.pth'),
            '--expected','4096' if step==10000 else '192'],check=True)
    for name in ('config.json','train_log.jsonl','DONE.json','K1248_DONE.json'):
        shutil.copy2(run/name,dest/name)
    write(OUT/(arm+'_ARCHIVE_READY.json'),dict(directory=str(dest),time=time.time(),status='ready_for_git_sync'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'reports/i_20260915/queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard()
        gpu_admission()
        if not a.execute:print('DRY_RUN_PASSED');return
        assert not (OUT/'state.json').exists(), 'Inspect existing queue instead of duplicating it'
        try:
            for arm in ('I3','I4'):
                runid=f'{arm}-V0916-S3407'
                run(arm+'_TRAIN',['-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                    'scripts/train_i34.py','--arm',arm,'--run-id',runid,'--difficulty-manifest',str(PREP/'difficulty_manifest.json')],lock)
                run(arm+'_K1248',['-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                    'scripts/eval_i34_k1248.py','--arm',arm,'--run-id',runid],lock)
                archive_arm(arm,runid)
                if arm=='I3':
                    e12=ROOT/'runs/E12C-V0916-S3407'
                    run('E12C_TRAIN',['scripts/train_e12c.py','--out',str(e12)],lock,{'CUDA_VISIBLE_DEVICES':'0'})
                    done=read(e12/'DONE.json');assert not done['smoke'] and done['A_steps']==2000 and done['B_steps']==1000
                    target=ROOT/'reports/experiments/E/E12c/V0916'
                    target.mkdir(parents=True,exist_ok=False)
                    for name in ('config.json','train_log.jsonl','internal_test.json','DONE.json'):
                        shutil.copy2(e12/name,target/name)
                    write(target/'ARCHIVE.json',dict(files_sha256={p.name:sha(p) for p in target.iterdir()},
                          checkpoint=str(e12),manifest_sha256=done['manifest_sha256']))
                    write(OUT/'E12C_ARCHIVE_READY.json',dict(directory=str(target),time=time.time()))
            write(OUT/'state.json',dict(state='COMPUTE_COMPLETE_ARCHIVE_PENDING',time=time.time(),pid=os.getpid()))
        except BaseException as exc:
            write(OUT/'failure.json',dict(error=repr(exc),time=time.time()))
            raise


if __name__=='__main__':main()
