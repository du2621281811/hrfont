"""I5 only: wait for I4, preflight, train<=10k, same inference, archive. I6 not authorized."""
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
PREP=ROOT/'artifacts/i56_20260916';OUT=ROOT/'reports/i56_20260916'
STORE=Path('/root/data1/hrfont_i56_20260916')
PY='/root/miniforge3/envs/boogu/bin/python';RUNID='I5-V0916-S3407'


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def stop_requested():
    return any(p.exists() for p in (OUT/'STOP',ROOT/'reports/i_20260915/STOP'))
def guard():
    assert not stop_requested(),'STOP requested'
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage(STORE).free)>10*2**30,'Disk safety floor'
    ready=read(PREP/'ARMED.json');assert ready['authorized_arms']==['I5']
    assert ready['run_id']==RUNID and ready['cpu_tests_passed']
    assert sha(PREP/'detail_manifest.json')==ready['detail_sha256']
    assert read(PREP/'detail_manifest.json')['status']=='REVIEWED'
    for p,digest in ready['code_sha256'].items():assert sha(CODE/p)==digest,'Code drift '+p


def gpu_admission():
    # Preserve unrelated idle host-namespace CUDA contexts; never terminate them.
    baseline={1397110,1397113,1397116,1397119,1397122,1397125,1397128,1397131}
    pids={int(s.strip()) for s in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if s.strip()}
    assert pids<=baseline and not any(Path(f'/proc/{p}').exists() for p in pids),'Unexpected GPU ownership'
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    rows=[[int(v.strip()) for v in line.split(',')] for line in lines]
    assert len(rows)==8 and all(f>=12000 and u<=5 for _,f,u in rows),rows
    write(OUT/'gpu_admission.json',dict(time=time.time(),rows=rows,preserved_host_pids=sorted(pids)))


def run(name,args,lock,runid=None,single=False):
    guard();gpu_admission();log=OUT/(name+'.log');assert not log.exists(),'Existing stage requires inspection: '+name
    env=os.environ.copy()
    if single:env['CUDA_VISIBLE_DEVICES']='0'
    with log.open('x') as f:
        proc=subprocess.Popen([PY,*args],cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,
                              start_new_session=True,pass_fds=(lock.fileno(),))
        started=time.time();stopping=None
        while proc.poll() is None:
            low=min(shutil.disk_usage(ROOT).free,shutil.disk_usage(STORE).free)<10*2**30
            if stop_requested() or low:
                if runid and (ROOT/'runs'/runid).exists():(ROOT/'runs'/runid/'STOP').touch(exist_ok=True)
                if stopping is None:stopping=time.time()
                if not runid or time.time()-stopping>120:os.killpg(proc.pid,signal.SIGTERM)
            write(OUT/'state.json',dict(state=name,pid=os.getpid(),child_pid=proc.pid,started=started,
                checked=time.time(),command=[PY,*args],stop_requested=stopping is not None))
            time.sleep(10)
        rc=proc.wait()
    assert rc==0 and stopping is None,(name,rc)
    write(OUT/(name+'_EXIT.json'),dict(returncode=rc,time=time.time(),log_sha256=sha(log)))


def archive():
    from archive_i34_k1248 import archive as matched_archive
    runpath=ROOT/'runs'/RUNID;done=read(runpath/'DONE.json')
    assert done['arm']=='I5' and done['step']==10000 and done['inference_complete'] and not done['smoke']
    assert read(runpath/'K1248_DONE.json')['images']==4888
    matched_archive('I5',RUNID)
    dest=ROOT/'reports/experiments/I/I5'
    for step in (2000,4000,6000,8000,10000):
        target=dest/f'step{step:08d}';assert not target.exists()
        shutil.copytree(runpath/f'eval_step_{step}',target)
        subprocess.run([PY,str(CODE/'scripts/seal_i_eval_archive.py'),str(target),'--run-id',RUNID,
            '--step',str(step),'--ema-sha256',sha(runpath/f'global_step_{step}/ema.pth'),
            '--expected','4096' if step==10000 else '192'],check=True)
    for name in ('config.json','train_log.jsonl','DONE.json','K1248_DONE.json','exposure.json','REVIEW_4K.json'):
        shutil.copy2(runpath/name,dest/name)
    write(OUT/'I5_ARCHIVE_READY.json',dict(directory=str(dest),time=time.time(),status='ready_for_git_sync'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);STORE.mkdir(parents=True,exist_ok=True)
    # Separate instance lock prevents two waiting supervisors, including while I4 holds GPU lock.
    with (OUT/'supervisor.lock').open('a+') as own,(ROOT/'reports/i_20260915/queue.lock').open('a+') as lock:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);guard()
        if not a.execute:print('ARMED_CONFIGURATION_VALID');return
        assert not (OUT/'state.json').exists(),'Inspect existing supervisor before restarting'
        try:
            while True:
                guard()
                old=ROOT/'reports/i34_20260916'
                if (old/'failure.json').exists():raise RuntimeError('I4 predecessor failed; review rather than steal resources')
                ready=(old/'I4_ARCHIVE_READY.json').exists() and read(old/'state.json')['state']=='COMPUTE_COMPLETE_ARCHIVE_PENDING'
                if ready:
                    d=read(ROOT/'runs/I4-V0916-S3407/DONE.json');k=read(ROOT/'runs/I4-V0916-S3407/K1248_DONE.json')
                    assert d['step']==10000 and d['inference_complete'] and k['images']==4888
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:pass
                write(OUT/'state.json',dict(state='WAITING_FOR_I4_ARCHIVE',pid=os.getpid(),checked=time.time(),
                    note='No GPU reserved. I4 train + formal4096 + matched4888 and archive finish first; I6 not launched.'))
                time.sleep(30)
            guard();gpu_admission()
            manifest=str(PREP/'detail_manifest.json');calibration=str(PREP/'calibration.json')
            run('CALIBRATE',['scripts/calibrate_i56.py','--manifest',manifest,'--out',calibration],lock,single=True)
            run('DIAGNOSTIC_BEFORE',['scripts/diagnose_i56.py','--manifest',manifest,'--out',str(PREP/'before')],lock,single=True)
            pre='I5-PREFLIGHT-V0916-S3407'
            cmd=['-m','torch.distributed.run','--standalone','--nproc_per_node=8','scripts/train_i56.py',
                 '--arm','I5','--run-id',pre,'--detail-manifest',manifest,'--calibration',calibration,'--smoke','--overfit']
            run('SMOKE_300',cmd+['--limit','300'],lock,pre)
            run('RESUME_304',cmd+['--limit','304','--resume',str(ROOT/'runs'/pre/'last_state')],lock,pre)
            run('DIAGNOSTIC_AFTER',['scripts/diagnose_i56.py','--manifest',manifest,'--out',str(PREP/'after'),
                '--checkpoint',str(ROOT/'runs'/pre/'global_step_304')],lock,single=True)
            run('PREFLIGHT_VERIFY',['scripts/verify_i56_preflight.py'],lock,single=True)
            assert read(PREP/'PREFLIGHT_RESULT.json')['status']=='PASSED'
            run('I5_TRAIN',['-m','torch.distributed.run','--standalone','--nproc_per_node=8','scripts/train_i56.py',
                '--arm','I5','--run-id',RUNID,'--detail-manifest',manifest,'--calibration',calibration],lock,RUNID)
            if (ROOT/'runs'/RUNID/'STOPPED.json').exists():
                write(OUT/'state.json',dict(state='REVIEW_REQUIRED',pid=os.getpid(),checked=time.time(),
                    review=read(ROOT/'runs'/RUNID/'REVIEW_4K.json')));return
            run('I5_K1248',['-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                'scripts/eval_i56_k1248.py','--arm','I5','--run-id',RUNID],lock)
            archive()
            write(OUT/'state.json',dict(state='COMPUTE_COMPLETE_ARCHIVE_PENDING',pid=os.getpid(),checked=time.time()))
        except BaseException as exc:
            write(OUT/'failure.json',dict(error=repr(exc),time=time.time()))
            write(OUT/'state.json',dict(state='REVIEW_REQUIRED',pid=os.getpid(),checked=time.time(),error=repr(exc)))
            raise


if __name__=='__main__':main()
