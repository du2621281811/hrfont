"""Effect-first H queue, seven independent 10k arms, inference inside each run."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/root/projects/hrfont')
CODE = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/h_20260915'
PY = '/root/miniforge3/envs/boogu/bin/python'
ARMS = ['H3','H0','H4','H2','H1','H-D+','H-D-']


def status(**payload):
    tmp = OUT/'queue.status.tmp'
    tmp.write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),code=str(CODE),**payload),indent=2)+'\n')
    tmp.replace(OUT/'status.json')


def run(tag,cmd):
    with (OUT/(tag+'.log')).open('a') as log:
        child = subprocess.Popen(cmd,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',NCCL_IB_DISABLE='1',PYTHONUNBUFFERED='1'))
        while child.poll() is None:
            status(state='RUNNING',task=tag,child_pid=child.pid,command=cmd)
            time.sleep(10)
    if child.returncode:
        raise RuntimeError(f'{tag} failed rc={child.returncode}; no automatic retry')


def train_cmd(arm,name,limit=10000,smoke=False,resume=None):
    cmd=[PY,'-m','torch.distributed.run','--nproc_per_node=8','--master_port=29572',
         str(CODE/'scripts/train_h.py'),'--arm',arm,'--run-id',name,'--limit',str(limit)]
    if smoke:
        cmd+=['--smoke','--transition','20','--state-interval','100','--overfit']
    if resume:
        cmd+=['--resume',str(resume)]
    return cmd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args()
    if not a.execute:
        print(json.dumps([train_cmd(arm,f'{arm}-V0915-S3407') for arm in ARMS],indent=2));return
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert json.loads((ROOT/'artifacts/h_20260915/manifest.json').read_text())['state']=='COMPLETE'
        if not (OUT/'preflight.PASS.json').exists():
            run('unit',[PY,str(CODE/'tests/test_h_memory.py'),'-v'])
            run('integration',[PY,str(CODE/'scripts/test_h_integration.py')])
            smoke='H-SMOKE-H3-V0915-S3407'
            run('smoke100',train_cmd('H3',smoke,100,True))
            run('resume120',train_cmd('H3',smoke,120,True,ROOT/'runs'/smoke/'last_state'))
            rows=[json.loads(x) for x in (ROOT/'runs'/smoke/'train_log.jsonl').read_text().splitlines()]
            assert rows[-1]['step']==100 and rows[-1]['reader_grad']>0 and rows[-1]['ddp_spread']<1e-4
            done=json.loads((ROOT/'runs'/smoke/'DONE.json').read_text())
            assert done['step']==120
            status(state='PREFLIGHT_COMPLETE',smoke_updates=120)
            (OUT/'preflight.PASS.json').write_text(json.dumps(dict(smoke=smoke,done=done,unit=True,integration=True))+'\n')
        if a.preflight_only:
            return
        for arm in ARMS:
            if (OUT/'STOP').exists():
                status(state='PAUSED_BY_REVIEW',next_arm=arm);return
            if shutil.disk_usage(ROOT).free<8*2**30:
                raise RuntimeError('under8GiB free; do not delete user data')
            name=f'{arm}-V0915-S3407'
            dest=ROOT/'runs'/name
            if (dest/'DONE.json').exists():
                done=json.loads((dest/'DONE.json').read_text())
                assert done['step']==10000 and done['inference_complete']
                continue
            if dest.exists():
                raise RuntimeError(f'existing incomplete {name}: review before resume')
            run(arm,train_cmd(arm,name))
            done=json.loads((dest/'DONE.json').read_text())
            assert done['step']==10000 and done['inference_complete']
        status(state='COMPLETE',arms=ARMS,total_updates=70000)
    except Exception as e:
        status(state='NEEDS_ATTENTION',error=str(e))
        raise


if __name__=='__main__':
    main()
