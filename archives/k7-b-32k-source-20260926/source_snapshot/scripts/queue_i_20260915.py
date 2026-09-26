"""I1 then I2, all eight V100s; stop on failed verification, never restart old H."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path('/root/projects/hrfont');CODE=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/i_20260915';PY='/root/miniforge3/envs/boogu/bin/python'

def status(**kw):
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),code=str(CODE),**kw),indent=2)+'\n');tmp.replace(OUT/'status.json')

def execute(tag,cmd):
    with (OUT/f'{tag}.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',NCCL_IB_DISABLE='1',PYTHONWARNINGS='ignore'))
        while child.poll() is None:
            status(state='RUNNING',task=tag,child_pid=child.pid,command=cmd)
            time.sleep(10)
    if child.returncode:raise RuntimeError(f'{tag} failed rc={child.returncode}; review before retry')

def train(arm,name,limit=10000,resume=None):
    cmd=[PY,'-m','torch.distributed.run','--nproc_per_node=8','--master_port=29573',str(CODE/'scripts/train_i.py'),
         '--arm',arm,'--run-id',name,'--limit',str(limit),'--microbatch','8']
    if limit<10000:cmd+=['--smoke','--state-interval','100']
    if resume:cmd+=['--resume',str(resume)]
    return cmd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    if not a.execute:print(json.dumps([train(i,f'{i}-V0915-S3407') for i in ('I1','I2')],indent=2));return
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert (OUT/'integration/PASS.json').exists()
        smoke=ROOT/'runs/I-SMOKE-I1-V0915-S3407'
        assert json.loads((smoke/'DONE.json').read_text())['step']==120
        rows=[json.loads(s) for s in (smoke/'train_log.jsonl').read_text().splitlines()]
        last=rows[-1]
        assert last['step']==100 and last['router_grad']>0 and last['encoder_grad']>0 and last['ddp_spread']<1e-4
        for arm in ('I1','I2'):
            if arm=='I2':
                assert (ROOT/'artifacts/i_20260915/cn_content/COMPLETE.json').exists()
                if not (ROOT/'runs/I-SMOKE-I2-V0915-S3407/DONE.json').exists():
                    execute('I2-preflight',train('I2','I-SMOKE-I2-V0915-S3407',20))
                assert json.loads((ROOT/'runs/I-SMOKE-I2-V0915-S3407/DONE.json').read_text())['step']==20
            if (OUT/'STOP').exists():status(state='PAUSED',next_arm=arm);return
            if shutil.disk_usage(ROOT).free<10*2**30:raise RuntimeError('less than10GiB disk free; do not delete user data')
            name=f'{arm}-V0915-S3407';dest=ROOT/'runs'/name
            if (dest/'DONE.json').exists():
                done=json.loads((dest/'DONE.json').read_text());assert done['step']==10000 and done['inference_complete'];continue
            if dest.exists():raise RuntimeError(f'existing incomplete run {dest}; explicit resume review required')
            execute(arm,train(arm,name))
            done=json.loads((dest/'DONE.json').read_text());assert done['step']==10000 and done['inference_complete']
            execute(f'archive-{arm}',[PY,str(CODE/'scripts/archive_experiments.py')])
        baseline=ROOT/'reports/experiments/I/I0/step00010000'
        if not (baseline/'DONE.json').exists():
            execute('I0-inference',[PY,'-m','torch.distributed.run','--nproc_per_node=8','--master_port=29573',
                str(CODE/'scripts/eval_i_checkpoint.py'),'--baseline-i0','--out',str(baseline)])
        status(state='COMPLETE',arms=['I1','I2'],total_updates=20000,baseline_inference=True)
    except Exception as exc:
        status(state='NEEDS_ATTENTION',error=str(exc));raise

if __name__=='__main__':main()
