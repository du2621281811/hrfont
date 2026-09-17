"""Durable original K1 queue; no K2 or auxiliary task is scheduled implicitly."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
ROOT=Path('/root/projects/hrfont')
RUN='K1-ORIGINAL-V0917-S3407'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',default=RUN);a=ap.parse_args()
    control=ROOT/'reports/k_original_queue_20260917';control.mkdir(parents=True,exist_ok=True)
    lock=(control/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=json.loads((CODE/'K_PREFLIGHT_PASSED.json').read_text())
    assert preflight['status']=='PASSED'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',PYTHONWARNINGS='ignore::FutureWarning')
    launcher=[str(Path(sys.executable).parent/'torchrun'),'--standalone','--nproc_per_node=8']
    tasks=[('K0_VAL192','scripts/k_eval.py',['--prepare','--out',str(ROOT/'reports/k0_original_val192_20260917')],ROOT/'reports/k0_original_val192_20260917'),
           ('K0_TEST','scripts/k_eval.py',['--manifest',str(ROOT/'experiments/K/K_TEST.json'),
             '--out',str(ROOT/'reports/k0_original_test_20260917')],ROOT/'reports/k0_original_test_20260917'),
           ('K1_TRAIN_AND_EMA_INFERENCE','scripts/train_k.py',['--run-id',a.run_id],ROOT/'runs'/a.run_id)]
    def status(stage,**extra):
        tmp=control/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),run_id=a.run_id,**extra),indent=2));tmp.replace(control/'status.json')
    for stage,script,args,out in tasks:
        if (control/'STOP').exists():status('STOPPED_BEFORE_'+stage);return
        if (out/'DONE.json').exists():continue
        if stage.startswith('K1') and (out/'STOP').exists():status('RUN_STOP_PENDING');return
        if stage.startswith('K1') and (out/'last_state').exists():args+=['--resume',str(out/'last_state')]
        status(stage)
        with (control/(stage+'.log')).open('a') as log:
            p=subprocess.Popen(launcher+[script]+args,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(stage,pid=p.pid);result=p.wait()
        if result!=0:status('FAILED',failed_stage=stage,returncode=result);return
        if not (out/'DONE.json').exists():status('STOPPED',stopped_stage=stage);return
    status('COMPLETED',final_review=str(ROOT/'reports/k1_final_review/index.html'))


if __name__=='__main__':main()
