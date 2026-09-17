"""Authorized full 10k K3 training followed by default inference. No quality gate."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]
ROOT=Path('/root/projects/hrfont')
RUN='K3-PURENOISE-V0917-S3407'

def main():
    control=ROOT/'reports/k3_queue_20260917';control.mkdir(exist_ok=True)
    lock=(control/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    approval=json.loads((CODE/'K3_AUTHORIZATION.json').read_text());assert approval['approved']
    env=dict(os.environ,PYTHONPATH=str(CODE),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONWARNINGS='ignore::FutureWarning')
    launcher=[str(Path(sys.executable).parent/'torchrun'),'--standalone','--nproc_per_node=8']
    def status(stage,**kw):
        p=control/'status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(stage=stage,time=time.time(),run_id=RUN,**kw),indent=2));q.replace(p)
    tasks=[('OBJECTIVE_CHECK',[sys.executable,'scripts/test_k3_components.py']),
           ('TRAIN_10000',launcher+['scripts/train_k3.py','--run-id',RUN,'--limit','10000']),
           ('DEFAULT_INFERENCE',launcher+['scripts/k3_default_eval.py','--arm','K3'])]
    for stage,args in tasks:
        if (control/'STOP').exists():status('STOPPED');return
        marker=control/(stage+'_DONE.json')
        if marker.exists():continue
        if stage=='TRAIN_10000':
            run=ROOT/'runs'/RUN
            if (run/'DONE.json').exists():continue
            if (run/'last_state').exists():args+=['--resume',str(run/'last_state')]
        with (control/(stage+'.log')).open('a') as f:
            c=subprocess.Popen(args,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
            status(stage,pid=c.pid);rc=c.wait()
        if rc:status('FAILED',failed_stage=stage,returncode=rc);return
        if stage=='TRAIN_10000' and not (ROOT/'runs'/RUN/'DONE.json').exists():status('STOPPED');return
        marker.write_text(json.dumps(dict(status='completed',time=time.time())))
    status('COMPLETED')

if __name__=='__main__':main()
