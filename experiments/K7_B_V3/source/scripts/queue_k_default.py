"""Durable execution of the user-approved default inference protocol."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parent
ROOT=Path('/root/projects/hrfont');OUT=ROOT/'reports/k_default_k1248_20260917'
def main():
    lock=(OUT/'QUEUE.LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**kw):
        p=OUT/'queue_status.tmp';p.write_text(json.dumps(dict(stage=stage,time=time.time(),pid=os.getpid(),**kw),indent=2));p.replace(OUT/'queue_status.json')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',PYTHONWARNINGS='ignore::FutureWarning')
    torchrun=str(Path(sys.executable).parent/'torchrun')
    tasks=[('K0_SMOKE',[sys.executable,str(CODE/'k_default_eval.py'),'--arm','K0','--smoke'],OUT/'K0/SMOKE_DONE.json')]
    for arm in ['K0','K1']:tasks.append((arm+'_INFERENCE',[torchrun,'--standalone','--nproc_per_node=8',str(CODE/'k_default_eval.py'),'--arm',arm],OUT/arm/'DONE.json'))
    tasks.append(('BUILD_REVIEW',[sys.executable,str(CODE/'build_k_default_board.py')],OUT/'DONE.json'))
    for stage,cmd,done in tasks:
        if (OUT/'STOP').exists():status('STOPPED');return
        if done.exists():continue
        with (OUT/(stage+'.log')).open('a') as log:
            p=subprocess.Popen(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT);status(stage,child_pid=p.pid);rc=p.wait()
        if rc or not done.exists():status('FAILED',failed_stage=stage,returncode=rc);return
    status('COMPLETED',review=str(OUT/'index.html'))
if __name__=='__main__':main()
