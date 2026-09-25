"""Approved serial E12-c execution after both original and expanded K inference."""
import fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from data import STORE,ROOT,atomic,sha
CODE=Path(__file__).resolve().parent

def main():
    STORE.mkdir(parents=True,exist_ok=True);lock=(STORE/'QUEUE.LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    approval=json.loads((CODE/'AUTHORIZATION.json').read_text());assert approval['approved'] and approval['scope']=='E12-c R1/R2/R3 and frozen GT ranking'
    identity={p.name:sha(p) for p in CODE.glob('*.py')};atomic(STORE/'CODE_LOCK.json',identity)
    if (STORE/'EVALUATION_DONE.json').exists():atomic(STORE/'queue_status.json',dict(stage='COMPLETED',time=time.time()));return
    def status(stage,**kw):atomic(STORE/'queue_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
    status('WAITING_FOR_K_DEFAULT_INFERENCE')
    while True:
        if (STORE/'STOP').exists():status('STOPPED');return
        p=ROOT/'reports/k_default_k1248_20260917/queue_status.json';d=json.loads(p.read_text()) if p.exists() else {}
        if d.get('stage')=='COMPLETED':break
        if d.get('stage') in ['FAILED','STOPPED']:status('DEPENDENCY_FAILED',dependency=d);return
        time.sleep(15)
    for path,expect in [('reports/k_original_queue_20260917/status.json',{'stage':'COMPLETED'}),
        ('runs/K1-ORIGINAL-V0917-S3407/DONE.json',{'step':10000,'inference_complete':True}),
        ('runs/K1-ORIGINAL-V0917-S3407/eval_step_10000/DONE.json',{'images':2816}),
        ('reports/k1_final_val192_20260917/DONE.json',{'images':192}),
        ('reports/k1_final_review/DONE.json',{'rows':2816}),('reports/k1_final_val192_review/DONE.json',{'rows':192})]:
        obj=json.loads((ROOT/path).read_text());assert all(obj.get(k)==v for k,v in expect.items()),path
    assert shutil.disk_usage(STORE).free>20*2**30 and shutil.disk_usage(ROOT).free>2*2**30
    # Context-only GPU allocations belong to another namespace on this host.
    # Check observed compute activity rather than killing unrelated contexts.
    samples=[]
    for _ in range(3):
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        samples.append({int(x.split(',')[0]):(int(x.split(',')[1]),int(x.split(',')[2])) for x in lines});time.sleep(2)
    devices=[i for i in samples[0] if all(s[i][0]<5 and s[i][1]>16000 for s in samples)]
    if not devices:status('NO_IDLE_GPU');return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(devices[0]),OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',PYTHONWARNINGS='ignore::FutureWarning')
    start=time.time()
    tasks=[('CPU_TESTS','tests.py',[]),('PREPARE_NEIGHBORS','neighbors.py',[]),
        ('SMOKE_RESUME_A','train.py',['--arm','R1','--smoke','--run-prefix','SMOKE_RESUME_','--stop-after-A']),
        ('SMOKE_RESUME_B','train.py',['--arm','R1','--smoke','--run-prefix','SMOKE_RESUME_','--resume']),
        ('SMOKE_R1','train.py',['--arm','R1','--smoke']),('SMOKE_R2','train.py',['--arm','R2','--smoke']),('SMOKE_R3','train.py',['--arm','R3','--smoke']),
        ('VERIFY_RESUME','verify_resume.py',[]),('R1','train.py',['--arm','R1']),('R2','train.py',['--arm','R2']),('R3','train.py',['--arm','R3']),
        ('FROZEN_RANKING','evaluate.py',[])]
    for stage,script,args in tasks:
        assert all(sha(CODE/n)==h for n,h in identity.items()),'Source drift'
        marker=STORE/(stage+'_PIPELINE_DONE.json')
        if marker.exists():continue
        if (STORE/'STOP').exists():status('STOPPED');return
        if time.time()-start>90*60:status('BUDGET_REVIEW_REQUIRED');return
        if script=='train.py' and '--resume' not in args:
            arm=args[args.index('--arm')+1];prefix=args[args.index('--run-prefix')+1] if '--run-prefix' in args else ('SMOKE_' if '--smoke' in args else '')
            if (STORE/(prefix+arm)/'config.json').exists():args=args+['--resume']
        with (STORE/(stage+'.log')).open('a') as log:
            p=subprocess.Popen([sys.executable,str(CODE/script)]+args,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(stage,child_pid=p.pid,gpu=devices[0])
            try:rc=p.wait(timeout=max(1,90*60-(time.time()-start)))
            except subprocess.TimeoutExpired:
                (STORE/'STOP').write_text('90-minute pipeline budget reached\n')
                p.terminate();p.wait();status('BUDGET_REVIEW_REQUIRED',stopped_stage=stage);return
        if rc:status('FAILED',failed_stage=stage,returncode=rc,log=str(STORE/(stage+'.log')));return
        atomic(marker,dict(status='completed',code_identity=identity))
    assert (STORE/'EVALUATION_DONE.json').exists();status('COMPLETED',review=str(STORE/'review/index.html'))

if __name__=='__main__':main()
