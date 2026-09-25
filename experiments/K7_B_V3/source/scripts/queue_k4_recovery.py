"""Resume the authorized K4 queue with bounded retries and explicit completion checks."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from k4_runtime import ROOT,CODE,STORE,RUNS,atomic_json,code_identity
from scripts.k4_numerics import compatible_identity

def main():
    control=STORE/'control';lock=(control/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=code_identity();auth=json.loads((CODE/'experiments/K4/AUTHORIZATION.json').read_text())
    assert auth['approved'] and auth['order']==['K4-A','K4-C','K4-B']
    pre=json.loads((control/'PREFLIGHT_PASSED.json').read_text());assert pre['status']=='PASS' and compatible_identity(pre['identity'],identity)
    precision=json.loads((control/'ATTENTION_PRECISION_PASSED.json').read_text())
    assert precision['status']=='PASS' and compatible_identity(precision['identity'],identity)
    for key in ['code/variants/cn2west_f123_rsi/FontDiffuser/src/modules/attention.py','scripts/test_k4_attention_precision.py']:
        assert precision['identity']['files'][key]==identity['files'][key]
    budget=auth['successful_updates'];assert budget==20000
    env=dict(os.environ,PYTHONPATH=str(CODE),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC='120')
    py=sys.executable;launch=[str(Path(py).parent/'torchrun'),'--standalone','--nproc_per_node=8']
    def status(stage,**kw):atomic_json(control/'status.json',dict(stage=stage,time=time.time(),order=auth['order'],identity=identity['commit'],**kw))
    def run(stage,cmd,training=False):
        marker=control/(stage+'_DONE.json')
        if marker.exists():
            assert json.loads(marker.read_text())['identity']==identity
            if training:
                d=json.loads((ROOT/'runs'/cmd[cmd.index('--run-id')+1]/'DONE.json').read_text());assert d['step']==budget and not d['smoke']
            return
        for retry in range(4):
            if (control/'STOP').exists():raise RuntimeError('Explicit queue STOP requested')
            args=list(cmd)
            if training:
                directory=ROOT/'runs'/args[args.index('--run-id')+1]
                if (directory/'STOP').exists():raise RuntimeError('Explicit run STOP requested')
                if (directory/'last_state').exists():args+=['--resume',str(directory/'last_state')]
            with (control/(stage+'.log')).open('a') as log:
                child=subprocess.Popen(args,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status(stage,pid=child.pid,retry=retry);rc=child.wait()
            if rc==0:
                if training:
                    d=json.loads((directory/'DONE.json').read_text());assert d['step']==budget and not d['smoke']
                atomic_json(marker,dict(status='completed',time=time.time(),identity=identity));return
            with (control/'RETRY_HISTORY.jsonl').open('a') as log:log.write(json.dumps(dict(stage=stage,retry=retry,exitcode=rc,time=time.time()))+'\n')
            # Numerical persistence needs diagnosis, not endless identical retries.
            tail=(control/(stage+'.log')).read_text()[-24000:]
            if any(s in tail for s in ['Nonfinite loss persists','Disk space floor','Source drift','Unapproved source migration']):break
            if retry<3:time.sleep(10)
        raise RuntimeError(f'{stage}: requires recovery after exit {rc}; evidence retained')
    try:
        run('BUDGET_20K_TEST',[py,'scripts/test_k4_budget.py'])
        run('NUMERICAL_RECOVERY_TEST',launch+['scripts/test_k4_numerics.py'])
        for arm in auth['order']:
            extra=['--checkpoint-root',auth['budget_extension']['B_checkpoint_root']] if arm=='K4-B' else []
            run(arm+'_TRAIN_'+str(budget),launch+['scripts/train_k4.py','--arm',arm,'--run-id',RUNS[arm],'--limit',str(budget),'--state-interval','200']+extra,True)
        for arm in ['K0','K1','K3',*auth['order']]:run(arm+'_V2_INFERENCE',launch+['scripts/k4_eval.py','--arm',arm])
        run('V2_FROZEN_DIFFICULTY',[py,'scripts/k4_gt_difficulty.py','--out',str(STORE/'difficulty'),'--workers','8'])
        for arm in ['K0','K1','K3',*auth['order']]:run(arm+'_METRICS',launch+['scripts/k4_metrics.py','--arm',arm])
        run('REPORT',[py,'scripts/k4_metrics.py','--report'])
        status('COMPLETED',review=str(STORE/'review/index.html'))
    except Exception as e:
        old=json.loads((control/'status.json').read_text()) if (control/'status.json').exists() else {}
        status('NEEDS_RECOVERY',failed_stage=old.get('stage'),reason=str(e));raise
if __name__=='__main__':main()
