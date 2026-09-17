"""Persistent authorized family-filtered queue: A -> C -> B -> full-v2 evaluation."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]
ROOT=Path('/root/projects/hrfont');STORE=Path('/root/data1/hrfont_k4_weight_20260918')
CONTROL=STORE/'control'

def main():
    CONTROL.mkdir(parents=True,exist_ok=True)
    lock=(CONTROL/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    auth=json.loads((CODE/'experiments/K4/AUTHORIZATION.json').read_text())
    assert auth['approved'] and auth['order']==['K4-A','K4-C','K4-B']
    identity=json.loads((CODE/'K4_CODE_IDENTITY.json').read_text())
    env=dict(os.environ,PYTHONPATH=str(CODE),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONWARNINGS='ignore::FutureWarning',PYTHONUNBUFFERED='1')
    py=sys.executable;launcher=[str(Path(py).parent/'torchrun'),'--standalone','--nproc_per_node=8']
    def status(stage,**kw):
        p=CONTROL/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),order=auth['order'],identity=identity['commit'],**kw),indent=2));tmp.replace(p)
    def run(stage,cmd):
        marker=CONTROL/(stage+'_DONE.json')
        if marker.exists():
            assert json.loads(marker.read_text())['identity']==identity
            return
        if (CONTROL/'STOP').exists():raise RuntimeError('Queue STOP requested')
        with (CONTROL/(stage+'.log')).open('a') as f:
            child=subprocess.Popen(cmd,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            status(stage,pid=child.pid);rc=child.wait()
        if rc:raise RuntimeError(f'{stage}: exit {rc}')
        if stage.endswith('_TRAIN_10000'):
            directory=ROOT/'runs'/cmd[cmd.index('--run-id')+1]
            done=directory/'DONE.json'
            if not done.exists():raise RuntimeError(f'{stage}: stopped before full training completion; no completion marker written')
            result=json.loads(done.read_text())
            assert result['step']==10000 and not result['smoke']
        marker.write_text(json.dumps(dict(status='completed',time=time.time(),identity=identity)))
    try:
        run('CONTRACTS',[py,'scripts/test_k4_preflight.py','contracts'])
        run('FAMILY_POLICY',[py,'scripts/test_k4_family.py'])
        run('GPU_PARITY',[py,'scripts/test_k4_preflight.py','gpu'])
        pre=['--smoke','--state-interval','100','--checkpoint-root',str(STORE/'preflight_checkpoints')]
        run('C_REFERENCE_120',launcher+['scripts/train_k4.py','--arm','K4-C','--run-id','K4-WEIGHT-PREFLIGHT-C','--limit','120']+pre)
        resume=ROOT/'runs/K4-WEIGHT-PREFLIGHT-C-RESUME';resume.mkdir(parents=True,exist_ok=True)
        cfg=(ROOT/'runs/K4-WEIGHT-PREFLIGHT-C/config.json').read_text()
        if (resume/'config.json').exists():assert (resume/'config.json').read_text()==cfg
        else:(resume/'config.json').write_text(cfg)
        run('C_RESUME_100_TO_120',launcher+['scripts/train_k4.py','--arm','K4-C','--run-id',resume.name,'--limit','120','--resume',str(ROOT/'runs/K4-WEIGHT-PREFLIGHT-C/checkpoints/state_step_100'),'--eval-smoke']+pre)
        run('B_FULL_GRADIENT_17',launcher+['scripts/train_k4.py','--arm','K4-B','--run-id','K4-WEIGHT-PREFLIGHT-B','--limit','17']+pre)
        run('A_WARMSTART_2',launcher+['scripts/train_k4.py','--arm','K4-A','--run-id','K4-WEIGHT-PREFLIGHT-A','--limit','2']+pre)
        run('PREFLIGHT_ACCEPTANCE',[py,'scripts/test_k4_preflight.py','training_checks'])
        runs={'K4-C':'K4-C-K1RECIPE-V2-WEIGHT-S3407','K4-B':'K4-B-K3RECIPE-V2-WEIGHT-S3407','K4-A':'K4-A-K1FT-0917-WEIGHT-S3407'}
        for arm in auth['order']:
            args=launcher+['scripts/train_k4.py','--arm',arm,'--run-id',runs[arm],'--limit','10000']
            directory=ROOT/'runs'/runs[arm]
            if (directory/'last_state').exists():args+=['--resume',str(directory/'last_state')]
            run(arm+'_TRAIN_10000',args)
            done=json.loads((directory/'DONE.json').read_text())
            assert done['step']==10000 and not done['smoke']
        for arm in ['K0','K1','K3',*auth['order']]:
            run(arm+'_V2_INFERENCE',launcher+['scripts/k4_eval.py','--arm',arm])
        run('V2_FROZEN_DIFFICULTY',[py,'scripts/k4_gt_difficulty.py','--out',str(STORE/'difficulty'),'--workers','8'])
        for arm in ['K0','K1','K3',*auth['order']]:
            run(arm+'_METRICS',launcher+['scripts/k4_metrics.py','--arm',arm])
        run('REPORT',[py,'scripts/k4_metrics.py','--report'])
        status('COMPLETED',review=str(STORE/'review/index.html'))
    except Exception as e:
        previous=json.loads((CONTROL/'status.json').read_text()) if (CONTROL/'status.json').exists() else {}
        status('FAILED',failed_stage=previous.get('stage'),reason=str(e));raise

if __name__=='__main__':main()
