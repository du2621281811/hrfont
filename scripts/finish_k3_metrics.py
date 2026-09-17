"""One-off downstream K3 result job: wait for its inference, score, build board."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/root/projects/hrfont');CODE=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'reports/k3_publish_20260917';out.mkdir(exist_ok=True)
    lock=(out/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**kw):
        p=out/'status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw),indent=2));q.replace(p)
    status('WAIT_FOR_K3_DEFAULT_INFERENCE')
    while True:
        p=ROOT/'reports/k3_queue_20260917/status.json';s=json.loads(p.read_text())
        if s['stage']=='COMPLETED':break
        if s['stage'] in ['FAILED','STOPPED']:
            status('DEPENDENCY_'+s['stage']);return
        time.sleep(30)
    assert json.loads((ROOT/'reports/k_default_k1248_20260917/K3/DONE.json').read_text())['status']=='completed'
    status('WAIT_FOR_IDLE_SCORING_GPU')
    while True:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        idle=[int(x.split(',')[0]) for x in lines if int(x.split(',')[1])<10 and int(x.split(',')[2])>4000]
        if idle:break
        time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(idle[0]),OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    for stage,script,args in [('SCORE_K3','k_paper_metrics.py',['--arm','K3']),('BUILD_K3_BOARD','build_k3_complete_board.py',[])]:
        status(stage,gpu=idle[0])
        with (out/(stage+'.log')).open('a') as f:
            rc=subprocess.call([sys.executable,str(CODE/'scripts'/script)]+args,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
        if rc:status('FAILED',failed_stage=stage,returncode=rc);return
    status('COMPLETED',review='/root/data1/hrfont_k3_complete_review_20260917/index.html',tables='/root/data1/hrfont_k3_complete_review_20260917/paper_metrics.csv')
if __name__=='__main__':main()
