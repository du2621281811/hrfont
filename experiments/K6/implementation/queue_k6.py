import os,sys,json,time,fcntl,subprocess,hashlib
from pathlib import Path
C=Path('/root/projects/hrfont_k6_20260920');CTRL=Path('/root/data1/hrfont_k6_20260920/control');RUN=Path('/root/projects/hrfont/runs')
lock=(CTRL/'QUEUE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def decision(action,**kw):
 row=dict(time=time.time(),action=action,**kw)
 with (CTRL/'DECISIONS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 with (CTRL/'DECISIONS.md').open('a') as f:f.write('\n'+json.dumps(row,ensure_ascii=False)+'\n')
identity=json.loads((C/'K6_CODE_IDENTITY.json').read_text())
env=dict(os.environ,PYTHONPATH=str(C),OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
launch=[str(Path(sys.executable).parent/'torchrun'),'--standalone','--nproc_per_node=8',str(C/'scripts/train_k6.py')]
def run(arm,name,smoke=False,resume=False,stop=0):
 storage=Path('/root/projects/hrfont_k6_preflight_states') if smoke else Path('/root/data1' if arm=='K6-A' else '/root/data2')/'hrfont_k6_20260920/checkpoints'
 args=launch+['--arm',arm,'--run-id',name,'--limit','4' if smoke else '10000','--state-interval','2' if smoke else '200','--checkpoint-root',str(storage)]
 if smoke:args+=['--smoke']
 if resume:args+=['--resume',str(RUN/name/'last_state')]
 if stop:args+=['--stop-after',str(stop)]
 if (CTRL/'STOP').exists():raise RuntimeError('K6 user STOP')
 with (CTRL/(name+('_resume' if resume else '')+'.log')).open('a') as f:
  p=subprocess.Popen(args,cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  write(CTRL/'status.json',dict(stage=name,pid=p.pid,smoke=smoke,resume=resume,time=time.time()))
  decision('launch',arm=arm,name=name,pid=p.pid,resume=resume);rc=p.wait()
 if rc:raise RuntimeError(f'{name} exit {rc}; inspect own log, no blind retry')
 return json.loads((RUN/name/('STOPPED.json' if stop else 'DONE.json')).read_text())
try:
 write(CTRL/'status.json',dict(stage='WAIT_K6_0_AND_GRADIENT_PREFLIGHT',time=time.time()))
 deadline=time.time()+1800
 while not all((CTRL.parent/'donor_audit'/f'DONE_rank{i}.json').exists() for i in range(8)) or not (CTRL/'GRADIENT_EQUIVALENCE.json').exists():
  if time.time()>deadline:raise RuntimeError('Prerequisite deadline; diagnose audit/gradient tests')
  time.sleep(10)
 files=list((CTRL.parent/'donor_audit'/'K5-B').glob('case*/*/result.json'));assert len(files)==192
 for p in files:
  d=json.loads(p.read_text());assert hashlib.sha256((p.parent/'prediction.png').read_bytes()).hexdigest()==d['prediction_sha256']
 assert json.loads((CTRL/'GRADIENT_EQUIVALENCE.json').read_text())['status']=='PASS'
 decision('k6_0_generation_accepted',count=192,note='Operational donor proxies; no contamination conclusion; unchanged training donor policy')
 for arm in ['K6-A','K6-B']:
  name=arm+'-PREFLIGHT-20260920';assert run(arm,name,True,False,2)['step']==2;assert run(arm,name,True,True)['step']==4
  rows=[]
  for rank in range(8):
   rr=[json.loads(x) for x in (RUN/name/f'rank{rank}.jsonl').read_text().splitlines()];assert rr[-1]['step']==4 and all(r['ddp_spread']<1e-4 for r in rr);assert any(r['es_adapter_grad']>0 for r in rr);rows+=rr
  assert any(r['k6_eligible']>0 for r in rows),'No eligible supervision in real preflight'
 write(CTRL/'PREFLIGHT_PASSED.json',dict(status='PASS',identity=identity,time=time.time(),checks=['same sampler 100 episodes','8rank relational gradient parity','A/B full state resume 2 to 4','real eligible supervision','Es adapter gradients/rank sync']))
 for arm in ['K6-A','K6-B']:
  d=run(arm,arm+'-V2-K0-S3407');assert d['status']=='completed' and d['step']==10000 and not d['smoke'];decision('training_completed',arm=arm,step=d['step'],attempt=d['attempt'])
 write(CTRL/'status.json',dict(stage='TRAINING_COMPLETED_EVALUATION_PENDING',time=time.time()))
except Exception as e:
 decision('NEEDS_DIAGNOSIS',error=str(e));write(CTRL/'status.json',dict(stage='NEEDS_RECOVERY',error=str(e),time=time.time()));raise
