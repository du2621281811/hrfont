import os,sys,json,time,fcntl,subprocess
from pathlib import Path
C=Path('/root/projects/hrfont_k5_20260919');CTRL=Path('/root/data1/hrfont_k5_20260919/control');RUN=Path('/root/projects/hrfont/runs');CK=Path('/root/data2/hrfont_k5_20260919/checkpoints')
lock=(CTRL/'QUEUE.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def decision(action,**kw):
 with (CTRL/'DECISIONS.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),action=action,**kw))+'\n')
env=dict(os.environ,PYTHONPATH=str(C),OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
launch=[str(Path(sys.executable).parent/'torchrun'),'--standalone','--nproc_per_node=8',str(C/'scripts/train_k5.py')]
identity=json.loads((C/'K5_CODE_IDENTITY.json').read_text())
def run(arm,name,smoke=False,resume=False,stop=0):
 args=launch+['--arm',arm,'--run-id',name,'--limit','4' if smoke else '10000','--state-interval','2' if smoke else '200','--checkpoint-root',str(CK)]
 if smoke:args+=['--smoke']
 if resume:args+=['--resume',str(RUN/name/'last_state')]
 if stop:args+=['--stop-after',str(stop)]
 if (CTRL/'STOP').exists():raise RuntimeError('K5 explicit STOP')
 log=CTRL/(name+('_resume' if resume else '')+'.log')
 with log.open('a') as f:
  p=subprocess.Popen(args,cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  write(CTRL/'status.json',dict(stage=name,pid=p.pid,smoke=smoke,resume=resume,time=time.time()))
  decision('launch',name=name,pid=p.pid,resume=resume,log=str(log));rc=p.wait()
 if rc:raise RuntimeError(f'{name} exit {rc}; diagnose {log}')
 return json.loads((RUN/name/('STOPPED.json' if stop else 'DONE.json')).read_text())
try:
 # Prior A preflight was explicitly launched and verified by the initiating agent.
 first=RUN/'K5-A-PREFLIGHT-20260919'
 while not (first/'STOPPED.json').exists():
  if time.time()-first.stat().st_mtime>1800:raise RuntimeError('Initial preflight did not finish; inspect preflight_A.log')
  time.sleep(10)
 # Wait for torchrun to release resources before resuming.
 while subprocess.run(['pgrep','-f','torchrun.*train_k5.py.*K5-A-PREFLIGHT-20260919'],stdout=subprocess.DEVNULL).returncode==0:time.sleep(3)
 assert run('K5-A',first.name,True,True)['step']==4
 b='K5-B-PREFLIGHT-20260919';assert run('K5-B',b,True,False,2)['step']==2;assert run('K5-B',b,True,True)['step']==4
 for name in [first.name,b]:
  for rank in range(8):
   rows=[json.loads(x) for x in (RUN/name/f'rank{rank}.jsonl').read_text().splitlines()]
   assert rows[-1]['step']==4 and all(x['ddp_spread']<1e-4 for x in rows)
   assert any(x['es_adapter_grad']>0 for x in rows),('missing adapter gradient',name,rank)
   if name.startswith('K5-B'):assert any(x['es_gate_grad']>0 for x in rows)
 write(CTRL/'PREFLIGHT_PASSED.json',dict(status='PASS',identity=identity,time=time.time(),checks=['8rank real input forward/backward A+B','adapter and dual gate gradients','complete state save/resume each arm','rank synchronization']))
 decision('preflight_pass',source=str(C),checks='See PREFLIGHT_PASSED.json')
 for arm in ['K5-A','K5-B']:
  name=arm+'-V2-K1RECIPE-S3407';d=run(arm,name)
  assert d['step']==10000 and d['status']=='completed' and not d['smoke']
  decision('training_completed',arm=arm,step=d['step'],attempt=d['attempt'])
 write(CTRL/'status.json',dict(stage='TRAINING_COMPLETED_EVALUATION_PENDING',time=time.time()))
except Exception as e:
 decision('requires_diagnosis',error=str(e));write(CTRL/'status.json',dict(stage='NEEDS_RECOVERY',error=str(e),time=time.time()));raise
