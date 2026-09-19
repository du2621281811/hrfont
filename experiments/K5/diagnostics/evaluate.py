import os,sys,json,time,hashlib
from pathlib import Path
CODE=Path('/root/projects/hrfont_k5_20260919_r2')
sys.path[:0]=[str(CODE/'scripts'),str(CODE)]
import torch
import torch.distributed as dist
from k5_runtime import model_for,DataContext,ROOT,T,dataset,atomic_json,sha256_file
from k4_eval import run_jobs,expand_queries,sample_batch,get_sample,batch_to
OUT=Path('/root/data1/hrfont_k5_20260919/evaluation')
rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(rank);torch.set_num_threads(1);dist.init_process_group('nccl')
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
for arm in ['K5-A','K5-B']:
 dest=OUT/arm;dest.mkdir(parents=True,exist_ok=True)
 run=ROOT/'runs'/f'{arm}-V2-K1RECIPE-S3407'
 assert json.loads((run/'DONE.json').read_text())['step']==10000
 weights=run/'global_step_10000/ema.pth'
 model,args=model_for(arm,torch.device('cuda',rank),dest/f'provenance_rank{rank}',evaluation=True)
 model.load_train_state(torch.load(weights,map_location='cpu',weights_only=True));model.eval()
 data=DataContext(args,torch.device('cuda',rank),model);scheduler=T.build_ddpm_scheduler(args)
 train=json.loads(Path('/root/data1/hrfont_default_inference_20260917/train_manifest.json').read_text())['jobs']
 jobs=train+expand_queries('K1')
 protocol=dict(arm=arm,source=str(CODE),source_identity=json.loads((CODE/'K5_CODE_IDENTITY.json').read_text()),weights=str(weights),weights_sha256=sha256_file(weights),jobs=jobs,bank='v2/train',exclusion='weight-only',shots=[1,2,4,8],steps=20,order=2,cfg=1,batch=4,train_protocol='legacy default fixed queries',val_test_protocol='frozen full v2 queries',runner_sha256=sha256_file(Path(__file__)))
 if rank==0:
  p=dest/'PROTOCOL.json'
  if p.exists():assert json.loads(p.read_text())==protocol
  else:atomic_json(p,protocol)
 dist.barrier()
 for split in ['train','val','test']:
  selected=[j for j in jobs if j['split']==split];ds=dataset(args,split)
  assert all((j['font'],j['cp']) in ds.lookup and all(c in ds.style_by_font_char[j['font']] for c in j['refs']) for j in selected)
  run_jobs(model,data,args,scheduler,selected,dest/split,rank=rank,world=world)
 if rank==0:atomic_json(dest/'DONE.json',dict(status='completed',rows=len(jobs),time=time.time()))
 del model,data;torch.cuda.empty_cache();dist.barrier()
if rank==0:atomic_json(OUT/'DONE.json',dict(status='inference_completed_metrics_board_pending',time=time.time()))
dist.destroy_process_group()
