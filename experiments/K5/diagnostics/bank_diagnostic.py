import sys,pathlib,json,types,inspect,copy,time,fcntl,dataclasses
C=pathlib.Path('/root/projects/hrfont_k5_20260919_r2');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch,numpy as np
from PIL import Image
from k5_runtime import model_for,DataContext,ROOT,T,dataset,batch_to,atomic_json,sha256_file
import k4_eval as E
from scripts.hrfont_i import SetOffset
from chunk_router import chunk_forward
CTRL=pathlib.Path('/root/data1/hrfont_k5_20260919/control');OUT=CTRL.parent/'bank_diagnostic';OUT.mkdir(exist_ok=True)
lock=(CTRL/'BANK.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((CTRL/'REAL_GENERATION_FP32_PARITY.json').read_text())['status']=='PASS'
exec(inspect.getsource(E.sample_batch).replace("torch.autocast('cuda',dtype=torch.float16)","torch.autocast('cuda',enabled=False)"),E.__dict__)
torch.set_num_threads(1);torch.manual_seed(3407);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
plan=json.loads(pathlib.Path('/root/data1/hrfont_k4a_intervention_matched_20260918/PLAN.json').read_text());cases=plan['jobs'];seeds=plan['seeds'];modes=['top10','top10_uniform','top50','all','all_uniform','off']
atomic_json(OUT/'PLAN.json',dict(cases=cases,seeds=seeds,modes=modes,precision='full FP32 denoiser and chunked offsets, fresh baselines',source=str(C),bank='v2/train weight-only',chunk=16,script_sha256=sha256_file(pathlib.Path(__file__))))
for arm in ['K5-A','K5-B']:
 model,args=model_for(arm,torch.device('cuda:0'),OUT/(arm+'_provenance'),evaluation=True);weights=ROOT/'runs'/f'{arm}-V2-K1RECIPE-S3407/global_step_10000/ema.pth';model.load_train_state(torch.load(weights,map_location='cpu',weights_only=True));model.eval();data=DataContext(args,torch.device('cuda:0'),model);ds=dataset(args,'train');scheduler=T.build_ddpm_scheduler(args)
 for m in model.modules():
  if isinstance(m,SetOffset):m.forward=types.MethodType(chunk_forward,m)
 select=data.library.select;conditions=data.conditions;state={};traces=[]
 def selection(cp,refs,q,font,cfg):
  cfg=copy.deepcopy(cfg);mode=state['mode']
  if mode=='top50':cfg=dataclasses.replace(cfg,k_top=50,k_max=50,mode='topk')
  if mode.startswith('all'):cfg=dataclasses.replace(cfg,k_top=len(data.library.fonts),k_max=len(data.library.fonts),mode='topk')
  fs,a=select(cp,refs,q,font,cfg)
  if mode.endswith('uniform'):a=torch.ones_like(a)/len(a)
  traces.append(dict(font=font,cp=cp,mode=mode,donors=fs,alpha=a.tolist()));return fs,a
 def cond(*a,**kw):
  if state['mode']=='off':kw['no_delta']=True
  return conditions(*a,**kw)
 data.library.select=selection;data.conditions=cond
 for i,c in enumerate(cases):
  for seed in seeds:
   j=dict(split='train',font=c['font'],cp=c['cp'],refs=c['refs'],k=len(c['refs']),seed=seed,script='western');sample=E.get_sample(ds,j);batch=batch_to([sample],data.device)
   for mode in modes:
    state['mode']=mode;dest=OUT/arm/f'case{i:02d}_seed{seed}'/mode;dest.mkdir(parents=True,exist_ok=True);p=dest/'prediction.png'
    if (dest/'result.json').exists():
     old=json.loads((dest/'result.json').read_text());assert sha256_file(p)==old['prediction_sha256'];continue
    traces.clear();start=time.time();pred=E.sample_batch(model,data,batch,scheduler,[seed]);Image.fromarray(pred[0]).save(p)
    row=E.score_rows(pred,batch,[j],[sample],[p],data)[0];row.update(mode=mode,case=i,arm=arm,seconds=time.time()-start,selection=traces,weights_sha256=sha256_file(weights));atomic_json(dest/'result.json',row);atomic_json(OUT/'PROGRESS.json',dict(arm=arm,case=i,seed=seed,mode=mode,time=time.time()));print(arm,i,seed,mode,flush=True)
 del data,model;torch.cuda.empty_cache()
atomic_json(OUT/'DONE.json',dict(status='generation_complete_analysis_pending',expected=2*len(cases)*len(seeds)*len(modes),time=time.time()))
