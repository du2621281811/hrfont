import sys,pathlib,json,types,inspect,copy,time,fcntl,dataclasses
C=pathlib.Path('/root/projects/hrfont_k5_20260919_r2');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch,numpy as np
from PIL import Image
from k5_runtime import model_for,DataContext,ROOT,T,dataset,batch_to,atomic_json,sha256_file
import k4_eval as E
from scripts.hrfont_i import SetOffset
from audit_router import audited_forward
import random,os
CTRL=pathlib.Path('/root/data1/hrfont_k6_20260920/control');OUT=CTRL.parent/'donor_audit';OUT.mkdir(exist_ok=True)
rank=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));torch.cuda.set_device(rank)
lock=(CTRL/f'DONOR_AUDIT_{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((pathlib.Path('/root/data1/hrfont_k5_20260919/control')/'REAL_GENERATION_FP32_PARITY.json').read_text())['status']=='PASS'
exec(inspect.getsource(E.sample_batch).replace("torch.autocast('cuda',dtype=torch.float16)","torch.autocast('cuda',enabled=False)"),E.__dict__)
torch.set_num_threads(1);torch.manual_seed(3407);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
plan=json.loads(pathlib.Path('/root/data1/hrfont_k4a_intervention_matched_20260918/PLAN.json').read_text());cases=plan['jobs'];seeds=plan['seeds'];modes=['matched','neutral','random','adversarial']
atomic_json(OUT/f'PLAN_rank{rank}.json',dict(cases=cases,seeds=seeds,modes=modes,precision='full FP32 denoiser and chunked offsets, fresh baselines',source=str(C),bank='v2/train weight-only',chunk=16,script_sha256=sha256_file(pathlib.Path(__file__))))
for arm in ['K5-B']:
 model,args=model_for(arm,torch.device('cuda',rank),OUT/(arm+f'_provenance_rank{rank}'),evaluation=True);weights=ROOT/'runs'/f'{arm}-V2-K1RECIPE-S3407/global_step_10000/ema.pth';model.load_train_state(torch.load(weights,map_location='cpu',weights_only=True));model.eval();data=DataContext(args,torch.device('cuda',rank),model);ds=dataset(args,'train');scheduler=T.build_ddpm_scheduler(args)
 audit=[]
 for name,m in model.named_modules():
  if isinstance(m,SetOffset):
   m.audit_name=name;m.audit_sink=audit;m.forward=types.MethodType(audited_forward,m)
 select=data.library.select;conditions=data.conditions;state={};traces=[]
 detail=json.loads((C/'experiments/K4/detail_manifest.json').read_text())['confirmed_detail']
 calibration=json.loads((CTRL/'CALIBRATION.json').read_text()); ordinary={}
 for row in calibration['rows']:
  ordinary.setdefault(row['font'],[]).append(sum(x/y for x,y in zip(row['distances'],calibration['scales']))/len(calibration['scales']))
 ordinary={f:sum(v)/len(v) for f,v in ordinary.items()}
 def selection(cp,refs,q,font,cfg):
  cfg=dataclasses.replace(cfg,k_top=len(data.library.fonts),k_max=len(data.library.fonts),mode='topk')
  fs,a=select(cp,refs,q,font,cfg); mode=state['mode']
  scores=dict(zip(fs,a.tolist())); ordered=sorted(fs,key=lambda f:(-scores[f],f))
  if mode=='matched':chosen=ordered[:10]
  elif mode=='random':chosen=random.Random(73407+state['case']).sample(sorted(fs),10)
  elif mode=='neutral':
   candidates=[f for f in fs if detail.get(f+'|latin',{}).get('decision')!='confirmed_detail' and f in ordinary]
   assert len(candidates)>=10
   chosen=sorted(candidates,key=lambda f:(ordinary[f],f))[:10]
  else:chosen=list(reversed(ordered))[:10]
  data.library.family_policy.verify(font,chosen)
  traces.append(dict(font=font,cp=cp,mode=mode,donors=chosen,original_scores=[scores[f] for f in chosen],alpha=[.1]*10,definition='matched=Chinese reference near; neutral=unconfirmed-detail low train GT-neutral distance proxy; adversarial=Chinese reference far, not ground-truth Western adversarial'))
  return chosen,torch.ones(10)/10
 def cond(*a,**kw):
  if state['mode']=='off':kw['no_delta']=True
  return conditions(*a,**kw)
 data.library.select=selection;data.conditions=cond
 for i,c in enumerate(cases):
  if i%world!=rank:continue
  for seed in seeds:
   j=dict(split='train',font=c['font'],cp=c['cp'],refs=c['refs'],k=len(c['refs']),seed=seed,script='western');sample=E.get_sample(ds,j);batch=batch_to([sample],data.device)
   for mode in modes:
    state['mode']=mode;state['case']=i;dest=OUT/arm/f'case{i:02d}_seed{seed}'/mode;dest.mkdir(parents=True,exist_ok=True);p=dest/'prediction.png'
    if (dest/'result.json').exists():
     old=json.loads((dest/'result.json').read_text());assert sha256_file(p)==old['prediction_sha256'];continue
    traces.clear();audit.clear();start=time.time();pred=E.sample_batch(model,data,batch,scheduler,[seed]);Image.fromarray(pred[0]).save(p)
    row=E.score_rows(pred,batch,[j],[sample],[p],data)[0];row.update(router_audit=audit,mode=mode,case=i,arm=arm,seconds=time.time()-start,selection=traces,weights_sha256=sha256_file(weights));atomic_json(dest/'result.json',row);atomic_json(OUT/f'PROGRESS_rank{rank}.json',dict(arm=arm,case=i,seed=seed,mode=mode,time=time.time()));print(arm,i,seed,mode,flush=True)
 del data,model;torch.cuda.empty_cache()
atomic_json(OUT/f'DONE_rank{rank}.json',dict(rank=rank,status='generation_complete_analysis_pending',expected=len(cases)*len(seeds)*len(modes),time=time.time()))
