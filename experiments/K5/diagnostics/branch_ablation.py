import sys,pathlib,json,types,inspect,copy,time,fcntl,dataclasses,os
C=pathlib.Path('/root/projects/hrfont_k5_20260919_r2');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch,numpy as np
from PIL import Image
from k5_runtime import model_for,DataContext,ROOT,T,dataset,batch_to,atomic_json,sha256_file
import k4_eval as E
from scripts.hrfont_i import SetOffset
from chunk_router import chunk_forward
CTRL=pathlib.Path('/root/data1/hrfont_k5_20260919/control');OUT=CTRL.parent/'branch_ablation';OUT.mkdir(exist_ok=True)
rank=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));torch.cuda.set_device(rank)
lock=(CTRL/f'BRANCH_ABLATION_{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((CTRL/'REAL_GENERATION_FP32_PARITY.json').read_text())['status']=='PASS'
exec(inspect.getsource(E.sample_batch).replace("torch.autocast('cuda',dtype=torch.float16)","torch.autocast('cuda',enabled=False)"),E.__dict__)
torch.set_num_threads(1);torch.manual_seed(3407);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
plan=json.loads(pathlib.Path('/root/data1/hrfont_k4a_intervention_matched_20260918/PLAN.json').read_text());cases=plan['jobs'];seeds=plan['seeds'];modes=['baseline','ec_only','es_only','zero_offset']
atomic_json(OUT/f'PLAN_rank{rank}.json',dict(cases=cases,seeds=seeds,modes=modes,precision='full FP32 original offsets with forward-hook branch interventions; fresh baseline',source=str(C),bank='v2/train weight-only',chunk=None,script_sha256=sha256_file(pathlib.Path(__file__))))
for arm in ['K5-B']:
 model,args=model_for(arm,torch.device('cuda',rank),OUT/(arm+f'_provenance_rank{rank}'),evaluation=True);weights=ROOT/'runs'/f'{arm}-V2-K1RECIPE-S3407/global_step_10000/ema.pth';model.load_train_state(torch.load(weights,map_location='cpu',weights_only=True));model.eval();data=DataContext(args,torch.device('cuda',rank),model);ds=dataset(args,'train');scheduler=T.build_ddpm_scheduler(args)
 select=data.library.select;conditions=data.conditions;state={};traces=[]
 for name,m in model.named_modules():
  if not hasattr(m,'es_gate'):continue
  def save_ec(module,inputs,output,parent=m):parent.ablate_ec=output
  def save_es(module,inputs,output,parent=m):parent.ablate_es=output
  m.ec_branch.register_forward_hook(save_ec);m.es_branch.register_forward_hook(save_es)
  def intervene(module,inputs,output):
   mode=state['mode']
   if mode=='ec_only':return module.ablate_ec
   if mode=='es_only':return (module.es_gate.float().tanh()*module.ablate_es.float()).to(output.dtype)
   if mode=='zero_offset':return torch.zeros_like(output)
   return output
  m.register_forward_hook(intervene)
 def selection(cp,refs,q,font,cfg):
  fs,a=select(cp,refs,q,font,cfg);traces.append(dict(font=font,cp=cp,mode=state['mode'],donors=fs,alpha=a.tolist()));return fs,a
 def cond(*a,**kw):
  if state['mode']=='off':kw['no_delta']=True
  return conditions(*a,**kw)
 data.library.select=selection;data.conditions=cond
 for i,c in enumerate(cases):
  if i%world!=rank:continue
  for seed in seeds:
   j=dict(split='train',font=c['font'],cp=c['cp'],refs=c['refs'],k=len(c['refs']),seed=seed,script='western');sample=E.get_sample(ds,j);batch=batch_to([sample],data.device)
   for mode in modes:
    state['mode']=mode;dest=OUT/arm/f'case{i:02d}_seed{seed}'/mode;dest.mkdir(parents=True,exist_ok=True);p=dest/'prediction.png'
    if (dest/'result.json').exists():
     old=json.loads((dest/'result.json').read_text());assert sha256_file(p)==old['prediction_sha256'];continue
    traces.clear();start=time.time();pred=E.sample_batch(model,data,batch,scheduler,[seed]);Image.fromarray(pred[0]).save(p)
    row=E.score_rows(pred,batch,[j],[sample],[p],data)[0];row.update(mode=mode,case=i,arm=arm,seconds=time.time()-start,selection=traces,weights_sha256=sha256_file(weights));atomic_json(dest/'result.json',row);atomic_json(OUT/f'PROGRESS_rank{rank}.json',dict(arm=arm,case=i,seed=seed,mode=mode,time=time.time()));print(arm,i,seed,mode,flush=True)
 del data,model;torch.cuda.empty_cache()
atomic_json(OUT/f'DONE_rank{rank}.json',dict(status='generation_complete_analysis_pending',expected=len(cases)*len(seeds)*len(modes),time=time.time()))
