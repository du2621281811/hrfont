import sys,json,time,dataclasses,fcntl
from pathlib import Path
C=Path('/root/projects/hrfont_k5_20260919_r2');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch
from k5_runtime import model_for,DataContext,ROOT,T,dataset,batch_to,atomic_json,sha256_file
import k4_eval as E
ctrl=Path('/root/data1/hrfont_k5_20260919/control');lock=(ctrl/'REPRESENTATION_AUDIT.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(1);torch.manual_seed(3407)
model,args=model_for('K5-B',torch.device('cuda'),ctrl/'representation_audit_provenance',evaluation=True)
model.eval();data=DataContext(args,torch.device('cuda'),model);ds=dataset(args,'train');rows=[]
manifest=dict(font='FZBangSKLTJW',chars=['u0041','u0067','u0033'],refs=['u6C38','u548C','u4E66','u98CE'],source=str(C),bank='V2 train weight-only exclusion',scope='frozen encoder raw spatial differences; not generated quality',time=time.time())
atomic_json(ctrl/'REPRESENTATION_AUDIT_PLAN.json',manifest)
for cp in manifest['chars']:
 j=dict(split='train',font=manifest['font'],cp=cp,refs=manifest['refs'],k=4,seed=3407,script='western');s=E.get_sample(ds,j);b=batch_to([s],data.device)
 with torch.no_grad():_,queries,*_=T._style_conditions(data.es,b,data.device)
 fs,alpha=data.library.select(cp,j['refs'],queries[0],j['font'],dataclasses.replace(data.alpha_cfg,k_top=len(data.library.fonts),k_max=len(data.library.fonts),mode='topk'))
 top,_=data.library.select(cp,j['refs'],queries[0],j['font'],data.alpha_cfg)
 features=data.ec.features_many('target',[(f,cp) for f in fs]);neutral=data.ec.features_many('content',[('',cp)])
 for level,c in [(1,64),(2,128)]:
  for branch in ['Ec','Es']:
   offset=0 if branch=='Ec' else c
   x=torch.cat([features[f,cp][level][:,offset:offset+c]-neutral['',cp][level] for f in fs]).float().flatten(1)
   n,d=x.shape;raw_rms=float(x.square().mean().sqrt());x-=x.mean(0,keepdim=True);gram=torch.zeros(n,n,device='cuda',dtype=torch.float64)
   for start in range(0,d,8192):
    z=x[:,start:start+8192].cuda().double();gram+=z@z.T/d
   ev=torch.linalg.eigvalsh(gram).clamp_min(0);trace=ev.sum();prob=ev/trace.clamp_min(1e-30)
   idx=torch.tensor([fs.index(f) for f in top],device='cuda');sub=gram[idx][:,idx];cross=gram[:,idx];captured=(cross@torch.linalg.pinv(sub,rtol=1e-8)*cross).sum()/trace.clamp_min(1e-30)
   rows.append(dict(cp=cp,level=level,branch=branch,donors=n,dimension=d,raw_delta_rms=raw_rms,centered_rms=float(x.square().mean().sqrt()),effective_rank=float((-(prob*prob.clamp_min(1e-30).log()).sum()).exp()),participation_rank=float(trace.square()/ev.square().sum().clamp_min(1e-30)),top10_linear_span_variance_fraction=float(captured),top10=top,all_donors=fs))
   print(cp,level,branch,rows[-1]['effective_rank'],rows[-1]['top10_linear_span_variance_fraction'],flush=True)
   del x,gram
 atomic_json(ctrl/'REPRESENTATION_AUDIT_PROGRESS.json',dict(rows=rows,time=time.time()))
 del features
atomic_json(ctrl/'REPRESENTATION_AUDIT_DONE.json',dict(status='complete',plan=manifest,rows=rows,interpretation='Descriptive centered raw spatial encoder geometry. Projection uses unrestricted signed coefficients, NOT convex router weights or realizable generation. No target GT in selection.',script_sha256=sha256_file(Path(__file__)),time=time.time()))
