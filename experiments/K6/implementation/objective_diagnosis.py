import sys,json,time
from pathlib import Path
C=Path('/root/projects/hrfont_k6_20260920');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch
from k6_runtime import *
from k6_objective import paired_batch,distance
from scripts.k_components import KSampler,raw_x0
from scripts.k3_components import rollout
ctrl=STORE/'control';torch.set_num_threads(1);torch.manual_seed(3407)
model,args=model_for('K6-A',torch.device('cuda'),ctrl/'objective_diagnosis_provenance');model.eval();data=DataContext(args,torch.device('cuda'),model);ds=dataset(args,'train');sampler=KSampler(ds,C/'experiments/K4/detail_manifest.json')
cal=json.loads((ctrl/'CALIBRATION.json').read_text());eligible=[r for r in cal['rows'] if r['script']=='western' and sum(x/y for x,y in zip(r['distances'],cal['scales']))/3>cal['tau']['western']];r=next((r for r in eligible if r['font']=='FZBangSKLTJW'),eligible[0]);ia=r['index'];ib=next(i for i,e in enumerate(sampler.entries) if e[1]==r['cp'] and e[0]!=r['font'] and not e[3]);ss=[ds.sample(i) for i in [ia,ib]];batch=batch_to(ss,data.device)
vgg=T.ContentPerceptualLoss().VGG.cuda().eval().requires_grad_(False);scheduler=T.build_ddpm_scheduler(args);cal=json.loads((ctrl/'CALIBRATION.json').read_text());scales=cal['scales']
with torch.no_grad():
 gt=vgg(T.normalize_mean_std(batch['nonorm_target_image']));neutral=vgg(T.normalize_mean_std(batch['content_image']/2+.5));delta=distance(gt,neutral,scales);off=torch.zeros(2,device=data.device,dtype=torch.bool);conditions=data.conditions(batch,off,off,need_refs=True);style,refs,query,keep,content,structure=conditions
rows=[]
for t in [50,250,500,750]:
 with torch.no_grad():
  ts=torch.full((2,),t,device=data.device,dtype=torch.long);noise=torch.randn_like(batch['target_image']);noisy=scheduler.add_noise(batch['target_image'],noise,ts)
  with torch.autocast('cuda',dtype=torch.float16):pred,_,_=model(noisy,ts,style,refs,query,keep,content,structure,off,1.)
  clean,alpha=raw_x0(noisy,pred,scheduler.alphas_cumprod,ts);feat=vgg(T.normalize_mean_std(clean.clamp(0,1)));pos=distance(feat,gt,scales);neg=distance(feat,neutral,scales)
  rows.append(dict(kind='GT_noised',t=t,alpha=alpha.flatten().tolist(),pos=pos.tolist(),neg=neg.tolist(),margin=(.2*delta).tolist(),rank=(.2*delta+pos-neg).relu().tolist()))
# Diagnostic proposal only: this does not change the authorized training objective.
y,path=rollout(model,data,batch,scheduler,1.,763407,amp_enabled=True)
feat=vgg(T.normalize_mean_std(y.clamp(0,1)));pos=distance(feat,gt,scales);neg=distance(feat,neutral,scales);enabled=(delta>cal['tau']['western']);loss=((.2*delta+pos-neg).relu()*enabled).mean()
param=next(p for n,p in model.named_parameters() if n.startswith('base.unet.') and p.requires_grad)
grad=torch.autograd.grad(loss,param,allow_unused=True)[0]
rows.append(dict(kind='PROPOSED_pure_noise_DDIM8_not_training',pos=pos.detach().tolist(),neg=neg.detach().tolist(),rank=float(loss),unet_parameter_gradient=0 if grad is None else float(grad.float().norm())))
atomic_json(ctrl/'OBJECTIVE_DIAGNOSIS.json',dict(fonts=batch['font_stem'],chars=batch['char_cp'],rows=rows,eligible=enabled.tolist(),delta=delta.tolist(),tau=cal['tau']['western'],time=time.time(),training_changed=False))
print(json.dumps(rows),flush=True)
