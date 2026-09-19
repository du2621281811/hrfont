import sys,json,types,time
from pathlib import Path
C=Path('/root/projects/hrfont_k5_20260919_r2');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch
from k5_runtime import model_for,DataContext,ROOT,T,dataset,batch_to,atomic_json
from scripts.hrfont_i import SetOffset
import k4_eval as E
import inspect
exec(inspect.getsource(E.sample_batch).replace("torch.autocast('cuda',dtype=torch.float16)","torch.autocast('cuda',enabled=False)"),E.__dict__)
torch.set_num_threads(1);torch.manual_seed(3407)
ctrl=Path('/root/data1/hrfont_k5_20260919/control');results=[]
for arm in ['K5-B']:
 model,args=model_for(arm,torch.device('cuda'),ctrl/(arm+'_offset_combination_probe'),evaluation=True)
 model.load_train_state(torch.load(ROOT/'runs'/f'{arm}-V2-K1RECIPE-S3407/global_step_10000/ema.pth',map_location='cpu',weights_only=True));model.eval();data=DataContext(args,torch.device('cuda'),model);ds=dataset(args,'train');sched=T.build_ddpm_scheduler(args)
 gates={n:float(m.es_gate.tanh()) for n,m in model.named_modules() if hasattr(m,'es_gate')}
 for name,m in model.named_modules():
  if not isinstance(m,SetOffset):continue
  original=m.forward
  def wrapped(self,h,payload,original=original,name=name):
   out=original(h,payload)
   self.probe_last=out.detach()
   call=getattr(self,'probe_calls',0);self.probe_calls=call+1
   if call in [0,5,10,15,19]:
    d,*rest=payload
    row=dict(call=call,arm=arm,layer=name,t=rest[-1].flatten().tolist(),delta_rms=float(d.float().square().mean().sqrt()),output_rms=float(out.float().square().mean().sqrt()))
    for key,alt in [('zero',torch.zeros_like(d)),('flip_spatial',d.flip(-1))]:
     changed=original(h,(alt,*rest));row[key+'_rms_diff']=float((changed.float()-out.float()).square().mean().sqrt());row[key+'_max_diff']=float((changed.float()-out.float()).abs().max())
    results.append(row)
   return out
  m.forward=types.MethodType(wrapped,m)
 for name,m in model.named_modules():
  if not hasattr(m,'es_gate'):continue
  def hook(module,inputs,output,name=name):
   call=getattr(module,'combined_probe_calls',0);module.combined_probe_calls=call+1
   if call not in [0,5,10,15,19]:return
   ec=module.ec_branch.probe_last.float();es=module.es_branch.probe_last.float();g=module.es_gate.tanh().float();weighted=g*es
   rms=lambda x:float(x.square().mean().sqrt())
   results.append(dict(arm=arm,layer=name,kind='combined',call=call,gate=float(g),ec_rms=rms(ec),es_gated_rms=rms(weighted),combined_rms=rms(output.float()),sum_parity_max=float((output.float()-ec-weighted).abs().max()),ec_es_cosine=float(torch.nn.functional.cosine_similarity(ec.flatten(),weighted.flatten(),dim=0)),cancellation_ratio=rms(output.float())/max(rms(ec)+rms(weighted),1e-30)))
  m.register_forward_hook(hook)
 j=dict(split='train',font='FZBangSKLTJW',cp='u0041',refs=['u6C38','u548C','u4E66','u98CE'],k=4,seed=3407,script='western');s=E.get_sample(ds,j)
 with torch.no_grad():E.sample_batch(model,data,batch_to([s],data.device),sched,[3407])
 atomic_json(ctrl/f'{arm}_OFFSET_COMBINATION_PROBE.json',dict(rows=[x for x in results if x['arm']==arm],gates=gates,scope='one real glyph frozen-hidden probe at 5 denoising calls; descriptive not quality',time=time.time()))
 del data,model;torch.cuda.empty_cache()
atomic_json(ctrl/'OFFSET_COMBINATION_PROBE_DONE.json',dict(status='complete',rows=results,time=time.time()))
