import sys,json,pathlib,types
C=pathlib.Path('/root/projects/hrfont_k5_20260919_r2');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch,numpy as np,inspect
import k4_eval as E
original_sample=sample_batch if False else E.sample_batch
exec(inspect.getsource(E.sample_batch).replace("torch.autocast('cuda',dtype=torch.float16)","torch.autocast('cuda',enabled=False)"),E.__dict__)
from k5_runtime import model_for,DataContext,ROOT,T,dataset,batch_to
from k4_eval import sample_batch,get_sample
from scripts.hrfont_i import SetOffset
from chunk_router import chunk_forward
O=pathlib.Path('/root/data1/hrfont_k5_20260919/control');torch.set_num_threads(1);torch.manual_seed(3407);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;results=[]
for arm in ['K5-A','K5-B']:
 model,args=model_for(arm,torch.device('cuda:0'),O/(arm+'_real_parity'),evaluation=True);model.load_train_state(torch.load(ROOT/'runs'/f'{arm}-V2-K1RECIPE-S3407/global_step_10000/ema.pth',map_location='cpu',weights_only=True));model.eval();data=DataContext(args,torch.device('cuda:0'),model);ds=dataset(args,'train');scheduler=T.build_ddpm_scheduler(args)
 modules=[m for m in model.modules() if isinstance(m,SetOffset)];originals=[m.forward for m in modules]
 def install(chunk):
  for m,fn in zip(modules,originals):
   def wrapped(self,h,p,_fn=fn):
    d,a,n,act,ref,t=p
    with torch.autocast('cuda',enabled=False):return (chunk_forward(self,h.float(),(d.float(),a.float(),n.float(),act,ref.float(),t),chunk=3) if chunk else _fn(h.float(),(d.float(),a.float(),n.float(),act,ref.float(),t))).to(h.dtype)
   m.forward=types.MethodType(wrapped,m)
 for font in ['FZBangSKLTJW','FZPANGPBJW','FZPTYJW']:
  j=dict(split='train',font=font,cp='u0041',k=4,refs=[f'u{ord(c):04X}' for c in '永和书风'],seed=3407);batch=batch_to([get_sample(ds,j)],data.device)
  for m,fn in zip(modules,originals):m.forward=fn
  amp=original_sample(model,data,batch,scheduler,[3407]);install(False);dense=sample_batch(model,data,batch,scheduler,[3407]);install(True);chunk=sample_batch(model,data,batch,scheduler,[3407]);diff=np.abs(dense.astype(float)-chunk.astype(float));row=dict(arm=arm,font=font,cp='u0041',max_pixel_difference=float(diff.max()),mean_pixel_difference=float(diff.mean()),changed_fraction=float((diff>0).mean()),amp_vs_fp32_mean=float(np.abs(amp.astype(float)-dense.astype(float)).mean()))
  assert diff.max()<=1 and diff.mean()<.01,row;results.append(row);print(row,flush=True)
 del data,model;torch.cuda.empty_cache()
(O/'REAL_GENERATION_FP32_PARITY.json').write_text(json.dumps(dict(status='PASS',rows=results,protocol='20-step DPM++ order2 CFG1; common FP32 offsets; original AMP comparison retained'),indent=2));print('PASS',flush=True)
