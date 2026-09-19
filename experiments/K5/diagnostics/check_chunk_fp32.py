import sys,json,pathlib
C=pathlib.Path('/root/projects/hrfont_k5_20260919_r2');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch
from k5_runtime import model_for,ROOT
from chunk_router import chunk_forward
from scripts.hrfont_i import SetOffset
O=pathlib.Path('/root/data1/hrfont_k5_20260919/control');torch.set_num_threads(1);torch.manual_seed(3407);results=[]
for arm in ['K5-A','K5-B']:
 model,args=model_for(arm,torch.device('cuda:0'),O/(arm+'_chunk_provenance'),evaluation=True);model.load_train_state(torch.load(ROOT/'runs'/f'{arm}-V2-K1RECIPE-S3407/global_step_10000/ema.pth',map_location='cpu',weights_only=True));model.eval()
 for name,module in model.named_modules():
  if not isinstance(module,SetOffset):continue
  c=module.original.style_proj_in.in_channels;h=48 if c==64 else 24;hc=module.query.in_channels
  for m in [10,37]:
   hidden=torch.randn(1,hc,h,h,device='cuda');d=torch.randn(1,m,c,h,h,device='cuda');a=torch.rand(1,m,device='cuda');a=a/a.sum(1,keepdim=True);payload=(d,a,torch.randn(1,c,h,h,device='cuda'),torch.ones(1,device='cuda'),torch.randn(1,1024,device='cuda'),torch.tensor([500.],device='cuda'))
   with torch.no_grad(),torch.autocast('cuda',enabled=False):
    dense=module(hidden,payload);chunked=chunk_forward(module,hidden,payload)
   error=float((dense.float()-chunked.float()).abs().max());torch.testing.assert_close(chunked,dense,atol=2e-5,rtol=2e-5);results.append(dict(arm=arm,module=name,donors=m,max_abs=error))
 del model;torch.cuda.empty_cache()
(O/'CHUNK_FP32_EQUIVALENCE.json').write_text(json.dumps(dict(status='PASS',tests=results,note='Synthetic payloads on trained modules; real-input full-generation parity still required before bank diagnosis'),indent=2));print('PASS',len(results))
