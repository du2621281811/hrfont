import os,sys,json
from pathlib import Path
C=Path('/root/projects/hrfont_k6_20260920');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch,torch.distributed as dist
from k6_objective import extra_objective,distance
from k6_runtime import atomic_json,STORE
rank=int(os.environ['RANK']);dist.init_process_group('gloo');torch.set_num_threads(1)
torch.manual_seed(887);x=torch.randn(64,3,4,4);gt=torch.randn_like(x);pairs=[(i,63-i) for i in range(8)];cal={'scales':[1.], 'tau':{'western':0.}}
w=torch.tensor(1.1,requires_grad=True);sl=slice(rank*8,(rank+1)*8)
loss,rec=extra_objective('K6-B',[x[sl]*w],[gt[sl]],None,torch.ones(8),torch.zeros(8,dtype=torch.bool),['western']*8,pairs,cal,1000)
loss.backward();gradient=w.grad.clone();dist.all_reduce(gradient);gradient/=8
ref=torch.tensor(1.1,requires_grad=True);i=torch.tensor([a for a,b in pairs]);j=torch.tensor([b for a,b in pairs]);expected=.05*(distance([x[i]*ref],[x[j]*ref],[1.])-distance([gt[i]],[gt[j]],[1.])).abs().mean();expected.backward()
assert torch.allclose(gradient,ref.grad,atol=1e-7,rtol=1e-5),(gradient,ref.grad)
# Exact zero when no sample is eligible, but autograd collectives remain connected.
w2=torch.tensor(1.1,requires_grad=True);z,_=extra_objective('K6-B',[x[sl]*w2],[gt[sl]],None,torch.zeros(8),torch.zeros(8,dtype=torch.bool),['western']*8,pairs,cal,1000);z.backward();assert w2.grad==0
if rank==0:atomic_json(STORE/'control/GRADIENT_EQUIVALENCE.json',dict(status='PASS',gradient=float(gradient),reference=float(ref.grad),all_masked_zero=True,world=8))
dist.destroy_process_group()
