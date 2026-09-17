import json,os,sys
from pathlib import Path
sys.path.insert(0,'scripts')
import torch
from k4_runtime import *
from scripts.k3_components import PairSampler,rollout,pair_loss,isolated_rng
torch.set_num_threads(1);torch.cuda.set_device(0)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
device=torch.device('cuda:0')
m,args=model_for('K4-B',device,STORE/'control/router_probe_parent')
d=DataContext(args,device);ds=dataset(args,'train');s=KSampler(ds,ASSETS/'detail_manifest.json')
ps=PairSampler(ds,s,json.loads((ASSETS/'aliases.json').read_text()))
params=[p for p in m.parameters() if p.requires_grad]
names=[n for n,p in m.named_parameters() if p.requires_grad]
sch=T.build_ddpm_scheduler(args);results=[]
for run,step in [('K4-PREFLIGHT-B',17),('K4-PREFLIGHT-C',120)]:
    m.load_train_state(torch.load(ROOT/'runs'/run/'last_state/model.pth',map_location=device,weights_only=True))
    m.train();m.base.style_encoder.eval();m.base.content_encoder.eval()
    for rank in range(8):
        with isolated_rng(993407+rank):
            b,meta=ps.draw(0,rank,device,batch_to)
            out,path=rollout(m,d,b,sch,step/1000,993407+rank,trace=True)
            loss,parts=pair_loss(out,b['nonorm_target_image'])
            gg=torch.autograd.grad(loss,params,allow_unused=True)
            norms={token:sum(float(g.float().square().sum()) for n,g in zip(names,gg) if token in n and g is not None)**.5 for token in ['reader.','sc_interpreter_offsets.','zero_convs.']}
            row=dict(run=run,step=step,rank=rank,loss=float(loss),norms=norms,rollout_epsilon_grad_norms=[float(t.grad.norm()) for t in path])
            results.append(row);print(json.dumps(row),flush=True)
            del out,path,loss,gg,b
atomic_json(STORE/'control/ROUTER_PROBE.json',dict(results=results))
