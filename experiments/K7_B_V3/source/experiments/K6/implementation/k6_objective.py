import json,inspect
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from scripts.k_components import KSampler
# Same random calls, same shuffled occurrences, explicit pair occurrence identities.
src=inspect.getsource(KSampler.batch)
src=src.replace('    def batch(self, attempt):','def paired_batch(self, attempt):',1)
src='\n'.join([src.splitlines()[0]]+[x[4:] for x in src.splitlines()[1:]])
src=src.replace('pairs.append(pair)','pairs.append((len(ids)-2,len(ids)-1))')
src=src.replace('rng.shuffle(ids)','ordered=list(enumerate(ids)); rng.shuffle(ordered)\n    remap={old:new for new,(old,_) in enumerate(ordered)}\n    ids=[i for _,i in ordered]')
src=src.replace('return ids, report',"report['pair_positions']=[(remap[i],remap[j]) for i,j in pairs]\n    return ids, report")
from scripts.k_components import random,SCRIPT,COMPLEX,PAIRS
exec(src)

def distance(xs,ys,scales):
    return sum((x.float()-y.float()).abs().flatten(1).mean(1)/s for x,y,s in zip(xs,ys,scales))/len(scales)

def extra_objective(arm,pred,gt,neutral,alpha,cfg,scripts,pairs,cal,step):
    with torch.autocast('cuda',enabled=False):
        alpha=alpha.float().reshape(-1);valid=(alpha>=.5)&(~cfg)
        scales=cal['scales'];tau=alpha.new_tensor([cal['tau'][s] for s in scripts])
        if arm=='K6-A':
            delta=distance(gt,neutral,scales).detach();pos=distance(pred,gt,scales);neg=distance(pred,neutral,scales)
            valid=valid&(delta>tau)
            values=(.2*delta+pos-neg).relu()*valid*alpha.sqrt()
            loss=values.mean();eligible=valid.sum()
        else:
            assert dist.get_world_size()==8
            pp=[torch.cat(all_gather(x.float()),0) for x in pred]
            def gather(x):
                r=[torch.empty_like(x) for _ in range(8)];dist.all_gather(r,x.contiguous());return torch.cat(r,0)
            gg=[gather(x.float().detach()) for x in gt];aa=gather(alpha);vv=gather(valid);tt=gather(tau)
            ii=torch.tensor([x[0] for x in pairs],device=alpha.device);jj=torch.tensor([x[1] for x in pairs],device=alpha.device)
            target=distance([x[ii] for x in gg],[x[jj] for x in gg],scales).detach()
            actual=distance([x[ii] for x in pp],[x[jj] for x in pp],scales)
            enabled=vv[ii]&vv[jj]&(target>torch.maximum(tt[ii],tt[jj]))
            loss=((actual-target).abs()*enabled*torch.minimum(aa[ii],aa[jj]).sqrt()).sum()/8
            eligible=enabled.sum()
        result=.05*min(1.,step/1000)*loss
        return result,dict(k6_loss=float(loss.detach()),k6_weighted=float(result.detach()),k6_eligible=int(eligible))
