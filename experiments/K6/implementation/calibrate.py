import sys,json,random,time
from pathlib import Path
C=Path('/root/projects/hrfont_k6_20260920');sys.path[:0]=[str(C/'scripts'),str(C)]
import torch
from k6_runtime import *
from k6_objective import distance,paired_batch
from scripts.k_components import KSampler
from k4_runtime import read_unit
import numpy as np
ctrl=STORE/'control';torch.set_num_threads(1);torch.manual_seed(3407)
model,args=model_for('K6-A',torch.device('cuda'),ctrl/'calibration_provenance');ds=dataset(args,'train');del model;torch.cuda.empty_cache()
vgg=T.ContentPerceptualLoss().VGG.cuda().eval().requires_grad_(False)
sampler=KSampler(ds,C/'experiments/K4/detail_manifest.json')
for attempt in range(100):
 a,_=sampler.batch(attempt);b,q=paired_batch(sampler,attempt);assert a==b
 for i,j in q['pair_positions']:assert sampler.entries[a[i]][1]==sampler.entries[a[j]][1] and sampler.entries[a[i]][0]!=sampler.entries[a[j]][0]
rng=random.Random(63407);indices=sorted(rng.sample(range(len(ds)),min(4096,len(ds))))
rows=[]
with torch.no_grad():
 for start in range(0,len(indices),32):
  ids=indices[start:start+32];ss=[ds.sample(i) for i in ids]
  g=vgg(T.normalize_mean_std(torch.stack([x['nonorm_target_image'] for x in ss]).cuda()))
  n=vgg(T.normalize_mean_std(torch.stack([x['content_image']/2+.5 for x in ss]).cuda()))
  dd=torch.stack([(x.float()-y.float()).abs().flatten(1).mean(1) for x,y in zip(g,n)],1).cpu().numpy()
  rows.extend(dict(index=i,font=sampler.entries[i][0],cp=sampler.entries[i][1],script=sampler.entries[i][2],distances=d.tolist()) for i,d in zip(ids,dd))
  print('calibration',start+len(ids),flush=True)
a=np.array([r['distances'] for r in rows]);scales=np.maximum(np.median(a,axis=0),1e-6);d=(a/scales).mean(1)
tau={s:float(np.median(d[[r['script']==s for r in rows]])) for s in ['western','kana','bopomofo']}
atomic_json(ctrl/'CALIBRATION.json',dict(status='PASS',scales=scales.tolist(),tau=tau,rows=rows,seed=63407,split='train',sampler_equivalence_100_episodes=True,encoder=str(PARENT0),time=time.time()))
print('CALIBRATION_COMPLETE',scales,tau,flush=True)
