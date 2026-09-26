"""Independent ImageNet and geometry views; freeze neighbors before E12 scores."""
import collections,hashlib,json,math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt,zoom
from data import STORE,ROOT,SCRIPTS,REF8,Images,atomic,sha
from model import Encoder
from train import encode

def geometry(path):
    im=np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255.;ink=1-im;mask=ink>.1;ys,xs=np.where(mask)
    if not len(xs):box=[0,0,0];width=[0,0,0]
    else:
        w,h=(xs.max()-xs.min()+1)/96,(ys.max()-ys.min()+1)/96;box=[w,h,w/max(h,1/96)]
        dt=distance_transform_edt(mask);width=(np.quantile(dt[mask],[.25,.5,.75])/96).tolist()
    gy,gx=np.gradient(ink);mag=np.hypot(gx,gy);angle=np.mod(np.arctan2(gy,gx),np.pi)
    hist=np.histogram(angle,bins=8,range=(0,np.pi),weights=mag)[0];hist/=max(hist.sum(),1e-6)
    proj=lambda x:x.reshape(16,6).mean(1)
    return np.concatenate([[ink.mean()],box,proj(ink.mean(0)),proj(ink.mean(1)),width,hist]).astype(np.float32)

def fixed_features():
    inv=json.loads((STORE/'inventory_all.json').read_text());rows=inv['rows'];dest=STORE/'independent_features.pt'
    if dest.exists():
        obj=torch.load(dest,map_location='cpu',weights_only=False);assert obj['inventory_sha256']==sha(STORE/'inventory_all.json');return inv,obj
    ids=[i for i,r in enumerate(rows) if r['role']=='target'];torch.set_num_threads(4);torch.cuda.set_device(0)
    encoder=Encoder(False,True).cuda().eval().requires_grad_(False)
    z=torch.zeros(len(rows),512);z[ids]=encode(encoder,Images(rows),ids,torch.device('cuda:0'))
    with ThreadPoolExecutor(max_workers=8) as pool:g=np.stack(list(pool.map(geometry,[rows[i]['path'] for i in ids])))
    m=json.loads((STORE/'manifest.json').read_text());fit=np.array([m['split'].get(rows[i]['font'])=='train' for i in ids])
    mean=g[fit].mean(0);std=np.maximum(g[fit].std(0),1e-3);gg=np.zeros((len(rows),g.shape[1]),np.float32);gg[ids]=(g-mean)/std
    obj=dict(z=z,g=torch.from_numpy(gg),geometry_mean=mean,geometry_std=std,inventory_sha256=sha(STORE/'inventory_all.json'),
        pretrained_sha256=sha(Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'),geometry_version='ink-bbox-proj16-distancewidth-orientation8-v1')
    tmp=dest.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(dest)
    return inv,obj

class Neighbors:
    def __init__(self,inv,obj):
        self.inv=inv;self.rows=inv['rows'];self.z=obj['z'];self.g=obj['g'];self.lookup={(r['main_split'],r['font'],r['role'],r['cp']):i for i,r in enumerate(self.rows)}
        self.cps=collections.defaultdict(set)
        for r in self.rows:
            if r['role']=='target':self.cps[r['main_split'],r['font'],r['script']].add(r['cp'])
    def pools(self,sp,font,script,pool,exclude=()):
        fs=[font]+[f for f in pool if f!=font and self.inv['groups'][f]!=self.inv['groups'][font] and len(self.cps[sp,f,script])>=4]
        if len(fs)<3:return [],[],dict(reason='insufficient_distinct_fonts')
        shared=set.intersection(*(self.cps[sp,f,script] for f in fs))-set(exclude)
        anchors=sorted(shared,key=lambda c:hashlib.sha256(('anchors3407:'+c).encode()).hexdigest())[:8]
        if len(anchors)<4:return [],[],dict(reason='insufficient_common_anchors',anchors=anchors)
        idx=torch.tensor([[self.lookup[sp,f,'target',c] for c in anchors] for f in fs])
        z=F.normalize(self.z[idx].mean(1),dim=1);g=self.g[idx].mean(1)
        d1=(1-z[0]@z[1:].T).numpy();d2=(g[1:]-g[0]).square().mean(1).sqrt().numpy();n=max(1,len(d1)//3)
        if np.ptp(d1)<1e-6 or np.ptp(d2)<1e-6:return [],[],dict(reason='indistinguishable_distances',anchors=anchors)
        near=set(np.argsort(d1,kind='stable')[:n])&set(np.argsort(d2,kind='stable')[:n])
        far=set(np.argsort(d1,kind='stable')[-n:])&set(np.argsort(d2,kind='stable')[-n:])
        return [fs[i+1] for i in sorted(near)],[fs[i+1] for i in sorted(far)],dict(anchors=anchors,distances={f:[float(x),float(y)] for f,x,y in zip(fs[1:],d1,d2)})

def build():
    inv,obj=fixed_features();n=Neighbors(inv,obj);m=json.loads((STORE/'manifest.json').read_text());pool=[f for f in m['split'] if m['split'][f]=='train']
    neighbors={}
    for f in pool:
        neighbors[f]={}
        for g in SCRIPTS:
            eligible=[other for other in pool if len(n.cps['train',other,g])>=4]
            if f not in eligible:continue
            near,_,_=n.pools('train',f,g,eligible);neighbors[f][g]=near
    atomic(STORE/'train_neighbors.json',dict(neighbors=neighbors,independent_features_sha256=sha(STORE/'independent_features.pt')))
    # External lists are frozen before scoring or E12 model selection.
    r0=json.loads((ROOT/'reports/k0_original_test_20260917/metrics.json').read_text())['rows']
    r1=json.loads((ROOT/'runs/K1-ORIGINAL-V0917-S3407/eval_step_10000/metrics.json').read_text())['rows']
    key=lambda r:(r['font'],r['cp'],r['k']);maps=[{key(r):r for r in rr} for rr in (r0,r1)]
    assert maps[0].keys()==maps[1].keys() and len(r0)==2816
    pairs=sorted({(r['font'],r['cp']) for r in r0});fixed={}
    for f,c in pairs:
        idx=n.lookup['test',f,'target',c];r=n.rows[idx];g=r['script']
        otherc=[x for x in n.cps['test',f,g] if x!=c and n.rows[n.lookup['test',f,'target',x]]['rgb_sha256']!=r['rgb_sha256']]
        otherc.sort(key=lambda x:hashlib.sha256(('X:'+f+c+x).encode()).hexdigest())
        if not otherc:fixed[f,c]=dict(status='NO_DISTINCT_X',A=r['path']);continue
        x=otherc[0];pool=[ff for ff in inv['main_fonts']['test'] if c in n.cps['test',ff,g] and
            n.rows[n.lookup['test',ff,'target',c]]['rgb_sha256']!=r['rgb_sha256']]
        near,far,meta=n.pools('test',f,g,pool,exclude=[c,x]);choose=lambda xs:sorted(xs,key=lambda ff:hashlib.sha256((f+c+ff).encode()).hexdigest())[0] if xs else None
        nn,ff=choose(near),choose(far)
        fixed[f,c]=dict(status='RESOLVED' if nn and ff and nn!=ff else 'NEAR_FAR_UNRESOLVED',A=r['path'],
            X=n.rows[n.lookup['test',f,'target',x]]['path'],N=n.rows[n.lookup['test',nn,'target',c]]['path'] if nn else None,
            F=n.rows[n.lookup['test',ff,'target',c]]['path'] if ff else None,near_font=nn,far_font=ff,other_cp=x,neighbor_evidence=meta)
    episodes=[]
    for arm,mp,directory in [('K0',maps[0],ROOT/'reports/k0_original_test_20260917'),('K1',maps[1],ROOT/'runs/K1-ORIGINAL-V0917-S3407/eval_step_10000')]:
        for f,c,k in sorted(mp):
            row=mp[f,c,k];cand=dict(fixed[f,c]);cand['G']=str(directory/row['png'])
            refs=[n.rows[n.lookup['test',f,'style',cc]]['path'] for cc in REF8]
            episodes.append(dict(id=f'{arm}:{f}:{c}:k{k}',arm=arm,font=f,cp=c,k=k,script=row['script'],refs=refs,**cand))
    assert len(episodes)==5632
    payload=dict(version='E12C-FIVE-CANDIDATE-v2',episodes=episodes,base_pairs=704,
        independent_features_sha256=sha(STORE/'independent_features.pt'),manifest_sha256=sha(STORE/'manifest.json'),
        counts=dict(collections.Counter(e['status'] for e in episodes)),selection_uses_e12_scores=False)
    dest=STORE/'candidates.json'
    if dest.exists():assert json.loads(dest.read_text())==payload,'Candidate drift'
    else:atomic(dest,payload)
    print('CANDIDATES',json.dumps(payload['counts']),flush=True)

if __name__=='__main__':build()
