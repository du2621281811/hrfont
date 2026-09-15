"""Offline character-calibrated cross-script difficulty; writes REVIEW_PENDING."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from new_data_inventory import PREP, sha, write_new


class Images(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['path']) as im:
            assert im.size == (96,96)
            x=np.array(im.convert('RGB'),copy=True)
        return torch.from_numpy(x).permute(2,0,1).float()/127.5-1


def percentile(values):
    a=np.asarray(values)
    return ((a[:,None]>a[None,:]).sum(1)+.5*(a[:,None]==a[None,:]).sum(1))/len(a)


def prepare(device):
    torch.set_num_threads(4)
    from h_runtime import args_for, PARENT0, T
    from scripts.hrfont_feature_cache import EsCache
    invpath=PREP/'inventory.json'; inv=json.loads(invpath.read_text())
    targets=[r for r in inv['records'] if r['role']=='target']
    fpath=PREP/'target_es.pt'
    args=args_for(PARENT0)
    if fpath.exists():
        cached=torch.load(fpath,map_location='cpu',weights_only=True)
        assert cached['inventory_sha256']==sha(invpath)
        assert cached['encoder_sha256']==sha(PARENT0/'style_encoder.pth')
        features=cached['features']
    else:
        encoder=T.build_style_encoder(args).to(device).eval().requires_grad_(False)
        encoder.load_state_dict(torch.load(PARENT0/'style_encoder.pth',map_location=device,weights_only=True))
        batches=[]
        loader=DataLoader(Images(targets),batch_size=256,num_workers=4,pin_memory=True)
        with torch.no_grad():
            for n,x in enumerate(loader):
                with torch.autocast(device_type='cuda',dtype=torch.float16):
                    spatial,y,_=encoder(x.to(device,non_blocking=True))
                assert y.ndim==2 and y.shape[1]==1024
                batches.append(F.normalize(y.float(),dim=1).cpu())
                if n%20==0: print('ES_TARGET_BATCH',n,len(loader),flush=True)
        features=torch.cat(batches)
        torch.save(dict(features=features,inventory_sha256=sha(invpath),encoder_sha256=sha(PARENT0/'style_encoder.pth')),fpath)
        del encoder
    fonts=inv['fonts']; fi={f:i for i,f in enumerate(fonts)}
    es=EsCache(Path(args.es_cache_path))
    styles=collections.defaultdict(list)
    for r in inv['records']:
        if r['role']=='style': styles[r['font']].append(r['cp'])
    common=sorted(set.intersection(*(set(v) for v in styles.values())))
    assert len(common)>=32
    rng=np.random.default_rng(3407)
    panel=list(rng.choice(common,min(64,len(common)),replace=False))
    def proto(chars):
        return F.normalize(torch.stack([F.normalize(torch.stack([es.pooled_tensor('train',f,c) for c in chars]),dim=1).mean(0) for f in fonts]),dim=1)
    proto_all=proto(panel); proto_a=proto(panel[::2]); proto_b=proto(panel[1::2])
    bycp=collections.defaultdict(list); byfg=collections.defaultdict(list)
    for i,r in enumerate(targets):
        bycp[r['cp']].append(i); byfg[(r['font'],r['script'])].append(i)
    group_sums={k:features[v].sum(0) for k,v in byfg.items()}
    scores=np.zeros((len(targets),3)); observable=np.zeros(len(targets))
    raw_rank=np.zeros(len(targets)); raw_margin=np.zeros(len(targets))
    for cp,ids in bycp.items():
        if len(ids)<16: continue
        ff=[targets[i]['font'] for i in ids]; own=torch.tensor([fi[f] for f in ff])
        x=features[ids]
        for j,p in enumerate((proto_all,proto_a,proto_b)):
            sim=x@p.T; ownsim=sim[torch.arange(len(ids)),own]
            # Rank error: a higher fraction of other fonts beating own font is harder.
            raw=(sim>ownsim[:,None]).float().sum(1)/(len(fonts)-1)
            if j==0:
                raw_rank[ids]=raw.numpy()
                rivals=sim.clone();rivals[torch.arange(len(ids)),own]=-torch.inf
                raw_margin[ids]=(ownsim-rivals.max(1).values).numpy()
            scores[ids,j]=percentile(raw.numpy())
        protos=[]
        for i in ids:
            r=targets[i]; key=(r['font'],r['script'])
            protos.append(F.normalize(group_sums[key]-features[i],dim=0))
        sim=x@torch.stack(protos).T
        ownsim=sim.diagonal()
        observable[ids]=1-(sim>ownsim[:,None]).float().sum(1).numpy()/(len(ids)-1)
    result={}; cp_group={r['cp']:r['script'] for r in targets}
    for (font,group),ids in byfg.items():
        usable=[i for i in ids if len(bycp[targets[i]['cp']])>=16 and observable[i]>=.6]
        stable=float(np.median(np.abs(scores[usable,1]-scores[usable,2]))) if usable else 1.
        target_gap=abs(float(np.median(scores[usable[::2],0]))-float(np.median(scores[usable[1::2],0]))) if len(usable)>=2 else 1.
        confident=len(usable)>=8 and stable<=.25 and target_gap<=.25
        score=float(np.median(scores[usable,0])) if usable else .5
        result[font+'|'+group]=dict(font=font,group=group,difficulty=score,confident=confident,
            comparable_chars=len(usable),stability_gap=stable,target_half_gap=target_gap,
            observable_mean=float(observable[ids].mean()),raw_margin_median=float(np.median(raw_margin[ids])),weight=1.,bucket='uncertain')
    for group in ('latin','kana','bopomofo'):
        pool=[r for r in result.values() if r['group']==group and r['confident']]
        assert len(pool)>=3,(group,len(pool))
        low,high=np.quantile([r['difficulty'] for r in pool],[1/3,2/3])
        for r in pool:
            r['bucket']='easy' if r['difficulty']<low else 'hard' if r['difficulty']>high else 'medium'
            r['weight']={'easy':1.,'medium':1.5,'hard':2.}[r['bucket']]
    review=[]
    for group in ('latin','kana','bopomofo'):
        rows=sorted([r for r in result.values() if r['group']==group],key=lambda r:r['difficulty'])
        for ix in np.linspace(0,len(rows)-1,12,dtype=int): review.append(rows[ix])
    for page in range(3):
        sheet=Image.new('RGB',(850,12*110),'white'); draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(review[page*12:(page+1)*12]):
            label=f"{row['font']} {row['group']} {row['bucket']} {row['difficulty']:.3f}"
            draw.text((2,j*110),label,fill='black')
            imgs=[next(r for r in inv['records'] if r['font']==row['font'] and r['role']=='style' and r['cp']==cp) for cp in panel[:3]]
            ids=byfg[(row['font'],row['group'])]
            imgs += [targets[ids[int(i)]] for i in np.linspace(0,len(ids)-1,5)]
            for col,r in enumerate(imgs):
                with Image.open(r['path']) as im: sheet.paste(im,(col*100,j*110+14))
        sheet.save(PREP/f'difficulty_review_{page}.png')
    details=[dict(font=r['font'],cp=r['cp'],group=r['script'],raw_rank=float(raw_rank[i]),
                  raw_margin=float(raw_margin[i]),calibrated=float(scores[i,0]),observable=float(observable[i]),
                  comparable_fonts=len(bycp[r['cp']])) for i,r in enumerate(targets)]
    write_new(PREP/'difficulty_char_scores.json',details)
    manifest=dict(status='REVIEW_PENDING',version='same-character-calibrated-v1',font_groups=result,
        character_scores_sha256=sha(PREP/'difficulty_char_scores.json'),
        cp_group=cp_group,sources_sha256=inv['sources_sha256'],inventory_sha256=sha(invpath),
        encoder_sha256=sha(PARENT0/'style_encoder.pth'),features_sha256=sha(fpath),chinese_panel=panel,
        thresholds=dict(min_comparable_fonts=16,min_reliable_chars=8,within_script_rank=.6,max_half_gap=.25),
        review_rows=review,review='Awaiting visual panel inspection; no automatic approval')
    write_new(PREP/'difficulty_manifest.json',manifest)
    print(json.dumps(collections.Counter(r['bucket'] for r in result.values())),flush=True)


if __name__=='__main__':
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    prepare(torch.device(a.device))
