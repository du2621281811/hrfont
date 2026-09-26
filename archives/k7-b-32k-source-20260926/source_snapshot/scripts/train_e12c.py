"""E12-c light A2000/B1000, independent frozen evaluation and explicit manifests."""
import argparse
import collections
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from e12c_metrics import roc_auc_score
from e12c_model import Encoder, SetHead, multi_positive
from new_data_inventory import PREP, ROOT, sha, write_new


class Images(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['path']) as im:
            assert im.size==(96,96)
            x=torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()/255
        return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def save(path,obj):
    storage=Path('/root/data1/hrfont_i34_20260915')/path.parent.name
    storage.mkdir(parents=True,exist_ok=True)
    dest=storage/path.name
    tmp=dest.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(dest)
    if not path.exists():path.symlink_to(dest)
    assert path.is_symlink() and path.resolve()==dest.resolve()


@torch.no_grad()
def encode(encoder,images,ids,device):
    encoder.eval();result=[]
    loader=DataLoader(torch.utils.data.Subset(images,ids),batch_size=256,num_workers=4,pin_memory=True)
    for x in loader:
        with torch.autocast('cuda',dtype=torch.float16): y=encoder(x.to(device,non_blocking=True))
        result.append(y.cpu())
    return torch.cat(result)


def rank1(features,protocol):
    fonts=sorted(protocol['gallery'])
    gallery=torch.stack([torch.nn.functional.normalize(features[protocol['gallery'][f]].mean(0),dim=0) for f in fonts])
    by=collections.defaultdict(list)
    for q in protocol['queries']:
        best=fonts[int((features[q['index']]@gallery.T).argmax())]
        by[q['script'],q['font']].append(float(best==q['font']))
    groups=collections.defaultdict(list)
    for (g,f),v in by.items(): groups[g].append(float(np.mean(v)))
    return float(np.mean([np.mean(v) for v in groups.values()])),{g:float(np.mean(v)) for g,v in groups.items()}


def episode_tensors(episodes,features,device):
    q=[];refs=[];keep=[];labels=[]
    for ep in episodes:
        r=features[ep['refs']]; pad=torch.zeros(8,512);pad[:len(r)]=r
        mask=torch.arange(8)<len(r)
        for idx,label in ((ep['pos'],1.),(ep['neg'],0.)):
            q.append(features[idx]);refs.append(pad);keep.append(mask);labels.append(label)
    return (torch.stack(q).to(device),torch.stack(refs).to(device),torch.stack(keep).to(device),torch.tensor(labels,device=device))


@torch.no_grad()
def auc(head,features,protocol,device):
    head.eval();cells=collections.defaultdict(lambda:[[],[]]);episodes=protocol['episodes']
    for st in range(0,len(episodes),128):
        eps=episodes[st:st+128];q,r,k,y=episode_tensors(eps,features,device)
        scores,_=head(q,r,k)
        for n,e in enumerate(eps):
            cells[e['script'],e['k']][0].extend([1,0]);cells[e['script'],e['k']][1].extend(scores[n*2:n*2+2].cpu().tolist())
    metrics={f'{g}_k{k}':float(roc_auc_score(y,s)) for (g,k),(y,s) in cells.items()}
    assert len(metrics)==12
    return float(np.mean(list(metrics.values()))),metrics


def rng(): return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,default=PREP/'e12c_manifest.json')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if a.out.exists() and not a.resume: raise RuntimeError('Existing E12-c run requires explicit resume')
    if a.resume:assert (a.out/'config.json').exists() and not (a.out/'DONE.json').exists()
    a.out.mkdir(parents=True,exist_ok=a.resume)
    torch.set_num_threads(4);random.seed(3407);np.random.seed(3407);torch.manual_seed(3407)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    m=json.loads(a.manifest.read_text());digest=sha(a.manifest)
    assert not set(m['split']) & set(m['excluded_main_eval_fonts'])
    for f,sp in m['split'].items():
        for other,osp in m['split'].items():
            if f in m['known_families'] and other in m['known_families'] and m['known_families'][f]==m['known_families'][other]: assert sp==osp
    data=Images(m['rows']);pool=sorted(f for f in m['split'] if m['split'][f]=='train')
    assert len(pool)>=32
    by=m['byfont'];known=m['known_families'];famnames=sorted({known.get(f,'instance:'+f) for f in pool})
    family={f:famnames.index(known.get(f,'instance:'+f)) for f in pool}
    lookup={(r['font'],r['role'],r['cp']):i for i,r in enumerate(m['rows'])}
    encoder=Encoder(pretrained=True).to(device)
    pretrained=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    config=dict(manifest_sha256=digest,pretrained_sha256=sha(pretrained),preprocessing=m['preprocessing'],
                code_sha256={p:sha(Path(__file__).parent/p) for p in ('train_e12c.py','e12c_model.py','e12c_data.py','e12c_metrics.py')},
                A_steps=20 if a.smoke else 2000,B_steps=20 if a.smoke else 1000,smoke=a.smoke,
                main_eval_excluded=True,known_family_fonts=len(known),unknown_family_fonts=len(m['unknown_family_fonts']))
    if a.resume:assert config==json.loads((a.out/'config.json').read_text()), 'Resume config drift'
    else:atomic(a.out/'config.json',config)
    def safety():
        import shutil
        if (a.out/'STOP').exists() or min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/root/data1/hrfont_i34_20260915').free)<10*2**30: raise RuntimeError('STOP or low disk; no later phase')
    optimizer=torch.optim.AdamW(encoder.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024.)
    best=-1.;step=attempt=skips=0;start=time.time()
    if a.resume and (a.out/'A_last.pt').exists():
        state=torch.load(a.out/'A_last.pt',map_location='cpu',weights_only=False)
        assert state['manifest_sha256']==digest
        encoder.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        step,attempt,skips=state['step'],state['attempt'],state['skips'];best=state['best'];restore_rng(state['rng'])
    # Fixed validation gallery/queries, no access to internal test for model selection.
    val=m['validation']['val'];vids=sorted({i for v in val['gallery'].values() for i in v}|{q['index'] for q in val['queries']})
    while step<config['A_steps']:
        safety();rr=random.Random(3407+attempt);fonts=rr.sample(pool,32)
        cn=[];target=[];inst=[];families=[]
        for n,f in enumerate(fonts):
            choices=[g for g in ('latin','kana','bopomofo') if len(by[f][g])>=2]
            g=choices[(attempt+n)%len(choices)]
            cn.extend(rr.sample(by[f]['han'],2));target.extend(rr.sample(by[f][g],2));inst.extend([n,n]);families.extend([family[f],family[f]])
        x=torch.stack([data[i] for i in cn+target]).to(device)
        encoder.train();optimizer.zero_grad(set_to_none=True)
        lr=1e-5*min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-100)/max(1,config['A_steps']-100))))
        for group in optimizer.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.float16): z=encoder(x)
        loss=multi_positive(z[:64],z[64:],torch.tensor(inst,device=device),torch.tensor(families,device=device))
        if not torch.isfinite(loss):raise RuntimeError('A nonfinite loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        grad=torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.)
        attempt+=1
        if not torch.isfinite(grad):
            skips+=1;scaler.update(new_scale=scaler.get_scale()/2)
            if skips>10:raise RuntimeError('A excessive AMP skips')
            continue
        scaler.step(optimizer);scaler.update();step+=1
        if step%10==0:
            record=dict(stage='A',step=step,attempt=attempt,loss=float(loss),grad=float(grad),scale=scaler.get_scale(),skips=skips,seconds=time.time()-start)
            with (a.out/'train_log.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic(a.out/'heartbeat.json',record);print('E12C',json.dumps(record),flush=True)
        if step%(10 if a.smoke else 250)==0 or step==config['A_steps']:
            ef=encode(encoder,data,vids,device);feat=torch.zeros(len(data),512);feat[vids]=ef
            metric,per=rank1(feat,val)
            state=dict(model=encoder.state_dict(),step=step,metric=metric,per_script=per,manifest_sha256=digest)
            if metric>best:best=metric;save(a.out/'A_best.pt',state)
            save(a.out/'A_last.pt',dict(**state,attempt=attempt,skips=skips,best=best,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng()))
    encoder.load_state_dict(torch.load(a.out/'A_best.pt',map_location=device,weights_only=False)['model'])
    encoder.eval().requires_grad_(False)
    frozen={k:v.cpu().clone() for k,v in encoder.state_dict().items()}
    if a.resume and (a.out/'features.pt').exists():
        cached=torch.load(a.out/'features.pt',map_location='cpu',weights_only=True)
        assert cached['encoder_sha256']==sha(a.out/'A_best.pt') and cached['manifest_sha256']==digest
        features=cached['features']
    else:
        features=encode(encoder,data,list(range(len(data))),device)
        torch.save(dict(features=features,encoder_sha256=sha(a.out/'A_best.pt'),manifest_sha256=digest),a.out/'features.pt')
    head=SetHead().to(device);optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    best=-1.;bstep=0
    if a.resume and (a.out/'B_last.pt').exists():
        state=torch.load(a.out/'B_last.pt',map_location='cpu',weights_only=False)
        assert state['encoder_sha256']==sha(a.out/'A_best.pt') and state['manifest_sha256']==digest
        head.load_state_dict(state['head']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        bstep=state['step'];best=state['best']
    for step in range(bstep+1,config['B_steps']+1):
        safety();rr=random.Random(100000+3407+step);episodes=[]
        for _ in range(16):
            f=rr.choice(pool);g=rr.choice([g for g in ('latin','kana','bopomofo') if by[f][g]])
            pos=rr.choice(by[f][g]);cp=m['rows'][pos]['cp']
            legal=[other for other in pool if family[other]!=family[f] and (other,'target',cp) in lookup]
            assert legal
            other=rr.choice(legal);k=rr.choice((1,2,4,8));refs=rr.sample(by[f]['han'],k)
            episodes.append(dict(pos=pos,neg=lookup[other,'target',cp],refs=refs))
        q,r,k,y=episode_tensors(episodes,features,device);head.train();optimizer.zero_grad(set_to_none=True)
        factor=min(1.,step/50)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-50)/max(1,config['B_steps']-50))))
        for group in optimizer.param_groups:group['lr']=1e-4*factor
        scores,_=head(q,r,k);loss=torch.nn.functional.binary_cross_entropy_with_logits(scores,y)
        if not torch.isfinite(loss):raise RuntimeError('B nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError('B nonfinite gradients')
        optimizer.step()
        if step%10==0:
            record=dict(stage='B',step=step,loss=float(loss),grad=float(grad),seconds=time.time()-start)
            with (a.out/'train_log.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic(a.out/'heartbeat.json',record);print('E12C',json.dumps(record),flush=True)
        if step%(10 if a.smoke else 250)==0 or step==config['B_steps']:
            metric,per=auc(head,features,val,device)
            state=dict(head=head.state_dict(),step=step,metric=metric,per_cell=per,encoder_sha256=sha(a.out/'A_best.pt'),manifest_sha256=digest)
            if metric>best:best=metric;save(a.out/'B_best.pt',state)
            save(a.out/'B_last.pt',dict(**state,best=best,optimizer=optimizer.state_dict(),rng=rng()))
    assert all(torch.equal(v.cpu(),frozen[k]) for k,v in encoder.state_dict().items())
    head.load_state_dict(torch.load(a.out/'B_best.pt',map_location=device,weights_only=False)['head'])
    # Locked model: internal test is evaluated once, not used to switch primary score.
    r1,rs=rank1(features,m['validation']['test']);score,cells=auc(head,features,m['validation']['test'],device)
    atomic(a.out/'internal_test.json',dict(R1=r1,R1_scripts=rs,AUC=score,AUC_cells=cells,main_score='logit',diagnostic='cosine'))
    atomic(a.out/'DONE.json',dict(status='completed',smoke=a.smoke,A_steps=config['A_steps'],B_steps=config['B_steps'],
           encoder_sha256=sha(a.out/'A_best.pt'),head_sha256=sha(a.out/'B_best.pt'),manifest_sha256=digest,seconds=time.time()-start))


if __name__=='__main__':main()
