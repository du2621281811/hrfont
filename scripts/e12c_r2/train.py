"""Approved R1/R2/R3; internal validation selection, recoverable state."""
import argparse,collections,json,math,random,time,shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from data import STORE,ROOT,Images,atomic,sha,SCRIPTS
from model import Encoder,SetHead,multi_positive,cosine

def save(p,state):
    tmp=p.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(p)
def rng():return dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
def restore(s):
    random.setstate(s['py']);np.random.set_state(s['np']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['cuda'])
@torch.no_grad()
def encode(e,data,ids,device):
    e.eval();out=[]
    for x in DataLoader(Subset(data,ids),batch_size=256,num_workers=4,pin_memory=True):
        with torch.autocast('cuda',dtype=torch.float16):z=e(x.to(device,non_blocking=True))
        out.append(z.float().cpu())
    return torch.cat(out)
def tensors(episodes,features,device):
    q=[];r=[];k=[];y=[];dim=features.shape[1]
    for ep in episodes:
        f=features[ep['refs']];pad=torch.zeros(8,dim);pad[:len(f)]=f;keep=torch.arange(8)<len(f)
        for idx,label in [(ep['pos'],1.),(ep['neg'],0.)]:q.append(features[idx]);r.append(pad);k.append(keep);y.append(label)
    return torch.stack(q).to(device),torch.stack(r).to(device),torch.stack(k).to(device),torch.tensor(y,device=device)
def auc_value(y,s):
    y=np.asarray(y);s=np.asarray(s);a=s[y==1];b=s[y==0]
    assert len(a) and len(b)
    return float(((a[:,None]>b[None,:]).mean()+.5*(a[:,None]==b[None,:]).mean()))
@torch.no_grad()
def auc(head,features,protocol,device,mode='head'):
    if head is not None:head.eval()
    cells=collections.defaultdict(lambda:[[],[]]);episodes=protocol['episodes']
    for st in range(0,len(episodes),128):
        eps=episodes[st:st+128];q,r,k,y=tensors(eps,features,device)
        scores=cosine(q,r,k) if mode=='cosine' else head(q,r,k)[0]
        for i,e in enumerate(eps):
            cells[e['script'],e['k']][0]+=[1,0];cells[e['script'],e['k']][1]+=scores[2*i:2*i+2].cpu().tolist()
    values={f'{g}_k{k}':auc_value(*v) for (g,k),v in cells.items()};assert len(values)==12
    return float(np.mean(list(values.values()))),values
@torch.no_grad()
def rank1(features,protocol,device,groups):
    fonts=sorted(protocol['gallery']);rr=features[torch.tensor([protocol['gallery'][f] for f in fonts])].to(device)
    keep=torch.ones(len(fonts),8,dtype=torch.bool,device=device);by=collections.defaultdict(list)
    for q in protocol['queries']:
        z=features[q['index']].to(device)[None].expand(len(fonts),-1)
        best=fonts[int(cosine(z,rr,keep).argmax())];by[q['script'],q['font']].append(float(groups[best]==groups[q['font']]))
    per=collections.defaultdict(list)
    for (g,f),v in by.items():per[g].append(float(np.mean(v)))
    return float(np.mean([np.mean(v) for v in per.values()]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['R1','R2','R3'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-A',action='store_true');p.add_argument('--run-prefix');a=p.parse_args()
    torch.set_num_threads(4);random.seed(3407);np.random.seed(3407);torch.manual_seed(3407);torch.cuda.set_device(0)
    prefix=a.run_prefix if a.run_prefix is not None else ('SMOKE_' if a.smoke else '')
    if a.run_prefix is not None:assert a.smoke and prefix.startswith('SMOKE_')
    device=torch.device('cuda:0');out=STORE/(prefix+a.arm)
    if (out/'DONE.json').exists():print('ALREADY_DONE',out,flush=True);return
    if out.exists() and not a.resume:raise RuntimeError('Existing run requires --resume')
    out.mkdir(exist_ok=True)
    manifest=STORE/'manifest.json';m=json.loads(manifest.read_text());digest=sha(manifest);data=Images(m['rows']);by=m['byfont']
    units=collections.defaultdict(list)
    for f,sp in m['split'].items():
        if sp=='train':units[m['groups'][f]].append(f)
    pool=sorted(f for fs in units.values() for f in fs);ukeys=sorted(units);assert len(ukeys)>=32
    group={f:ukeys.index(m['groups'][f]) for f in pool};lookup={(r['font'],r['role'],r['cp']):i for i,r in enumerate(m['rows'])}
    multi=a.arm!='R1';dim=768 if multi else 512;A_steps=2 if a.smoke else 2000;B_steps=2 if a.smoke else 1000;interval=1 if a.smoke else 250
    config=dict(arm=a.arm,multiscale=multi,dim=dim,A_steps=0 if a.arm=='R3' else A_steps,B_steps=B_steps,smoke=a.smoke,
        manifest_sha256=digest,code_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},seed=3407,
        pretrained_sha256=sha(Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'))
    if a.resume:assert json.loads((out/'config.json').read_text())==config,'Resume identity drift'
    else:atomic(out/'config.json',config)
    start=time.time()
    def safety():
        if (out/'STOP').exists() or (STORE/'STOP').exists():raise RuntimeError('STOP requested')
        if shutil.disk_usage(STORE).free<10*2**30 or shutil.disk_usage(ROOT).free<2*2**30:raise RuntimeError('Low disk')
        if time.time()-start>90*60:raise RuntimeError('90 minute phase budget exceeded; last state retained')
    e=Encoder(multi,True).to(device)
    if a.arm=='R3':
        src=STORE/(prefix+'R2')
        assert (src/'DONE.json').exists()
        if not (out/'A_best.pt').exists():(out/'A_best.pt').symlink_to(src/'A_best.pt')
        if not (out/'features.pt').exists():(out/'features.pt').symlink_to(src/'features.pt')
    else:
        opt=torch.optim.AdamW(e.parameters(),lr=1e-5,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
        step=attempt=skips=0;best=-1.
        if a.resume and (out/'A_last.pt').exists():
            s=torch.load(out/'A_last.pt',map_location='cpu',weights_only=False);assert s['manifest_sha256']==digest
            e.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);scaler.load_state_dict(s['scaler']);restore(s['rng'])
            step,attempt,skips,best=s['step'],s['attempt'],s['skips'],s['best']
        val=m['validation']['val'];vids=sorted({i for rr in val['gallery'].values() for i in rr}|{q['index'] for q in val['queries']})
        while step<A_steps:
            safety();rr=random.Random(3407+attempt);fonts=[rr.choice(units[u]) for u in rr.sample(ukeys,32)]
            cn=[];target=[];inst=[];families=[]
            for n,f in enumerate(fonts):
                choices=[g for g in SCRIPTS if len(by[f][g])>=2];g=choices[(attempt+n)%len(choices)]
                cn+=rr.sample(by[f]['han'],2);target+=rr.sample(by[f][g],2);inst+=[n,n];families+=[group[f],group[f]]
            x=torch.stack([data[i] for i in cn+target]).to(device);e.train();opt.zero_grad(set_to_none=True)
            factor=min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-100)/max(1,A_steps-100))))
            for gr in opt.param_groups:gr['lr']=1e-5*factor
            with torch.autocast('cuda',dtype=torch.float16):z=e(x);loss=multi_positive(z[:64],z[64:],torch.tensor(inst,device=device),torch.tensor(families,device=device))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite A loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);grad=torch.nn.utils.clip_grad_norm_(e.parameters(),1.);attempt+=1
            if not torch.isfinite(grad):
                skips+=1;scaler.update(new_scale=scaler.get_scale()/2)
                if skips>10:raise RuntimeError('Too many AMP skips')
                continue
            scaler.step(opt);scaler.update();step+=1
            record=dict(stage='A',step=step,loss=float(loss),grad=float(grad),skips=skips,seconds=time.time()-start)
            if step%10==0 or a.smoke:atomic(out/'heartbeat.json',record);print(a.arm,json.dumps(record),flush=True)
            if step%interval==0 or step==A_steps:
                vf=torch.zeros(len(data),dim);vf[vids]=encode(e,data,vids,device);metric=rank1(vf,val,device,m['groups'])
                s=dict(model=e.state_dict(),step=step,metric=metric,manifest_sha256=digest)
                if metric>best:best=metric;save(out/'A_best.pt',s)
                save(out/'A_last.pt',dict(**s,attempt=attempt,skips=skips,best=best,optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=rng()))
    if a.stop_after_A:
        assert a.smoke and a.arm!='R3'
        atomic(out/'PAUSED_AFTER_A.json',dict(status='paused_for_resume_test'));return
    e.load_state_dict(torch.load(out/'A_best.pt',map_location=device,weights_only=False)['model']);e.eval().requires_grad_(False)
    encsha=sha(out/'A_best.pt')
    if (out/'features.pt').exists():
        cache=torch.load(out/'features.pt',map_location='cpu',weights_only=True);assert cache['encoder_sha256']==encsha and cache['manifest_sha256']==digest
        features=cache['features']
    else:
        features=encode(e,data,list(range(len(data))),device);save(out/'features.pt',dict(features=features,encoder_sha256=encsha,manifest_sha256=digest))
    head=SetHead(dim,a.arm=='R3').to(device);opt=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01);bstep=0;best=-1.
    if a.resume and (out/'B_last.pt').exists():
        s=torch.load(out/'B_last.pt',map_location='cpu',weights_only=False);assert s['encoder_sha256']==encsha and s['manifest_sha256']==digest
        head.load_state_dict(s['head']);opt.load_state_dict(s['optimizer']);restore(s['rng']);bstep,best=s['step'],s['best']
    if a.arm=='R3':
        near=json.loads((STORE/'train_neighbors.json').read_text())['neighbors']
        if bstep==0:
            best,_=auc(head,features,m['validation']['val'],device)
            save(out/'B_best.pt',dict(head=head.state_dict(),step=0,metric=best,encoder_sha256=encsha,manifest_sha256=digest))
    fallback=0
    for step in range(bstep+1,B_steps+1):
        safety();rr=random.Random(103407+step);episodes=[]
        for n in range(16):
            f=rr.choice(pool);g=rr.choice([g for g in SCRIPTS if by[f][g]]);pos=rr.choice(by[f][g]);cp=m['rows'][pos]['cp']
            legal=[o for o in pool if group[o]!=group[f] and (o,'target',cp) in lookup and m['rows'][lookup[o,'target',cp]]['rgb_sha256']!=m['rows'][pos]['rgb_sha256']]
            assert legal
            choices=legal
            if a.arm=='R3' and n<8:
                choices=[o for o in near.get(f,{}).get(g,[]) if o in legal]
                if not choices:choices=legal;fallback+=1
            other=rr.choice(choices);k=rr.choice((1,2,4,8));refs=rr.sample(by[f]['han'],k)
            episodes.append(dict(pos=pos,neg=lookup[other,'target',cp],refs=refs))
        q,r,k,y=tensors(episodes,features,device);head.train();opt.zero_grad(set_to_none=True)
        factor=min(1.,step/50)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-50)/max(1,B_steps-50))))
        for gr in opt.param_groups:gr['lr']=1e-4*factor
        scores,base=head(q,r,k)
        loss=(F.softplus((scores[1::2]-scores[::2])/.1).mean()+1e-3*(scores-base).square().mean()) if a.arm=='R3' else F.binary_cross_entropy_with_logits(scores,y)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite B loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(grad);opt.step()
        record=dict(stage='B',step=step,loss=float(loss),grad=float(grad),near_fallback=fallback,seconds=time.time()-start)
        if step%10==0 or a.smoke:atomic(out/'heartbeat.json',record);print(a.arm,json.dumps(record),flush=True)
        if step%interval==0 or step==B_steps:
            metric,per=auc(head,features,m['validation']['val'],device)
            s=dict(head=head.state_dict(),step=step,metric=metric,per_cell=per,encoder_sha256=encsha,manifest_sha256=digest)
            if metric>best:best=metric;save(out/'B_best.pt',s)
            save(out/'B_last.pt',dict(**s,best=best,optimizer=opt.state_dict(),rng=rng()))
    head.load_state_dict(torch.load(out/'B_best.pt',map_location=device,weights_only=False)['head'])
    ca,cc=auc(head,features,m['validation']['val'],device,'cosine');ha,hc=auc(head,features,m['validation']['val'],device)
    atomic(out/'validation.json',dict(cosine_AUC=ca,head_AUC=ha,cosine_cells=cc,head_cells=hc))
    atomic(out/'DONE.json',dict(status='completed',arm=a.arm,smoke=a.smoke,A_steps=config['A_steps'],B_steps=B_steps,encoder_sha256=encsha,
        head_sha256=sha(out/'B_best.pt'),manifest_sha256=digest,seconds=time.time()-start,near_fallback=fallback))

if __name__=='__main__':main()
