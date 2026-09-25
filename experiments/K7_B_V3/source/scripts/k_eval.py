"""Frozen clean K_VAL192 / test manifests and matched CFG1 DPM20 inference."""
import argparse
import collections
import datetime
import hashlib
import html
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from skimage.metrics import structural_similarity
from k_runtime import ROOT,PARENT0,T,DataContext,model_for,dataset,batch_to,atomic_json,sha256_file
from scripts.k_components import detail_distances
from src.dpm_solver.dpm_solver_pytorch import DPM_Solver,NoiseScheduleVP


def freeze_manifests(args):
    from scripts.v0913_clean_lib import load_cp_group
    groups=load_cp_group(args.v0913_clean_map)
    refs=[f'u{ord(c):04X}' for c in '永和书风骨韵天地']
    directory=ROOT/'experiments/K';directory.mkdir(parents=True,exist_ok=True)
    prototype=json.loads((ROOT/'reports/g_v0913_shot_k1248/PROTOCOL.json').read_text())
    panel={f'u{ord(c):04X}' for c in prototype['chars']}
    for split,name in [('val','K_VAL192'),('test','K_TEST')]:
        ds=dataset(args,split); rows=[]
        for i,p in enumerate(ds.target_images):
            path=Path(p);font=path.parent.name;cp=path.stem[len(font)+1:]
            group=groups[cp];group='western' if group=='latin' else group
            if split=='test' and cp not in panel:continue
            assert cp not in refs and all(c in ds.style_by_font_char[font] for c in refs)
            rows.append(dict(index=i,font=font,cp=cp,script=group,split=split))
        if split=='val':
            rng=random.Random(3407);selected=[]
            for group in ('western','kana','bopomofo'):
                pool=sorted([r for r in rows if r['script']==group],key=lambda r:(r['font'],r['cp']))
                selected.extend(rng.sample(pool,64))
            rows=selected
        jobs=[]
        for r in sorted(rows,key=lambda r:(r['script'],r['font'],r['cp'])):
            for shot in ([4] if split=='val' else [1,2,4,8]):
                seed=3407+int(hashlib.sha256((r['font']+r['cp']).encode()).hexdigest()[:8],16)%1000000
                jobs.append(dict(**r,k=shot,refs=refs[:shot],seed=seed))
        assert len(jobs)==(192 if split=='val' else 2816),len(jobs)
        payload=dict(protocol='K-original-CFG1-DPM20-v1',split=split,cfg=1.,steps=20,order=2,
                     seed=3407,clean_sha256=T._clean_map_sha256(args.v0913_clean_map),jobs=jobs)
        path=directory/(name+'.json')
        if path.exists():assert json.loads(path.read_text())==payload,'Frozen manifest differs'
        else:atomic_json(path,payload)


@torch.no_grad()
def sample(model,data,samples,scheduler,seed):
    device=data.device;active=torch.zeros(1,device=device,dtype=torch.bool)
    style,refs,query,keep,content,structure=data.conditions(samples,active,active,
        no_delta=model.arm=='K0',official=model.arm=='K2-Delta-',need_refs=model.use_tc)
    ctx=model.conditions(style,refs,query,keep,active)
    schedule=NoiseScheduleVP(schedule='discrete',betas=scheduler.betas)
    def model_fn(x,continuous_t):
        t=(continuous_t-1./schedule.total_N)*1000.
        with torch.autocast('cuda',dtype=torch.float16):
            pred,_=model.denoise(x,t,style,content,structure,ctx,1.)
        return pred.float()
    solver=DPM_Solver(model_fn=model_fn,noise_schedule=schedule,algorithm_type='dpmsolver++')
    noise=torch.randn((1,3,96,96),device=device,generator=torch.Generator(device=device).manual_seed(seed))
    pred=solver.sample(x=noise,steps=20,order=2,skip_type='time_uniform',method='multistep')
    assert torch.isfinite(pred).all()
    return ((pred[0].float()/2+.5).clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')


@torch.no_grad()
def evaluate(model,data,args,scheduler,manifest_path,out,step,smoke=False):
    rank,world=dist.get_rank(),dist.get_world_size()
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(manifest_path).read_text());jobs=manifest['jobs']
    if smoke:
        jobs=[next(j for j in jobs if j['script']==g) for g in ('western','kana','bopomofo')]
    ds=dataset(args,manifest['split']);model.eval();begin=time.time();rows=[]
    if rank==0:
        atomic_json(out/'protocol.json',dict(**{k:v for k,v in manifest.items() if k!='jobs'},
            jobs=jobs,arm=model.arm,step=step,weights='K0 parent' if model.arm=='K0' else 'EMA',
            manifest_sha256=sha256_file(manifest_path),smoke=smoke))
    for j in jobs[rank::world]:
        s=ds[j['index']]
        assert (s['font_stem'],s['char_cp'])==(j['font'],j['cp'])
        s['ref_chars']=j['refs'];s['ref_image_paths']=[str(ds.style_by_font_char[j['font']][c]) for c in j['refs']]
        b=batch_to([s],data.device)
        pred=sample(model,data,b,scheduler,j['seed'])
        name=f"{j['font']}__{j['cp']}__k{j['k']}.png"
        Image.fromarray(pred).save(out/name)
        gt=np.asarray(Image.open(s['target_image_path']).convert('L'),dtype='float32')/255
        gray=np.asarray(Image.fromarray(pred).convert('L'),dtype='float32')/255
        tensor=torch.from_numpy(pred.copy()).permute(2,0,1).float()[None].to(data.device)/255
        d=detail_distances(tensor,b['content_image']/2+.5,b['nonorm_target_image'])
        change=detail_distances(b['content_image']/2+.5,b['content_image']/2+.5,b['nonorm_target_image'])
        edge=lambda x:np.diff(x,axis=0,prepend=x[:1])**2+np.diff(x,axis=1,prepend=x[:,:1])**2
        rows.append(dict(**j,png=name,target=s['target_image_path'],refs_paths=s['ref_image_paths'],
            content=str(ds.root/ds.phase/'ContentImage'/(j['cp']+'.png')),
            l1=float(np.abs(gray-gt).mean()),ssim=float(structural_similarity(gray,gt,data_range=1.,win_size=7)),
            edge=float(np.abs(edge(gray)-edge(gt)).mean()),ink_fraction=float((gray<.95).mean()),
            change_magnitude=float(change['D_change'][0]),**{k:float(v[0]) for k,v in d.items()}))
        if len(rows)%20==0:print('K_INFER',model.arm,rank,len(rows),len(jobs[rank::world]),flush=True)
    atomic_json(out/f'rank{rank}.json',dict(rows=rows,seconds=time.time()-begin))
    dist.barrier()
    if rank==0:
        allrows=sum([json.loads((out/f'rank{i}.json').read_text())['rows'] for i in range(world)],[])
        assert len(allrows)==len(jobs) and len({r['png'] for r in allrows})==len(jobs)
        keys=['l1','ssim','edge','D_region','D_change','D_add','D_remove','D_high']
        summary={}
        for group in ['all','western','kana','bopomofo']:
            selected=[r for r in allrows if group=='all' or r['script']==group]
            summary[group]={k:float(np.mean([r[k] for r in selected])) for k in keys}
        stratified={}
        for field in ['font','k']:
            stratified[field]={str(value):{k:float(np.mean([r[k] for r in allrows if r[field]==value]))
                for k in keys} for value in sorted({r[field] for r in allrows})}
        ordered=sorted(allrows,key=lambda r:r['change_magnitude'])
        stratified['change_quartile']={str(i+1):{k:float(np.mean([r[k] for r in chunk])) for k in keys}
            for i,chunk in enumerate(np.array_split(ordered,4)) if len(chunk)}
        atomic_json(out/'metrics.json',dict(summary=summary,stratified=stratified,rows=allrows))
        atomic_json(out/'DONE.json',dict(status='completed',images=len(allrows),step=step,seconds=time.time()-begin))
    dist.barrier()


def make_board(baseline,candidate,out):
    import shutil
    base=json.loads((baseline/'metrics.json').read_text())
    cand=json.loads((candidate/'metrics.json').read_text())
    key=lambda r:(r['font'],r['cp'],r['k'])
    lookup={key(r):r for r in base['rows']};out.mkdir(parents=True,exist_ok=True)
    items=[]
    def copy(path):
        path=Path(path);name=hashlib.sha256(str(path).encode()).hexdigest()[:16]+'.png'
        if not (out/name).exists():shutil.copy2(path,out/name)
        return name
    for r in cand['rows']:
        b=lookup[key(r)];assert (b['seed'],b['refs'])==(r['seed'],r['refs'])
        items.append(dict(**r,delta=b['D_change']-r['D_change'],baseline=copy(baseline/b['png']),
            prediction=copy(candidate/r['png']),gt=copy(r['target']),neutral=copy(r['content']),
            reference=[copy(p) for p in r['refs_paths']]))
    atomic_json(out/'metrics.json',dict(baseline=base['summary'],candidate=cand['summary'],rows=items))
    payload=json.dumps(items,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><meta charset="utf-8"><title>K matched review</title>
<style>body{font:15px system-ui;margin:24px;background:#f4f5f6;color:#17202a}img{width:96px;height:96px;background:white}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.refs img{width:48px;height:48px}select{padding:8px;margin:5px}</style>
<h1>K0 / K1 matched review</h1><p>Clean pairs · EMA candidate · CFG 1 · DPM++ 20 · same noise and nested references. All rows retained.</p>
<div id="filters"></div><table><thead><tr><th>Sample</th><th>Reference</th><th>Content</th><th>K0</th><th>Candidate</th><th>GT</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data=PAYLOAD;const fields=['script','font','cp','k'];const state={};
for(const field of fields){let s=document.createElement('select');s.innerHTML='<option value="">All '+field+'</option>';for(const x of [...new Set(data.map(r=>r[field]))].sort()){let o=document.createElement('option');o.value=x;o.textContent=x;s.appendChild(o)}s.onchange=()=>{state[field]=s.value;render()};document.getElementById('filters').appendChild(s)}
let sort=document.createElement('select');sort.innerHTML='<option value="">Default order</option><option value="delta">K0 to candidate improvement</option><option value="change_magnitude">Content to GT change</option>';sort.onchange=render;document.getElementById('filters').appendChild(sort);
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){let selected=data.filter(r=>fields.every(f=>!state[f]||String(r[f])===state[f]));if(sort.value)selected.sort((a,b)=>b[sort.value]-a[sort.value]);document.getElementById('rows').innerHTML=selected.map(r=>'<tr><td>'+esc(r.font)+'<br>'+esc(r.cp)+' / '+r.k+' shot<br>ΔDchange '+r.delta.toFixed(4)+'</td><td class="refs">'+r.reference.map(p=>'<img loading="lazy" src="'+p+'">').join('')+'</td>'+['neutral','baseline','prediction','gt'].map(k=>'<td><img loading="lazy" src="'+r[k]+'"></td>').join('')+'</tr>').join('')}render();</script>'''
    (out/'index.html').write_text(page.replace('PAYLOAD',payload))
    atomic_json(out/'DONE.json',dict(status='completed',rows=len(items)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',default='K0');ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--manifest',type=Path,default=ROOT/'experiments/K/K_VAL192.json')
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--prepare',action='store_true');ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    model,args=model_for(a.arm,torch.device('cuda',rank),a.out/f'provenance_rank{rank}')
    if a.prepare and rank==0:freeze_manifests(args)
    dist.barrier()
    if a.checkpoint:model.load_train_state(torch.load(a.checkpoint/'ema.pth',map_location=f'cuda:{rank}',weights_only=True))
    data=DataContext(args,torch.device('cuda',rank))
    evaluate(model,data,args,T.build_ddpm_scheduler(args),a.manifest,a.out,10000,smoke=a.smoke)
    dist.destroy_process_group()


if __name__=='__main__':main()
