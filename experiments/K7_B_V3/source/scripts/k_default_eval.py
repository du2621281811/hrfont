"""Default clean Train/Val/Test x 128 requested characters x shots 1/2/4/8.

Uses the frozen original K backend; never modifies old manifests or results.
"""
import argparse, collections, hashlib, json, os, shutil, sys, time
from pathlib import Path

BACKEND=Path(os.environ.get('HRFONT_K_BACKEND','/root/projects/hrfont_k_20260917'))
sys.path[:0]=[str(BACKEND/'scripts'),str(BACKEND)]
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from k_runtime import ROOT,PARENT0,T,DataContext,model_for,dataset,batch_to,atomic_json,sha256_file
from h_runtime import args_for
from k_eval import sample as serial_sample
from scripts.k_components import detail_distances
from scripts.v0913_clean_lib import load_cp_group
from src.dpm_solver.dpm_solver_pytorch import DPM_Solver,NoiseScheduleVP
from skimage.metrics import structural_similarity

OUT=ROOT/'reports/k_default_k1248_20260917'
REFS=[f'u{ord(c):04X}' for c in '永和书风骨韵天地']

def prepare():
    storage=Path('/root/data1/hrfont_default_inference_20260917')
    storage.mkdir(parents=True,exist_ok=True)
    if not OUT.exists(): OUT.symlink_to(storage,target_is_directory=True)
    assert OUT.resolve()==storage.resolve()
    args=args_for(PARENT0); groups=load_cp_group(args.v0913_clean_map)
    old=json.loads((ROOT/'reports/g_v0913_shot_k1248/PROTOCOL.json').read_text())
    oldc=[f'u{ord(c):04X}' for c in old['chars']]
    selected=[]
    for group,n in [('latin',80),('kana',32),('bopomofo',16)]:
        pool=sorted(c for c,g in groups.items() if g==group)
        base=[c for c in oldc if c in pool]
        extra=sorted(set(pool)-set(base),key=lambda c:hashlib.sha256(('default128:'+c).encode()).hexdigest())
        chosen=base+extra[:n-len(base)]
        assert len(chosen)==n,(group,len(pool),len(chosen))
        selected+=chosen
    manifests={}; missing=[]
    for split,key in [('train','train5'),('val','val5'),('test','test')]:
        fonts=old['fonts'][key]; ds=dataset(args,split)
        lookup={}
        for i,p in enumerate(ds.target_images):
            p=Path(p);f=p.parent.name;c=p.stem[len(f)+1:]
            lookup[f,c]=i
        jobs=[]
        for f in fonts:
            for c in selected:
                if (f,c) not in lookup:
                    missing.append(dict(split=split,font=f,cp=c,reason='not_a_clean_legal_pair'));continue
                assert all(r in ds.style_by_font_char[f] for r in REFS)
                seed=3407+int(hashlib.sha256((f+c).encode()).hexdigest()[:8],16)%1000000
                for k in (1,2,4,8):
                    jobs.append(dict(split=split,font=f,cp=c,index=lookup[f,c],k=k,refs=REFS[:k],seed=seed,
                        script='western' if groups[c]=='latin' else groups[c]))
        assert jobs and len({(j['font'],j['cp'],j['k']) for j in jobs})==len(jobs)
        manifests[split]=dict(protocol='HRFONT-DEFAULT-128-CLEAN-K1248-v1',split=split,jobs=jobs,
            cfg=1.,steps=20,order=2,seed=3407,fonts=fonts,requested_cps=selected,
            clean_sha256=T._clean_map_sha256(args.v0913_clean_map),backend=str(BACKEND),batch_size=4)
        p=OUT/(split+'_manifest.json')
        if p.exists(): assert json.loads(p.read_text())==manifests[split],'Immutable manifest drift'
        else: atomic_json(p,manifests[split])
    protocol=dict(version='HRFONT-DEFAULT-128-CLEAN-K1248-v1',shots=[1,2,4,8],ref8=REFS,
        sampling=dict(sampler='dpmsolver++',cfg=1.,steps=20,order=2,batch_size=4),
        fonts={sp:m['fonts'] for sp,m in manifests.items()},requested_characters=128,
        character_counts=dict(western=80,kana=32,bopomofo=16),cps=selected,
        jobs_per_model={sp:len(m['jobs']) for sp,m in manifests.items()},missing_pairs=missing,
        note='Clean pairs only; no cross-split fallback. Train is diagnostic, not generalization evidence.',
        manifest_sha256={sp:sha256_file(OUT/(sp+'_manifest.json')) for sp in manifests},
        source_sha256=sha256_file(Path(__file__)))
    atomic_json(OUT/'PROTOCOL.json',protocol)
    print(json.dumps({k:protocol[k] for k in ['version','requested_characters','jobs_per_model']},ensure_ascii=False),flush=True)

@torch.no_grad()
def sample_batch(model,data,samples,scheduler,seeds):
    b=len(seeds);device=data.device;active=torch.zeros(b,device=device,dtype=torch.bool)
    style,refs,query,keep,content,structure=data.conditions(samples,active,active,no_delta=model.arm=='K0',
        official=False,need_refs=model.use_tc)
    ctx=model.conditions(style,refs,query,keep,active)
    schedule=NoiseScheduleVP(schedule='discrete',betas=scheduler.betas)
    def fn(x,t):
        t=((t-1./schedule.total_N)*1000.).expand(b)
        with torch.autocast('cuda',dtype=torch.float16): pred,_=model.denoise(x,t,style,content,structure,ctx,1.)
        return pred.float()
    noise=torch.cat([torch.randn((1,3,96,96),device=device,generator=torch.Generator(device=device).manual_seed(s)) for s in seeds])
    solver=DPM_Solver(model_fn=fn,noise_schedule=schedule,algorithm_type='dpmsolver++')
    pred=solver.sample(x=noise,steps=20,order=2,skip_type='time_uniform',method='multistep')
    assert torch.isfinite(pred).all()
    return ((pred.float()/2+.5).clamp(0,1).permute(0,2,3,1).cpu().numpy()*255).round().astype('uint8')

def get_sample(ds,j):
    s=ds[j['index']]; assert (s['font_stem'],s['char_cp'])==(j['font'],j['cp'])
    s['ref_chars']=j['refs'];s['ref_image_paths']=[str(ds.style_by_font_char[j['font']][c]) for c in j['refs']]
    return s

def infer(arm,smoke=False):
    rank=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    if world>1:dist.init_process_group('nccl')
    device=torch.device('cuda',rank);out=OUT/arm;out.mkdir(exist_ok=True)
    model,args=model_for(arm,device,out/f'provenance_rank{rank}')
    checkpoint=ROOT/'runs/K1-ORIGINAL-V0917-S3407/global_step_10000/ema.pth'
    if arm=='K1':model.load_train_state(torch.load(checkpoint,map_location=device,weights_only=True))
    model.eval();data=DataContext(args,device);scheduler=T.build_ddpm_scheduler(args)
    allstats=[]
    for split in ('train','val','test'):
        m=json.loads((OUT/(split+'_manifest.json')).read_text());jobs=m['jobs'];ds=dataset(args,split)
        if smoke:jobs=[next(j for j in jobs if j['script']==g and j['k']==k) for g,k in [('western',1),('western',2),('kana',4),('bopomofo',8)]]
        directory=out/(split+('_smoke' if smoke else ''));directory.mkdir(exist_ok=True)
        shard=jobs[rank::world];rows=[];start=time.time()
        progress=directory/f'rank{rank}.jsonl';existing={}
        if progress.exists():
            for line in progress.read_text().splitlines():
                r=json.loads(line);existing[r['png']]=r
        for st in range(0,len(shard),4):
            chunk=shard[st:st+4];pending=[]
            for j in chunk:
                name=f"{j['font']}__{j['cp']}__k{j['k']}.png"
                if name in existing:
                    with Image.open(directory/name) as im:im.load();assert im.size==(96,96)
                    rows.append(existing[name])
                else:pending.append(j)
            if not pending:continue
            samples=[get_sample(ds,j) for j in pending];batch=batch_to(samples,device)
            preds=sample_batch(model,data,batch,scheduler,[j['seed'] for j in pending])
            if smoke:
                errors=[]
                for p,s,j in zip(preds,samples,pending):
                    reference=serial_sample(model,data,batch_to([s],device),scheduler,j['seed'])
                    errors.append(float(np.abs(p.astype(float)-reference.astype(float)).mean()/255))
                assert max(errors)<.005,('batch versus serial mismatch',errors)
                print('BATCH_PARITY',arm,split,errors,flush=True)
            predt=torch.from_numpy(preds.copy()).permute(0,3,1,2).float().to(device)/255
            detail=detail_distances(predt,batch['content_image']/2+.5,batch['nonorm_target_image'])
            with progress.open('a') as log:
                for n,(j,s,pred) in enumerate(zip(pending,samples,preds)):
                    name=f"{j['font']}__{j['cp']}__k{j['k']}.png";Image.fromarray(pred).save(directory/name)
                    gt=np.array(Image.open(s['target_image_path']).convert('L'),dtype=float)/255
                    gray=np.array(Image.fromarray(pred).convert('L'),dtype=float)/255
                    r=dict(**j,png=name,target=s['target_image_path'],refs_paths=s['ref_image_paths'],
                        content=str(ds.root/ds.phase/'ContentImage'/(j['cp']+'.png')),
                        l1=float(np.abs(gray-gt).mean()),ssim=float(structural_similarity(gray,gt,data_range=1.,win_size=7)),
                        **{k:float(v[n]) for k,v in detail.items()})
                    log.write(json.dumps(r)+'\n');rows.append(r)
            atomic_json(directory/f'progress_rank{rank}.json',dict(done=len(rows),total=len(shard),seconds=time.time()-start))
        atomic_json(directory/f'rank{rank}.json',dict(rows=rows,seconds=time.time()-start))
        if world>1:dist.barrier()
        if rank==0:
            total=sum([json.loads((directory/f'rank{i}.json').read_text())['rows'] for i in range(world)],[])
            assert len(total)==len(jobs) and len({r['png'] for r in total})==len(jobs)
            metrics={}
            for g in ('all','western','kana','bopomofo'):
                rr=[r for r in total if g=='all' or r['script']==g]
                metrics[g]={key:float(np.mean([r[key] for r in rr])) for key in ['l1','ssim','D_region','D_change','D_high']}
            atomic_json(directory/'metrics.json',dict(rows=total,summary=metrics))
            atomic_json(directory/'DONE.json',dict(status='completed',rows=len(total),smoke=smoke,arm=arm,
                manifest_sha256=sha256_file(OUT/(split+'_manifest.json')),weights_sha256=sha256_file(checkpoint) if arm=='K1' else sha256_file(PARENT0/'unet.pth'),
                script_sha256=sha256_file(Path(__file__)),backend=str(BACKEND)))
            allstats.append(dict(split=split,rows=len(total),summary=metrics))
    if rank==0:atomic_json(out/('SMOKE_DONE.json' if smoke else 'DONE.json'),dict(status='completed',splits=allstats))
    if world>1:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--arm',choices=['K0','K1']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.arm:infer(a.arm,a.smoke)
    else:p.error('Specify --prepare or --arm')

if __name__=='__main__':main()
