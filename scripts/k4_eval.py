"""Frozen full-v2 evaluation and Val192 monitoring for the authorized K4 queue."""
import argparse, collections, copy, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from skimage.metrics import structural_similarity
from k4_runtime import (ROOT,CODE,DATA,STORE,ASSETS,RUNS,PARENT0,K1_EMA,T,DataContext,model_for,
                        dataset,batch_to,atomic_json,sha256_file,code_identity,data_identity)
from scripts.k_components import detail_distances
from src.dpm_solver.dpm_solver_pytorch import DPM_Solver,NoiseScheduleVP

@torch.no_grad()
def sample_batch(model,data,samples,scheduler,seeds):
    b=len(seeds);device=data.device;active=torch.zeros(b,device=device,dtype=torch.bool)
    style,refs,query,keep,content,structure=data.conditions(samples,active,active,no_delta=model.arm=='K0',need_refs=model.use_tc)
    ctx=model.conditions(style,refs,query,keep,active)
    schedule=NoiseScheduleVP(schedule='discrete',betas=scheduler.betas)
    def fn(x,t):
        t=((t-1./schedule.total_N)*1000.).expand(b)
        with torch.autocast('cuda',dtype=torch.float16):pred,_=model.denoise(x,t,style,content,structure,ctx,1.)
        return pred.float()
    noise=torch.cat([torch.randn((1,3,96,96),device=device,generator=torch.Generator(device=device).manual_seed(s)) for s in seeds])
    solver=DPM_Solver(model_fn=fn,noise_schedule=schedule,algorithm_type='dpmsolver++')
    pred=solver.sample(x=noise,steps=20,order=2,skip_type='time_uniform',method='multistep')
    assert torch.isfinite(pred).all()
    return ((pred.float()/2+.5).clamp(0,1).permute(0,2,3,1).cpu().numpy()*255).round().astype('uint8')

def checkpoint_for(arm):
    if arm=='K0':return PARENT0/'unet.pth'
    if arm=='K1':return K1_EMA
    if arm=='K3':return ROOT/'runs/K3-PURENOISE-V0917-S3407/global_step_10000/ema.pth'
    return ROOT/'runs'/RUNS[arm]/'global_step_10000/ema.pth'

def expand_queries(arm):
    shots=[1] if arm=='K0' else [1,2,4,8]
    rows=[json.loads(l) for l in (ASSETS/'v2_val_test_queries.jsonl').read_text().splitlines()]
    return [dict(**r,k=k,refs=r['ref8'][:k]) for r in rows for k in shots]

def filename(j):return f"{j['font']}__{j['cp']}__k{j['k']}.png"

def get_sample(ds,j):return ds.sample(ds.lookup[j['font'],j['cp']],j['refs'])

def score_rows(preds,batch,jobs,samples,paths,data,scheduler=None):
    tensor=torch.from_numpy(preds.copy()).permute(0,3,1,2).float().to(data.device)/255.
    detail=detail_distances(tensor,batch['content_image']/2+.5,batch['nonorm_target_image'])
    rows=[]
    for n,(p,j,s,path) in enumerate(zip(preds,jobs,samples,paths)):
        with Image.open(s['target_image_path']) as im:gt=np.array(im.convert('L'),dtype=float)/255.
        gray=np.array(Image.fromarray(p).convert('L'),dtype=float)/255.
        rows.append(dict(**j,png=path.name,prediction=str(path),prediction_sha256=sha256_file(path),
            target=s['target_image_path'],content=str(DATA/'v2'/j['split']/'ContentImage'/(j['cp']+'.png')),
            refs_paths=s['ref_image_paths'],l1=float(np.abs(gray-gt).mean()),
            ssim=float(structural_similarity(gray,gt,data_range=1.,win_size=7)),
            **{k:float(v[n]) for k,v in detail.items()}))
    return rows

def run_jobs(model,data,args,scheduler,jobs,out,reuse=None,rank=0,world=1):
    out.mkdir(parents=True,exist_ok=True);shard=jobs[rank::world];datasets={sp:dataset(args,sp) for sp in {j['split'] for j in jobs}}
    progress=out/f'rank{rank}.jsonl';existing={}
    if progress.exists():
        for l in progress.read_text().splitlines():
            try:r=json.loads(l)
            except json.JSONDecodeError:continue  # interrupted last line: recompute its job
            existing[r['png']]=r
    rows=[];start=time.time()
    for st in range(0,len(shard),4):
        chunk=shard[st:st+4];pending=[]
        for j in chunk:
            name=filename(j);p=out/name;r=existing.get(name)
            if r is not None and p.is_file() and sha256_file(p)==r['prediction_sha256']:
                assert (r['seed'],r['refs'])==(j['seed'],j['refs'])
                rows.append(r)
            else:pending.append(j)
        if not pending:continue
        # Keep old batch4 positions for recomputed outputs; reused rows are scored
        # independently. Batched/serial equivalence is checked before launch.
        computed=[j for j in pending if not reuse or (j['split'],j['font'],j['cp']) not in reuse]
        predictions={}
        if computed:
            samples=[get_sample(datasets[j['split']],j) for j in computed]
            batch=batch_to(samples,data.device)
            pp=sample_batch(model,data,batch,scheduler,[j['seed'] for j in computed])
            for j,p in zip(computed,pp):
                path=out/filename(j);tmp=path.with_suffix('.tmp.png');Image.fromarray(p).save(tmp);tmp.replace(path)
                predictions[filename(j)]=p
        for j in pending:
            if filename(j) in predictions:continue
            r=reuse[j['split'],j['font'],j['cp']];src=Path(r['path']);assert sha256_file(src)==r['sha256']
            path=out/filename(j)
            if path.exists():path.unlink()  # this run's incomplete job only
            path.symlink_to(src)
            with Image.open(path) as im:im.load();assert im.size==(96,96);predictions[filename(j)]=np.array(im.convert('RGB'),copy=True)
        samples=[get_sample(datasets[j['split']],j) for j in pending];batch=batch_to(samples,data.device)
        new=score_rows(np.stack([predictions[filename(j)] for j in pending]),batch,pending,samples,[out/filename(j) for j in pending],data)
        for r in new:r['reused']=bool(reuse and (r['split'],r['font'],r['cp']) in reuse)
        with progress.open('a') as f:
            f.write(''.join(json.dumps(r)+'\n' for r in new));f.flush()
        rows+=new
        atomic_json(out/f'progress_rank{rank}.json',dict(done=len(rows),total=len(shard),seconds=time.time()-start,time=time.time()))
    atomic_json(out/f'rank{rank}.json',dict(rows=rows,seconds=time.time()-start))
    if dist.is_initialized():dist.barrier()
    if rank==0:
        allrows=sum([json.loads((out/f'rank{r}.json').read_text())['rows'] for r in range(world)],[])
        assert len(allrows)==len(jobs)==len({(r['font'],r['cp'],r['k']) for r in allrows})
        atomic_json(out/'metrics.json',dict(rows=allrows))
        atomic_json(out/'DONE.json',dict(status='completed',rows=len(allrows),reused=sum(r.get('reused',False) for r in allrows)))
    return rows

def evaluate_monitor(model,data,args,scheduler,out,step):
    jobs=json.loads((ASSETS/'v2_val192_monitor.json').read_text())['jobs']
    jobs=[dict(**j,k=j['shot']) for j in jobs]
    was_training=model.training;model.eval()
    run_jobs(model,data,args,scheduler,jobs,out,rank=dist.get_rank(),world=dist.get_world_size())
    model.train(was_training)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['K0','K1','K3',*RUNS],required=True);a=ap.parse_args()
    rank=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if world>1:dist.init_process_group('nccl')
    device=torch.device('cuda',rank);out=STORE/'inference'/a.arm;out.mkdir(parents=True,exist_ok=True)
    identity=code_identity();model,args=model_for(a.arm,device,out/f'provenance_rank{rank}',evaluation=True)
    weights=checkpoint_for(a.arm)
    if a.arm!='K0':model.load_train_state(torch.load(weights,map_location=device,weights_only=True))
    protocol=dict(arm=a.arm,weights_sha256=sha256_file(weights),identity=identity,data=data_identity(args),
        queries_sha256=sha256_file(ASSETS/'v2_val_test_queries.jsonl'),cfg=1.,steps=20,order=2,batch=4,
        source_encoders={n:sha256_file(PARENT0/n) for n in ['style_encoder.pth','content_encoder.pth']},
        sampler='dpmsolver++',donors='v2/train',shots=[1] if a.arm=='K0' else [1,2,4,8])
    if rank==0:
        p=out/'PROTOCOL.json'
        if p.exists():assert json.loads(p.read_text())==protocol,'Inference identity changed'
        else:atomic_json(p,protocol)
    if world>1:dist.barrier()
    model.eval();data=DataContext(args,device);scheduler=T.build_ddpm_scheduler(args)
    reuse=None
    if a.arm=='K0':
        passed=json.loads((STORE/'control/PREFLIGHT_PASSED.json').read_text())
        if passed['k0_reuse_approved']:
            reuse={ (r['split'],r['font'],r['cp']):r for r in [json.loads(l) for l in (ASSETS/'k0_reuse_candidates.jsonl').read_text().splitlines()] }
    jobs=expand_queries(a.arm)
    for split in ['val','test']:
        run_jobs(model,data,args,scheduler,[j for j in jobs if j['split']==split],out/split,reuse,rank,world)
    if rank==0:atomic_json(out/'DONE.json',dict(status='completed',rows=len(jobs),protocol_sha256=sha256_file(out/'PROTOCOL.json')))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
