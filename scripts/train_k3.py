"""Authorized full K3: shared K0 parent, 10k updates and differentiable reference interventions."""
import argparse
import collections
import datetime
import json
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scripts.k3_components import PairSampler, pair_loss, rollout, isolated_rng, average_gradients, PAIR_VERSION
from k_runtime import (ROOT,CODE,CACHE,PARENT0,T,DataContext,model_for,dataset,train_sample,
                       batch_to,seed_episode,atomic_json,sha256_file,code_identity)
from scripts.hrfont_h import lr_factor
from scripts.k_components import KSampler,raw_x0,extra_losses


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())


def set_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def checkpoint(raw,ema,optimizer,scaler,out,step,attempt,exposure,identity,milestone=False):
    rngs=[None]*dist.get_world_size(); dist.all_gather_object(rngs,rng_state())
    if dist.get_rank()==0:
        storage=(out/'checkpoints').resolve()
        dest=storage/f'state_step_{step}'
        if dest.exists(): raise RuntimeError('Checkpoint collision: '+str(dest))
        tmp=storage/f'.state_step_{step}.tmp'; tmp.mkdir()
        torch.save(raw.train_state(),tmp/'model.pth'); torch.save(ema,tmp/'ema.pth')
        torch.save(dict(step=step,attempt=attempt,optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                        rngs=rngs,exposure=dict(exposure),sampler=KSampler.version,identity=identity),tmp/'trainer.pt')
        atomic_json(tmp/'checkpoint.json',dict(step=step,attempt=attempt,arm=raw.arm,complete=True,
                                               sampler=KSampler.version,code_commit=identity['commit']))
        tmp.rename(dest)
        link=out/'.last_state.next'; link.symlink_to('checkpoints/'+dest.name,target_is_directory=True)
        link.replace(out/'last_state')
        if milestone:
            m=storage/f'global_step_{step}'; m.mkdir()
            for name in ('ema.pth','checkpoint.json'): os.link(dest/name,m/name)
            (out/m.name).symlink_to('checkpoints/'+m.name,target_is_directory=True)
        # Retain two complete rolling states, only remove this runner's known files.
        states=sorted(storage.glob('state_step_*'),key=lambda p:int(p.name.split('_')[-1]))
        for old in states[:-2]:
            assert old.parent==storage and not old.is_symlink()
            for name in ('model.pth','ema.pth','trainer.pt','checkpoint.json'): (old/name).unlink()
            old.rmdir()
    dist.barrier()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=['K3'],default='K3')
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--detail-manifest',type=Path,default=ROOT/'artifacts/i56_20260916/detail_manifest.json')
    ap.add_argument('--limit',type=int,default=10000)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--state-interval',type=int,default=1000)
    ap.add_argument('--eval-smoke',action='store_true')
    ap.add_argument('--checkpoint-root',type=Path,default=Path('/root/data1/hrfont_k3_20260917'))
    a=ap.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    local=int(os.environ['LOCAL_RANK']); device=torch.device('cuda',local)
    assert 1<=a.limit<=10000 and (world==8 or (a.smoke and world==1))
    torch.cuda.set_device(device); torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    identity=code_identity()
    out=ROOT/'runs'/a.run_id
    if rank==0:
        if out.exists() and any(out.iterdir()) and not a.resume: raise RuntimeError('Existing run; refusing overwrite')
        out.mkdir(parents=True,exist_ok=True)
        storage=a.checkpoint_root/a.run_id
        storage.mkdir(parents=True,exist_ok=True)
        if not (out/'checkpoints').exists(): (out/'checkpoints').symlink_to(storage,target_is_directory=True)
        assert (out/'checkpoints').resolve()==storage
        atomic_json(out/'heartbeat.json',dict(state='INITIALIZING',step=0,time=time.time()))
    dist.barrier()
    model,args=model_for('K1',device,out/f'provenance_rank{rank}')
    model.arm='K3'
    data=DataContext(args,device); train=dataset(args,'train')
    sampler=KSampler(train,a.detail_manifest)
    aliases=json.loads((Path('/root/data1/hrfont_e12c_r2_20260917')/'inventory_all.json').read_text())['groups']
    pair_sampler=PairSampler(train,sampler,aliases)
    aux_weight=None
    calibration_path=out/'PAIR_CALIBRATION.json'
    scheduler=T.build_ddpm_scheduler(args)
    vgg=T.ContentPerceptualLoss().VGG.to(device).eval().requires_grad_(False)
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        peak=1e-4 if name.startswith('reader.') or ('sc_interpreter_offsets.' in name and '.original.' not in name) else 2e-5
        groups.setdefault((peak,.01 if p.ndim>1 else 0.),[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,lr=lr,peak_lr=lr,weight_decay=wd) for (lr,wd),ps in groups.items()],
                                betas=(.9,.999),eps=1e-8)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    raw=model; params=[p for p in raw.parameters() if p.requires_grad]
    for tensor in raw.state_dict().values():dist.broadcast(tensor,src=0)
    names={n for n,p in raw.named_parameters() if p.requires_grad}
    ema={k:v.detach().clone() for k,v in raw.train_state().items()}
    step=attempt=0; exposure=collections.Counter()
    config=dict(arm=a.arm,global_batch=64,microbatch=8,world_size=world,accumulation=8//world,
        parent=str(PARENT0),parent_unet_sha256=sha256_file(PARENT0/'unet.pth'),
        detail_sha256=sha256_file(a.detail_manifest),clean_sha256=T._clean_map_sha256(args.v0913_clean_map),
        teacher_stats_sha256=sha256_file(CACHE/'teacher_stats.pt'),identity=identity,
        sampler=KSampler.version,seed=3407,joint_cfg=.02,source_drop=.05,beta='.8*min(update/1000,1)',
        comp=.01,detail=.05,offset=.25,vgg=.01,schedule='warmup500/flat5000/cosine10pct10000',
        smoke=a.smoke,precision='fp16/FP32 logits and loss',ema='min(.999,1-1/(update+1))',
        pair=dict(version=PAIR_VERSION,steps=8,interval=16,pairs_per_rank=1,eta=1.,ramp=500,
            gradient_target=.10,weight_bounds=[1e-6,.5],same_ref_codepoints=True,
            pure_noise=True,full_rollout_gradient=True,activation_checkpointing='nonreentrant',
            aliases_sha256=sha256_file(Path('/root/data1/hrfont_e12c_r2_20260917/inventory_all.json'))),
        gradient_sync='explicit global average after main plus auxiliary',authorized_updates=a.limit)
    if a.resume:
        assert json.loads((out/'config.json').read_text())==config,'Resume configuration/source changed'
        state=torch.load(a.resume/'trainer.pt',map_location='cpu',weights_only=False)
        assert state['identity']==identity and state['sampler']==sampler.version
        assert len(state['rngs'])==world
        raw.load_train_state(torch.load(a.resume/'model.pth',map_location=device,weights_only=True))
        ema=torch.load(a.resume/'ema.pth',map_location=device,weights_only=True)
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        step,attempt=state['step'],state['attempt']; exposure.update(state['exposure'])
        assert sum(exposure.values())==64*step
        set_rng(state['rngs'][rank]); del state
        aux_weight=json.loads(calibration_path.read_text())['weight']
    elif rank==0: atomic_json(out/'config.json',config)
    if not a.smoke:
        authorization=json.loads((CODE/'K3_AUTHORIZATION.json').read_text())
        assert authorization['approved'] and authorization['successful_updates']==10000
        assert authorization['parent']==str(PARENT0)
    begin=time.time(); accumulation=8//world
    window_drops=torch.zeros(2,device=device);window_samples=0
    while step<a.limit:
        if step%100==0 and (shutil.disk_usage(out/'checkpoints').free<8*2**30 or shutil.disk_usage(ROOT).free<2*2**30):
            raise RuntimeError('Disk space floor; retain last complete checkpoint')
        start=time.time(); raw.train(); raw.base.style_encoder.eval(); raw.base.content_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        ids,quota=sampler.batch(attempt)
        total=torch.zeros(11,device=device); drops=torch.zeros(2,device=device)
        local_count=0
        for micro in range(accumulation):
            seed_episode(3407+attempt*1000+micro*world+rank)
            indices=ids[(micro*world+rank)*8:(micro*world+rank+1)*8]
            samples=batch_to([train_sample(train,i) for i in indices],device)
            assert len(samples['font_stem'])==8
            local_count+=8
            source=torch.rand(8,device=device)<.05; cfg=torch.rand(8,device=device)<.02
            drops+=torch.stack([cfg.sum(),source.sum()])
            style,refs,query,keep,content,structure=data.conditions(samples,cfg,source,
                official=a.arm=='K2-Delta-',need_refs=raw.use_tc)
            noise=torch.randn_like(samples['target_image']); ts=torch.randint(0,1000,(8,),device=device)
            noisy=scheduler.add_noise(samples['target_image'],noise,ts)
            gate=min(1.,(step+1)/1000)
            with torch.autocast('cuda',dtype=torch.float16):
                pred,offset,appearance=model(noisy,ts,style,refs,query,keep,content,structure,cfg,gate)
                clean,alpha=raw_x0(noisy,pred,scheduler.alphas_cumprod,ts)
                gen=vgg(T.normalize_mean_std(clean.clamp(0,1)))
                with torch.no_grad(): gt=vgg(T.normalize_mean_std(samples['nonorm_target_image']))
                teacher=F.adaptive_avg_pool2d(gt[1].float(),12).flatten(2).transpose(1,2)
                teacher=(teacher-raw.teacher_mean)/raw.teacher_std
                extra,comp,detail,d=extra_losses(appearance,teacher,clean,samples['content_image']/2+.5,
                    samples['nonorm_target_image'],~cfg,alpha,step+1)
                eps=F.mse_loss(pred.float(),noise.float())
                perceptual=sum(F.mse_loss(x.float(),y.float()) for x,y in zip(gen,gt))/3
                loss=eps+.01*perceptual+.25*offset.float()+extra
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            scaler.scale(loss/accumulation).backward()
            total+=torch.stack([loss,eps,perceptual,comp,detail,offset.float(),
                d['D_region'].mean(),d['D_change'].mean(),d['D_add'].mean(),d['D_remove'].mean(),
                d['D_high'].mean()]).detach()/accumulation
        count=torch.tensor(local_count,device=device); dist.all_reduce(count); assert count.item()==64
        aux_record={};is_aux=(step%16==0)
        if is_aux:
            aux_start=time.time()
            with isolated_rng(993407+attempt*1009+rank):
                pair,meta=pair_sampler.draw(step//16,rank,device,batch_to)
                generated,path=rollout(raw,data,pair,scheduler,gate,993407+attempt*1009+rank,trace=(step==0))
                aux_loss,parts=pair_loss(generated,pair['nonorm_target_image'])
                if not torch.isfinite(aux_loss):raise RuntimeError('Nonfinite full-rollout loss')
                aux_grads=torch.autograd.grad(aux_loss,params,allow_unused=True)
                shared=[i for i,(n,p) in enumerate((n,p) for n,p in raw.named_parameters() if p.requires_grad) if n.startswith('base.unet.')]
                norms=torch.stack([sum((params[i].grad.float().square().sum() for i in shared if params[i].grad is not None),generated.new_zeros(())),
                    sum((aux_grads[i].float().square().sum() for i in shared if aux_grads[i] is not None),generated.new_zeros(()))])
                dist.all_reduce(norms);norms=(norms/world).sqrt()/norms.new_tensor([scaler.get_scale(),1.])
                if aux_weight is None and bool(torch.isfinite(norms).all()) and float(norms[1])>0:
                    aux_weight=float((.1*norms[0]/norms[1]).clamp(1e-6,.5))
                    calibration=dict(weight=aux_weight,base_unet_grad=float(norms[0]),pair_unet_grad=float(norms[1]),
                        nominal_ratio=aux_weight*float(norms[1])/max(float(norms[0]),1e-20),
                        shared_parameter_group='UNet',calibration_update=step+1,train_only=True,code_commit=identity['commit'])
                    if rank==0:atomic_json(calibration_path,calibration)
                weight=aux_weight if aux_weight is not None else 1e-6
                ramp=min(1.,(step+1)/500)
                for p,g in zip(params,aux_grads):
                    if g is not None:
                        if p.grad is None:p.grad=g.detach().mul(weight*ramp*scaler.get_scale())
                        else:p.grad.add_(g.detach(),alpha=weight*ramp*scaler.get_scale())
                aux_only_norm=lambda token:float(torch.stack([g.float().square().sum() for (n,p),g in zip(((n,p) for n,p in raw.named_parameters() if p.requires_grad),aux_grads) if token in n and g is not None]).sum().sqrt())
                aux_record=dict(pair_reader_grad=aux_only_norm('reader.'),pair_router_grad=aux_only_norm('sc_interpreter_offsets.'),pair_loss=float(aux_loss.detach()),pair_weight=weight,pair_ramp=ramp,
                    pair_unet_grad=float(norms[1]),base_unet_grad=float(norms[0]),
                    pair_seconds=time.time()-aux_start,pair_meta=meta,
                    **{'pair_'+k:float(v.detach()) for k,v in parts.items()})
                if path:
                    trace_norms=[float(t.grad.float().norm()) if t.grad is not None else 0. for t in path]
                    aux_record['rollout_epsilon_grad_norms']=trace_norms
                    if bool(torch.isfinite(norms).all()):assert all(v>0 for v in trace_norms),trace_norms
                del aux_grads,generated,path,pair,aux_loss,parts
        attempt+=1
        average_gradients(params)
        scaler.unscale_(optimizer)
        grad=torch.nn.utils.clip_grad_norm_(params,1.)
        finite=torch.tensor(int(torch.isfinite(grad)),device=device); dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():
            scaler.update(new_scale=scaler.get_scale()/2)
            if rank==0:
                with (out/'amp_skips.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,attempt=attempt,scale=scaler.get_scale()))+'\n')
            if attempt-step>10 and (attempt-step)/attempt>.01: raise RuntimeError('Excess AMP skips')
            continue
        def gradnorm(selector):
            values=[p.grad.detach().float().square().sum() for n,p in raw.named_parameters() if selector(n) and p.grad is not None]
            return float(torch.stack(values).sum().sqrt()) if values else 0.
        report_grad=(a.smoke or step<10 or (step+1)%100==0 or is_aux)
        grads={}
        if report_grad:
            grads=dict(reader_grad=gradnorm(lambda n:n.startswith('reader.')),
                online_encoder_grad=gradnorm(lambda n:n.startswith('reader.blocks.')),
                router_grad=gradnorm(lambda n:'sc_interpreter_offsets.' in n and '.original.' not in n))
        for g in optimizer.param_groups:g['lr']=g['peak_lr']*lr_factor(step+1)
        scaler.step(optimizer); scaler.update(); step+=1
        exposure.update(quota['fonts'])
        window_drops+=drops;window_samples+=local_count
        with torch.no_grad():
            decay=min(.999,1-1/(step+1))
            for name,value in raw.train_state().items():
                if name in names:ema[name].lerp_(value.detach(),1-decay)
                else:ema[name].copy_(value)
        if report_grad:
            dist.all_reduce(total); total/=world; dist.all_reduce(window_drops)
            fingerprint=torch.stack([p.detach().float().sum() for p in params])
            high,low=fingerprint.clone(),fingerprint.clone()
            dist.all_reduce(high,op=dist.ReduceOp.MAX);dist.all_reduce(low,op=dist.ReduceOp.MIN)
            spread=float((high-low).abs().max()); assert spread<1e-4,spread
            metrics=dict(zip(['loss','epsilon_loss','vgg_loss','completion_loss','detail_loss','offset_loss',
                'D_region','D_change','D_add','D_remove','D_high'],map(float,total)))
            scales=[m._k_scale for m in raw.up_attn if m._k_scale is not None]
            scale_stats=[torch.quantile(s.float(),torch.tensor([.1,.5,.9],device=device)).tolist() for s in scales]
            record=dict(step=step,attempt=attempt,**metrics,**grads,**aux_record,grad_norm=float(grad),
                beta=.8*gate,tc_scale_p10_p50_p90=scale_stats,
                ddp_spread=spread,lr=[g['lr'] for g in optimizer.param_groups],scale=scaler.get_scale(),
                skips=attempt-step,global_batch=64,local_batch=8,script=quota['script'],complex=quota['complex'],
                other={k:quota['script'][k]-quota['complex'][k] for k in quota['script']},
                pairs=quota['pairs'],cfg_drop=int(window_drops[0]),source_drop=int(window_drops[1]),
                drop_window_samples=window_samples*world,
                cfg_drop_rate=float(window_drops[0])/(window_samples*world),
                source_drop_rate=float(window_drops[1])/(window_samples*world),
                update_seconds=time.time()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,
                elapsed_seconds=time.time()-begin)
            with (out/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if rank==0:
                with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                atomic_json(out/'heartbeat.json',dict(state='TRAINING',time=time.time(),**record))
                print('K3_UPDATE',json.dumps(record),flush=True)
            window_drops.zero_();window_samples=0
        final=step==a.limit
        stop=torch.tensor(int((out/'STOP').exists()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        infer=(not a.smoke and (step%2000==0 or final)) or (a.eval_smoke and final)
        if step%a.state_interval==0 or final or stop.item() or infer:
            checkpoint(raw,ema,optimizer,scaler,out,step,attempt,exposure,identity,
                       milestone=infer or (not a.smoke and step==5000))
            if rank==0:
                atomic_json(out/f'exposure_{step}.json',dict(step=step,samples=sum(exposure.values()),counts=dict(exposure)))
        if infer:
            from k_eval import evaluate,make_board
            saved=rng_state(); backup={k:v.detach().cpu().clone() for k,v in raw.train_state().items()}
            raw.load_train_state(ema)
            manifest=ROOT/'experiments/K/K_VAL192.json'
            if final and not a.smoke:manifest=ROOT/'experiments/K/K_TEST.json'
            evaluate(raw,data,args,scheduler,manifest,out/f'eval_step_{step}',step,smoke=a.eval_smoke)
            if not a.smoke and rank==0:
                baseline=ROOT/'reports'/('k0_original_test_20260917' if final else 'k0_original_val192_20260917')
                board=ROOT/'reports'/('k3_final_review' if final else f'k3_step{step}_review')
                if not (baseline/'DONE.json').exists():raise RuntimeError('Missing matched K0 baseline')
                make_board(baseline,out/f'eval_step_{step}',board)
            dist.barrier()
            raw.load_train_state(backup);del backup;set_rng(saved)
        if stop.item():
            if rank==0:atomic_json(out/'STOPPED.json',dict(step=step,attempt=attempt,status='safely_stopped'))
            dist.destroy_process_group();return
    if rank==0:atomic_json(out/'DONE.json',dict(status='completed',step=step,attempt=attempt,arm=a.arm,
        smoke=a.smoke,inference_complete=not a.smoke or a.eval_smoke,code_commit=identity['commit']))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
