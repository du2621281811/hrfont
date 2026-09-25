"""Authorized family-filtered K4-A -> K4-C -> K4-B restart."""
import argparse
import hashlib
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
from k6_runtime import (ROOT,CODE,CACHE,PARENT0,T,DataContext,model_for,dataset,train_sample,
                       batch_to,seed_episode,atomic_json,sha256_file,code_identity)
from k6_runtime import ASSETS, STORE, data_identity, configure
from scripts.hrfont_h import lr_factor
def update_factor(step,*args):return lr_factor(step)
def schedule_spec(*args):return "warmup500/flat5000/cosine10pct10000"

from scripts.k_components import KSampler,raw_x0,extra_losses
from k6_objective import paired_batch,extra_objective,pure_objective
from scripts.k4_numerics import guarded_forward,compatible_identity


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
    ap.add_argument('--arm',choices=['K6-A','K6-B'],required=True)
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--detail-manifest',type=Path,default=CODE/'experiments/K4/detail_manifest.json')
    ap.add_argument('--limit',type=int,default=10000)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--state-interval',type=int,default=1000)
    ap.add_argument('--eval-smoke',action='store_true')
    ap.add_argument('--stop-after',type=int,default=0)
    ap.add_argument('--checkpoint-root',type=Path,default=STORE/'checkpoints')
    a=ap.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    local=int(os.environ['LOCAL_RANK']); device=torch.device('cuda',local)
    assert 1<=a.limit<=10000 and (world==8 or (a.smoke and world==1))
    torch.cuda.set_device(device); torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=60))
    identity=code_identity()
    authorization=json.loads((ASSETS/'AUTHORIZATION.json').read_text())
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
    model,args=model_for(a.arm,device,out/f'provenance_rank{rank}')
    data=DataContext(args,device,model); train=dataset(args,'train')
    sampler=KSampler(train,a.detail_manifest)
    calibration=json.loads((STORE/'control/CALIBRATION.json').read_text())
    aliases={}
    pair_sampler=PairSampler(train,sampler,aliases) if a.arm=='NEVER_AUX' else None
    aux_weight=None
    calibration_path=out/'PAIR_CALIBRATION.json'
    scheduler=T.build_ddpm_scheduler(args)
    vgg=T.ContentPerceptualLoss().VGG.to(device).eval().requires_grad_(False)
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        peak=1e-4 if name.startswith('reader.') or ('sc_interpreter_offsets.' in name and ('.original.' not in name or '.es_branch.' in name)) else 2e-5
        groups.setdefault((peak,.01 if p.ndim>1 else 0.),[]).append(p)
    optimizer=torch.optim.AdamW([dict(params=ps,lr=lr,peak_lr=lr,weight_decay=wd) for (lr,wd),ps in groups.items()],
                                betas=(.9,.999),eps=1e-8)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    raw=model; params=[p for p in raw.parameters() if p.requires_grad]
    for tensor in raw.state_dict().values():dist.broadcast(tensor,src=0)
    names={n for n,p in raw.named_parameters() if p.requires_grad}
    ema={k:v.detach().clone() for k,v in raw.train_state().items()}
    step=attempt=0; exposure=collections.Counter()
    config=dict(calibration_sha256=sha256_file(STORE/'control/CALIBRATION.json'),k6_objective='pure-noise-v2-every16-8step-lambda005-ramp1000',arm=a.arm,global_batch=64,microbatch=8,world_size=world,accumulation=8//world,
        parent=str(K1_EMA if a.arm=='NEVER_FINETUNE' else PARENT0),parent_unet_sha256=sha256_file(PARENT0/'unet.pth'),
        warmstart_ema_sha256=sha256_file(K1_EMA) if a.arm=='NEVER_FINETUNE' else None,
        data_identity=data_identity(args),
        detail_sha256=sha256_file(a.detail_manifest),clean_sha256=T._clean_map_sha256(args.v0913_clean_map),
        teacher_stats_sha256=sha256_file(CACHE/'teacher_stats.pt'),identity=identity,
        sampler=KSampler.version,seed=3407,joint_cfg=.02,source_drop=.05,beta='.8' if a.arm=='NEVER_FINETUNE' else '.8*min(update/1000,1)',
        comp=.01,detail=.05,offset=.25,vgg=.01,schedule=schedule_spec(authorization,a.arm),
        smoke=a.smoke,precision='fp16/FP32 logits and loss',ema='min(.999,1-1/(update+1))',
        pair=dict(version=PAIR_VERSION,steps=8,interval=16,pairs_per_rank=1,eta=1.,ramp=500,
            gradient_target=.10,weight_bounds=[1e-6,.5],same_ref_codepoints=True,
            pure_noise=True,full_rollout_gradient=True,activation_checkpointing='nonreentrant',
            aliases_sha256=sha256_file(ASSETS/'aliases.json')) if a.arm=='NEVER_AUX' else None,
        gradient_sync='explicit global average after main plus auxiliary',authorized_updates=a.limit)
    if a.resume:
        assert json.loads((out/'config.json').read_text())==config,'Resume configuration/source changed'
        state=torch.load(a.resume/'trainer.pt',map_location='cpu',weights_only=False)
        assert state['identity']==identity and len(state['rngs'])==world
        raw.load_train_state(torch.load(a.resume/'model.pth',map_location=device,weights_only=True))
        ema=torch.load(a.resume/'ema.pth',map_location=device,weights_only=True)
        optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        step,attempt=state['step'],state['attempt'];exposure.update(state['exposure'])
        assert sum(exposure.values())==64*step
        set_rng(state['rngs'][rank]);del state
    elif rank==0:atomic_json(out/'config.json',config)
    if not a.smoke:
        pre=json.loads((STORE/'control/PREFLIGHT_PASSED.json').read_text())
        assert pre['status']=='PASS' and pre['identity']==identity
        assert authorization[a.arm]['authorized'] and a.limit==authorization[a.arm]['successful_updates']
    begin=time.time(); accumulation=8//world
    window_drops=torch.zeros(2,device=device);window_samples=0
    while step<a.limit:
        if step%100==0 and (shutil.disk_usage(out/'checkpoints').free<2*2**30 or shutil.disk_usage(ROOT).free<2*2**30):
            raise RuntimeError('Disk space floor; retain last complete checkpoint')
        start=time.time(); raw.train(); raw.base.style_encoder.eval(); raw.base.content_encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        ids,quota=paired_batch(sampler,attempt)
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
            episode_hash=hashlib.sha256(json.dumps([samples['font_stem'],samples['char_cp'],samples['ref_chars']]).encode()+noise.detach().cpu().numpy().tobytes()+ts.cpu().numpy().tobytes()+cfg.cpu().numpy().tobytes()+source.cpu().numpy().tobytes()).hexdigest() if (a.smoke or step<2) else None
            objective_step=10000+step+1 if a.arm=='NEVER_FINETUNE' else step+1
            gate=min(1.,objective_step/1000)
            k6_record={}
            def objective(amp):
                with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
                    pred,offset,appearance=model(noisy,ts,style,refs,query,keep,content,structure,cfg,gate)
                    clean,alpha=raw_x0(noisy,pred,scheduler.alphas_cumprod,ts)
                    gen=vgg(T.normalize_mean_std(clean.clamp(0,1)))
                    with torch.no_grad(): gt=vgg(T.normalize_mean_std(samples['nonorm_target_image']))
                    teacher=F.adaptive_avg_pool2d(gt[1].float(),12).flatten(2).transpose(1,2)
                    teacher=(teacher-raw.teacher_mean)/raw.teacher_std
                    extra,comp,detail,d=extra_losses(appearance,teacher,clean,samples['content_image']/2+.5,
                        samples['nonorm_target_image'],~cfg,alpha,objective_step)
                    eps=F.mse_loss(pred.float(),noise.float())
                    perceptual=sum(F.mse_loss(x.float(),y.float()) for x,y in zip(gen,gt))/3
                    loss=eps+.01*perceptual+.25*offset.float()+extra
                return loss,(eps,perceptual,comp,detail,offset,extra,d),dict(loss=loss,epsilon=eps,vgg=perceptual,completion=comp,detail=detail,offset=offset,extra=extra)
            loss,payload,_=guarded_forward(objective,out,dict(step=step,attempt=attempt,kind='main',fonts=samples['font_stem'],chars=samples['char_cp'],refs=samples['ref_chars'],timesteps=ts.tolist()))
            eps,perceptual,comp,detail,offset,extra,d=payload
            scaler.scale(loss/accumulation).backward()
            total+=torch.stack([loss,eps,perceptual,comp,detail,offset.float(),
                d['D_region'].mean(),d['D_change'].mean(),d['D_add'].mean(),d['D_remove'].mean(),
                d['D_high'].mean()]).detach()/accumulation
        count=torch.tensor(local_count,device=device); dist.all_reduce(count); assert count.item()==64
        aux_record=dict(k6_loss=0.,k6_eligible=0,k6_aux=False)
        is_aux=(step%16==0)
        if is_aux:
            aux_start=time.time()
            with isolated_rng(993407+attempt*1009+rank):
                positions=quota['pair_positions'][rank]
                pair_indices=[ids[i] for i in positions]
                selected=[]
                for position in positions:
                    owner=position//8
                    with isolated_rng(3407+attempt*1000+owner):
                        batch_samples=[train_sample(train,i) for i in ids[owner*8:(owner+1)*8]]
                    selected.append(batch_samples[position%8])
                pair=batch_to(selected,device)
                assert pair['char_cp'][0]==pair['char_cp'][1]
                def auxiliary(amp):
                    generated,path=rollout(raw,data,pair,scheduler,gate,993407+attempt*1009+rank,trace=True,amp_enabled=amp)
                    with torch.autocast('cuda',enabled=False):
                        features=vgg(T.normalize_mean_std(generated.float().clamp(0,1)))
                        with torch.no_grad():
                            target=vgg(T.normalize_mean_std(pair['nonorm_target_image'].float()))
                            neutral=vgg(T.normalize_mean_std(pair['content_image'].float()/2+.5))
                        aux_loss,parts=pure_objective(a.arm,features,target,neutral,[sampler.entries[i][2] for i in pair_indices],calibration)
                    return aux_loss,(generated,path,parts),dict(loss=aux_loss)
                aux_loss,(generated,path,parts),_=guarded_forward(auxiliary,out,dict(step=step,attempt=attempt,kind='pure_noise_auxiliary',indices=pair_indices))
                aux_grads=torch.autograd.grad(aux_loss,params,allow_unused=True)
                grad_sq=sum((g.float().square().sum() for g in aux_grads if g is not None),generated.new_zeros(()))
                base_sq=sum((p.grad.float().square().sum() for p in params if p.grad is not None),generated.new_zeros(()))/scaler.get_scale()**2
                weight=.05*min(1.,(step+1)/1000)
                for p,g in zip(params,aux_grads):
                    if g is not None:
                        if p.grad is None:p.grad=g.detach().mul(weight*scaler.get_scale())
                        else:p.grad.add_(g.detach(),alpha=weight*scaler.get_scale())
                aux_record=dict(k6_aux=True,k6_loss=float(aux_loss.detach()),k6_weight=weight,k6_aux_grad=float(grad_sq.sqrt()),k6_base_grad=float(base_sq.sqrt()),k6_weighted_grad_ratio=float(weight*grad_sq.sqrt()/base_sq.sqrt().clamp_min(1e-20)),k6_seconds=time.time()-aux_start,k6_indices=pair_indices,**parts)
                aux_record['rollout_epsilon_grad_norms']=[float(x.grad.float().norm()) if x.grad is not None else 0. for x in path]
                if float(grad_sq)>0:assert all(x>0 for x in aux_record['rollout_epsilon_grad_norms'])
                del aux_grads,generated,path,pair,aux_loss,aux_start
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
                router_grad=gradnorm(lambda n:'sc_interpreter_offsets.' in n and '.original.' not in n),
                es_adapter_grad=gradnorm(lambda n:'es_adapter' in n),
                es_gate_grad=gradnorm(lambda n:'es_gate' in n))
        for g in optimizer.param_groups:g['lr']=g['peak_lr']*update_factor(step+1,authorization,a.arm)
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
            record=dict(step=step,attempt=attempt,episode_sha256=episode_hash,**metrics,**grads,**aux_record,grad_norm=float(grad),
                beta=.8*gate,tc_scale_p10_p50_p90=scale_stats,
                ddp_spread=spread,lr=[g['lr'] for g in optimizer.param_groups],scale=scaler.get_scale(),
                skips=attempt-step,global_batch=64,local_batch=8,script=quota['script'],complex=quota['complex'],
                other={k:quota['script'][k]-quota['complex'][k] for k in quota['script']},
                pairs=quota['pairs'],cfg_drop=int(window_drops[0]),source_drop=int(window_drops[1]),
                drop_window_samples=window_samples*world,
                cfg_drop_rate=float(window_drops[0])/(window_samples*world),
                source_drop_rate=float(window_drops[1])/(window_samples*world),
                update_seconds=time.time()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,
                elapsed_seconds=time.time()-begin,donor_family_guard=data.library.family_policy.snapshot())
            with (out/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if rank==0:
                with (out/'train_log.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                atomic_json(out/'heartbeat.json',dict(state='TRAINING',time=time.time(),**record))
                print(a.arm+'_UPDATE',json.dumps(record),flush=True)
            window_drops.zero_();window_samples=0
        final=step==a.limit
        stop=torch.tensor(int((out/'STOP').exists() or (a.stop_after>0 and step>=a.stop_after)),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        infer=(not a.smoke and (step%2000==0 or final))
        if step%a.state_interval==0 or final or stop.item() or infer:
            checkpoint(raw,ema,optimizer,scaler,out,step,attempt,exposure,identity,
                       milestone=infer or (not a.smoke and step==5000))
            if rank==0:
                atomic_json(out/f'exposure_{step}.json',dict(step=step,samples=sum(exposure.values()),counts=dict(exposure)))
        if stop.item():
            if rank==0:atomic_json(out/'STOPPED.json',dict(step=step,attempt=attempt,status='safely_stopped'))
            dist.destroy_process_group();return
    if rank==0:atomic_json(out/'DONE.json',dict(status='completed',step=step,attempt=attempt,arm=a.arm,
        smoke=a.smoke,inference_complete=False,monitor_complete=False,code_commit=identity['commit']))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    try:main()
    finally:
        if dist.is_initialized():dist.destroy_process_group()
