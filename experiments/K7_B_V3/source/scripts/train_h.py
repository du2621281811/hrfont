"""H 10k runner: deterministic global64 episodes, DDP, joint CFG, EMA, inference.

Separate entry point: the incumbent G trainer and its snapshots are never edited.
"""
import argparse
import contextlib
import datetime
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from h_runtime import (ROOT, CODE, CACHE, PARENT, PARENT0, T, DataContext, model_for,
                       dataset, batch_to, seed_episode, atomic_json, sha256_file)
from scripts.hrfont_h import lr_factor
from h_eval import evaluate


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def set_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state(s['cuda'])


def checkpoint(model, ema, optimizer, scaler, out, step, attempt, milestone=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    rngs = [None] * world
    dist.all_gather_object(rngs, rng_state())
    if rank == 0:
        dest = out / f'state_step_{step}'
        if dest.exists():
            raise RuntimeError(f'checkpoint collision: {dest}')
        tmp = out / f'.state_step_{step}.tmp'
        tmp.mkdir()
        torch.save(model.train_state(), tmp/'model.pth')
        torch.save(ema, tmp/'ema.pth')
        torch.save(dict(step=step, attempt=attempt, optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), rngs=rngs, sampler='step-keyed-global64-v1'), tmp/'trainer.pt')
        atomic_json(tmp/'checkpoint.json', dict(step=step, attempt=attempt, arm=model.arm, complete=True))
        tmp.rename(dest)
        link = out/'.last_state.next'
        link.symlink_to(dest.name, target_is_directory=True)
        link.replace(out/'last_state')
        if milestone:
            m = out/f'global_step_{step}'
            m.mkdir()
            # Hardlinks preserve inference files when rolling state dirs expire.
            for name in ('model.pth','ema.pth','checkpoint.json'):
                os.link(dest/name, m/name)
        states = sorted(out.glob('state_step_*'), key=lambda p: int(p.name.split('_')[-1]))
        for old in states[:-2]:
            assert old.parent == out and not old.is_symlink() and old.name.startswith('state_step_')
            # Only four known files created by this H runner, never recursive deletion.
            for name in ('model.pth','ema.pth','trainer.pt','checkpoint.json'):
                (old/name).unlink()
            old.rmdir()
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['H3','H0','H4','H2','H1','H-D+','H-D-'], required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--limit', type=int, default=10000, help='successful updates, hard maximum 10k')
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--transition', type=int, default=1000)
    ap.add_argument('--eval-interval', type=int, default=2000)
    ap.add_argument('--state-interval', type=int, default=500)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--overfit', action='store_true')
    a = ap.parse_args()
    if not 1 <= a.limit <= 10000:
        raise ValueError('user cap is 10k per model')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    assert world == 8, 'H global batch64 requires eight ranks'
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=45))
    out = ROOT/'runs'/a.run_id
    if rank == 0:
        if out.exists() and any(out.iterdir()) and not a.resume:
            raise RuntimeError('run exists; no overwrite')
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out/'heartbeat.json', dict(state='INITIALIZING', step=0, time=time.time()))
    dist.barrier()
    # Each rank writes its own parent-load manifest; no racing shared JSON file.
    model, args = model_for(a.arm, device, out/f'provenance_rank{rank}')
    model.base.style_encoder.eval()
    model.base.content_encoder.eval()
    data = DataContext(args, device)
    train = dataset(args, 'train')
    val = dataset(args, 'val')
    weights = torch.tensor(train.sample_weights, dtype=torch.double)
    noise_scheduler = T.build_ddpm_scheduler(args)
    vgg = T.ContentPerceptualLoss().VGG.to(device).eval().requires_grad_(False)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        peak = 2e-5 if name.startswith('base.') else 1e-4
        decay = .01 if p.ndim > 1 else 0.
        groups.setdefault((peak, decay), []).append(p)
    optimizer = torch.optim.AdamW([dict(params=p, lr=peak, peak_lr=peak, weight_decay=wd)
        for (peak, wd), p in groups.items()], betas=(.9,.999), eps=1e-8)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
    raw = model.module
    ema = {k: v.detach().clone() for k,v in raw.train_state().items()}
    parameter_names = {n for n,p in raw.named_parameters() if p.requires_grad}
    step = attempt = 0
    if a.resume:
        state = torch.load(a.resume/'trainer.pt', map_location='cpu', weights_only=False)
        raw.load_train_state(torch.load(a.resume/'model.pth', map_location=device, weights_only=True))
        ema = torch.load(a.resume/'ema.pth', map_location=device, weights_only=True)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        step, attempt = state['step'], state['attempt']
        set_rng(state['rngs'][rank])
    if rank == 0:
        parent = PARENT0 if a.arm.startswith('H-D') else PARENT
        atomic_json(out/'config.json', dict(arm=a.arm, limit=a.limit, global_batch=64, per_gpu=4, accumulation=2,
            parent=str(parent), parent_unet_sha256=sha256_file(parent/'unet.pth'), code_root=str(CODE),
            code_sha256={str(p.relative_to(CODE)):sha256_file(p) for p in
                [CODE/'scripts/train_h.py',CODE/'scripts/hrfont_h.py',CODE/'scripts/h_runtime.py',CODE/'scripts/h_eval.py']},
            cache_manifest_sha256=sha256_file(CACHE/'manifest.json'), clean_sha256=T._clean_map_sha256(args.v0913_clean_map),
            seed=3407, schedule='500 warmup; flat to5k; cosine to10% at10k', unet_lr=2e-5, new_lr=1e-4,
            comp_weight=0 if a.arm in ('H0','H1','H2','H4') else .01, transition=a.transition,
            joint_cfg=.1, source_drop=.25, precision='fp16 scaler1024 FP32 logits/loss/optimizer',
            native_resolution=96, teacher='train-only VGG enc2 pool12 with fixed channel calibration',
            smoke=a.smoke, overfit=a.overfit, resume=str(a.resume) if a.resume else None))
    params = [p for p in raw.parameters() if p.requires_grad]
    skips = 0
    begin = time.time()
    while step < a.limit:
        raw.train()
        raw.base.style_encoder.eval()
        raw.base.content_encoder.eval()
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        key = 0 if a.overfit else attempt
        indices = torch.multinomial(weights, 64, replacement=True,
            generator=torch.Generator().manual_seed(3407+key)).tolist()
        total = torch.zeros(4, device=device)
        for micro in range(2):
            seed_episode(3407 + key*1000 + micro*world + rank)
            ids = indices[(micro*world+rank)*4:(micro*world+rank+1)*4]
            samples = batch_to([train[i] for i in ids], device)
            source = torch.rand(4, device=device)<.25
            _support_draw = torch.rand(4, device=device)
            cfg = torch.rand(4, device=device)<.1
            style, refs, query, keep, content, structure = data.conditions(samples, cfg, source, a.arm=='H-D-')
            noise = torch.randn_like(samples['target_image'])
            t = torch.randint(0, noise_scheduler.num_train_timesteps, (4,), device=device)
            noisy = noise_scheduler.add_noise(samples['target_image'], noise, t)
            gate = min(1., (step+1)/a.transition)
            with model.no_sync() if micro == 0 else contextlib.nullcontext():
                with torch.autocast('cuda', dtype=torch.float16):
                    pred, offset, appearance = model(noisy, t, style, refs, query, keep, content, structure, cfg, gate)
                    x0 = T.reNormalize_img(T.x0_from_epsilon(noise_scheduler, pred, noisy, t))
                    gen = vgg(T.normalize_mean_std(x0))
                    with torch.no_grad():
                        gt = vgg(T.normalize_mean_std(samples['nonorm_target_image']))
                    diffusion = F.mse_loss(pred.float(), noise.float())
                    perceptual = sum(F.mse_loss(x.float(),y.float()) for x,y in zip(gen,gt))/3
                    comp = pred.float().sum()*0
                    if appearance is not None:
                        teacher = F.adaptive_avg_pool2d(gt[1].float(),12).flatten(2).transpose(1,2)
                        teacher = (teacher-raw.teacher_mean)/raw.teacher_std
                        comp = F.smooth_l1_loss(appearance.float(), teacher.detach())
                    coef = 0. if a.arm in ('H0','H1','H2','H4') else .01 * min(1.,(step+1)/1000)
                    loss = diffusion + .01*perceptual + .5*(offset.float()/2) + coef*comp
                if not torch.isfinite(loss):
                    raise RuntimeError(f'nonfinite loss rank{rank} update{step}')
                scaler.scale(loss/2).backward()
            total += torch.stack([loss.detach(), diffusion.detach(), perceptual.detach(), comp.detach()]).float()/2
        attempt += 1
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(params, 1.)
        finite = torch.tensor(float(torch.isfinite(grad)), device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            scaler.update(new_scale=scaler.get_scale()/2)
            skips += 1
            if rank == 0:
                with (out/'amp_skips.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step,attempt=attempt,scale=scaler.get_scale()))+'\n')
            if skips > 10 and skips/attempt > .01:
                raise RuntimeError('AMP skip rate exceeds1%; stop for diagnosis')
            continue
        for group in optimizer.param_groups:
            group['lr'] = group['peak_lr'] * lr_factor(step+1)
        reader_grad = sum(float(p.grad.float().square().sum()) for n,p in raw.named_parameters()
                          if n.startswith(('reader.','local.')) and p.grad is not None)**.5
        scaler.step(optimizer)
        scaler.update()
        step += 1
        with torch.no_grad():
            decay = min(.999, 1-1/(step+1))
            for name, value in raw.train_state().items():
                if name in parameter_names:
                    ema[name].lerp_(value.detach(), 1-decay)
                else:
                    ema[name].copy_(value)
        if step <= 10 or step % 100 == 0:
            dist.all_reduce(total)
            total /= world
            # DDP correctness: cross-rank trainable module fingerprints.
            fingerprint = torch.stack([p.detach().float().sum() for n,p in raw.named_parameters()
                if p.requires_grad and (n.startswith(('reader.','local.')) or n=='base.unet.conv_in.weight')])
            hi, lo = fingerprint.clone(), fingerprint.clone()
            dist.all_reduce(hi,op=dist.ReduceOp.MAX)
            dist.all_reduce(lo,op=dist.ReduceOp.MIN)
            spread = float((hi-lo).abs().max())
            if spread > 1e-4:
                raise RuntimeError(f'DDP weight divergence {spread}')
            record = dict(step=step, attempt=attempt, loss=float(total[0]), eps=float(total[1]),
                perceptual=float(total[2]), completion=float(total[3]), comp_coef=coef,
                grad_norm=float(grad), reader_grad=reader_grad, gate=gate,
                lr=[g['lr'] for g in optimizer.param_groups], skips=skips, scale=scaler.get_scale(),
                ddp_spread=spread, update_seconds=time.time()-start,
                peak_mib=torch.cuda.max_memory_allocated()/2**20, elapsed_seconds=time.time()-begin)
            with (out/f'rank{rank}.jsonl').open('a') as f:
                f.write(json.dumps(dict(record,fonts=samples['font_stem'],chars=samples['char_cp']))+'\n')
            if rank == 0:
                with (out/'train_log.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
                atomic_json(out/'heartbeat.json', dict(state='TRAINING',time=time.time(),**record))
                print('H_UPDATE', json.dumps(record), flush=True)
        final = step == a.limit
        infer = not a.smoke and (step % a.eval_interval == 0 or final)
        stop = torch.tensor(int((out/'STOP').exists()), device=device)
        dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if step%a.state_interval == 0 or final or infer or stop.item():
            checkpoint(raw,ema,optimizer,scaler,out,step,attempt,milestone=infer or final)
        if infer:
            saved_rng = rng_state()
            backup = {k:v.detach().cpu().clone() for k,v in raw.train_state().items()}
            raw.load_train_state(ema)
            if rank == 0:
                atomic_json(out/'heartbeat.json',dict(state='INFERENCE',step=step,time=time.time()))
            evaluate(raw,data,val,noise_scheduler,out/f'eval_step_{step}',step,standard=final)
            raw.load_train_state(backup)
            del backup
            set_rng(saved_rng)
        if stop.item():
            if rank == 0:
                atomic_json(out/'STOPPED.json',dict(step=step,attempt=attempt,status='safely_stopped'))
            dist.destroy_process_group()
            return
    if rank == 0:
        if not a.smoke:
            assert json.loads((out/f'eval_step_{step}/DONE.json').read_text())['images'] == 4096
        atomic_json(out/'DONE.json',dict(status='completed',step=step,attempt=attempt,arm=a.arm,
            inference_complete=not a.smoke,smoke=a.smoke,seconds=time.time()-begin))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
