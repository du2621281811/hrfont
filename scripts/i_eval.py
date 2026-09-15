"""Immediate distributed DPM20 inference; fixed nested reference episodes."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from skimage.metrics import structural_similarity
from i_runtime import T, batch_to, atomic_json
from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP


def episodes(ds, standard=False):
    by_font = {}
    for i, p in enumerate(ds.target_images):
        path = Path(p)
        font = path.parent.name
        cp = path.stem[len(font)+1:]
        by_font.setdefault(font, {})[cp] = i
    jobs = []
    preferred = list('0Agé')
    if standard:
        preferred += list('1BamZあカアㄅㄆㄇㄈ')
    for font, available in sorted(by_font.items()):
        # Fixed selection from valid pairs only, never from visual outputs.
        chosen = [f'u{ord(c):04X}' for c in preferred if f'u{ord(c):04X}' in available]
        count = 16 if standard else 4
        chosen += [c for c in sorted(available) if c not in chosen][:max(0, count-len(chosen))]
        chosen = chosen[:count]
        assert len(chosen) == count
        pool = sorted(ds.style_by_font_char[font])
        base = [f'u{ord(c):04X}' for c in '永和书风骨韵天地']
        assert all(c in pool for c in base)
        refsets = [base]
        for n in range(1, 4):
            refsets.append(np.random.default_rng(3407+n).choice(pool, 8, replace=False).tolist())
        for cp in chosen:
            for group, refs in enumerate(refsets if standard else refsets[:1]):
                for k in ((1, 2, 4, 8) if standard else (1, 4, 8)):
                    jobs.append(dict(index=available[cp], font=font, cp=cp, group=group, k=k, refs=refs[:k]))
    assert len(jobs) == (4096 if standard else 192)
    return jobs


@torch.no_grad()
def sample(model, data, samples, noise_scheduler, seed, gate=1.):
    device = data.device
    cfg = torch.zeros(1, device=device, dtype=torch.bool)
    style, refs, query, keep, content, structure = data.conditions(samples, cfg, cfg, model.arm == 'H-D-')
    # Precompute reader only once, reuse memory for all denoising steps.
    ctx = model.conditions(style, refs, query, keep, cfg)
    un = model.conditions(style, refs, query, keep, ~cfg)
    context = tuple(torch.cat([u, c], 0) if c is not None else None for u, c in zip(un, ctx))
    styles = torch.cat([torch.zeros_like(style), style], 0)
    contents = [torch.cat([torch.zeros_like(c), c], 0) for c in content]
    # Donors/alpha are fixed candidates. Mask only active route for unconditional CFG.
    structures = [(torch.cat([d,d]),torch.cat([a,a]),torch.cat([c,c]),
                   torch.cat([torch.zeros_like(active),active])) for d,a,c,active in structure]
    schedule = NoiseScheduleVP(schedule='discrete', betas=noise_scheduler.betas)
    def model_fn(x, continuous_t):
        t = (continuous_t - 1. / schedule.total_N) * 1000.
        with torch.autocast('cuda', dtype=torch.float16):
            pred, _ = model.denoise(torch.cat([x, x]), torch.cat([t, t]), styles,
                                    contents, structures, context, gate)
        uncond, cond = pred.float().chunk(2)
        return uncond + 7.5 * (cond-uncond)
    solver = DPM_Solver(model_fn=model_fn, noise_schedule=schedule, algorithm_type='dpmsolver++')
    noise = torch.randn((1,3,96,96), device=device, generator=torch.Generator(device=device).manual_seed(seed))
    x = solver.sample(x=noise, steps=20, order=2, skip_type='time_uniform', method='multistep')
    return ((x[0].float()/2+.5).clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')


@torch.no_grad()
def evaluate(model, data, ds, scheduler, out, step, standard=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    out.mkdir(parents=True, exist_ok=True)
    jobs = episodes(ds, standard)
    if rank == 0:
        atomic_json(out/'protocol.json', dict(step=step, arm=model.arm,
            weights='G0b parent unet.pth unchanged' if model.arm=='I0' else 'EMA', jobs=jobs,
            seed=3407, cfg=7.5, dpm_steps=20, split='val', gate=min(1.,step/1000),
            note='nested refs within each set; Delta uses the same refs as the appearance path'))
    model.eval()
    rows = []
    begin = time.time()
    for j in jobs[rank::world]:
        s = ds[j['index']]
        s['ref_chars'] = j['refs']
        s['ref_image_paths'] = [str(ds.style_by_font_char[j['font']][c]) for c in j['refs']]
        b = batch_to([s], data.device)
        seed = 3407 + int(hashlib.sha256((j['font']+j['cp']).encode()).hexdigest()[:8],16) % 1000000
        with torch.autocast('cuda', dtype=torch.float16):
            pred = sample(model, data, b, scheduler, seed, min(1.,step/1000))
        name = f"{j['font']}__{j['cp']}__r{j['group']}__k{j['k']}.png"
        Image.fromarray(pred).save(out/name)
        gt = np.asarray(Image.open(s['target_image_path']).convert('L'), dtype='float32')/255
        gray = np.asarray(Image.fromarray(pred).convert('L'), dtype='float32')/255
        edge = lambda im: np.diff(im, axis=0, prepend=im[:1])**2 + np.diff(im, axis=1, prepend=im[:,:1])**2
        rows.append(dict(**j, png=name, seed=seed, target=s['target_image_path'],
            l1=float(np.abs(gray-gt).mean()), ssim=float(structural_similarity(gray,gt,data_range=1.,win_size=7)),
            edge_l1=float(np.abs(edge(gray)-edge(gt)).mean())))
        if len(rows)%20 == 0:
            print('INFER', rank, len(rows), len(jobs[rank::world]), flush=True)
    atomic_json(out/f'rank{rank}.json', dict(rows=rows, seconds=time.time()-begin))
    dist.barrier()
    if rank == 0:
        combined = sum([json.loads((out/f'rank{i}.json').read_text())['rows'] for i in range(world)], [])
        assert len(combined) == len(jobs) and all((out/r['png']).is_file() for r in combined)
        summary = {}
        for k in (1,2,4,8):
            selected = [r for r in combined if r['k']==k]
            if selected:
                summary[k] = {metric: float(np.mean([r[metric] for r in selected])) for metric in ('l1','ssim','edge_l1')}
        atomic_json(out/'metrics.json', dict(summary=summary, rows=combined))
        html = ['<!doctype html><meta charset="utf-8"><style>body{font:14px sans-serif}img{width:96px}td{padding:8px}</style>',
                f'<h1>{model.arm} step {step}</h1><p>Weights: see protocol.json; DPM20 / CFG7.5 / fixed val refs</p><table>']
        for r in sorted(combined, key=lambda r:(r['font'],r['cp'],r['group'],r['k'])):
            # Copy GT alongside generated images; HTML works outside the server.
            gt_name = f"GT__{r['font']}__{r['cp']}.png"
            if not (out/gt_name).exists():
                Image.open(r['target']).save(out/gt_name)
            html.append(f"<tr><td>{r['font']} {r['cp']} refs{r['group']} k{r['k']}</td><td>GT<img src='{gt_name}'></td><td>Pred<img src='{r['png']}'></td></tr>")
        (out/'review.html').write_text('\n'.join(html)+ '</table>')
        atomic_json(out/'DONE.json', dict(status='completed', images=len(combined), step=step, seconds=time.time()-begin))
    dist.barrier()
