"""Read-only I0-gradient calibration on training batches; no model update."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from i56_runtime import ROOT, CODE, PARENT0, model_for, DataContext, dataset, batch_to, seed_episode, T, atomic_json, sha256_file
from scripts.i56_components import raw_x0, region_detail_distance
from scripts.i56_sampling import load_sampling_manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists();a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    model,args=model_for('I5',device,a.out.parent/'calibration_load');model.train()
    model.base.style_encoder.eval();model.base.content_encoder.eval()
    ds=dataset(args,'train');data=DataContext(args,device)
    target,flags,keys,digest=load_sampling_manifest(a.manifest,ds,args)
    sched=T.build_ddpm_scheduler(args);vgg=T.ContentPerceptualLoss().VGG.to(device).eval().requires_grad_(False)
    # Same parameters for both gradients. Explicit subset, not a claim about full-network norms.
    selected=[(n,p) for n,p in model.named_parameters() if p.requires_grad and
        (n.startswith('base.unet.conv_out.') or (n.startswith('base.unet.up_blocks.') and
         '.attn2.' in n and (n.endswith('to_q.weight') or n.endswith('to_out.0.weight'))))]
    assert selected
    params=[p for _,p in selected];records=[]
    for index in range(32):
        seed_episode(93407+index)
        weights=target.clone()
        if index%2==0:weights*=torch.tensor(flags,dtype=torch.float64)
        ids=torch.multinomial(weights,4,replacement=True).tolist()
        batch=batch_to([ds[i] for i in ids],device)
        cfg=torch.zeros(4,dtype=torch.bool,device=device);source=torch.rand(4,device=device)<.25
        style,refs,q,keep,content,structure=data.conditions(batch,cfg,source)
        noise=torch.randn_like(batch['target_image'])
        # Equal quartiles reproduce uniform timestep mass while exposing extremes.
        bucket=index%4;t=torch.randint(bucket*250,(bucket+1)*250,(4,),device=device)
        noisy=sched.add_noise(batch['target_image'],noise,t)
        with torch.autocast('cuda',dtype=torch.float16):
            pred,offset,_=model(noisy,t,style,refs,q,keep,content,structure,cfg,1.)
            x=T.reNormalize_img(T.x0_from_epsilon(sched,pred,noisy,t))
            gen=vgg(T.normalize_mean_std(x))
            with torch.no_grad():gt=vgg(T.normalize_mean_std(batch['nonorm_target_image']))
            base=F.mse_loss(pred.float(),noise.float())+.01*sum(F.mse_loss(u.float(),v.float()) for u,v in zip(gen,gt))/3+.25*offset.float()
            clean,alpha=raw_x0(noisy,pred,sched.alphas_cumprod,t)
            render=(region_detail_distance(clean,batch['nonorm_target_image'])*alpha).mean()
        # Match the fp16 scaler regime; unscaled fp16 backward can underflow.
        gb=torch.autograd.grad(base*1024.,params,retain_graph=True,allow_unused=True)
        gr=torch.autograd.grad(render*1024.,params,allow_unused=True)
        gb=tuple(None if x is None else x.float()/1024. for x in gb)
        gr=tuple(None if x is None else x.float()/1024. for x in gr)
        bn=sum(float(x.float().square().sum()) for x in gb if x is not None)**.5
        rn=sum(float(x.float().square().sum()) for x in gr if x is not None)**.5
        dot=sum(float((x.float()*y.float()).sum()) for x,y in zip(gb,gr) if x is not None and y is not None)
        assert all(math.isfinite(x) for x in (bn,rn,dot)) and bn>0 and rn>0
        rec=dict(batch=index,t_bucket=bucket,base_norm=bn,render_unit_norm=rn,unit_ratio=rn/bn,
            cosine=dot/(bn*rn),base_loss=float(base),render_loss=float(render),fonts=batch['font_stem'])
        records.append(rec);print('CALIBRATE',json.dumps(rec),flush=True)
        del gb,gr,base,render,pred,gen,gt
    candidates={str(w):dict(median=float(np.median([w*r['unit_ratio'] for r in records])),
        p95=float(np.quantile([w*r['unit_ratio'] for r in records],.95))) for w in (.1,.2,.5)}
    eligible=[w for w in (.1,.2,.5) if .02<=candidates[str(w)]['median']<=.8 and candidates[str(w)]['p95']<=2.]
    if not eligible:
        atomic_json(a.out.parent/'calibration_failed.json',dict(status='REVIEW_REQUIRED',candidates=candidates,records=records))
        raise RuntimeError('No candidate has acceptable measured gradient strength; review, do not auto-launch')
    chosen=min(eligible,key=lambda w:abs(math.log(candidates[str(w)]['median']/.3)))
    atomic_json(a.out,dict(status='CALIBRATED',arm='I5',render_weight=chosen,detail_sha256=digest,
        parent_unet_sha256=sha256_file(PARENT0/'unet.pth'),parameter_subset=[n for n,_ in selected],
        objective_code_sha256={f'scripts/{n}':sha256_file(CODE/'scripts'/n) for n in
            ('i56_components.py','i56_sampling.py','i56_runtime.py','i34_components.py','i34_runtime.py','hrfont_i.py','i_runtime.py')},
        candidates=candidates,target_median_ratio=.3,acceptable_median_interval=[.02,.8],max_p95=2.,
        records=records,note='Fixed32 training batches; I0 initial weights, no optimizer update; full-strength gate; shared-parameter subset norms, not total-network gradient norms'))
    print('CALIBRATION_COMPLETE',chosen,candidates,flush=True)


if __name__=='__main__':main()
