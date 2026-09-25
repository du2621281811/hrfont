"""Actual TC readout + DPM20 generation on fixed overfit-training episodes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from i56_runtime import ROOT, model_for, DataContext, dataset, batch_to, seed_episode, T, atomic_json, sha256_file
from scripts.i56_sampling import load_sampling_manifest, episode_indices
from scripts.i56_components import region_detail_distance
from i_eval import sample


def image_tensor(array,device):
    return torch.as_tensor(np.asarray(array).copy(),device=device).permute(2,0,1).float()[None]/255


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--checkpoint',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    model,args=model_for('I5',device,a.out/'load')
    step=0
    if a.checkpoint:
        meta=json.loads((a.checkpoint/'checkpoint.json').read_text());step=meta['step']
        model.load_train_state(torch.load(a.checkpoint/'ema.pth',map_location=device,weights_only=True))
    model.eval();data=DataContext(args,device);ds=dataset(args,'train')
    target,flags,keys,digest=load_sampling_manifest(a.manifest,ds,args)
    ids=episode_indices(ds.sample_weights,target,flags,0,0,True)
    scheduler=T.build_ddpm_scheduler(args);rows=[]
    # Reconstruct the first two samples of each rank's fixed8 training batch.
    for rank in range(8):
        seed_episode(3407+rank)
        samples=[ds[i] for i in ids[rank*8:(rank+1)*8]]
        for j,s in enumerate(samples[:2]):
            b=batch_to([s],device);mask=torch.zeros(1,dtype=torch.bool,device=device)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                style,refs,q,keep,_,_=data.conditions(b,mask,mask)
                tokens,_=model.reader(refs,q,keep)
                tc=model.reader.ink(tokens)  # Explicit readout; eval placeholder is not a prediction.
                pred=sample(model,data,b,scheduler,3407,1.)
                gain=model.local_gain.detach().clone()
                model.local_gain.zero_()
                off=sample(model,data,b,scheduler,3407,1.)
                model.local_gain.copy_(gain)
                target_image=b['nonorm_target_image']
                tc_d=float(region_detail_distance(tc,target_image))
                pred_d=float(region_detail_distance(image_tensor(pred,device),target_image))
            ident=f'{rank}_{j}';tc_array=(tc[0,0].float().clamp(0,1).cpu().numpy()*255).round().astype('uint8')
            Image.fromarray(pred).save(a.out/f'{ident}_pred.png');Image.fromarray(off).save(a.out/f'{ident}_local_off.png')
            Image.fromarray(tc_array).save(a.out/f'{ident}_tc.png')
            with Image.open(s['target_image_path']) as im:im.save(a.out/f'{ident}_gt.png')
            with Image.open(s['ref_image_paths'][0]) as im:im.save(a.out/f'{ident}_ref0.png')
            row=dict(id=ident,font=s['font_stem'],cp=s['char_cp'],refs=s['ref_chars'],
                tc_detail=tc_d,pred_detail=pred_d,local_intervention_l1=float(np.abs(pred.astype(float)-off.astype(float)).mean()/255),
                finite=bool(np.isfinite(pred).all()),pred_ink_fraction=float((pred.mean(2)<242).mean()))
            rows.append(row);print('DIAGNOSTIC',json.dumps(row),flush=True)
    canvas=Image.new('RGB',(660,40+110*len(rows)),'white');draw=ImageDraw.Draw(canvas)
    for col,label in enumerate(('Ref0','GT','TC readout','DPM20','local off')):draw.text((155+100*col,10),label,fill='black')
    for i,row in enumerate(rows):
        y=35+i*110;draw.text((2,y+20),row['font'][:22],fill='black');draw.text((2,y+40),row['cp'],fill='black')
        for col,kind in enumerate(('ref0','gt','tc','pred','local_off')):
            with Image.open(a.out/f'{row["id"]}_{kind}.png') as im:canvas.paste(im.convert('RGB'),(155+100*col,y))
    canvas.save(a.out/'review.png')
    atomic_json(a.out/'metrics.json',dict(step=step,checkpoint=str(a.checkpoint) if a.checkpoint else 'I0-init',
        detail_sha256=digest,rows=rows,mean_tc=float(np.mean([r['tc_detail'] for r in rows])),
        mean_pred=float(np.mean([r['pred_detail'] for r in rows])),
        warning='Seen training episodes; optimization/condition-use diagnostic, NOT generalization evidence'))


if __name__=='__main__':main()
