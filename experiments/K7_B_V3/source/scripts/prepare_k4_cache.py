"""Incremental frozen-encoder cache for the authorized v2 experiment.

Only 0917 reference images and train-only donor targets are encoded. Existing
0913 features are read through by k4_runtime, never copied or rewritten.
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from h_runtime import ROOT, PARENT0, T, args_for, atomic_json, sha256_file
from scripts.hrfont_feature_cache import EC_SHAPES, ES_SPATIAL

DATA = Path('/root/data1/hrfont_dataset_v2_20260917')
DEST = Path('/root/data1/hrfont_k4_20260917/cache')

def jobs_for(kind):
    mapping = DATA/'manifests/v0917'
    if kind == 'es':
        split = json.loads((mapping/'split.json').read_text())['stems']
        pools = json.loads((mapping/'style_pool.json').read_text())
        return [(f'es|{sp}|{f}|{c}', DATA/'v0917'/sp/'StyleImage'/f/f'{f}+{c}.png')
                for sp in ('train','val','test') for f in sorted(split[sp]) for c in sorted(pools[f])]
    with (mapping/'pairs_train.tsv').open() as h:
        return [(f"ec|target|{r['font']}|{r['cp']}", DATA/'v0917/train/TargetImage'/r['font']/f"{r['font']}+{r['cp']}.png")
                for r in csv.DictReader(h,delimiter='\t')]

def read_image(path):
    with Image.open(path) as im:
        assert im.size == (96,96) and im.mode=='RGB', str(path)
        return torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float()/127.5-1

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['es','ec'],required=True)
    p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(a.gpu)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda',a.gpu);out=DEST/a.kind;out.mkdir(parents=True,exist_ok=True)
    jobs=jobs_for(a.kind);keys=[k for k,_ in jobs]
    assert len(keys)==len(set(keys))==({'es':78070,'ec':39547}[a.kind])
    keytext='\n'.join(keys)+'\n';kp=out/'keys.txt'
    if kp.exists():assert kp.read_text()==keytext
    else:kp.write_text(keytext)
    args=args_for(PARENT0)
    enc=T.build_style_encoder(args) if a.kind=='es' else T.build_content_encoder(args)
    weight=PARENT0/('style_encoder.pth' if a.kind=='es' else 'content_encoder.pth')
    enc.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    enc=enc.to(device).eval().requires_grad_(False)
    source_sha={n:sha256_file(DATA/'manifests/v0917'/n) for n in ['split.json','style_pool.json','pairs_train.tsv']}
    identity=dict(kind=a.kind,entries=len(keys),encoder_sha256=sha256_file(weight),source_sha256=source_sha,
                  keys_sha256=sha256_file(kp),precision='float32 encoder / float16 storage',levels=[1,2] if a.kind=='ec' else None)
    ip=out/'identity.json'
    if ip.exists():assert json.loads(ip.read_text())==identity
    else:atomic_json(ip,identity)
    if (out/'COMPLETE.json').exists():
        m=json.loads((out/'COMPLETE.json').read_text());assert m['identity']==identity
        for name,h in m['files_sha256'].items():assert sha256_file(out/name)==h
        return
    specs={'spatial':ES_SPATIAL,'pooled':(1024,)} if a.kind=='es' else {f's{i}':EC_SHAPES[i] for i in [1,2]}
    arrays={}
    for n,shape in specs.items():
        file=out/(n+'.dat');mode='r+' if file.exists() else 'w+'
        if file.exists():assert file.stat().st_size==len(keys)*int(np.prod(shape))*2
        arrays[n]=np.memmap(file,dtype=np.float16,mode=mode,shape=(len(keys),*shape))
    progress=out/'progress.json';done=json.loads(progress.read_text())['done'] if progress.exists() else 0
    start=time.time();pool=ThreadPoolExecutor(max_workers=8);batch=64 if a.kind=='es' else 32
    with torch.no_grad():
        for st in range(done,len(jobs),batch):
            chunk=jobs[st:st+batch];x=torch.stack(list(pool.map(read_image,[p for _,p in chunk]))).to(device)
            if a.kind=='es':
                spatial,pooled,_=enc(x);values={'spatial':spatial,'pooled':F.normalize(pooled.float(),dim=1)}
            else:
                final,residuals=enc(x);values={f's{i}':residuals[i] for i in [1,2]}
            for n,v in values.items():
                assert torch.isfinite(v).all()
                arrays[n][st:st+len(chunk)]=v.detach().half().cpu().numpy()
            if st//batch%16==0 or st+len(chunk)==len(jobs):
                for v in arrays.values():v.flush()
                atomic_json(progress,dict(done=st+len(chunk),total=len(jobs),seconds=time.time()-start))
                print(a.kind,st+len(chunk),len(jobs),flush=True)
    # Cache parity against fresh encoder computation after disk writes.
    checks=[]
    with torch.no_grad():
        for i in [0,len(jobs)//2,len(jobs)-1]:
            x=read_image(jobs[i][1])[None].to(device)
            if a.kind=='es':
                s,h,_=enc(x);values={'spatial':s,'pooled':F.normalize(h.float(),dim=1)}
            else:
                _,r=enc(x);values={f's{j}':r[j] for j in [1,2]}
            for n,v in values.items():
                delta=np.abs(np.asarray(arrays[n][i],dtype=np.float32)-v[0].float().cpu().numpy())
                relative=float(delta.mean()/max(float(v.abs().mean()),1e-8))
                assert relative<.005,(n,i,relative)
                checks.append(dict(row=i,array=n,relative_mae=relative))
    files={n+'.dat':sha256_file(out/(n+'.dat')) for n in arrays}
    atomic_json(out/'COMPLETE.json',dict(status='COMPLETE',identity=identity,files_sha256=files,checks=checks))

if __name__=='__main__':main()
