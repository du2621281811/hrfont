#!/usr/bin/env python3
"""训练跨语系风格编码器 φ_s2（对称 InfoNCE）。"""
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import *
from data import *
from models import PhiS2
from train_utils import *

def evaluate(model, dataset, device):
    model.eval(); scores=[]; labels=[]
    with torch.no_grad():
        families=dataset.families
        for i in range(min(len(dataset),128)):
            a,b,f=dataset[i]; scores.append(float((model(a[None].to(device))*model(b[None].to(device))).sum())); labels.append(1)
            wrong=families[(families.index(f)+1)%len(families)]
            if (wrong,dataset.latin[i%len(dataset.latin)]) in dataset.by:
                wb,_=dataset.glyphs[dataset.by[(wrong,dataset.latin[i%len(dataset.latin)])]]; scores.append(float((model(a[None].to(device))*model(wb[None].to(device))).sum())); labels.append(0)
    return binary_auc(labels,scores)

def main(argv=None):
    cli=parse_cli(argv); cfg=load_config(cli["config"],cli.get("override"),cli["set"]); seed_all(cfg.train.seed); out=Path(cfg.output.dir); sha=save_resolved_config(cfg,out,cli["config"])
    manifest=load_manifest(cfg.data.cache_dir); splits=split_families(manifest,cfg.data.split_ratios,cfg.data.split_seed); assert_disjoint_splits(splits)
    train=CrossScriptPairDataset(cfg.data.cache_dir,cfg.data.chinese_chars,cfg.data.latin_chars,splits["train"],cfg.model.in_channels,cfg.train.seed)
    val_f=splits["val"] or splits["test"] or splits["train"]; val=CrossScriptPairDataset(cfg.data.cache_dir,cfg.data.chinese_chars,cfg.data.latin_chars,val_f,cfg.model.in_channels,cfg.train.seed)
    loader=DataLoader(train,batch_size=min(cfg.train.batch_size,len(train)),shuffle=True,generator=make_generator(cfg.train.seed),num_workers=cfg.data.workers,worker_init_fn=worker_init_fn(cfg.train.seed))
    device=device_from_config(cfg); model=PhiS2(cfg.model.in_channels,cfg.model.feature_dim).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.train.lr,betas=tuple(cfg.train.betas),weight_decay=cfg.train.weight_decay,eps=cfg.train.eps)
    total=cfg.train.max_steps if cfg.train.max_steps>0 else cfg.train.epochs*max(1,len(loader)); warm=cfg.train.warmup_steps if cfg.train.warmup_steps>=0 else cfg.train.warmup_epochs*max(1,len(loader)); sch=scheduler(opt,total,warm); scaler=torch.amp.GradScaler("cuda",enabled=cfg.train.fp16 and device.type=="cuda"); log=JSONLLogger(out/"curves.jsonl"); best=-1.; iterator=infinite(loader)
    for step in range(1,total+1):
        model.train(); a,b,_=next(iterator); a,b=a.to(device),b.to(device); opt.zero_grad(set_to_none=True)
        with amp_context(device,cfg.train.fp16):
            za,zb=model(a),model(b); logits=za@zb.T/cfg.train.temperature; target=torch.arange(len(a),device=device); loss=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.grad_clip); scaler.step(opt); scaler.update(); sch.step(); log.log(step=step,loss=float(loss),lr=sch.get_last_lr()[0])
        if step%cfg.train.checkpoint_every==0 or step==total:
            auc=evaluate(model,val,device); payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sch.state_dict(),"step":step,"config_sha256":sha,"val_auc":auc}; torch.save(payload,out/f"step_{step:06d}.pt")
            if auc>=best: best=auc; torch.save(payload,out/"best.pt")
    print(f"phi_s2 complete steps={total} best_auc={best:.4f} device={device}")
if __name__=="__main__": main()
