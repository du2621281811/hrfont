#!/usr/bin/env python3
"""训练跨语系风格编码器 φ_s2（对称 InfoNCE）。"""
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import *
from data import *
from data import group_table, pick_negative, as_chars
from models import PhiS2
from train_utils import *

def evaluate(model, dataset, device, groups=None, cross_group=True, neg_pool=None):
    """Val AUC with cross-typeface negatives drawn from `neg_pool`.

    Two problems this fixes. The old negative was the alphabetical neighbour
    (`families[(i+1)%n]`), which on a weight-heavy pool is another weight of the same
    typeface -- an unwinnable negative that says nothing about the encoder. And once
    the split is by typeface group, the val split can be a single group, so a
    cross-typeface negative does not exist inside val at all.

    `neg_pool` spans train+val and never test: `best.pt` is selected on this number, so
    test fonts here would leak into checkpoint selection. Negatives are therefore fonts
    the encoder has seen, while the anchor stays held out.
    """
    model.eval(); scores=[]; labels=[]
    pool = neg_pool if neg_pool is not None else dataset
    with torch.no_grad():
        for i in range(min(len(dataset),128)):
            a,b,f=dataset[i]; scores.append(float((model(a[None].to(device))*model(b[None].to(device))).sum())); labels.append(1)
            wrong=pick_negative(f,pool.families,groups,cross_group)
            if wrong is None: continue
            key=(wrong,dataset.latin[i%len(dataset.latin)])
            if key in pool.by:
                wb,_=pool.glyphs[pool.by[key]]; scores.append(float((model(a[None].to(device))*model(wb[None].to(device))).sum())); labels.append(0)
    if 0 not in labels:
        raise RuntimeError("val AUC has no negatives: widen neg_pool or disable split_by_group")
    return binary_auc(labels,scores)

def main(argv=None):
    cli=parse_cli(argv); cfg=load_config(cli["config"],cli.get("override"),cli["set"]); seed_all(cfg.train.seed); out=Path(cfg.output.dir); sha=save_resolved_config(cfg,out,cli["config"])
    manifest=load_manifest(cfg.data.cache_dir); by_group=bool(getattr(cfg.data,"split_by_group",True)); splits=split_families(manifest,cfg.data.split_ratios,cfg.data.split_seed,by_group=by_group); groups=group_table(manifest); assert_disjoint_splits(splits,groups if by_group else None)
    train=CrossScriptPairDataset(cfg.data.cache_dir,as_chars(cfg.data.chinese_chars),as_chars(cfg.data.latin_chars),splits["train"],cfg.model.in_channels,cfg.train.seed)
    val_f=splits["val"] or splits["test"] or splits["train"]; val=CrossScriptPairDataset(cfg.data.cache_dir,as_chars(cfg.data.chinese_chars),as_chars(cfg.data.latin_chars),val_f,cfg.model.in_channels,cfg.train.seed)
    neg_pool=CrossScriptPairDataset(cfg.data.cache_dir,as_chars(cfg.data.chinese_chars),as_chars(cfg.data.latin_chars),sorted(set(splits["train"])|set(val_f)),cfg.model.in_channels,cfg.train.seed)
    loader=DataLoader(train,batch_size=min(cfg.train.batch_size,len(train)),shuffle=True,generator=make_generator(cfg.train.seed),num_workers=cfg.data.workers,worker_init_fn=worker_init_fn(cfg.train.seed))
    device=device_from_config(cfg); model=PhiS2(cfg.model.in_channels,cfg.model.feature_dim).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.train.lr,betas=tuple(cfg.train.betas),weight_decay=cfg.train.weight_decay,eps=cfg.train.eps)
    total=cfg.train.max_steps if cfg.train.max_steps>0 else cfg.train.epochs*max(1,len(loader)); warm=cfg.train.warmup_steps if cfg.train.warmup_steps>=0 else cfg.train.warmup_epochs*max(1,len(loader)); sch=scheduler(opt,total,warm); scaler=torch.amp.GradScaler("cuda",enabled=cfg.train.fp16 and device.type=="cuda"); log=JSONLLogger(out/"curves.jsonl"); best=-1.; stale=0; patience=int(getattr(cfg.train,"early_stop_evals",0) or 0); iterator=infinite(loader)
    for step in range(1,total+1):
        model.train(); a,b,_=next(iterator); a,b=a.to(device),b.to(device); opt.zero_grad(set_to_none=True)
        with amp_context(device,cfg.train.fp16):
            za,zb=model(a),model(b); logits=za@zb.T/cfg.train.temperature; target=torch.arange(len(a),device=device); loss=(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.grad_clip); scaler.step(opt); scaler.update(); sch.step(); log.log(step=step,loss=float(loss),lr=sch.get_last_lr()[0])
        if step%cfg.train.checkpoint_every==0 or step==total:
            auc=evaluate(model,val,device,groups,by_group,neg_pool); payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sch.state_dict(),"step":step,"config_sha256":sha,"val_auc":auc}; torch.save(payload,out/f"step_{step:06d}.pt")
            log.log(step=step,val_auc=auc)
            if auc>=best: best=auc; stale=0; torch.save(payload,out/"best.pt")
            else: stale+=1
            # Train loss keeps falling long after val AUC stops moving on a small pool;
            # select on val AUC, never on train loss.
            if patience and stale>=patience:
                print(f"early stop at step={step}: val AUC flat for {stale} evals"); break
    print(f"phi_s2 complete steps={step} best_auc={best:.4f} device={device}")
if __name__=="__main__": main()
