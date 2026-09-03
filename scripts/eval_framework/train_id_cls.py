#!/usr/bin/env python3
"""训练字符身份分类器；严格按字体族拆分。"""
from pathlib import Path
import json, torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import *
from data import *
from models import IDCLS
from train_utils import *

def evaluate(model,ds,device,n):
    cm=torch.zeros(n,n,dtype=torch.long); model.eval()
    with torch.no_grad():
        for x,y,_,_ in DataLoader(ds,batch_size=64):
            pred=model(x.to(device)).argmax(1).cpu()
            for a,b in zip(y,pred): cm[a,b]+=1
    per=(cm.diag()/cm.sum(1).clamp_min(1)).tolist(); return {"macro_accuracy":sum(per)/len(per),"per_class_accuracy":per,"confusion_matrix":cm.tolist()}

def main(argv=None):
    cli=parse_cli(argv); cfg=load_config(cli["config"],cli.get("override"),cli["set"]); seed_all(cfg.train.seed); out=Path(cfg.output.dir); sha=save_resolved_config(cfg,out,cli["config"])
    splits=split_families(load_manifest(cfg.data.cache_dir),cfg.data.split_ratios,cfg.data.split_seed); assert_disjoint_splits(splits)
    train=GlyphClassDataset(cfg.data.cache_dir,cfg.data.chars,splits["train"],cfg.model.in_channels); vf=splits["val"] or splits["test"] or splits["train"]; val=GlyphClassDataset(cfg.data.cache_dir,cfg.data.chars,vf,cfg.model.in_channels)
    loader=DataLoader(train,batch_size=min(cfg.train.batch_size,len(train)),shuffle=True,generator=make_generator(cfg.train.seed),num_workers=cfg.data.workers); device=device_from_config(cfg); model=IDCLS(len(cfg.data.chars),cfg.model.in_channels,cfg.model.feature_dim).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.train.lr,betas=tuple(cfg.train.betas),weight_decay=cfg.train.weight_decay,eps=cfg.train.eps); total=cfg.train.max_steps if cfg.train.max_steps>0 else cfg.train.epochs*len(loader); warm=cfg.train.warmup_steps if cfg.train.warmup_steps>=0 else cfg.train.warmup_epochs*len(loader); sch=scheduler(opt,total,warm); it=infinite(loader); log=JSONLLogger(out/"curves.jsonl")
    for step in range(1,total+1):
        x,y,_,_=next(it); x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=F.cross_entropy(logits,y,label_smoothing=cfg.train.label_smoothing); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.grad_clip); opt.step(); sch.step(); log.log(step=step,loss=float(loss),lr=sch.get_last_lr()[0])
        if step%cfg.train.checkpoint_every==0 or step==total:
            metrics=evaluate(model,val,device,len(cfg.data.chars)); (out/"val_metrics.json").write_text(json.dumps(metrics,indent=2)); torch.save({"model":model.state_dict(),"step":step,"chars":list(cfg.data.chars),"config_sha256":sha},out/"best.pt")
    print(f"id_cls complete steps={total} macro={metrics['macro_accuracy']:.4f} device={device}")
if __name__=="__main__": main()
