#!/usr/bin/env python3
"""训练 style membership verifier，并在 held-out 字体族上做温度缩放。"""
from pathlib import Path
import json, math, torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import *
from data import *
from models import MembershipVerifier, load_phi_checkpoint
from train_utils import *

def collect(model,ds,device):
    logits=[]; labels=[]; model.eval()
    with torch.no_grad():
        for q,r,y,_,_ in DataLoader(ds,batch_size=32): logits.extend(model(q.to(device),r.to(device)).cpu().tolist()); labels.extend(y.tolist())
    return torch.tensor(logits),torch.tensor(labels)

def fit_temperature(logits,labels):
    log_t=torch.zeros((),requires_grad=True); opt=torch.optim.LBFGS([log_t],lr=.1,max_iter=50)
    def closure(): opt.zero_grad(); loss=F.binary_cross_entropy_with_logits(logits/log_t.exp(),labels); loss.backward(); return loss
    opt.step(closure); return float(log_t.exp().clamp(.05,20))

def metrics(logits,labels,temp=1.):
    probs=torch.sigmoid(logits/temp); return {"roc_auc":binary_auc(labels,probs),"pr_auc":pr_auc(labels,probs),"brier":float(((probs-labels)**2).mean()),"ece":ece(labels,probs),"temperature":temp}

def main(argv=None):
    cli=parse_cli(argv); cfg=load_config(cli["config"],cli.get("override"),cli["set"]); seed_all(cfg.train.seed); out=Path(cfg.output.dir); sha=save_resolved_config(cfg,out,cli["config"]); device=device_from_config(cfg)
    splits=split_families(load_manifest(cfg.data.cache_dir),cfg.data.split_ratios,cfg.data.split_seed); assert_disjoint_splits(splits); train_f=splits["train"]
    # tiny pools may have only one train family; borrow val solely in smoke mode, never silently in formal configs.
    if len(train_f)<2 and cfg.train.smoke_mode: train_f=(train_f+(splits["val"] or splits["test"]))[:2]
    train=MembershipDataset(cfg.data.cache_dir,cfg.data.query_chars,cfg.data.ref_chars,train_f,cfg.model.in_channels,cfg.train.seed,cfg.train.episodes)
    val_f=splits["val"]+splits["test"]; val_f=val_f if len(val_f)>=2 else list(load_manifest(cfg.data.cache_dir)["fonts"] and sorted({x["family"] for x in load_manifest(cfg.data.cache_dir)["fonts"]}))
    val=MembershipDataset(cfg.data.cache_dir,cfg.data.query_chars,cfg.data.ref_chars,val_f,cfg.model.in_channels,cfg.train.seed+99,cfg.eval.episodes)
    encoder=load_phi_checkpoint(cfg.model.phi_checkpoint,cfg.model.in_channels,cfg.model.feature_dim,device).to(device); model=MembershipVerifier(encoder,cfg.model.feature_dim,cfg.model.hidden_dim,True).to(device)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay); loader=DataLoader(train,batch_size=min(cfg.train.batch_size,len(train)),shuffle=True,generator=make_generator(cfg.train.seed)); it=infinite(loader); total=cfg.train.max_steps if cfg.train.max_steps>0 else cfg.train.epochs*len(loader); log=JSONLLogger(out/"curves.jsonl")
    for step in range(1,total+1):
        q,r,y,_,_=next(it); opt.zero_grad(); z=model(q.to(device),r.to(device)); loss=F.binary_cross_entropy_with_logits(z,y.to(device)); loss.backward(); opt.step(); log.log(step=step,loss=float(loss))
    logits,labels=collect(model,val,device); temp=fit_temperature(logits,labels) if cfg.calibration.method=="temperature" else 1.; report=metrics(logits,labels,temp); (out/"val_metrics.json").write_text(json.dumps(report,indent=2)); torch.save({"model":model.state_dict(),"temperature":temp,"config_sha256":sha},out/"best.pt"); print(f"membership complete steps={total} auc={report['roc_auc']:.4f}")
if __name__=="__main__": main()
