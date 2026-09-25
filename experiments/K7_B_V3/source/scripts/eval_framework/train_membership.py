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
    probs=torch.sigmoid(logits/temp); return {"roc_auc":binary_auc(labels,logits),"pr_auc":pr_auc(labels,logits),"brier":float(((probs-labels)**2).mean()),"ece":ece(labels,probs),"temperature":temp}

def membership_splits(manifest, ratios, seed, smoke=False):
    splits=split_families(manifest,ratios,seed); assert_disjoint_splits(splits)
    if all(len(splits[k])>=2 for k in ("train","val","test")):
        return splits, False
    if not smoke:
        raise ValueError("membership requires >=2 families in each train/val/test split; expand the pool or adjust ratios")
    # Tiny smoke caches exercise the code only; their scores are not held-out.
    families=sorted({x["family"] for x in manifest["fonts"]})
    if len(families)<2: raise ValueError("membership smoke requires >=2 families")
    return {k:list(families) for k in ("train","val","test")}, True

def main(argv=None):
    cli=parse_cli(argv); cfg=load_config(cli["config"],cli.get("override"),cli["set"]); seed_all(cfg.train.seed); out=Path(cfg.output.dir); sha=save_resolved_config(cfg,out,cli["config"]); device=device_from_config(cfg)
    splits,smoke_overlap=membership_splits(load_manifest(cfg.data.cache_dir),cfg.data.split_ratios,cfg.data.split_seed,bool(getattr(cfg.train,"smoke_mode",False))); train_f=splits["train"]
    train=MembershipDataset(cfg.data.cache_dir,cfg.data.query_chars,cfg.data.ref_chars,train_f,cfg.model.in_channels,cfg.train.seed,cfg.train.episodes)
    val=MembershipDataset(cfg.data.cache_dir,cfg.data.query_chars,cfg.data.ref_chars,splits["val"],cfg.model.in_channels,cfg.train.seed+99,cfg.eval.episodes)
    test=MembershipDataset(cfg.data.cache_dir,cfg.data.query_chars,cfg.data.ref_chars,splits["test"],cfg.model.in_channels,cfg.train.seed+199,cfg.eval.episodes)
    encoder=load_phi_checkpoint(cfg.model.phi_checkpoint,cfg.model.in_channels,cfg.model.feature_dim,device).to(device); model=MembershipVerifier(encoder,cfg.model.feature_dim,cfg.model.hidden_dim,True).to(device)
    model.encoder.eval()  # Frozen encoder also keeps BatchNorm buffers fixed.
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay); loader=DataLoader(train,batch_size=min(cfg.train.batch_size,len(train)),shuffle=True,generator=make_generator(cfg.train.seed)); it=infinite(loader); total=cfg.train.max_steps if cfg.train.max_steps>0 else cfg.train.epochs*len(loader); log=JSONLLogger(out/"curves.jsonl")
    for step in range(1,total+1):
        q,r,y,_,_=next(it); opt.zero_grad(); z=model(q.to(device),r.to(device)); loss=F.binary_cross_entropy_with_logits(z,y.to(device)); loss.backward(); opt.step(); log.log(step=step,loss=float(loss.detach()))
    logits,labels=collect(model,val,device); temp=fit_temperature(logits,labels) if cfg.calibration.method=="temperature" else 1.
    report=metrics(logits,labels,temp)
    report.update({"role":"temperature_fit","families":splits["val"],"smoke_overlap":smoke_overlap})
    (out/"val_metrics.json").write_text(json.dumps(report,indent=2))
    test_logits,test_labels=collect(model,test,device); test_report=metrics(test_logits,test_labels,temp)
    test_report.update({"role":"held_out_test" if not smoke_overlap else "smoke_only","families":splits["test"],"smoke_overlap":smoke_overlap})
    (out/"test_metrics.json").write_text(json.dumps(test_report,indent=2))
    torch.save({"model":model.state_dict(),"temperature":temp,"config_sha256":sha,"splits":splits,"smoke_overlap":smoke_overlap},out/"best.pt")
    print(f"membership complete steps={total} test_auc={test_report['roc_auc']:.4f} smoke_overlap={smoke_overlap}")
if __name__=="__main__": main()
