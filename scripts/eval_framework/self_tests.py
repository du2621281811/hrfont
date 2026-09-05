#!/usr/bin/env python3
"""E12 T1–T4 自动门；正式模式任一失败均返回非零。"""
from pathlib import Path
import json, math, numpy as np, torch
from common import *
from data import GlyphDataset, GlyphClassDataset, load_manifest, split_families, group_table, pick_negative, as_chars
from models import IDCLS, load_phi_checkpoint
from train_utils import binary_auc

def main(argv=None):
    cli=parse_cli(argv); cfg=load_config(cli["config"],cli.get("override"),cli["set"]); out=Path(cfg.output.dir); out.mkdir(parents=True,exist_ok=True); device=device_from_config(cfg); phi=load_phi_checkpoint(cfg.model.phi_checkpoint,cfg.model.in_channels,cfg.model.feature_dim,device).to(device).eval(); manifest=load_manifest(cfg.data.cache_dir); by_group=bool(getattr(cfg.data,"split_by_group",True)); splits=split_families(manifest,cfg.data.split_ratios,cfg.data.split_seed,by_group=by_group); groups=group_table(manifest); families=splits["test"] or splits["val"] or sorted({x["family"] for x in manifest["fonts"]}); all_f=sorted({x["family"] for x in manifest["fonts"]})
    if len(families)<2: families=all_f
    ds=GlyphDataset(cfg.data.cache_dir,all_f,as_chars(cfg.data.ref_chars)+as_chars(cfg.data.query_chars),cfg.model.in_channels); by={(r["family"],r["char"]):i for i,r in enumerate(ds.records)}
    same=[]; different=[]; drops=[]; within=[]  # within = same-typeface weight pair, diagnostic only
    with torch.no_grad():
        prototypes={f:torch.nn.functional.normalize(torch.stack([phi(ds[by[(f,c)]][0][None].to(device))[0] for c in as_chars(cfg.data.ref_chars)]).mean(0),dim=0) for f in all_f}
        for f in families:
            wrong=pick_negative(f,all_f,groups,by_group)
            near=pick_negative(f,[x for x in all_f if groups.get(x)==groups.get(f)],None,False)
            if wrong is None: continue
            for ch in as_chars(cfg.data.query_chars):
                z=phi(ds[by[(f,ch)]][0][None].to(device))[0]; a=float(z@prototypes[f]); b=float(z@prototypes[wrong]); same.append(a); different.append(b); drops.append(a-b)
                if near is not None: within.append(a-float(z@prototypes[near]))
    labels=[1]*len(same)+[0]*len(different); auc=binary_auc(labels,same+different); mean_drop=float(np.mean(drops)); sd=float(np.std(drops,ddof=1)) if len(drops)>1 else 0.; t=mean_drop/(sd/math.sqrt(len(drops))) if sd else (float("inf") if mean_drop>0 else 0.)
    p=torch.load(cfg.model.id_checkpoint,map_location=device,weights_only=False); chars=as_chars(p.get("chars",cfg.data.query_chars)); ident=IDCLS(len(chars),cfg.model.in_channels,cfg.model.feature_dim).to(device); ident.load_state_dict(p["model"]); ident.eval(); ids=GlyphClassDataset(cfg.data.cache_dir,chars,families,cfg.model.in_channels); correct=0
    with torch.no_grad():
        for x,y,_,_ in torch.utils.data.DataLoader(ids,batch_size=64): correct += int((ident(x.to(device)).argmax(1).cpu()==y).sum())
    id_acc=correct/max(1,len(ids)); values={"T1":{"median":float(np.median(same)),"p5":float(np.percentile(same,5))},"T2":{"auc":auc},"T3":{"accuracy":id_acc},"T4":{"mean_drop":mean_drop,"paired_t":t,"n":len(drops)},"D_within_typeface":{"mean_drop":float(np.mean(within)) if within else None,"n":len(within)}}
    gates={"T1":values["T1"]["median"]>=cfg.gates.t1_median,"T2":auc>=cfg.gates.t2_auc,"T3":id_acc>=cfg.gates.t3_accuracy,"T4":mean_drop>=cfg.gates.t4_mean_drop and t>=cfg.gates.t4_paired_t}; report={"mode":"smoke" if cfg.gates.relaxed else "formal","split_by_group":by_group,"negatives":"cross_typeface" if by_group else "any_family","test_families":list(families),"tests":values,"gates":gates,"passed":all(gates.values())}; (out/"self_tests.json").write_text(json.dumps(report,indent=2,allow_nan=False)); print(json.dumps(report)); raise SystemExit(0 if report["passed"] else 2)
if __name__=="__main__": main()
