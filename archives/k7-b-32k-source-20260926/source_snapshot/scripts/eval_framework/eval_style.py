#!/usr/bin/env python3
"""SC-R/SC-GT、episodic Rank@1/MRR/margin。输入可直接使用外部字体 cache。"""
from pathlib import Path
import json, numpy as np, torch
from common import *
from data import GlyphDataset, load_manifest
from models import load_phi_checkpoint

def percentiles(xs):
    return {"mean":float(np.mean(xs)),"median":float(np.median(xs)),"p5":float(np.percentile(xs,5)),"p95":float(np.percentile(xs,95))} if xs else {}

def main(argv=None):
    cli=parse_cli(argv); cfg=load_config(cli["config"],cli.get("override"),cli["set"]); out=Path(cfg.output.dir); out.mkdir(parents=True,exist_ok=True); device=device_from_config(cfg); model=load_phi_checkpoint(cfg.model.checkpoint,cfg.model.in_channels,cfg.model.feature_dim,device).to(device).eval()
    manifest=load_manifest(cfg.data.cache_dir); families=sorted({x["family"] for x in manifest["fonts"]}); ds=GlyphDataset(cfg.data.cache_dir,families,list(cfg.data.ref_chars)+list(cfg.data.query_chars),cfg.model.in_channels); by={(r["family"],r["char"]):i for i,r in enumerate(ds.records)}
    with torch.no_grad():
        protos={f:torch.nn.functional.normalize(torch.stack([model(ds[by[(f,c)]][0][None].to(device))[0] for c in cfg.data.ref_chars]).mean(0),dim=0) for f in families}
        rows=[]
        for f in families:
            for ch in cfg.data.query_chars:
                z=model(ds[by[(f,ch)]][0][None].to(device))[0]; values=torch.stack([z@protos[x] for x in families]); order=torch.argsort(values,descending=True); rank=int((order==families.index(f)).nonzero()[0])+1; sorted_values=values[order]
                rows.append({"family":f,"char":ch,"script":"latin" if ord(ch)<128 else "han","sc_r":float(z@protos[f]),"rank":rank,"rank1":rank==1,"mrr":1/rank,"margin":float(values[families.index(f)]-sorted_values[1 if order[0]==families.index(f) else 0]),"sc_gt":float(z@protos[f]) if cfg.data.compute_gt_from_cache else None})
    with (out/"samples.jsonl").open("w") as fp:
        for row in rows: fp.write(json.dumps(row,ensure_ascii=False)+"\n")
    agg={"n":len(rows),"sc_r":percentiles([x["sc_r"] for x in rows]),"rank_at_1":sum(x["rank1"] for x in rows)/len(rows),"mrr":sum(x["mrr"] for x in rows)/len(rows),"margin":percentiles([x["margin"] for x in rows]),"per_script":{s:percentiles([x["sc_r"] for x in rows if x["script"]==s]) for s in sorted({x["script"] for x in rows})}}
    (out/"aggregate.json").write_text(json.dumps(agg,ensure_ascii=False,indent=2)); print(json.dumps(agg))
if __name__=="__main__": main()
