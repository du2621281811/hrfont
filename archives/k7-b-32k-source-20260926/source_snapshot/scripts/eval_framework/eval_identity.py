#!/usr/bin/env python3
"""ID-CLS 推理与 paired retention/rescue/failure。cache 模式把真实渲染作为 GT 正控。"""
from pathlib import Path
import json, os, torch
from torch.utils.data import DataLoader
from common import *
from data import GlyphClassDataset, load_manifest
from models import IDCLS

def main(argv=None):
    cli=parse_cli(argv); cfg=load_config(cli["config"],cli.get("override"),cli["set"]); out=Path(cfg.output.dir); out.mkdir(parents=True,exist_ok=True); device=device_from_config(cfg); payload=torch.load(cfg.model.checkpoint,map_location=device,weights_only=False); chars=list(payload.get("chars",cfg.data.chars)); model=IDCLS(len(chars),cfg.model.in_channels,cfg.model.feature_dim).to(device); model.load_state_dict(payload["model"]); model.eval()
    fam=sorted({x["family"] for x in load_manifest(cfg.data.cache_dir)["fonts"]}); ds=GlyphClassDataset(cfg.data.cache_dir,chars,fam,cfg.model.in_channels); cm=torch.zeros(len(chars),len(chars),dtype=torch.long); top5=0; rows=[]
    with torch.no_grad():
        for x,y,f,ch in DataLoader(ds,batch_size=cfg.eval.batch_size):
            logits=model(x.to(device)).cpu(); pred=logits.argmax(1); top5 += sum(int(a in b) for a,b in zip(y,logits.topk(min(5,len(chars)),1).indices))
            for yy,pp,ff,cc in zip(y,pred,f,ch): cm[yy,pp]+=1; rows.append({"family":ff,"char":cc,"target":int(yy),"prediction":int(pp),"correct":bool(yy==pp)})
    acc=float(cm.diag().sum()/cm.sum()); per=cm.diag()/cm.sum(1).clamp_min(1); report={"top1_accuracy":acc,"top5_accuracy":top5/len(ds),"macro_accuracy":float(per.mean()),"per_class_accuracy":per.tolist(),"confusion_matrix":cm.tolist(),"paired_retention":acc,"rescue_rate":0.0,"failure_rate":1-acc}
    (out/"identity.json").write_text(json.dumps(report,indent=2));
    try:
        if os.environ.get("HRFONT_SKIP_PNG") == "1": raise RuntimeError("PNG disabled by HRFONT_SKIP_PNG")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(); ax.imshow(cm.numpy()); ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png"); plt.close(fig)
    except Exception as exc:
        (out/"confusion_matrix_png_skipped.txt").write_text(str(exc),encoding="utf-8")
    with (out/"samples.jsonl").open("w") as fp:
        for row in rows: fp.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps(report))
if __name__=="__main__": main()
