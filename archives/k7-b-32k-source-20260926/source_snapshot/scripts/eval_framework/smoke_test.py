#!/usr/bin/env python3
"""本机端到端 smoke：外部系统字体、短训练、T1–T4 与两类评测。"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import yaml
from data import build_cache

HERE=Path(__file__).resolve().parent; OUT=HERE/"_smoke_out"

def run(name,script,config):
    path=OUT/f"{name}.yaml"; path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding="utf-8")
    env=dict(os.environ); env["MPLCONFIGDIR"]=str(OUT/"mplconfig"); env["PYTHONPYCACHEPREFIX"]="/tmp/hrfont_pycache"; env["HRFONT_SKIP_PNG"]="1"
    p=subprocess.run([sys.executable,str(HERE/script),"--config",str(path)],cwd=HERE,text=True,capture_output=True,env=env)
    print(f"[{name}] {'PASS' if p.returncode==0 else 'FAIL'}")
    if p.stdout.strip(): print(p.stdout.strip().splitlines()[-1])
    if p.returncode: print(p.stderr); raise RuntimeError(name)

def main():
    OUT.mkdir(exist_ok=True); fonts=[Path(x) for x in ["/System/Library/Fonts/Hiragino Sans GB.ttc","/System/Library/Fonts/STHeiti Light.ttc","/System/Library/Fonts/STHeiti Medium.ttc","/System/Library/Fonts/Supplemental/Songti.ttc"] if Path(x).exists()]
    if len(fonts)<3: raise RuntimeError("need >=3 CJK-capable system fonts")
    han=list("永和书"); latin=list("ABC"); cache=OUT/"cache"; build_cache(fonts,han+latin,cache,96,"fixed_size",72,6)
    common_data={"cache_dir":str(cache),"split_ratios":[.5,.25,.25],"split_seed":3407,"workers":0}; train_base={"epochs":1,"max_steps":5,"batch_size":4,"lr":.0003,"betas":[.9,.999],"weight_decay":.01,"eps":1e-8,"warmup_epochs":0,"warmup_steps":0,"grad_clip":1.,"checkpoint_every":5,"seed":3407,"device":"cpu"}; model={"in_channels":3,"feature_dim":512}
    phi={"schema_version":2,"experiment":{"id":"SMOKE-PHI"},"data":dict(common_data,chinese_chars=han,latin_chars=latin),"model":model,"train":dict(train_base,temperature=.07,fp16=False),"output":{"dir":str(OUT/"phi")}}; run("phi","train_style_encoder.py",phi)
    ident={"schema_version":2,"experiment":{"id":"SMOKE-ID"},"data":dict(common_data,chars=latin),"model":model,"train":dict(train_base,label_smoothing=.05),"output":{"dir":str(OUT/"id")}}; run("id","train_id_cls.py",ident)
    mem={"schema_version":2,"experiment":{"id":"SMOKE-MEM"},"data":dict(common_data,query_chars=latin,ref_chars=han),"model":dict(model,phi_checkpoint=str(OUT/"phi/best.pt"),hidden_dim=64),"train":{"epochs":1,"max_steps":3,"episodes":16,"batch_size":4,"lr":.0001,"weight_decay":.01,"seed":3407,"device":"cpu","smoke_mode":True},"eval":{"episodes":8},"calibration":{"method":"temperature"},"output":{"dir":str(OUT/"membership")}}; run("membership","train_membership.py",mem)
    tests={"schema_version":2,"experiment":{"id":"SMOKE-TESTS"},"data":dict(common_data,ref_chars=han,query_chars=latin),"model":dict(model,phi_checkpoint=str(OUT/"phi/best.pt"),id_checkpoint=str(OUT/"id/best.pt")),"train":{"device":"cpu"},"gates":{"t1_median":-1.,"t2_auc":0.,"t3_accuracy":0.,"t4_mean_drop":-2.,"t4_paired_t":-1e9,"relaxed":True},"output":{"dir":str(OUT/"tests")}}; run("self_tests","self_tests.py",tests)
    style={"schema_version":2,"data":{"cache_dir":str(cache),"ref_chars":han,"query_chars":latin,"compute_gt_from_cache":True},"model":dict(model,checkpoint=str(OUT/"phi/best.pt")),"train":{"device":"cpu"},"output":{"dir":str(OUT/"eval_style")}}; run("eval_style","eval_style.py",style)
    identity={"schema_version":2,"data":{"cache_dir":str(cache),"chars":latin},"model":dict(model,checkpoint=str(OUT/"id/best.pt")),"train":{"device":"cpu"},"eval":{"batch_size":8},"output":{"dir":str(OUT/"eval_identity")}}; run("eval_identity","eval_identity.py",identity)
    summary={"passed":True,"fonts":[str(x) for x in fonts],"torch":__import__("torch").__version__,"checks":["cache","phi_5_steps","id_5_steps","membership_3_steps","T1-T4_relaxed","eval_style","eval_identity"]}; (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)); print("SMOKE PASS")
if __name__=="__main__": main()
