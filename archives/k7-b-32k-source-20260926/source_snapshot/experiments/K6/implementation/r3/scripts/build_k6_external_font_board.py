#!/usr/bin/env python3
"""Build an offline, filterable visual board for the K6 external-font OOD run."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
OUT = ROOT / "outputs/K6_EXTERNAL_FONT_OOD_20260921"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = read_json(OUT / "manifest.json")
    arms = [
        ("K6-A", "K6-A · RSI"),
        ("K6-B", "K6-B · RSI · offset loss=0"),
        ("K6-C", "K6-C · RSI + aux · offset loss=0"),
    ]
    payload = {"manifest": manifest, "arms": [], "rows": []}
    for arm, label in arms:
        metrics = read_json(OUT / "data" / arm / "metrics.json")
        payload["arms"].append({"id": arm, "label": label, "metrics": metrics["summary"], "by_font": metrics["stratified"]["font"], "by_k": metrics["stratified"]["k"]})
        for row in metrics["rows"]:
            row = dict(row)
            row["arm"] = arm
            row["target_rel"] = f"assets/TargetImage/{row['font']}/{row['font']}+{row['cp']}.png"
            row["pred_rel"] = f"data/{arm}/{row['png']}"
            row["refs_rel"] = [f"assets/StyleImage/{row['font']}/{row['font']}+{cp}.png" for cp in row["refs"]]
            payload["rows"].append(row)

    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = r'''<!doctype html><meta charset="utf-8"><title>K6 External Font OOD</title>
<style>
:root{--ink:#18212b;--muted:#66717d;--line:#dfe5ea;--bg:#f5f7f9;--card:#fff;--accent:#2457d6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:24px 30px;background:#111827;color:white}h1{margin:0 0 6px;font-size:25px}header p{margin:0;color:#cbd5e1}.wrap{padding:20px 30px;max-width:1800px;margin:auto}
.controls,.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:0 2px 8px #1d29370b}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin-bottom:14px}label{font-size:12px;color:var(--muted);display:grid;gap:4px}select{min-width:145px;padding:7px;border:1px solid #cbd5e1;border-radius:7px;background:white;color:var(--ink)}button{padding:8px 12px;border:0;border-radius:7px;background:var(--accent);color:#fff;cursor:pointer}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:14px}.arm h2{font-size:17px;margin:0 0 8px}.metric{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.metric div{background:#f1f5f9;border-radius:7px;padding:7px}.metric b{display:block;font-size:16px}.metric small{color:var(--muted)}
.note{color:var(--muted);margin:10px 0 16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}.item{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}.title{font-weight:650;display:flex;justify-content:space-between;gap:6px}.sub{color:var(--muted);font-size:12px;margin:2px 0 8px}.imgs{display:grid;grid-template-columns:1fr 1fr;gap:8px}.box{border:1px solid var(--line);border-radius:6px;padding:4px;text-align:center;background:#fafbfc}.box img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#fff}.box span{font-size:11px;color:var(--muted)}.refs{display:flex;gap:3px;margin-top:8px}.refs img{width:27px;height:27px;object-fit:contain;border:1px solid var(--line);background:#fff}.refs span{font-size:11px;color:var(--muted);align-self:center}
@media(max-width:900px){.cards{grid-template-columns:1fr}.wrap{padding:14px}.metric{grid-template-columns:repeat(2,1fr)}}
</style><header><h1>K6 External Font OOD Challenge</h1><p>域外字体独立评估 · 不进入 K_TEST / K_VAL192 主指标 · K-original-CFG1-DPM20-v1 · EMA</p></header><main class="wrap"><section class="controls">
<label>模型<select id="arm"></select></label><label>外部字体<select id="font"></select></label><label>目标字<select id="cp"></select></label><label>shot<select id="k"></select></label><button id="reset">重置筛选</button></section><section id="arms" class="cards"></section><p id="note" class="note"></p><section id="grid" class="grid"></section></main>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id);const uniq=a=>[...new Set(a)];
function fill(sel,vals,all){$(sel).innerHTML=(all?'<option value="">全部</option>':'')+vals.map(v=>`<option value="${v}">${v}</option>`).join('')}
function init(){fill('arm',DATA.arms.map(x=>x.id),'1');fill('font',DATA.manifest.fonts.map(x=>x.stem),'1');fill('cp',DATA.manifest.targets.map(x=>`u${x.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')}`),'1');fill('k',[1,2,4,8],'1');['arm','font','cp','k'].forEach(x=>$(x).onchange=render);$('reset').onclick=()=>{['arm','font','cp','k'].forEach(x=>$(x).value='');render()};render()}
function fmt(x){return Number(x).toFixed(4)}
function renderArms(){ $('arms').innerHTML=DATA.arms.map(a=>`<article class="card arm"><h2>${a.label}</h2><div class="metric"><div><b>${fmt(a.metrics.ssim)}</b><small>SSIM</small></div><div><b>${fmt(a.metrics.l1)}</b><small>L1</small></div><div><b>${fmt(a.metrics.D_change)}</b><small>D_change</small></div><div><b>${fmt(a.metrics.D_high)}</b><small>D_high</small></div></div></article>`).join('') }
function render(){renderArms();const av=$('arm').value,fv=$('font').value,cv=$('cp').value,kv=$('k').value;let rows=DATA.rows.filter(r=>(!av||r.arm===av)&&(!fv||r.font===fv)&&(!cv||r.cp===cv)&&(!kv||String(r.k)===kv));$('note').textContent=`显示 ${rows.length} / ${DATA.rows.length} 个 matched 外部样本。目标图来自外部字体；content 仍沿用冻结 V2 目标字和 encoder cache，外部字体只作为 style/target 域外因素。`;$('grid').innerHTML=rows.map(r=>`<article class="item"><div class="title"><span>${r.arm} · ${r.font}</span><span>k=${r.k}</span></div><div class="sub">target ${r.cp} · SSIM ${fmt(r.ssim)} · L1 ${fmt(r.l1)} · D_change ${fmt(r.D_change)}</div><div class="imgs"><div class="box"><img src="${r.pred_rel}"><span>生成</span></div><div class="box"><img src="${r.target_rel}"><span>外部 GT</span></div></div><div class="refs"><span>refs</span>${r.refs_rel.map((p,i)=>`<img src="${p}" title="${r.refs[i]}">`).join('')}</div></article>`).join('')}
init();</script>'''.replace("__DATA__", data)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
