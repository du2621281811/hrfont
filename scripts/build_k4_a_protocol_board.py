#!/usr/bin/env python3
"""Build the K4-A final Train/Val/Test fixed-protocol review board."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/root/projects/hrfont")
DATA = Path("/root/data1/hrfont_k4a_diagnostic_20260918")
SOURCE = Path("/root/data1/hrfont_dataset_v2_20260917/v2")
OUT = ROOT / "reports/k4_a_protocol_20260918"
ROUTE = "/k4_a_protocol"


def source_url(path: str) -> str:
    relative = Path(path).relative_to(SOURCE)
    return f"{ROUTE}/source/{relative.as_posix()}"


def build_rows() -> list[dict]:
    output: list[dict] = []
    for split in ("train", "val", "test"):
        payload = json.loads(
            (DATA / "default20k" / split / "metrics.json").read_text(encoding="utf-8")
        )
        for row in payload["rows"]:
            output.append(
                {
                    "split": split,
                    "font": row["font"],
                    "cp": row["cp"],
                    "k": int(row["k"]),
                    "script": row["script"],
                    "refs": row["refs"],
                    "prediction": f"{ROUTE}/default20k/{split}/{row['png']}",
                    "gt": source_url(row["target"]),
                    "content": source_url(row["content"]),
                    "reference": [source_url(path) for path in row["refs_paths"]],
                    "l1": row["l1"],
                    "ssim": row["ssim"],
                    "d_change": row.get("D_change"),
                    "d_add": row.get("D_add"),
                    "d_remove": row.get("D_remove"),
                    "d_high": row.get("D_high"),
                    "d_region": row.get("D_region"),
                }
            )
    return output


def build_document(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return """<!doctype html>
<meta charset="utf-8">
<title>K4-A 固定协议 Train / Val / Test</title>
<style>
body{font:14px system-ui,sans-serif;margin:22px;background:#f4f6f8;color:#17202a}
h1{margin:0 0 4px}.note{color:#536273;margin:5px 0 16px;max-width:1200px}
.toolbar{position:sticky;top:0;z-index:5;background:#f4f6f8;padding:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
select,button{padding:5px 8px;border:1px solid #aeb9c5;border-radius:4px;background:#fff}
label{color:#344454}.count{color:#536273;margin:8px 0}
.summary{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 14px}
.card{background:#fff;border:1px solid #ccd5df;border-radius:6px;padding:8px 12px;min-width:170px}
.card b{display:block;font-size:16px}.card.active{border-color:#3d78a8;box-shadow:0 0 0 1px #3d78a8 inset}
table{border-collapse:collapse;background:#fff;width:max-content;min-width:100%}
th,td{border:1px solid #ccd3db;padding:6px;vertical-align:top}
th{position:sticky;top:58px;background:#e8eef4;z-index:3}
td.sample{min-width:180px;position:sticky;left:0;background:#fff;z-index:2}
img{width:112px;height:112px;object-fit:contain;background:#fff;display:block}
.refs{display:flex;gap:3px;flex-wrap:wrap;width:190px}.refs img{width:43px;height:43px}
.pred{display:flex;gap:8px}.metric{font-size:12px;color:#536273;line-height:1.45;white-space:nowrap}
.metric strong{color:#17202a}.empty{color:#a33;padding:20px 4px}
.pager{display:flex;align-items:center;gap:8px;margin:12px 0}.pager button:disabled{opacity:.45}
small{color:#687787}
</style>
<h1>K4-A@20k · 固定协议 Train / Val / Test</h1>
<p class="note">最终 20k checkpoint，使用锁定的 default protocol；Train / Val / Test 分别为 2304 / 2112 / 7232 个样本。固定显示参考图、Content、GT、预测和输出指标。该看板读取当前执行机上的完整推理图。</p>
<div class="toolbar">
  <label>数据集 <select id="split"><option value="">全部</option></select></label>
  <label>字体 <select id="font"><option value="">全部</option></select></label>
  <label>script <select id="script"><option value="">全部</option></select></label>
  <label>shot <select id="shot"><option value="">全部</option></select></label>
  <label>排序 <select id="sort"><option value="sample">样本顺序</option><option value="l1">L1 升序</option><option value="ssim">SSIM 降序</option></select></label>
  <button id="reset">重置</button>
</div>
<div id="summary" class="summary"></div>
<div id="count" class="count"></div>
<div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div>
<div id="app"></div>
<script>
const data=__DATA__;
const state={split:"test",font:"",script:"",shot:"",sort:"sample",page:0};
const PAGE_SIZE=48;
const splitOrder=["train","val","test"];
const esc=x=>String(x).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const metric=x=>x==null?"—":Number(x).toFixed(4);
const img=src=>src?'<img loading="lazy" src="'+src+'">':'<span class="empty">缺失</span>';
const charLabel=cp=>{try{return String.fromCodePoint(parseInt(cp.slice(1),16))+" "+cp}catch{return cp}};
function fill(id,values){const el=document.getElementById(id);el.innerHTML='<option value="">全部</option>';for(const v of [...new Set(values)].sort()){const o=document.createElement("option");o.value=v;o.textContent=v;el.appendChild(o)}el.value=state[id]||""}
function syncFonts(){const values=data.filter(r=>!state.split||r.split===state.split).map(r=>r.font);fill("font",values);if(state.font&&!values.includes(state.font))state.font="" ;document.getElementById("font").value=state.font}
fill("split",splitOrder);fill("script",data.map(r=>r.script));fill("shot",data.map(r=>r.k));syncFonts();
document.getElementById("split").value=state.split;
for(const id of ["split","font","script","shot"]){document.getElementById(id).onchange=e=>{state[id]=e.target.value;state.page=0;if(id==="split"){state.font="";syncFonts()}render()}}
document.getElementById("sort").onchange=e=>{state.sort=e.target.value;state.page=0;render()};
document.getElementById("reset").onclick=()=>{state.split="test";state.font="";state.script="";state.shot="";state.sort="sample";state.page=0;document.getElementById("split").value="test";document.getElementById("script").value="";document.getElementById("shot").value="";document.getElementById("sort").value="sample";syncFonts();render()};
document.getElementById("prev").onclick=()=>{state.page--;render()};document.getElementById("next").onclick=()=>{state.page++;render()};
function selected(){const rows=data.filter(r=>(!state.split||r.split===state.split)&&(!state.font||r.font===state.font)&&(!state.script||r.script===state.script)&&(!state.shot||String(r.k)===state.shot));if(state.sort==="l1")rows.sort((a,b)=>a.l1-b.l1);if(state.sort==="ssim")rows.sort((a,b)=>b.ssim-a.ssim);return rows}
function render(){
 const rows=selected(),pages=Math.max(1,Math.ceil(rows.length/PAGE_SIZE));state.page=Math.max(0,Math.min(state.page,pages-1));const view=rows.slice(state.page*PAGE_SIZE,(state.page+1)*PAGE_SIZE);
 document.getElementById("count").textContent=rows.length+" / "+data.length+" 个样本";
 document.getElementById("page").textContent="第 "+(state.page+1)+" / "+pages+" 页";
 document.getElementById("prev").disabled=state.page===0;document.getElementById("next").disabled=state.page>=pages-1;
 document.getElementById("summary").innerHTML=splitOrder.map(split=>{const a=data.filter(r=>r.split===split);const l=a.reduce((s,x)=>s+x.l1,0)/a.length;const ss=a.reduce((s,x)=>s+x.ssim,0)/a.length;return '<div class="card '+(state.split===split?"active":"")+'"><b>'+split.toUpperCase()+'</b><span>'+a.length+' samples · L1 '+metric(l)+' · SSIM '+metric(ss)+'</span></div>'}).join("");
 const head='<tr><th>样本</th><th>参考 / Content</th><th>GT</th><th>K4-A@20k</th></tr>';
 const body=view.map(r=>'<tr><td class="sample"><b>'+esc(r.font)+'</b><br>'+esc(r.cp)+' · '+esc(charLabel(r.cp))+'<br>'+esc(r.script)+' · '+r.k+' shot · '+r.split+'</td><td><div class="refs">'+r.reference.map(img).join("")+'</div><small>refs: '+r.refs.join(", ")+'</small><div>'+img(r.content)+'</div></td><td>'+img(r.gt)+'</td><td><div class="pred">'+img(r.prediction)+'</div><div class="metric">L1 <strong>'+metric(r.l1)+'</strong> · SSIM <strong>'+metric(r.ssim)+'</strong><br>Δ change '+metric(r.d_change)+'<br>Δ add '+metric(r.d_add)+' · remove '+metric(r.d_remove)+'<br>region '+metric(r.d_region)+'</div></td></tr>').join("");
 document.getElementById("app").innerHTML='<table><thead>'+head+'</thead><tbody>'+body+'</tbody></table>';
}
render();
</script>
""".replace("__DATA__", payload)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(build_document(build_rows()), encoding="utf-8")
    print(OUT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
