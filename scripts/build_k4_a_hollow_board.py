#!/usr/bin/env python3
"""Build the K4-A hollow-glyph train-set cross-checkpoint board."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/root/projects/hrfont")
DATA = Path("/root/data1/hrfont_k4a_diagnostic_20260918")
SOURCE = Path("/root/data1/hrfont_dataset_v2_20260917/v2")
STEPS = (0, 2000, 4000, 5000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000)
OUT = ROOT / "reports/k4_a_hollow_compare_20260918"
ROUTE = "/k4_a_hollow"


def key(row: dict) -> tuple[str, str, int, str]:
    return row["font"], row["cp"], int(row["k"]), row["split"]


def source_url(path: str) -> str:
    relative = Path(path).relative_to(SOURCE)
    return f"{ROUTE}/source/{relative.as_posix()}"


def load_rows(step: int) -> list[dict]:
    path = DATA / "hollow" / f"step{step:05d}" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def build_data() -> list[dict]:
    by_step = {step: {key(row): row for row in load_rows(step)} for step in STEPS}
    keys = list(by_step[STEPS[0]])
    if any(set(rows) != set(keys) for rows in by_step.values()):
        raise RuntimeError("hollow checkpoint outputs do not share the same sample keys")

    output: list[dict] = []
    for sample_key in keys:
        base = by_step[STEPS[0]][sample_key]
        stages = []
        for step in STEPS:
            row = by_step[step][sample_key]
            stages.append(
                {
                    "label": f"K4-A@{step // 1000}k" if step else "K1 parent",
                    "src": f"{ROUTE}/hollow/step{step:05d}/{row['png']}",
                    "l1": row["l1"],
                    "ssim": row["ssim"],
                    "d_change": row.get("D_change"),
                    "d_add": row.get("D_add"),
                    "d_remove": row.get("D_remove"),
                    "d_region": row.get("D_region"),
                }
            )
        output.append(
            {
                "font": base["font"],
                "cp": base["cp"],
                "k": int(base["k"]),
                "script": base["script"],
                "bucket": base.get("bucket", ""),
                "refs": base["refs"],
                "reference": [source_url(path) for path in base["refs_paths"]],
                "content": source_url(base["content"]),
                "gt": source_url(base["target"]),
                "stages": stages,
            }
        )
    return output


def build_document(data: list[dict]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return """<!doctype html>
<meta charset="utf-8">
<title>K4-A 训练集空心字跨 checkpoint</title>
<style>
body{font:14px system-ui,sans-serif;margin:22px;background:#f4f6f8;color:#17202a}
h1{margin:0 0 4px}.note{color:#536273;margin:5px 0 16px;max-width:1200px}
.toolbar{position:sticky;top:0;z-index:5;background:#f4f6f8;padding:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
select,button{padding:5px 8px;border:1px solid #aeb9c5;border-radius:4px;background:#fff}
label{color:#344454}.count{color:#536273;margin:8px 0}
.summary{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 14px}.card{background:#fff;border:1px solid #ccd5df;border-radius:6px;padding:8px 12px;min-width:145px}
.card b{display:block;font-size:15px}.card.active{border-color:#3d78a8;box-shadow:0 0 0 1px #3d78a8 inset}
table{border-collapse:collapse;background:#fff;width:max-content;min-width:100%}
th,td{border:1px solid #ccd3db;padding:6px;vertical-align:top}th{position:sticky;top:58px;background:#e8eef4;z-index:3}
td.sample{min-width:175px;position:sticky;left:0;background:#fff;z-index:2}
img{width:108px;height:108px;object-fit:contain;background:#fff;display:block}
.refs{display:flex;gap:3px;flex-wrap:wrap;width:190px}.refs img{width:43px;height:43px}
.pred{display:flex;gap:8px}.metric{font-size:12px;color:#536273;line-height:1.4;white-space:nowrap}
.metric strong{color:#17202a}.delta-down{color:#176b3a}.delta-up{color:#a33}.empty{color:#a33;padding:20px 4px}
.pager{display:flex;align-items:center;gap:8px;margin:12px 0}.pager button:disabled{opacity:.45}small{color:#687787}
</style>
<h1>K4-A · 训练集空心字跨 checkpoint 人工检查</h1>
<p class="note">固定同一组 384 个训练集空心字样本、参考图、Content 与 GT；每个样本展示 1/2/4/8-shot，共 1536 行。横向比较 K1 parent 与 K4-A 从 2k 到 20k 的输出。L1 越低越好，SSIM 越高越好。</p>
<div class="toolbar">
  <label>字体 <select id="font"><option value="">全部</option></select></label>
  <label>字符 <select id="cp"><option value="">全部</option></select></label>
  <label>script <select id="script"><option value="">全部</option></select></label>
  <label>shot <select id="shot"><option value="">全部</option></select></label>
  <label>排序 <select id="sort"><option value="sample">样本顺序</option><option value="l1">20k L1</option><option value="delta">parent→20k L1 改善</option></select></label>
  <button id="reset">重置</button>
</div>
<div id="summary" class="summary"></div><div id="count" class="count"></div>
<div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div>
<div id="app"></div>
<script>
const data=__DATA__,steps=__STEPS__;
const state={font:"",cp:"",script:"",shot:"",sort:"sample",page:0},PAGE_SIZE=36;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const metric=x=>x==null?"—":Number(x).toFixed(4);
const img=src=>src?'<img loading="lazy" src="'+src+'">':'<span class="empty">缺失</span>';
const charLabel=cp=>{try{return String.fromCodePoint(parseInt(cp.slice(1),16))+" "+cp}catch{return cp}};
function fill(id,values){const el=document.getElementById(id);el.innerHTML='<option value="">全部</option>';for(const v of [...new Set(values)].sort()){const o=document.createElement("option");o.value=v;o.textContent=v;el.appendChild(o)}}
fill("font",data.map(r=>r.font));fill("cp",data.map(r=>r.cp));fill("script",data.map(r=>r.script));fill("shot",data.map(r=>r.k));
for(const id of ["font","cp","script","shot"]){document.getElementById(id).onchange=e=>{state[id]=e.target.value;state.page=0;render()}}
document.getElementById("sort").onchange=e=>{state.sort=e.target.value;state.page=0;render()};
document.getElementById("reset").onclick=()=>{for(const id of ["font","cp","script","shot"]){state[id]="";document.getElementById(id).value=""}state.sort="sample";state.page=0;document.getElementById("sort").value="sample";render()};
document.getElementById("prev").onclick=()=>{state.page--;render()};document.getElementById("next").onclick=()=>{state.page++;render()};
function selected(){const rows=data.filter(r=>(!state.font||r.font===state.font)&&(!state.cp||r.cp===state.cp)&&(!state.script||r.script===state.script)&&(!state.shot||String(r.k)===state.shot));const last=steps.length-1;if(state.sort==="l1")rows.sort((a,b)=>a.stages[last].l1-b.stages[last].l1);if(state.sort==="delta")rows.sort((a,b)=>(a.stages[0].l1-a.stages[last].l1)-(b.stages[0].l1-b.stages[last].l1));return rows}
function render(){const rows=selected(),pages=Math.max(1,Math.ceil(rows.length/PAGE_SIZE));state.page=Math.max(0,Math.min(state.page,pages-1));const view=rows.slice(state.page*PAGE_SIZE,(state.page+1)*PAGE_SIZE);document.getElementById("count").textContent=rows.length+" / "+data.length+" 个空心字样本";document.getElementById("page").textContent="第 "+(state.page+1)+" / "+pages+" 页";document.getElementById("prev").disabled=state.page===0;document.getElementById("next").disabled=state.page>=pages-1;
const sums=steps.map((step,i)=>{const a=data.map(r=>r.stages[i]);return {l1:a.reduce((s,x)=>s+x.l1,0)/a.length,ssim:a.reduce((s,x)=>s+x.ssim,0)/a.length}});document.getElementById("summary").innerHTML=sums.map((x,i)=>'<div class="card '+(i===steps.length-1?"active":"")+'"><b>'+data[0].stages[i].label+'</b><span>L1 '+metric(x.l1)+' · SSIM '+metric(x.ssim)+'</span></div>').join("");
const head='<tr><th>样本</th><th>参考 / Content</th><th>GT</th>'+data[0].stages.map(s=>'<th>'+s.label+'</th>').join("")+'</tr>';const body=view.map(r=>{const base=r.stages[0];return '<tr><td class="sample"><b>'+esc(r.font)+'</b><br>'+esc(r.cp)+' · '+esc(charLabel(r.cp))+'<br>'+esc(r.script)+' · '+r.k+' shot<br>'+esc(r.bucket)+'</td><td><div class="refs">'+r.reference.map(img).join("")+'</div><small>refs: '+r.refs.join(", ")+'</small><div>'+img(r.content)+'</div></td><td>'+img(r.gt)+'</td>'+r.stages.map((s,i)=>{const d=i?s.l1-base.l1:null;const cls=d==null?"":d<0?"delta-down":"delta-up";return '<td><div class="pred">'+img(s.src)+'</div><div class="metric">L1 <strong>'+metric(s.l1)+'</strong> · SSIM <strong>'+metric(s.ssim)+'</strong><br>vs parent: <span class="'+cls+'">'+(d==null?"—":(d>0?"+":"")+metric(d))+'</span><br>Δ change '+metric(s.d_change)+'</div></td>'}).join("")+'</tr>'}).join("");document.getElementById("app").innerHTML='<table><thead>'+head+'</thead><tbody>'+body+'</tbody></table>'}
render();
</script>
""".replace("__DATA__", payload).replace("__STEPS__", json.dumps(list(STEPS)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(build_document(build_data()), encoding="utf-8")
    print(OUT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
