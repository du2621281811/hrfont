#!/usr/bin/env python3
"""Build a matched fixed-protocol visual board for K0/K1/K5-A/K5-B."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/root/projects/hrfont")
LEGACY_DATA = ROOT / "data"
V2_DATA = Path("/root/data1/hrfont_dataset_v2_20260917/v2")
DEFAULT_ROOT = Path("/root/data1/hrfont_default_inference_20260917")
K5_ROOT = Path("/root/data1/hrfont_k5_20260919")
OUT = ROOT / "reports/k0_k1_k5_protocol_20260920"
ROUTE = "/k5_protocol"
ARMS = ("K0", "K1", "K5-A", "K5-B")


def source_url(path: str) -> str:
    value = Path(path)
    try:
        return f"{ROUTE}/source_legacy/{value.relative_to(LEGACY_DATA).as_posix()}"
    except ValueError:
        return f"{ROUTE}/source_v2/{value.relative_to(V2_DATA).as_posix()}"


def prediction_url(arm: str, split: str, png: str) -> str:
    return f"{ROUTE}/pred/{arm}/{split}/{png}"


def load_arm(arm: str) -> dict[str, dict[tuple[str, str, int], dict]]:
    root = DEFAULT_ROOT / arm if arm in {"K0", "K1"} else K5_ROOT / "evaluation" / arm
    output = {}
    for split in ("train", "val", "test"):
        rows = json.loads((root / split / "metrics.json").read_text(encoding="utf-8"))["rows"]
        output[split] = {
            (row["font"], row["cp"], int(row["k"])): row
            for row in rows
        }
    return output


def donor_info(arm: str, row: dict) -> dict:
    if arm in {"K0", "K1"}:
        return {
            "source": "fixed protocol",
            "refs": row["refs"],
            "note": "固定 reference，不记录动态 donor pool",
        }
    return {
        "source": row.get("source", "0913"),
        "refs": row["refs"],
        "ref8": row.get("ref8", []),
        "eligible": row.get("eligible_v2_donors_at_8shot"),
        "note": "固定协议只记录 donor pool 数量；未记录本行实际选中的 donor 名单",
    }


def build_rows() -> list[dict]:
    loaded = {arm: load_arm(arm) for arm in ARMS}
    output = []
    for split in ("train", "val", "test"):
        common = set.intersection(*(set(loaded[arm][split]) for arm in ARMS))
        for key in sorted(common):
            base = loaded["K0"][split][key]
            models = {}
            for arm in ARMS:
                row = loaded[arm][split][key]
                models[arm] = {
                    "prediction": prediction_url(arm, split, row["png"]),
                    "l1": row["l1"],
                    "ssim": row["ssim"],
                    "d_change": row.get("D_change"),
                    "d_add": row.get("D_add"),
                    "d_remove": row.get("D_remove"),
                    "donor": donor_info(arm, row),
                }
            output.append(
                {
                    "split": split,
                    "font": base["font"],
                    "cp": base["cp"],
                    "k": int(base["k"]),
                    "script": base["script"],
                    "refs": base["refs"],
                    "reference": [source_url(path) for path in base["refs_paths"]],
                    "content": source_url(base["content"]),
                    "gt": source_url(base["target"]),
                    "models": models,
                }
            )
    return output


def build_document(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return """<!doctype html>
<meta charset="utf-8">
<title>K0 / K1 / K5-A / K5-B 固定协议对比</title>
<style>
body{font:14px system-ui,sans-serif;margin:22px;background:#f4f6f8;color:#17202a}
h1{margin:0 0 4px}.note{color:#536273;margin:5px 0 16px;max-width:1250px}
.toolbar{position:sticky;top:0;z-index:5;background:#f4f6f8;padding:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
select,button{padding:5px 8px;border:1px solid #aeb9c5;border-radius:4px;background:#fff}
.summary{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 14px}.card{background:#fff;border:1px solid #ccd5df;border-radius:6px;padding:8px 12px;min-width:175px}
.card b{display:block;font-size:15px}.count{color:#536273;margin:8px 0}
table{border-collapse:collapse;background:#fff;width:max-content;min-width:100%}
th,td{border:1px solid #ccd3db;padding:6px;vertical-align:top}th{position:sticky;top:58px;background:#e8eef4;z-index:3}
td.sample{min-width:170px;position:sticky;left:0;background:#fff;z-index:2}
img{width:102px;height:102px;object-fit:contain;background:#fff;display:block}.refs{display:flex;gap:3px;flex-wrap:wrap;width:185px}.refs img{width:42px;height:42px}
.pred{display:flex;gap:7px}.metric{font-size:12px;color:#536273;line-height:1.4;white-space:nowrap}.metric strong{color:#17202a}
.donor{margin-top:5px;padding-top:4px;border-top:1px dashed #ccd3db;color:#536273;white-space:normal;max-width:175px}
.pager{display:flex;align-items:center;gap:8px;margin:12px 0}.pager button:disabled{opacity:.45}small{color:#687787}
</style>
<h1>K0 / K1 / K5-A / K5-B · 固定协议逐图对比</h1>
<p class="note">仅使用四个 arm 的精确交集样本，保证逐图可比：Train 2304、Val 2112、Test 7232。每个模型列出预测图、L1（逐图 image loss）、SSIM、D-change 以及 donor/reference 信息。K5 的 Val/Test 原始协议比 K0/K1 更大，本板只展示共同交集，额外覆盖量不混入对比。</p>
<div class="toolbar">
  <label>数据集 <select id="split"><option value="">全部</option></select></label>
  <label>字体 <select id="font"><option value="">全部</option></select></label>
  <label>script <select id="script"><option value="">全部</option></select></label>
  <label>shot <select id="shot"><option value="">全部</option></select></label>
  <label>排序 <select id="sort"><option value="sample">样本顺序</option><option value="K0">K0 L1</option><option value="K1">K1 L1</option><option value="K5-A">K5-A L1</option><option value="K5-B">K5-B L1</option></select></label>
  <button id="reset">重置</button>
</div>
<div id="summary" class="summary"></div><div id="count" class="count"></div>
<div class="pager"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div>
<div id="app"></div>
<script>
const data=__DATA__,arms=["K0","K1","K5-A","K5-B"],splits=["train","val","test"],state={split:"test",font:"",script:"",shot:"",sort:"sample",page:0},PAGE_SIZE=24;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));const metric=x=>x==null?"—":Number(x).toFixed(4);const img=s=>'<img loading="lazy" src="'+s+'">';const charLabel=cp=>{try{return String.fromCodePoint(parseInt(cp.slice(1),16))+" "+cp}catch{return cp}};
function fill(id,values){const e=document.getElementById(id);e.innerHTML='<option value="">全部</option>';for(const v of [...new Set(values)].sort()){const o=document.createElement("option");o.value=v;o.textContent=v;e.appendChild(o)}}
fill("split",splits);fill("font",data.map(r=>r.font));fill("script",data.map(r=>r.script));fill("shot",data.map(r=>r.k));
document.getElementById("split").value=state.split;
for(const id of ["split","font","script","shot"])document.getElementById(id).onchange=e=>{state[id]=e.target.value;state.page=0;render()};
document.getElementById("sort").onchange=e=>{state.sort=e.target.value;state.page=0;render()};document.getElementById("reset").onclick=()=>{Object.assign(state,{split:"test",font:"",script:"",shot:"",sort:"sample",page:0});for(const id of ["split","font","script","shot"])document.getElementById(id).value=id==="split"?"test":"";document.getElementById("sort").value="sample";render()};document.getElementById("prev").onclick=()=>{state.page--;render()};document.getElementById("next").onclick=()=>{state.page++;render()};
function selected(){const r=data.filter(x=>(!state.split||x.split===state.split)&&(!state.font||x.font===state.font)&&(!state.script||x.script===state.script)&&(!state.shot||String(x.k)===state.shot));if(arms.includes(state.sort))r.sort((a,b)=>a.models[state.sort].l1-b.models[state.sort].l1);return r}
function render(){const rows=selected(),pages=Math.max(1,Math.ceil(rows.length/PAGE_SIZE));state.page=Math.max(0,Math.min(state.page,pages-1));const view=rows.slice(state.page*PAGE_SIZE,(state.page+1)*PAGE_SIZE);document.getElementById("count").textContent=rows.length+" / "+data.length+" 个共同样本";document.getElementById("page").textContent="第 "+(state.page+1)+" / "+pages+" 页";document.getElementById("prev").disabled=state.page===0;document.getElementById("next").disabled=state.page>=pages-1;
document.getElementById("summary").innerHTML=arms.map(a=>{const scope=data.filter(r=>!state.split||r.split===state.split),l=scope.reduce((s,r)=>s+r.models[a].l1,0)/scope.length,ss=scope.reduce((s,r)=>s+r.models[a].ssim,0)/scope.length;return '<div class="card"><b>'+a+' · '+(state.split||"all")+'</b><span>'+scope.length+' samples · L1 '+metric(l)+' · SSIM '+metric(ss)+'</span></div>'}).join("");
const head='<tr><th>样本</th><th>参考 / Content</th><th>GT</th>'+arms.map(a=>'<th>'+a+'</th>').join("")+'</tr>';const body=view.map(r=>'<tr><td class="sample"><b>'+esc(r.font)+'</b><br>'+esc(r.cp)+' · '+esc(charLabel(r.cp))+'<br>'+esc(r.script)+' · '+r.k+' shot · '+r.split+'</td><td><div class="refs">'+r.reference.map(img).join("")+'</div><small>fixed refs: '+r.refs.join(", ")+'</small><div>'+img(r.content)+'</div></td><td>'+img(r.gt)+'</td>'+arms.map(a=>{const m=r.models[a],d=m.donor,donor=a.startsWith("K5")?d.source+" pool · eligible@8shot "+d.eligible+"<br>ref8: "+d.ref8.join(", "):"fixed refs: "+d.refs.join(", ");return '<td><div class="pred">'+img(m.prediction)+'</div><div class="metric">L1/image loss <strong>'+metric(m.l1)+'</strong><br>SSIM <strong>'+metric(m.ssim)+'</strong><br>Δ change '+metric(m.d_change)+' · add '+metric(m.d_add)+'<br>remove '+metric(m.d_remove)+'</div><div class="donor">'+donor+'<br><small>'+esc(d.note)+'</small></div></td>'}).join("")+'</tr>').join("");document.getElementById("app").innerHTML='<table><thead>'+head+'</thead><tbody>'+body+'</tbody></table>'}
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
