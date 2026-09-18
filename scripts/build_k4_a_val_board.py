#!/usr/bin/env python3
"""Build a same-sample, cross-step VAL board for K4-A."""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path


ROOT = Path("/root/projects/hrfont")
RUN = "K4-A-K1FT-0917-WEIGHT-S3407"
STEPS = (2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000)
ARCHIVE = ROOT / "reports/experiments/K" / RUN
OUT = ROOT / "reports/k4_a_val_compare_20260918"
ROUTE = "/k4_a_val"


def key(row: dict) -> tuple[str, str, int, str]:
    return (row["font"], row["cp"], int(row["k"]), row["split"])


def load_rows(step: int) -> list[dict]:
    path = ARCHIVE / f"step{step:08d}" / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["rows"]


def asset_ref(source: str, copied: dict[str, str]) -> str:
    source_path = Path(source)
    if source not in copied:
        token = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
        destination = OUT / "assets" / f"{token}_{source_path.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied[source] = f"{ROUTE}/assets/{destination.name}"
    return copied[source]


def build_data() -> list[dict]:
    by_step = {step: {key(row): row for row in load_rows(step)} for step in STEPS}
    keys = list(by_step[STEPS[0]])
    if any(set(rows) != set(keys) for rows in by_step.values()):
        raise RuntimeError("K4-A VAL step snapshots do not share the same sample keys")

    copied: dict[str, str] = {}
    output = []
    for sample_key in keys:
        base = by_step[STEPS[0]][sample_key]
        stages = []
        for step in STEPS:
            row = by_step[step][sample_key]
            stages.append(
                {
                    "label": f"K4-A@{step // 1000}k",
                    "src": (
                        f"{ROUTE}/experiments/K/{RUN}/step{step:08d}/"
                        f"{row['png']}"
                    ),
                    "l1": row["l1"],
                    "ssim": row["ssim"],
                    "d_change": row.get("D_change"),
                    "d_add": row.get("D_add"),
                    "d_remove": row.get("D_remove"),
                }
            )
        output.append(
            {
                "font": base["font"],
                "cp": base["cp"],
                "script": base["script"],
                "split": base["split"],
                "k": int(base["k"]),
                "char": base.get("char", ""),
                "refs": base["refs"],
                "reference": [asset_ref(path, copied) for path in base["refs_paths"]],
                "content": asset_ref(base["content"], copied),
                "gt": asset_ref(base["target"], copied),
                "stages": stages,
            }
        )
    return output


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":"))
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>K4-A VAL 跨 step 人工检查</title>
<style>
body{{font:14px system-ui,sans-serif;margin:22px;background:#f4f6f8;color:#17202a}}
h1{{margin:0 0 4px}} .note{{color:#536273;margin:5px 0 16px;max-width:1100px}}
.toolbar{{position:sticky;top:0;z-index:5;background:#f4f6f8;padding:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
select,button{{padding:5px 8px;border:1px solid #aeb9c5;border-radius:4px;background:#fff}}
label{{color:#344454}} .count{{color:#536273;margin:8px 0}}
.summary{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 14px}}
.card{{background:#fff;border:1px solid #ccd5df;border-radius:6px;padding:8px 12px;min-width:150px}}
.card b{{display:block;font-size:16px}} .good{{color:#176b3a}} .warn{{color:#9a5a00}}
table{{border-collapse:collapse;background:#fff;width:max-content;min-width:100%}}
th,td{{border:1px solid #ccd3db;padding:6px;vertical-align:top}}
th{{position:sticky;top:54px;background:#e8eef4;z-index:3}}
td.sample{{min-width:145px;position:sticky;left:0;background:#fff;z-index:2}}
img{{width:112px;height:112px;object-fit:contain;background:#fff;display:block}}
.refs{{display:flex;gap:3px;flex-wrap:wrap;width:190px}} .refs img{{width:43px;height:43px}}
.pred{{display:flex;gap:8px}} .metric{{font-size:12px;color:#536273;line-height:1.45;white-space:nowrap}}
.metric strong{{color:#17202a}} .delta-up{{color:#a33}} .delta-down{{color:#176b3a}}
.empty{{color:#a33;padding:20px 4px}} small{{color:#687787}}
</style>
<h1>K4-A VAL · 跨 step 人工检查</h1>
<p class="note">固定同一组 192 个 VAL 样本、参考图、Content 与 GT；横向比较 K4-A@2k / @4k / @6k / @8k / @10k / @12k / @14k / @16k / @18k / @20k。L1 越低越好，SSIM 越高越好。当前为完整 20k 训练过程的阶段 checkpoint 结果，不能替代 Train / Val / Test 固定协议评测。</p>
<div class="toolbar">
  <label>script <select id="script"><option value="">全部</option></select></label>
  <label>font <select id="font"><option value="">全部</option></select></label>
  <label>shot <select id="shot"><option value="">全部</option></select></label>
  <label>排序 <select id="sort"><option value="sample">样本顺序</option><option value="l1">20k L1</option><option value="delta">2k→20k L1 改善</option></select></label>
  <button id="reset">重置筛选</button>
</div>
<div id="summary" class="summary"></div>
<div id="count" class="count"></div>
<div id="app"></div>
<script>
const data={payload};
const state={{script:"",font:"",shot:"",sort:"sample"}};
const esc=x=>String(x).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
const img=src=>src?'<img loading="lazy" src="'+src+'">':'<span class="empty">缺失</span>';
const metric=x=>x==null?"—":Number(x).toFixed(4);
function fill(id,values){{const el=document.getElementById(id);for(const v of [...new Set(values)].sort()){{const o=document.createElement("option");o.value=v;o.textContent=v;el.appendChild(o)}}el.onchange=()=>{{state[id]=el.value;render()}}}}
fill("script",data.map(r=>r.script));fill("font",data.map(r=>r.font));fill("shot",data.map(r=>r.k));
document.getElementById("sort").onchange=e=>{{state.sort=e.target.value;render()}};
document.getElementById("reset").onclick=()=>{{for(const id of ["script","font","shot","sort"])document.getElementById(id).value=id==="sort"?"sample":"";state.script="";state.font="";state.shot="";state.sort="sample";render()}};
function selected(){{const rows=data.filter(r=>(!state.script||r.script===state.script)&&(!state.font||r.font===state.font)&&(!state.shot||String(r.k)===state.shot));const last=data[0].stages.length-1;if(state.sort==="l1")rows.sort((a,b)=>a.stages[last].l1-b.stages[last].l1);if(state.sort==="delta")rows.sort((a,b)=>(a.stages[0].l1-a.stages[last].l1)-(b.stages[0].l1-b.stages[last].l1));return rows}}
function render(){{const rows=selected();document.getElementById("count").textContent=rows.length+" / "+data.length+" 个固定 VAL 样本";
  const sums=data.length?data[0].stages.map((_,i)=>{{const a=rows.map(r=>r.stages[i]);return {{l1:a.reduce((s,x)=>s+x.l1,0)/Math.max(1,a.length),ssim:a.reduce((s,x)=>s+x.ssim,0)/Math.max(1,a.length)}}}}):[];
  document.getElementById("summary").innerHTML=sums.map((x,i)=>'<div class="card"><b>'+data[0].stages[i].label+'</b><span>L1 '+metric(x.l1)+' · SSIM '+metric(x.ssim)+'</span></div>').join("");
  const head='<tr><th>样本</th><th>参考 / Content</th><th>GT</th>'+data[0].stages.map(s=>'<th>'+s.label+'</th>').join("")+'</tr>';
  const body=rows.map(r=>{{const prev=r.stages[0];return '<tr><td class="sample"><b>'+esc(r.font)+'</b><br>'+esc(r.cp)+' · '+esc(r.char)+'<br>'+esc(r.script)+' · '+r.k+' shot</td><td><div class="refs">'+r.reference.map(img).join("")+'</div><small>refs: '+r.refs.join(", ")+'</small><div>'+img(r.content)+'</div></td><td>'+img(r.gt)+'</td>'+r.stages.map((s,i)=>{{const d=i?s.l1-prev.l1:null;const cls=d==null?"":d<0?"delta-down":"delta-up";return '<td><div class="pred">'+img(s.src)+'</div><div class="metric">L1 <strong>'+metric(s.l1)+'</strong> · SSIM <strong>'+metric(s.ssim)+'</strong><br>vs 2k: <span class="'+cls+'">'+(d==null?"—":(d>0?"+":"")+metric(d))+'</span><br>Δ change '+metric(s.d_change)+'</div></td>'}}).join("")+'</tr>'}}).join("");
  document.getElementById("app").innerHTML='<table><thead>'+head+'</thead><tbody>'+body+'</tbody></table>';
}}
render();
</script>
"""
    (OUT / "index.html").write_text(document, encoding="utf-8")
    print(OUT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
