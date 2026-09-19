#!/usr/bin/env python3
"""Build K5-A/K5-B per-step training telemetry board."""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path


ROOT = Path("/root/projects/hrfont")
CTRL = Path("/root/data1/hrfont_k5_20260919/control")
K5A = Path("/root/data1/hrfont_k5_20260919/checkpoints/K5-A-V2-K1RECIPE-S3407")
K5B = Path("/root/data2/hrfont_k5_20260919/checkpoints/K5-B-V2-K1RECIPE-S3407")
EVAL = Path("/root/data1/hrfont_k5_20260919/evaluation")
OUT = ROOT / "reports/k5_step_training_20260920"
ROUTE = "/k5_steps"
ARMS = ("K5-A", "K5-B")
IMAGE_ROOT = OUT / "image_data"
ASSET_ROOT = OUT / "assets"


def parse_log(arm: str) -> list[dict]:
    prefix = f"{arm}_UPDATE "
    rows = []
    for line in (CTRL / f"{arm}-V2-K1RECIPE-S3407.log").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.startswith(prefix):
            continue
        try:
            raw = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            continue
        guard = raw.get("donor_family_guard", {})
        rows.append(
            {
                "arm": arm,
                "step": int(raw["step"]),
                "loss": raw.get("loss"),
                "epsilon_loss": raw.get("epsilon_loss"),
                "vgg_loss": raw.get("vgg_loss"),
                "completion_loss": raw.get("completion_loss"),
                "detail_loss": raw.get("detail_loss"),
                "offset_loss": raw.get("offset_loss"),
                "D_change": raw.get("D_change"),
                "D_add": raw.get("D_add"),
                "D_remove": raw.get("D_remove"),
                "D_region": raw.get("D_region"),
                "grad_norm": raw.get("grad_norm"),
                "update_seconds": raw.get("update_seconds"),
                "skips": raw.get("skips", 0),
                "donor_calls": guard.get("calls"),
                "donor_candidates": guard.get("masked_family_candidates"),
                "donor_minimum_eligible": guard.get("minimum_eligible"),
                "donor_violations": guard.get("violations"),
            }
        )
    return rows


def eval_summary() -> dict:
    output = {}
    for arm in ARMS:
        output[arm] = {}
        for split in ("train", "val", "test"):
            rows = json.loads(
                (EVAL / arm / split / "metrics.json").read_text(encoding="utf-8")
            )["rows"]
            output[arm][split] = {
                "rows": len(rows),
                "l1": sum(row["l1"] for row in rows) / len(rows),
                "ssim": sum(row["ssim"] for row in rows) / len(rows),
            }
    return output


def checkpoint_steps() -> list[int]:
    values = set()
    for root in (K5A, K5B):
        for path in root.glob("global_step_*"):
            values.add(int(path.name.split("_")[-1]))
    return sorted(values)


def _asset_url(path: str, cache: dict[str, str]) -> str:
    source = Path(path)
    key = str(source.resolve())
    if key not in cache:
        suffix = source.suffix.lower() or ".png"
        name = hashlib.sha256(key.encode()).hexdigest()[:20] + suffix
        ASSET_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, ASSET_ROOT / name)
        cache[key] = f"assets/{name}"
    return cache[key]


def visual_rows(steps: list[int]) -> list[dict]:
    """Load a small, fixed VAL image panel when checkpoint inference exists."""
    cache: dict[str, str] = {}
    output = []
    for step in steps:
        for arm in ARMS:
            metrics = IMAGE_ROOT / arm / f"step{step:05d}" / "metrics.json"
            if not metrics.exists():
                continue
            for row in json.loads(metrics.read_text(encoding="utf-8"))["rows"]:
                output.append(
                    {
                        "step": step,
                        "arm": arm,
                        "font": row["font"],
                        "cp": row["cp"],
                        "script": row["script"],
                        "prediction": f"image_data/{arm}/step{step:05d}/{row['png']}",
                        "target": _asset_url(row["target"], cache),
                        "content": _asset_url(row["content"], cache),
                        "references": [_asset_url(p, cache) for p in row["refs_paths"]],
                        "l1": row["l1"],
                        "ssim": row["ssim"],
                    }
                )
    return output


def build_document(telemetry: list[dict], summary: dict, steps: list[int]) -> str:
    payload = json.dumps(
        {
            "telemetry": telemetry,
            "summary": summary,
            "checkpoints": steps,
            "visual": visual_rows(steps),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return """<!doctype html>
<meta charset="utf-8">
<title>K5-A / K5-B 逐 step 训练效果</title>
<style>
body{font:14px system-ui,sans-serif;margin:22px;background:#f4f6f8;color:#17202a}
h1{margin:0 0 4px}.note{color:#536273;max-width:1250px;line-height:1.5}
.toolbar{position:sticky;top:0;z-index:5;background:#f4f6f8;padding:10px 0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
select,button{padding:5px 8px;border:1px solid #aeb9c5;border-radius:4px;background:#fff}
.cards{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.card{background:#fff;border:1px solid #ccd5df;border-radius:6px;padding:9px 12px;min-width:205px}
.card b{display:block;font-size:16px}.section{margin:18px 0}.section h2{margin:8px 0}
.chart{background:#fff;border:1px solid #ccd5df;padding:8px;margin:8px 0;overflow:auto}.chart h3{margin:2px 0 4px;font-size:14px}
.arm-toggle{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.arm-toggle label{padding:4px 7px;background:#fff;border:1px solid #ccd5df;border-radius:4px;cursor:pointer}
svg{display:block;min-width:780px}.axis{stroke:#9aa8b5;stroke-width:1}.lineA{fill:none;stroke:#2468a2;stroke-width:2}.lineB{fill:none;stroke:#b66a1a;stroke-width:2}
.legend{font-size:12px;color:#536273}.legend span{margin-right:14px}.a{color:#2468a2}.b{color:#b66a1a}
table{border-collapse:collapse;background:#fff;width:max-content;min-width:100%}
th,td{border:1px solid #ccd3db;padding:6px 8px;vertical-align:top}th{position:sticky;top:58px;background:#e8eef4}
.good{color:#176b3a}.warn{color:#9a5a00}.muted{color:#687787}.count{color:#536273;margin:8px 0}
.visual-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px}
.visual-card{background:#fff;border:1px solid #ccd5df;border-radius:6px;padding:10px;overflow:auto}
.visual-card h3{margin:0 0 7px}.visual-static{display:flex;gap:5px;align-items:flex-start;margin-bottom:8px}
.visual-static img{width:58px;height:58px;object-fit:contain;background:#fff;border:1px solid #e1e5ea}
.visual-static .refs{display:flex;gap:3px;max-width:250px;flex-wrap:wrap}.visual-static .refs img{width:28px;height:28px}
.visual-card table{min-width:500px}.visual-card td,.visual-card th{padding:5px}
.step-pred{display:flex;gap:7px;align-items:flex-start}.step-pred img{width:92px;height:92px;object-fit:contain;background:#fff}
</style>
<h1>K5-A / K5-B · 逐 step 训练效果</h1>
<p class="note">本板同时展示训练 telemetry 和固定 VAL 样本的 checkpoint 图片。图像使用相同样本、相同 donor/reference 协议，便于直接观察 2k→10k 的变化；完整固定协议结果见 <a href="/k5_protocol/">K0/K1/K5 固定协议看板</a>。donor 信息来自每 step 的 family guard：calls、masked candidates、minimum eligible、violations。</p>
<div id="cards" class="cards"></div>
<div class="toolbar"><label>主图指标 <select id="metric"><option value="loss">loss</option><option value="detail_loss">detail_loss</option><option value="D_change">D_change</option><option value="completion_loss">completion_loss</option><option value="grad_norm">grad_norm</option><option value="update_seconds">update_seconds</option></select></label><label>显示 <select id="mode"><option value="chart">曲线</option><option value="table">checkpoint 表格</option></select></label><span class="arm-toggle" id="arms"></span><button id="reset">重置</button></div>
<div id="visual" class="section"></div><div id="charts" class="section"></div><div id="table" class="section"></div>
<script>
const payload=__DATA__,telemetry=payload.telemetry,arms=["K5-A","K5-B"],state={metric:"loss",mode:"chart",visible:{"K5-A":true,"K5-B":true}};
const fmt=x=>x==null?"—":Number(x).toFixed(6);const byArm=a=>telemetry.filter(x=>x.arm===a);
for(const a of arms){const l=document.createElement("label"),c=document.createElement("input");c.type="checkbox";c.checked=true;c.onchange=()=>{if(!c.checked&&!arms.some(x=>x!==a&&state.visible[x])){c.checked=true;return}state.visible[a]=c.checked;render()};l.append(c,a);document.getElementById("arms").append(l)}
document.getElementById("metric").onchange=e=>{state.metric=e.target.value;render()};document.getElementById("mode").onchange=e=>{state.mode=e.target.value;render()};document.getElementById("reset").onclick=()=>{state.metric="loss";state.mode="chart";state.visible={"K5-A":true,"K5-B":true};document.getElementById("metric").value="loss";document.getElementById("mode").value="chart";document.querySelectorAll("#arms input").forEach(c=>c.checked=true);render()};
function path(points,min,max,w,h,pad){return points.map((p,i)=>{const x=pad+(p.step-1)/(10000-1)*(w-pad*2);const y=h-pad-(p.v-min)/Math.max(1e-12,max-min)*(h-pad*2);return (i?"L":"M")+x.toFixed(1)+" "+y.toFixed(1)}).join(" ")}
function chart(metric){const w=900,h=250,pad=34,all=telemetry.map(x=>x[metric]).filter(x=>x!=null),min=Math.min(...all),max=Math.max(...all),shown=arms.filter(a=>state.visible[a]),lines=shown.map((a,i)=>'<path class="'+(i?"lineB":"lineA")+'" d="'+path(byArm(a).map(x=>({step:x.step,v:x[metric]})).filter(x=>x.v!=null),min,max,w,h,pad)+'"/>').join(""),legend=shown.map(a=>'<span class="'+(a==="K5-A"?"a":"b")+'">■ '+a+'</span>').join("");return '<div class="chart"><h3>'+metric+' vs training step</h3><svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+metric+' versus training step"><line class="axis" x1="'+pad+'" y1="'+(h-pad)+'" x2="'+(w-pad)+'" y2="'+(h-pad)+'"/><line class="axis" x1="'+pad+'" y1="'+pad+'" x2="'+pad+'" y2="'+(h-pad)+'"/>'+lines+'<text x="'+pad+'" y="'+(h-8)+'">step 0</text><text x="'+(w-pad-55)+'" y="'+(h-8)+'">step 10000</text><text x="4" y="'+(pad+4)+'">'+fmt(max)+'</text><text x="4" y="'+(h-pad)+'">'+fmt(min)+'</text><text x="'+(w/2-35)+'" y="'+(h-2)+'">training step</text><text transform="translate(12 '+(h/2)+') rotate(-90)">'+metric+'</text></svg><div class="legend">'+legend+'<span>Source: K5 training logs · 0–10000 steps</span></div></div>'}
function renderVisual(){const rows=payload.visual||[],shown=arms.filter(a=>state.visible[a]);if(!rows.length){document.getElementById("visual").innerHTML='<h2>逐 step 图片</h2><div class="count">图片推理尚未归档；当前先显示 telemetry。</div>';return}const keys=[...new Set(rows.map(r=>r.font+"|"+r.cp))].sort();const by=(key,step,arm)=>rows.find(r=>r.font+"|"+r.cp===key&&r.step===step&&r.arm===arm);document.getElementById("visual").innerHTML='<h2>逐 step 图片（固定 VAL 样本）</h2><div class="count">每个样本使用相同 4-shot donor/reference；下方按 checkpoint 对比 K5-A 与 K5-B。</div><div class="visual-grid">'+keys.map(key=>{const first=rows.find(r=>r.font+"|"+r.cp===key),steps=[...new Set(rows.filter(r=>r.font+"|"+r.cp===key).map(r=>r.step))].sort((a,b)=>a-b);return '<article class="visual-card"><h3>'+first.font+' · '+first.cp+' · '+first.script+'</h3><div class="visual-static"><img loading="lazy" src="'+first.content+'" title="content"><img loading="lazy" src="'+first.target+'" title="GT"><div class="refs">'+first.references.map(p=>'<img loading="lazy" src="'+p+'" title="reference">').join("")+'</div></div><table><thead><tr><th>step</th>'+shown.map(a=>'<th>'+a+'</th>').join("")+'</tr></thead><tbody>'+steps.map(step=>'<tr><td><b>'+step+'</b></td>'+shown.map(a=>{const r=by(key,step,a);return '<td>'+(!r?'<span class="muted">—</span>':'<div class="step-pred"><img loading="lazy" src="'+r.prediction+'"><span class="metric">L1 <strong>'+fmt(r.l1)+'</strong><br>SSIM '+fmt(r.ssim)+'</span></div>')+'</td>'}).join("")+'</tr>').join("")+'</tbody></table></article>'}).join("")+'</div>'}
function render(){const m=state.metric,shown=arms.filter(a=>state.visible[a]);document.getElementById("cards").innerHTML=shown.map(a=>{const s=payload.summary[a],v=s.test;const last=byArm(a).at(-1);return '<div class="card"><b>'+a+' · final step '+last.step+'</b><span>train loss '+fmt(last.loss)+' · detail '+fmt(last.detail_loss)+'<br>Test L1 '+fmt(v.l1)+' · SSIM '+fmt(v.ssim)+'<br>donor violations '+(last.donor_violations??"—")+'</span></div>'}).join("");renderVisual();if(state.mode==="chart"){document.getElementById("charts").innerHTML=[m,"detail_loss","D_change","completion_loss","grad_norm","donor_calls"].map(chart).join("");document.getElementById("table").innerHTML=""}else{const exact=payload.checkpoints.map(step=>'<tr><td>'+step+'</td>'+shown.map(a=>{const rows=byArm(a),r=rows.reduce((p,x)=>Math.abs(x.step-step)<Math.abs(p.step-step)?x:p);return '<td>loss '+fmt(r.loss)+'<br>detail '+fmt(r.detail_loss)+'<br>D_change '+fmt(r.D_change)+'<br>update '+fmt(r.update_seconds)+' s<br>donor calls '+fmt(r.donor_calls)+'<br>violations '+fmt(r.donor_violations)+'</td>'}).join("")+'</tr>').join("");document.getElementById("charts").innerHTML="";document.getElementById("table").innerHTML='<h2>checkpoint telemetry</h2><div class="count">每个 checkpoint 取最接近的训练日志记录；不代表该 checkpoint 的逐图推理 loss。</div><table><thead><tr><th>checkpoint</th>'+shown.map(a=>'<th>'+a+'</th>').join("")+'</tr></thead><tbody>'+exact+'</tbody></table>'}}
render();
</script>
""".replace("__DATA__", payload)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    telemetry = parse_log("K5-A") + parse_log("K5-B")
    (OUT / "index.html").write_text(
        build_document(telemetry, eval_summary(), checkpoint_steps()),
        encoding="utf-8",
    )
    print(OUT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
