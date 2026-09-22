"""Build a fixed-VAL192 K7-B checkpoint convergence board in F-series style."""
import hashlib
import html
import json
import shutil
from pathlib import Path


ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/k7_b_val_checkpoint_20260922"
STEPS = [2000, 5000, 10000, 17000, 18000, 20000]
ARMS = {
    "K7-B": Path("/root/projects/hrfont/data/v3_v2_plus_v0921_20260922/inference_checkpoint_board/K7-B"),
}
ARM_LABELS = {"K7-B": "K7-B · V3 · v0921 supplement"}
RUNS = {
    "K7-B": Path("/root/projects/hrfont/runs/K7-B-V3-K0-S3407-20K-R2/train_log.jsonl"),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def esc(value):
    return html.escape(str(value), quote=True)


def fmt(value):
    return "—" if value is None else f"{value:.6f}"


def copy_asset(source, prefix):
    if not source or not Path(source).is_file():
        return None
    source = Path(source)
    name = prefix + "_" + hashlib.sha256(str(source).encode()).hexdigest()[:16] + source.suffix.lower()
    target = OUT / "assets" / name
    if not target.exists():
        shutil.copy2(source, target)
    return "assets/" + name


def load_stage(arm, step):
    path = ARMS[arm] / f"step_{step}"
    metrics = read_json(path / "metrics.json")
    protocol = read_json(path / "protocol.json")
    rows = [dict(row, png=row.get("prediction", row.get("png"))) for row in metrics["rows"]]
    return metrics, protocol, rows, path


def load_training(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "step" in row:
            rows.append(row)
    return rows


def build_samples(stage_rows):
    common = set.intersection(*(set((r["font"], r["cp"], int(r["k"])) for r in rs) for rs in stage_rows.values()))
    indexed = {name: {(r["font"], r["cp"], int(r["k"])): r for r in rs} for name, rs in stage_rows.items()}
    samples = {}
    for font, cp, shot in sorted(common):
        base = indexed[next(iter(ARMS))][(font, cp, shot)]
        item = samples.setdefault(
            (font, cp),
            {
                "font": font,
                "cp": cp,
                "script": base["script"],
                "content": copy_asset(base.get("content"), "content"),
                "gt": copy_asset(base.get("target"), "gt"),
                "shots": {},
            },
        )
        shot_item = item["shots"].setdefault(str(shot), {"refs": [], "stages": {}})
        if not shot_item["refs"]:
            shot_item["refs"] = [copy_asset(path, "ref") for path in base.get("refs_paths", [])]
        for arm in ARMS:
            for step in STEPS:
                row = indexed[arm][(font, cp, shot)]
                shot_item["stages"].setdefault(str(step), {})[arm] = copy_asset(
                    ARMS[arm] / f"step_{step}" / row["png"], arm.replace("-", "_")
                )
    return sorted(samples.values(), key=lambda x: (x["font"], x["cp"])), common


def aggregate(rows, key):
    selected = [r for r in rows if key == "all" or r["script"] == key]
    return {
        metric: (sum(float(r[metric]) for r in selected) / len(selected) if selected else None)
        for metric in ["l1", "ssim", "D_region", "D_change", "D_add", "D_remove", "D_high"]
    }


def telemetry_summary(rows):
    if not rows:
        return {}
    tail = rows[-1]
    aux = [r for r in rows if r.get("k6_aux") and r.get("k6_eligible", 0) > 0]
    return {
        "updates": int(tail.get("step", 0)),
        "skips": int(sum(int(r.get("skips", 0)) for r in rows)),
        "last_loss": float(tail.get("loss", 0.0)),
        "last_update_seconds": float(tail.get("update_seconds", 0.0)),
        "aux_events": len(aux),
        "aux_nonzero_events": sum(1 for r in aux if float(r.get("k6_aux_grad", 0.0)) > 0),
        "last_grad_norm": float(tail.get("grad_norm", 0.0)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    stage_rows = {}
    metrics_payload = {"protocol": "K-original-CFG1-DPM20-v1", "split": "VAL192", "steps": STEPS, "arms": {}}
    for arm in ARMS:
        metrics_payload["arms"][arm] = {}
        for step in STEPS:
            metrics, protocol, rows, _ = load_stage(arm, step)
            assert protocol["protocol"] == "K-original-CFG1-DPM20-v1"
            assert protocol["split"] == "val"
            assert int(protocol["step"]) == step
            stage_rows[f"{arm}@{step}"] = rows
            metrics_payload["arms"][arm][str(step)] = {
                "summary": metrics["summary"],
                "protocol": protocol,
            }
    flat_for_samples = {}
    for arm in ARMS:
        for step in STEPS:
            flat_for_samples[f"{arm}@{step}"] = stage_rows[f"{arm}@{step}"]
    samples, common = build_samples({arm: stage_rows[f"{arm}@{10000}"] for arm in ARMS})
    # Fill stage images from every checkpoint while keeping one exact VAL192 intersection.
    indexed = {name: {(r["font"], r["cp"], int(r["k"])): r for r in rows} for name, rows in stage_rows.items()}
    for sample in samples:
        for shot, shot_item in sample["shots"].items():
            key = (sample["font"], sample["cp"], int(shot))
            for step in STEPS:
                for arm in ARMS:
                    row = indexed[f"{arm}@{step}"][key]
                    shot_item["stages"][str(step)][arm] = copy_asset(
                        ARMS[arm] / f"step_{step}" / row["png"], arm.replace("-", "_")
                    )
    training = {arm: load_training(path) for arm, path in RUNS.items()}
    payload = {
        "samples": samples,
        "steps": STEPS,
        "arms": list(ARMS),
        "arm_labels": ARM_LABELS,
        "metrics": metrics_payload,
        "training": training,
        "training_summary": {arm: telemetry_summary(rows) for arm, rows in training.items()},
        "common_rows": len(common),
        "common_samples": len(samples),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    safe_payload = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>K7-B VAL checkpoint convergence review</title>
<style>
:root{--bg:#14120f;--card:#1c1914;--line:#3a342c;--fg:#e4dcc8;--muted:#8d8473;--head:#221e18;--a:#6aa8a0;--b:#8fb4d9;--warn:#d7a85f;--good:#a7c98a}*{box-sizing:border-box}html,body{height:100%}body{margin:0;display:flex;flex-direction:column;overflow:hidden;font-family:ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}header{flex:0 0 auto;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px 8px;z-index:30}h1{font-size:1.05rem;margin:0 0 4px;font-weight:650}.meta,.note{color:var(--muted);font-size:12px;line-height:1.45;max-width:150ch}.bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-top:8px;font-size:13px}.bar label{cursor:pointer}select,button{background:var(--head);color:var(--fg);border:1px solid var(--line);padding:4px 8px;font:inherit}button:hover{border-color:var(--b);color:var(--b)}main{flex:1;min-height:0;display:flex;flex-direction:column;padding:8px 16px 12px;overflow:hidden}.note{flex:0 0 auto;margin:0 0 8px}.tabs{display:flex;gap:6px;flex:0 0 auto;margin-bottom:8px}.tab{cursor:pointer}.tab.active{border-color:var(--b);color:var(--b)}.view{display:none;flex:1;min-height:0;overflow:auto}.view.active{display:block}.scroll{overflow:auto;border:1px solid var(--line);background:#17140f;max-height:100%}table.sheet{border-collapse:separate;border-spacing:0;font-size:11px;min-width:max-content}th,td{border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:3px 4px;text-align:center;vertical-align:middle}thead th{position:sticky;top:0;z-index:6;background:var(--head);white-space:nowrap;padding:6px 5px 7px}.stub{left:0;z-index:10!important;background:var(--head)!important}.sample{position:sticky;left:0;z-index:3;background:var(--card);min-width:155px;max-width:155px;text-align:left;box-shadow:1px 0 0 var(--line)}.sample b{font-size:12px}.sample small{display:block;color:var(--muted);font-size:10px;margin-top:2px}.stephead{box-shadow:inset 0 -3px 0 var(--b)}.sub{display:block;font-size:9px;color:var(--muted);font-weight:400}.armA{color:var(--a)}.armB{color:var(--b)}img.g{width:56px;height:56px;image-rendering:pixelated;background:#fff;display:block}.refs{display:flex;gap:1px;max-width:180px;flex-wrap:wrap;justify-content:center}.refs img{width:22px;height:22px;image-rendering:pixelated;background:#fff}.miss{color:#666;font-size:18px}.count{color:var(--muted);font-size:12px;margin-left:auto}.panel{border:1px solid var(--line);background:#17140f;padding:10px;overflow:auto}.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(360px,1fr));gap:10px}.metric-card{border:1px solid var(--line);padding:8px;background:var(--card)}.metric-card h3{font-size:13px;margin:0 0 6px;font-weight:600}.metric-card svg{display:block;width:100%;height:190px}.legend{font-size:11px;color:var(--muted);display:flex;gap:14px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.table-num{border-collapse:collapse;font-size:11px;width:100%}.table-num th,.table-num td{padding:4px 6px;text-align:right}.table-num th:first-child,.table-num td:first-child{text-align:left}.health{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}.health span{border:1px solid var(--line);padding:5px 8px;color:var(--muted)}details{color:var(--muted);font-size:12px;margin-top:10px} @media(max-width:760px){.metric-grid{grid-template-columns:1fr}.bar{gap:6px 8px}main{padding:6px 8px 10px}header{padding:8px}.metric-card svg{height:170px}}
</style></head><body><header><h1>K7-B · fixed VAL192 checkpoint convergence</h1>
<p class="meta">同一冻结 K_VAL192、seed 3407、CFG 1.0、DPM++ 20 steps/order 2；每个 checkpoint 都是独立 EMA 推理。K7-B 使用 V3、v0921 supplement、K0 初始化；每个 checkpoint 独立加载 EMA。这里观察阶段性收敛，不替代最终 test 2816 行的主协议验收。</p>
<div class="bar"><label>视图 <select id="viewSel"><option value="font">按字体（字为行）</option><option value="char">按字（字体为行）</option></select></label><label>字体 <select id="fontSel"></select></label><label>脚本 <select id="scriptSel"></select></label><label><input type="checkbox" id="s1"> 1-shot</label><label><input type="checkbox" id="s2"> 2-shot</label><label><input type="checkbox" id="s4" checked> 4-shot</label><label><input type="checkbox" id="s8" checked> 8-shot</label><button type="button" id="allShots">全选 shot</button><button type="button" id="noneShots">清空 shot</button><span id="count" class="count"></span></div></header>
<main><p class="note">先看「指标曲线」判断是否进入平台，再到「固定样本」横向看同一 font/字/shot 是否仍发生结构性变化；图像指标和训练 telemetry 都是检查证据，不单独构成视觉质量结论。</p><div class="tabs"><button class="tab active" data-tab="metrics">指标曲线</button><button class="tab" data-tab="gallery">固定样本</button><button class="tab" data-tab="telemetry">训练健康</button></div>
<section id="metrics" class="view active"><div id="metricGrid" class="metric-grid"></div><details><summary>VAL192 数字表</summary><div id="metricTable"></div></details></section>
<section id="gallery" class="view"><div class="scroll"><table class="sheet" id="sheet"></table></div></section>
<section id="telemetry" class="view"><div id="health" class="health"></div><div class="panel"><h3>训练 loss / 梯度（按原始 train_log，曲线为抽样显示）</h3><div id="telemetryChart"></div><details><summary>说明</summary><p>训练 telemetry 展示 loss、grad_norm 和 K7-B 辅助事件的运行轨迹；它只能说明优化过程是否前进/是否异常，不能替代固定协议生成结果。K7-B 目标为 20000 update；辅助事件数量按日志中 eligible 且 k6_aux=true 的记录统计。</p></details></div></section></main>
<script>const DATA=PAYLOAD;const SHOTS=[4];const ARMS=DATA.arms;const state={view:'font',font:'',script:'',shots:new Set([4])};const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>x==null?'—':Number(x).toFixed(6);function fillSelect(id,values,all){const s=document.getElementById(id);s.innerHTML='<option value="">'+all+'</option>'+values.map(x=>'<option value="'+esc(x)+'">'+esc(x)+'</option>').join('')}fillSelect('fontSel',[...new Set(DATA.samples.map(x=>x.font))].sort(),'全部字体');fillSelect('scriptSel',[...new Set(DATA.samples.map(x=>x.script))].sort(),'全部脚本');document.getElementById('viewSel').onchange=e=>{state.view=e.target.value;renderGallery()};document.getElementById('fontSel').onchange=e=>{state.font=e.target.value;renderGallery()};document.getElementById('scriptSel').onchange=e=>{state.script=e.target.value;renderGallery()};for(const s of SHOTS){document.getElementById('s'+s).onchange=e=>{if(e.target.checked)state.shots.add(s);else state.shots.delete(s);renderGallery()}}document.getElementById('allShots').onclick=()=>{SHOTS.forEach(s=>{state.shots.add(s);document.getElementById('s'+s).checked=true});renderGallery()};document.getElementById('noneShots').onclick=()=>{SHOTS.forEach(s=>{state.shots.delete(s);document.getElementById('s'+s).checked=false});renderGallery()};document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===b.dataset.tab))});function image(src){return src?'<img class="g" loading="lazy" src="'+src+'">':'<span class="miss">∅</span>'}function refs(xs){return '<div class="refs">'+(xs||[]).map(x=>x?'<img loading="lazy" src="'+x+'">':'').join('')+'</div>'}function renderGallery(){let selected=DATA.samples.filter(x=>(!state.font||x.font===state.font)&&(!state.script||x.script===state.script));selected.sort((a,b)=>state.view==='font'?(a.font+a.cp).localeCompare(b.font+b.cp):(a.cp+a.font).localeCompare(b.cp+b.font));const shots=SHOTS.filter(s=>state.shots.has(s));let h='<thead><tr><th class="stub" rowspan="2">样本</th><th rowspan="2">Content</th><th rowspan="2">GT</th>'+shots.map(s=>'<th class="stephead" colspan="'+(DATA.steps.length+1)+'"><span>'+s+'-shot</span><span class="sub">Refs + checkpoints</span></th>').join('')+'</tr><tr>'+shots.map(()=>'<th>Refs</th>'+DATA.steps.map(step=>'<th><span>'+step+'</span><span class="sub">K7-B</span></th>').join('')).join('')+'</tr></thead><tbody>';for(const x of selected){h+='<tr><td class="sample"><b>'+esc(x.font)+'</b><small>'+esc(x.cp)+' · '+esc(x.script)+'</small></td><td>'+image(x.content)+'</td><td>'+image(x.gt)+'</td>';for(const s of shots){const q=x.shots[String(s)]||{};h+='<td>'+refs(q.refs)+'</td>'+DATA.steps.map(step=>{const p=q.stages[String(step)]||{};return '<td><div style="display:flex;gap:2px;justify-content:center">'+image(p['K7-B'])+'</div></td>'}).join('')}h+='</tr>'}h+='</tbody>';document.getElementById('sheet').innerHTML=h;document.getElementById('count').textContent='显示 '+selected.length+' / '+DATA.samples.length+' samples · '+shots.length+' shot'}function points(values,w,h,pad){const finite=values.filter(x=>x!=null);const lo=Math.min(...finite),hi=Math.max(...finite),span=hi-lo||1;return values.map((v,i)=>v==null?'':(pad+(i/(values.length-1))* (w-2*pad))+','+(h-pad-((v-lo)/span)*(h-2*pad))).join(' ')}function chart(metric,title,up){const w=620,h=190,p=30;const lines=ARMS.map((arm,ai)=>{const vals=DATA.steps.map(step=>DATA.metrics.arms[arm][String(step)].summary.all[metric]);return '<polyline fill="none" stroke="'+(ai?'#8fb4d9':'#6aa8a0')+'" stroke-width="2" points="'+points(vals,w,h,p)+'"/>'+vals.map((v,i)=>{const xy=points(vals,w,h,p).split(' ')[i].split(',');return '<circle cx="'+xy[0]+'" cy="'+xy[1]+'" r="3" fill="'+(ai?'#8fb4d9':'#6aa8a0')+'"><title>'+arm+' step '+DATA.steps[i]+': '+fmt(v)+'</title></circle>'}).join('')}).join('');const all=ARMS.flatMap(a=>DATA.steps.map(s=>DATA.metrics.arms[a][String(s)].summary.all[metric]));const lo=Math.min(...all),hi=Math.max(...all);return '<div class="metric-card"><h3>'+title+' '+(up?'↑':'↓')+'</h3><svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+title+' checkpoint curve"><line x1="'+p+'" y1="'+(h-p)+'" x2="'+(w-p)+'" y2="'+(h-p)+'" stroke="#3a342c"/><line x1="'+p+'" y1="'+p+'" x2="'+p+'" y2="'+(h-p)+'" stroke="#3a342c"/><text x="'+p+'" y="'+(h-8)+'" fill="#8d8473" font-size="10">'+DATA.steps[0]+'</text><text x="'+(w-p-28)+'" y="'+(h-8)+'" fill="#8d8473" font-size="10">'+DATA.steps.at(-1)+'</text>'+lines+'<text x="'+(p+4)+'" y="'+(p+10)+'" fill="#8d8473" font-size="10">'+fmt(up?hi:lo)+'</text><text x="'+(p+4)+'" y="'+(h-p-2)+'" fill="#8d8473" font-size="10">'+fmt(up?lo:hi)+'</text></svg><div class="legend"><span><i class="dot" style="background:#6aa8a0"></i>K7-B</span></div></div>'}function renderMetrics(){document.getElementById('metricGrid').innerHTML=[['l1','L1'],['ssim','SSIM'],['D_region','D_region'],['D_add','D_add'],['D_remove','D_remove'],['D_high','D_high']].map(x=>chart(x[0],x[1],x[0]==='ssim')).join('');let h='<table class="table-num"><tr><th>step</th>'+ARMS.map(a=>'<th>'+esc(DATA.arm_labels[a])+' L1↓</th><th>SSIM↑</th><th>D_region↓</th><th>D_add↓</th><th>D_remove↓</th></tr>').join('');for(const step of DATA.steps){h+='<tr><td>'+step+'</td>'+ARMS.map(a=>{const m=DATA.metrics.arms[a][String(step)].summary.all;return '<td>'+fmt(m.l1)+'</td><td>'+fmt(m.ssim)+'</td><td>'+fmt(m.D_region)+'</td><td>'+fmt(m.D_add)+'</td><td>'+fmt(m.D_remove)+'</td>'}).join('')+'</tr>'}h+='</table>';document.getElementById('metricTable').innerHTML=h}function renderHealth(){let h='';for(const arm of ARMS){const s=DATA.training_summary[arm];h+='<span><b>'+esc(DATA.arm_labels[arm])+'</b> · updates '+s.updates+' · skips '+s.skips+' · aux events '+s.aux_events+' · nonzero '+s.aux_nonzero_events+' · last loss '+fmt(s.last_loss)+'</span>'}document.getElementById('health').innerHTML=h}function renderTelemetry(){const w=900,h=220,p=34;const series=[['loss','#d7a85f'],['grad_norm','#a7c98a']];let out='<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="K7-B training telemetry"><line x1="'+p+'" y1="'+(h-p)+'" x2="'+(w-p)+'" y2="'+(h-p)+'" stroke="#3a342c"/><line x1="'+p+'" y1="'+p+'" x2="'+p+'" y2="'+(h-p)+'" stroke="#3a342c"/>';for(const [metric,color] of series){let all=ARMS.flatMap(a=>DATA.training[a].map(r=>Number(r[metric])).filter(Number.isFinite));let lo=Math.min(...all),hi=Math.max(...all),span=hi-lo||1;for(const [ai,arm] of ARMS.entries()){const rows=DATA.training[arm].filter((r,i)=>i%20===0||i===DATA.training[arm].length-1);const pts=rows.map((r,i)=>{const x=p+i/(rows.length-1)*(w-2*p);const y=h-p-((Number(r[metric])-lo)/span)*(h-2*p);return x+','+y}).join(' ');out+='<polyline fill="none" stroke="'+color+'" opacity="'+(ai?.45:'.85')+'" stroke-width="1.4" points="'+pts+'"/>'}}out+='</svg><div class="legend"><span><i class="dot" style="background:#d7a85f"></i>loss</span><span><i class="dot" style="background:#a7c98a"></i>grad_norm</span><span>透明度区分 K7-B</span></div>';document.getElementById('telemetryChart').innerHTML=out}renderMetrics();renderHealth();renderTelemetry();renderGallery();</script></body></html>'''
    page = page.replace("PAYLOAD", safe_payload)
    (OUT / "index.html").write_text(page)
    (OUT / "DONE.json").write_text(json.dumps({"status": "completed", "common_rows": len(common), "common_samples": len(samples), "steps": STEPS, "assets": len(list((OUT / "assets").glob("*")))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
