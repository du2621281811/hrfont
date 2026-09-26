"""Build the final K6 board in the established F-series sheet format.

The board keeps the old K0/K1 reference columns, replaces K6-A/K6-B with
the final 20k inference outputs, and adds K6-C.  It is intentionally
offline: all displayed images are copied into the output directory.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from pathlib import Path


REVIEW_ROOT = Path(__file__).resolve().parents[5].parent
OLD_BOARD = REVIEW_ROOT / "outputs/K6_PROTOCOL_REVIEW_20260920"
K5_BOARD = REVIEW_ROOT / "outputs/K5_EVALUATION_20260919"
STAGE = REVIEW_ROOT / "work/k6_final_board_remote"
OUT = REVIEW_ROOT / "outputs/K6_FINAL_REVIEW_20260921"

CURRENT = {
    "K6-A": "K6-A",
    "K6-B": "K6-B-RSI-NOOFFSETLOSS",
    "K6-C": "K6-C-RSI-AUX-NOOFFSETLOSS",
}
LABELS = {
    "K0": "K0",
    "K1": "K1",
    "K6-A": "K6-A · RSI · 20k",
    "K6-B": "K6-B · RSI · offset loss=0 · 20k",
    "K6-C": "K6-C · RSI + aux · offset loss=0 · 10k",
}
ARMS = ["K0", "K1", "K6-A", "K6-B", "K6-C"]


def read_json(path: Path):
    return json.loads(path.read_text())


def load_old_payload():
    text = (OLD_BOARD / "index.html").read_text()
    match = re.search(r"const DATA=(.*?);const SHOTS=", text, re.S)
    if not match:
        raise RuntimeError("cannot locate DATA payload in old K6 board")
    return json.JSONDecoder().raw_decode(match.group(1))[0]


def load_k5_rows():
    """K5 is the frozen source of the matched train/val/test panel metadata."""
    text = (K5_BOARD / "index.html").read_text()
    match = re.search(r"const rows=(.*?);let page=", text, re.S)
    if not match:
        raise RuntimeError("cannot locate K5 rows")
    rows = json.JSONDecoder().raw_decode(match.group(1))[0]
    return {(r["split"], r["font"], r["cp"], int(r["shot"])): r for r in rows}


def row_key(row):
    return row["font"], row["cp"], int(row["k"])


def copy_old_asset(rel: str) -> str:
    source = OLD_BOARD / rel
    target = OUT / rel
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return rel


def copy_k5_asset(rel: str) -> str:
    source = K5_BOARD / rel
    target = OUT / rel
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return rel


def copy_current_asset(source: Path, prefix: str) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    name = f"{prefix}_{hashlib.sha256(str(source).encode()).hexdigest()[:16]}{source.suffix.lower()}"
    target = OUT / "assets" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return f"assets/{name}"


def load_current(arm: str, split: str):
    directory = STAGE / CURRENT[arm] / split
    metrics = read_json(directory / "metrics.json")
    rows = {row_key(row): row for row in metrics["rows"]}
    return metrics, rows


def build_current_samples(split, current_metrics, k5_rows, content_by_cp=None):
    """Build a split gallery from the intersection of the three K6 metrics."""
    arms = list(current_metrics)
    grouped = {}
    for arm in arms:
        metrics, rows = current_metrics[arm]
        for row in metrics["rows"]:
            grouped.setdefault((row["font"], row["cp"]), {})[int(row["k"])] = row
    samples = []
    for (font, cp), shots in sorted(grouped.items()):
        first = next(iter(shots.values()))
        item = {"font": font, "cp": cp, "script": first["script"], "shots": {}}
        if content_by_cp and cp in content_by_cp:
            item["content"] = content_by_cp[cp]
        else:
            source = STAGE / "sources" / split / "ContentImage" / f"{cp.upper()}.png"
            item["content"] = copy_current_asset(source, f"content_{split}")
        for shot, first_row in sorted(shots.items()):
            meta = k5_rows.get((split, font, cp, shot))
            if meta is None:
                raise KeyError(f"missing K5 matched metadata: {split} {font} {cp} k{shot}")
            q = {
                "refs": [copy_k5_asset(x) for x in meta["refs"]],
                "pred": {},
            }
            q["gt"] = copy_k5_asset(meta["gt"])
            item.setdefault("gt", q["gt"])
            for arm in arms:
                row = current_metrics[arm][1].get((font, cp, shot))
                q["pred"][arm] = (
                    copy_current_asset(STAGE / CURRENT[arm] / split / row["png"], arm.replace("-", "_"))
                    if row is not None else None
                )
            item["shots"][str(shot)] = q
        samples.append(item)
    return samples


def fmt(value):
    return "—" if value is None else f"{float(value):.6f}"


def esc(value):
    return html.escape(str(value), quote=True)


def build_payload():
    old = load_old_payload()
    k5_rows = load_k5_rows()
    train_metrics = {arm: load_current(arm, "train") for arm in CURRENT}
    test_metrics = {arm: load_current(arm, "test") for arm in CURRENT}
    val_metrics = {arm: load_current(arm, "val") for arm in CURRENT}

    # Copy only the old assets still used by K0/K1 plus the common references.
    old_asset_refs = set()
    for sample in old["samples"]:
        old_asset_refs.update([sample["content"], sample["gt"]])
        for shot in sample["shots"].values():
            old_asset_refs.update(shot.get("refs", []))
            for arm in ("K0", "K1"):
                asset = shot.get("pred", {}).get(arm)
                if asset:
                    old_asset_refs.add(asset)
    for rel in sorted(old_asset_refs):
        copy_old_asset(rel)

    samples = old["samples"]
    for arm, (_, rows) in test_metrics.items():
        for sample in samples:
            for shot, item in sample["shots"].items():
                row = rows.get((sample["font"], sample["cp"], int(shot)))
                if row is None:
                    item.setdefault("pred", {})[arm] = None
                else:
                    item.setdefault("pred", {})[arm] = copy_current_asset(
                        STAGE / CURRENT[arm] / "test" / row["png"], arm.replace("-", "_")
                    )

    # K5 train/val rows provide the same frozen GT/ref provenance; content is
    # copied from the matching V2 split so every split remains self-contained.
    old_content_by_cp = {sample["cp"]: sample["content"] for sample in samples}
    train_samples = build_current_samples("train", train_metrics, k5_rows)
    val_samples = build_current_samples("val", val_metrics, k5_rows, old_content_by_cp)

    summary = dict(old["summary"])
    summary.update(
        {
            "status": "completed",
            "protocol": "K-original-CFG1-DPM20-v1",
            "common_rows": 2816,
            "common_samples": 704,
            "split_counts": {"train": 2304, "val": 192, "test": 2816},
            "arms": {name: summary["arms"][name] for name in ("K0", "K1")},
            "train": {},
            "val192": {},
            "fd_original": "excluded from the main table: available FD results use legacy p1/p2 protocol",
            "inference_root": "/root/data2/hrfont_k6_20260920/inference_final_20260921",
            "weights": "EMA",
            "training_checkpoints": {
                "K6-A": "global_step_20000",
                "K6-B": "global_step_20000",
                "K6-C": "global_step_10000",
            },
        }
    )
    for arm, (metrics, _) in test_metrics.items():
        summary["arms"][arm] = {
            "summary": metrics["summary"],
            "stratified": metrics.get("stratified", {}),
            "split": "test",
            "n": len(metrics["rows"]),
        }
    for arm, (metrics, _) in train_metrics.items():
        summary["train"][arm] = {
            "summary": metrics["summary"],
            "stratified": metrics.get("stratified", {}),
            "split": "train",
            "n": len(metrics["rows"]),
        }
    for arm, (metrics, _) in val_metrics.items():
        summary["val192"][arm] = {
            "summary": metrics["summary"],
            "stratified": metrics.get("stratified", {}),
            "split": "val",
            "n": len(metrics["rows"]),
        }

    return {
        "samples": samples,
        "samples_by_split": {"train": train_samples, "val": val_samples, "test": samples},
        "summary": summary,
        "arms": ARMS,
        "arms_by_split": {
            "train": list(CURRENT),
            "val": list(CURRENT),
            "test": ARMS,
        },
        "labels": LABELS,
        "shots": [1, 2, 4, 8],
        "split_meta": {
            "train": {"label": "Train · K5 fixed panel", "rows": 2304, "samples": len(train_samples), "shots": [1, 2, 4, 8], "default_shots": [1, 8]},
            "val": {"label": "VAL192 · fixed 4-shot panel", "rows": 192, "samples": len(val_samples), "shots": [4], "default_shots": [4]},
            "test": {"label": "Test · K_TEST", "rows": 2816, "samples": 704, "shots": [1, 2, 4, 8], "default_shots": [1, 8]},
        },
    }


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>K6 final normal-protocol review</title>
<style>
:root{--bg:#14120f;--card:#1c1914;--line:#3a342c;--fg:#e4dcc8;--muted:#8d8473;--head:#221e18;--k0:#a7a7a7;--k1:#b7c98a;--a:#6aa8a0;--b:#8fb4d9;--c:#d9a86c}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;display:flex;flex-direction:column;overflow:hidden;font-family:ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{flex:0 0 auto;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px 8px;z-index:30}h1{font-size:1.05rem;margin:0 0 4px;font-weight:650}.meta,.note{color:var(--muted);font-size:12px;line-height:1.45;max-width:150ch}.bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-top:8px;font-size:13px}.bar label{cursor:pointer}select,button{background:var(--head);color:var(--fg);border:1px solid var(--line);padding:4px 8px;font:inherit}button:hover{border-color:var(--b);color:var(--b)}
details.nums{margin-top:8px;color:var(--muted);font-size:12px}details.nums table{border-collapse:collapse;margin:8px 0}details.nums th,details.nums td{border:1px solid var(--line);padding:3px 8px}main{flex:1;min-height:0;display:flex;flex-direction:column;padding:8px 16px 12px}.note{flex:0 0 auto;margin:0 0 8px}.scroll{flex:1;min-height:0;overflow:auto;border:1px solid var(--line);background:#17140f}table.sheet{border-collapse:separate;border-spacing:0;font-size:11px;min-width:max-content}th,td{border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:3px 4px;text-align:center;vertical-align:middle}thead th{position:sticky;top:0;z-index:6;background:var(--head);white-space:nowrap;padding:6px 5px 7px}thead tr:nth-child(2) th{top:34px;padding:4px 5px}.stub{left:0;z-index:10!important;background:var(--head)!important}.shot{display:block;font-size:10px;font-weight:600}.sub{display:block;font-size:9px;color:var(--muted);font-weight:400}.sample{position:sticky;left:0;z-index:3;background:var(--card);min-width:155px;max-width:155px;text-align:left;box-shadow:1px 0 0 var(--line)}.sample b{font-size:12px}.sample small{display:block;color:var(--muted);font-size:10px;margin-top:2px}.shothead{box-shadow:inset 0 -3px 0 var(--b)}.k0{color:var(--k0)}.k1{color:var(--k1)}.ka{color:var(--a)}.kb{color:var(--b)}.kc{color:var(--c)}img.g{width:56px;height:56px;image-rendering:pixelated;background:#fff;display:block}.refs{display:flex;gap:1px;max-width:180px;flex-wrap:wrap;justify-content:center}.refs img{width:22px;height:22px;image-rendering:pixelated;background:#fff}.miss{color:#666;font-size:18px}.num{text-align:right;font-variant-numeric:tabular-nums}.count{color:var(--muted);font-size:12px;margin-left:auto}.delta-good{color:#a7c98a}.delta-bad{color:#d48b7e}
</style></head>
<body>
<header>
<h1>K0 · K1 · K6-A · K6-B · K6-C · F-series sheet · Train / VAL192 / Test</h1>
<p class="meta">同一冻结协议下分区查看：Train 使用 K5 fixed train panel（2304 行），VAL192 是固定 4-shot 辅助 panel（192 行），Test 是 K_TEST（2816 行 = 704 个语义样本 × 1/2/4/8-shot）。seed 3407、CFG 1.0、DPM++ 20 steps/order 2、EMA；FD 原版不放主表，因为现有结果是 legacy p1/p2 协议。</p>
<div class="bar"><label>分区 <select id="splitSel"><option value="test">Test · K_TEST</option><option value="train">Train · K5 fixed</option><option value="val">VAL192 · fixed 4-shot</option></select></label><label>视图 <select id="viewSel"><option value="font">按字体（字为行）</option><option value="char">按字（字体为行）</option></select></label><label>字体 <select id="fontSel"></select></label><label>脚本 <select id="scriptSel"></select></label><label><input type="checkbox" id="s1" checked> 1-shot</label><label><input type="checkbox" id="s2"> 2-shot</label><label><input type="checkbox" id="s4"> 4-shot</label><label><input type="checkbox" id="s8" checked> 8-shot</label><button type="button" id="allShots">全选 shot</button><button type="button" id="noneShots">清空 shot</button><span id="bk" class="count"></span></div>
<details class="nums"><summary>数字表（点开）</summary><p>数字按当前分区显示；Train / VAL192 / Test 的样本数和 shot 范围明确分开，不能跨分区直接混读。</p><div id="trainNums"></div><div id="valNums"></div><div id="testNums"></div></details>
</header>
<main><p class="note">先用脚本、字体和 shot 过滤缩小范围，再横向看同一 font/字形的三条 K6 输出。首列和表头固定；图像缺失会显示 ∅，不会静默替换成旧 checkpoint。</p><div class="scroll"><table class="sheet" id="sheet"></table></div></main>
<script>
const DATA=PAYLOAD;const SHOTS=DATA.shots;const state={split:'test',view:'font',font:'',script:'',shots:new Set([1,8])};
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors={K0:'k0',K1:'k1','K6-A':'ka','K6-B':'kb','K6-C':'kc'};
function fillSelect(id,values,all){const s=document.getElementById(id);s.innerHTML='<option value="">'+all+'</option>'+values.map(x=>'<option value="'+esc(x)+'">'+esc(x)+'</option>').join('')}
function samples(){return DATA.samples_by_split[state.split]||[]}function arms(){return DATA.arms_by_split[state.split]||[]}function availableShots(){return DATA.split_meta[state.split].shots}
function syncControls(){const ss=samples();fillSelect('fontSel',[...new Set(ss.map(x=>x.font))].sort(),'全部字体');fillSelect('scriptSel',[...new Set(ss.map(x=>x.script))].sort(),'全部脚本');const avail=new Set(availableShots());for(const s of SHOTS){const el=document.getElementById('s'+s);el.disabled=!avail.has(s);el.checked=avail.has(s)&&state.shots.has(s)}document.getElementById('splitSel').value=state.split}
syncControls();
document.getElementById('splitSel').onchange=e=>{state.split=e.target.value;state.font='';state.script='';state.shots=new Set(DATA.split_meta[state.split].default_shots);syncControls();render();renderNumbers()};document.getElementById('viewSel').onchange=e=>{state.view=e.target.value;render()};document.getElementById('fontSel').onchange=e=>{state.font=e.target.value;render()};document.getElementById('scriptSel').onchange=e=>{state.script=e.target.value;render()};
for(const s of SHOTS){document.getElementById('s'+s).onchange=e=>{if(e.target.checked)state.shots.add(s);else state.shots.delete(s);render()}}
document.getElementById('allShots').onclick=()=>{availableShots().forEach(s=>{state.shots.add(s);document.getElementById('s'+s).checked=true});render()};document.getElementById('noneShots').onclick=()=>{SHOTS.forEach(s=>{state.shots.delete(s);document.getElementById('s'+s).checked=false});render()};
function image(src){return src?'<img class="g" loading="lazy" src="'+src+'">':'<span class="miss">∅</span>'}function refs(xs){return '<div class="refs">'+(xs||[]).map(x=>x?'<img loading="lazy" src="'+x+'">':'').join('')+'</div>'}
function render(){let ss=samples(),aa=arms();let selected=ss.filter(x=>(!state.font||x.font===state.font)&&(!state.script||x.script===state.script));selected.sort((a,b)=>state.view==='font'?(a.font+a.cp).localeCompare(b.font+b.cp):(a.cp+a.font).localeCompare(b.cp+b.font));let shots=SHOTS.filter(s=>availableShots().includes(s)&&state.shots.has(s));let h='<thead><tr><th class="stub" rowspan="2">样本</th><th rowspan="2">Content</th><th rowspan="2">GT</th>'+shots.map(s=>'<th class="shothead" colspan="'+(aa.length+1)+'"><span class="shot">'+s+'-shot</span><span class="sub">Refs + '+aa.length+' 臂</span></th>').join('')+'</tr><tr>'+shots.map(()=>'<th>Refs</th>'+aa.map(a=>'<th class="'+colors[a]+'">'+esc(a)+'</th>').join('')).join('')+'</tr></thead><tbody>';for(const x of selected){h+='<tr><td class="sample"><b>'+esc(x.font)+'</b><small>'+esc(x.cp)+' · '+esc(x.script)+'</small></td><td>'+image(x.content)+'</td><td>'+image(x.gt)+'</td>';for(const s of shots){let q=x.shots[String(s)]||{};h+='<td>'+refs(q.refs)+'</td>'+aa.map(a=>'<td>'+image(q.pred&&q.pred[a])+'</td>').join('')}h+='</tr>'}h+='</tbody>';document.getElementById('sheet').innerHTML=h;document.getElementById('bk').textContent=DATA.split_meta[state.split].label+' · 显示 '+selected.length+' / '+ss.length+' samples · '+shots.length+' shot'}
function metricTable(section,n){let h='<table><tr><th>方法</th><th>n</th><th>L1↓</th><th>SSIM↑</th><th>D_change↓</th><th>D_add↓</th><th>D_remove↓</th></tr>';const key=section==='val192'?'val':section==='arms'?'test':section;for(const arm of (DATA.arms_by_split[key]||[])){const m=DATA.summary[section][arm]?.summary?.all;if(!m)continue;h+='<tr><td><b>'+esc(DATA.labels[arm])+'</b></td><td class="num">'+n+'</td><td class="num">'+fmt(m.l1)+'</td><td class="num">'+fmt(m.ssim)+'</td><td class="num">'+fmt(m.D_change)+'</td><td class="num">'+fmt(m.D_add)+'</td><td class="num">'+fmt(m.D_remove)+'</td></tr>'}return h+'</table>'}function fmt(x){return x==null?'—':Number(x).toFixed(6)}function renderNumbers(){document.getElementById('trainNums').innerHTML='<p>Train · K5 fixed 2304 rows</p>'+metricTable('train',2304);document.getElementById('valNums').innerHTML='<p>VAL192 · fixed 4-shot 192 rows</p>'+metricTable('val192',192);document.getElementById('testNums').innerHTML='<p>Test · K_TEST 2816 rows</p>'+metricTable('arms',2816)}renderNumbers();render();
</script></body></html>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    safe = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    (OUT / "summary.json").write_text(json.dumps(payload["summary"], ensure_ascii=False, indent=2) + "\n")
    (OUT / "index.html").write_text(PAGE.replace("PAYLOAD", safe))
    (OUT / "DONE.json").write_text(
        json.dumps(
            {"status": "completed", "protocol": payload["summary"]["protocol"], "common_rows": 2816, "common_samples": 704, "arms": ARMS},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
