"""Build the K6 review board in the compact F-series sheet style."""
import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/k6_protocol_20260920"
ARMS = {
    "K0": ROOT / "reports/k0_original_test_20260917",
    "K1": ROOT / "reports/k1_final_review",
    "K6-A": Path("/root/data1/hrfont_k6_20260920/inference/K6-A/test"),
    "K6-B": Path("/root/data2/hrfont_k6_20260920/inference/K6-B/test"),
}
ARM_LABELS = {"K0": "K0", "K1": "K1", "K6-A": "K6-A", "K6-B": "K6-B · RSI · offset loss=0"}


def read_json(path):
    return json.loads(Path(path).read_text())


def row_key(row):
    return row["font"], row["cp"], int(row["k"])


def load(path, protocol_path=None):
    path = Path(path)
    protocol_file = path / "protocol.json"
    if not protocol_file.is_file():
        assert protocol_path is not None
        protocol_file = Path(protocol_path)
    metrics = read_json(path / "metrics.json")
    if "summary" not in metrics:
        metrics["summary"] = metrics["candidate"]
    metrics["rows"] = [dict(row, png=row.get("prediction", row.get("png"))) for row in metrics["rows"]]
    return metrics, read_json(protocol_file)


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


def build_data(rows):
    common = set.intersection(*(set(value) for value in rows.values()))
    by_sample = {}
    for font, cp, shot in sorted(common):
        source = {name: rows[name][(font, cp, shot)] for name in ARMS}
        item = by_sample.setdefault(
            (font, cp),
            {
                "font": font,
                "cp": cp,
                "script": source["K6-A"]["script"],
                "content": copy_asset(source["K6-A"]["content"], "content"),
                "gt": copy_asset(source["K6-A"]["target"], "gt"),
                "shots": {},
            },
        )
        item["shots"][str(shot)] = {
            "refs": [copy_asset(path, "ref") for path in source["K6-A"]["refs_paths"]],
            "pred": {name: copy_asset(ARMS[name] / source[name]["png"], name) for name in ARMS},
        }
    return sorted(by_sample.values(), key=lambda x: (x["font"], x["cp"])), common


def metric_table(summary, val_summary):
    test_rows = []
    for name in ARMS:
        m = summary["arms"][name]["summary"]["all"]
        test_rows.append(
            f"<tr><td><b>{esc(ARM_LABELS[name])}</b></td><td class='num'>2816</td>"
            f"<td class='num'>{fmt(m['l1'])}</td><td class='num'>{fmt(m['ssim'])}</td>"
            f"<td class='num'>{fmt(m['D_change'])}</td><td class='num'>{fmt(m['D_add'])}</td>"
            f"<td class='num'>{fmt(m['D_remove'])}</td></tr>"
        )
    val_rows = []
    for name, value in val_summary.items():
        m = value["summary"]["all"]
        val_rows.append(
            f"<tr><td>{esc(name)}</td><td class='num'>192</td><td class='num'>{fmt(m['l1'])}</td>"
            f"<td class='num'>{fmt(m['ssim'])}</td><td class='num'>{fmt(m['D_change'])}</td></tr>"
        )
    return "".join(test_rows), "".join(val_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    loaded = {"K0": load(ARMS["K0"])}
    loaded.update({name: load(path, ARMS["K0"] / "protocol.json") for name, path in ARMS.items() if name != "K0"})
    for _, (_, protocol) in loaded.items():
        assert protocol["protocol"] == "K-original-CFG1-DPM20-v1"
        assert protocol["split"] == "test"
    rows = {name: {row_key(row): row for row in metrics["rows"]} for name, (metrics, _) in loaded.items()}
    samples, common = build_data(rows)

    summary = {"status": "completed", "protocol": "K-original-CFG1-DPM20-v1", "common_rows": len(common), "common_samples": len(samples), "arms": {}, "val192": {}, "fd_original": "excluded from main table: available FD results use legacy p1/p2 protocol"}
    for name, (metrics, protocol) in loaded.items():
        summary["arms"][name] = {"summary": metrics["summary"], "protocol": protocol}
    for name, path in (("K6-A", ARMS["K6-A"].parent / "val"), ("K6-B", ARMS["K6-B"].parent / "val")):
        if (path / "metrics.json").is_file():
            metrics, protocol = load(path)
            summary["val192"][name] = {"summary": metrics["summary"], "protocol": protocol}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    test_rows, val_rows = metric_table(summary, summary["val192"])
    payload = json.dumps({"samples": samples, "summary": summary, "arms": list(ARMS), "shots": [1, 2, 4, 8]}, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>K6 normal protocol review</title>
<style>
:root{--bg:#14120f;--card:#1c1914;--line:#3a342c;--fg:#e4dcc8;--muted:#8d8473;--head:#221e18;--k0:#a7a7a7;--k1:#b7c98a;--a:#6aa8a0;--b:#8fb4d9}*{box-sizing:border-box}html,body{height:100%}body{margin:0;display:flex;flex-direction:column;overflow:hidden;font-family:ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}header{flex:0 0 auto;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px 8px;z-index:30}h1{font-size:1.05rem;margin:0 0 4px;font-weight:650}.meta,.note{color:var(--muted);font-size:12px;line-height:1.45;max-width:150ch}.bar{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-top:8px;font-size:13px}.bar label{cursor:pointer}select,button{background:var(--head);color:var(--fg);border:1px solid var(--line);padding:4px 8px;font:inherit}button:hover{border-color:var(--b);color:var(--b)}details.nums{margin-top:8px;color:var(--muted);font-size:12px}details.nums table{border-collapse:collapse;margin:8px 0}details.nums th,details.nums td{border:1px solid var(--line);padding:3px 8px}main{flex:1;min-height:0;display:flex;flex-direction:column;padding:8px 16px 12px}.note{flex:0 0 auto;margin:0 0 8px}.scroll{flex:1;min-height:0;overflow:auto;border:1px solid var(--line);background:#17140f}table.sheet{border-collapse:separate;border-spacing:0;font-size:11px;min-width:max-content}th,td{border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:3px 4px;text-align:center;vertical-align:middle}thead th{position:sticky;top:0;z-index:6;background:var(--head);white-space:nowrap;padding:6px 5px 7px}thead tr:nth-child(2) th{top:34px;padding:4px 5px}.stub{left:0;z-index:10!important;background:var(--head)!important}.shot{display:block;font-size:10px;font-weight:600}.sub{display:block;font-size:9px;color:var(--muted);font-weight:400}.sample{position:sticky;left:0;z-index:3;background:var(--card);min-width:155px;max-width:155px;text-align:left;box-shadow:1px 0 0 var(--line)}.sample b{font-size:12px}.sample small{display:block;color:var(--muted);font-size:10px;margin-top:2px}.shothead{box-shadow:inset 0 -3px 0 var(--b)}.k0{color:var(--k0)}.k1{color:var(--k1)}.ka{color:var(--a)}.kb{color:var(--b)}img.g{width:56px;height:56px;image-rendering:pixelated;background:#fff;display:block}.refs{display:flex;gap:1px;max-width:180px;flex-wrap:wrap;justify-content:center}.refs img{width:22px;height:22px;image-rendering:pixelated;background:#fff}.miss{color:#666;font-size:18px}.num{text-align:right;font-variant-numeric:tabular-nums}.count{color:var(--muted);font-size:12px;margin-left:auto}
</style></head><body><header><h1>K0 · K1 · K6-A · K6-B · Normal protocol · Test 704 samples × 4 shots</h1>
<p class='meta'>同一冻结 test manifest、seed 3407、CFG 1.0、DPM++ 20 steps/order 2、嵌套 refs「永 → 永和 → 永和书风 → 永和书风骨韵天地」。主表只显示四臂 exact semantic intersection；K6-B 保留 RSI，offset loss=0。</p>
<div class='bar'><label>视图 <select id='viewSel'><option value='font'>按字体（字为行）</option><option value='char'>按字（字体为行）</option></select></label><label>字体 <select id='fontSel'></select></label><label>脚本 <select id='scriptSel'></select></label><label><input type='checkbox' id='s1' checked> 1-shot</label><label><input type='checkbox' id='s2'> 2-shot</label><label><input type='checkbox' id='s4'> 4-shot</label><label><input type='checkbox' id='s8' checked> 8-shot</label><button type='button' id='allShots'>全选 shot</button><button type='button' id='noneShots'>清空 shot</button><span id='bk' class='count'></span></div>
<details class='nums'><summary>数字表（点开）</summary><p>Test 2816 行 = 704 个语义样本 × 1/2/4/8-shot。指标只用于固定协议检查，视觉判断请看下方逐行 panel。FD 原版因可用结果是 legacy p1/p2 协议，不进入主表。</p><table><tr><th>方法</th><th>n</th><th>L1↓</th><th>SSIM↑</th><th>D_change↓</th><th>D_add↓</th><th>D_remove↓</th></tr>TEST_ROWS</table><p>VAL192（辅助视图）：</p><table><tr><th>方法</th><th>n</th><th>L1↓</th><th>SSIM↑</th><th>D_change↓</th></tr>VAL_ROWS</table></details></header><main><p class='note'>身份/字形可优先看 western 的 <b>1 / l / Q / K / g</b>；kana 看 <b>き / シ</b>；bopomofo 看 <b>ㄅ / ㄚ / ㄩ</b>。横向滚动查看 shot 与 arm；首列和表头会固定。</p><div class='scroll'><table class='sheet' id='sheet'></table></div></main>
<script>const DATA=PAYLOAD;const SHOTS=DATA.shots;const ARMS=DATA.arms;const state={view:'font',font:'',script:'',shots:new Set([1,8])};const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function fillSelect(id,values,all){const s=document.getElementById(id);s.innerHTML='<option value="">'+all+'</option>'+values.map(x=>'<option value="'+esc(x)+'">'+esc(x)+'</option>').join('')}fillSelect('fontSel',[...new Set(DATA.samples.map(x=>x.font))].sort(),'全部字体');fillSelect('scriptSel',[...new Set(DATA.samples.map(x=>x.script))].sort(),'全部脚本');document.getElementById('viewSel').onchange=e=>{state.view=e.target.value;render()};document.getElementById('fontSel').onchange=e=>{state.font=e.target.value;render()};document.getElementById('scriptSel').onchange=e=>{state.script=e.target.value;render()};for(const s of SHOTS){document.getElementById('s'+s).onchange=e=>{if(e.target.checked)state.shots.add(s);else state.shots.delete(s);render()}}document.getElementById('allShots').onclick=()=>{SHOTS.forEach(s=>{state.shots.add(s);document.getElementById('s'+s).checked=true});render()};document.getElementById('noneShots').onclick=()=>{SHOTS.forEach(s=>{state.shots.delete(s);document.getElementById('s'+s).checked=false});render()};function image(src){return src?'<img class="g" loading="lazy" src="'+src+'">':'<span class="miss">∅</span>'}function refs(xs){return '<div class="refs">'+(xs||[]).map(x=>x?'<img loading="lazy" src="'+x+'">':'').join('')+'</div>'}function render(){let selected=DATA.samples.filter(x=>(!state.font||x.font===state.font)&&(!state.script||x.script===state.script));selected.sort((a,b)=>state.view==='font'?(a.font+a.cp).localeCompare(b.font+b.cp):(a.cp+a.font).localeCompare(b.cp+b.font));let shots=SHOTS.filter(s=>state.shots.has(s));let h='<thead><tr><th class="stub" rowspan="2">样本</th><th rowspan="2">Content</th><th rowspan="2">GT</th>'+shots.map(s=>'<th class="shothead" colspan="5"><span class="shot">'+s+'-shot</span><span class="sub">Refs + 四臂</span></th>').join('')+'</tr><tr>'+shots.map(()=>'<th>Refs</th><th class="k0">K0</th><th class="k1">K1</th><th class="ka">K6-A</th><th class="kb">K6-B</th>').join('')+'</tr></thead><tbody>';for(const x of selected){h+='<tr><td class="sample"><b>'+esc(x.font)+'</b><small>'+esc(x.cp)+' · '+esc(x.script)+'</small></td><td>'+image(x.content)+'</td><td>'+image(x.gt)+'</td>';for(const s of shots){let q=x.shots[String(s)]||{};h+='<td>'+refs(q.refs)+'</td>'+ARMS.map(a=>'<td>'+image(q.pred&&q.pred[a])+'</td>').join('')}h+='</tr>'}h+='</tbody>';document.getElementById('sheet').innerHTML=h;document.getElementById('bk').textContent='显示 '+selected.length+' / '+DATA.samples.length+' samples · '+shots.length+' shot'}render();</script></body></html>"""
    page = page.replace("TEST_ROWS", test_rows).replace("VAL_ROWS", val_rows).replace("PAYLOAD", payload)
    (OUT / "index.html").write_text(page)
    (OUT / "DONE.json").write_text(json.dumps({"status": "completed", "common_rows": len(common), "common_samples": len(samples), "assets": len(list((OUT / "assets").glob("*")))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
