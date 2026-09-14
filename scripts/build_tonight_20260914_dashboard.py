#!/usr/bin/env python3
"""Build overnight F0/F2-CLEAN + E12-b convergence dashboard (self-contained HTML).

Writes:
  reports/tonight_20260914_dashboard/index.html
  reports/tonight_20260914_dashboard/data.json
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/tonight_20260914_dashboard"
F0 = ROOT / "runs/F0-CLEAN-V0913-A-S3407"
F2 = ROOT / "runs/F2-CLEAN-V0913-A-S3407"
PHI = ROOT / "runs/e12_phi_s2_b_s3407"
MEM = ROOT / "runs/e12_membership_b_s3407"
STATUS = ROOT / "reports/f0f2_clean_v0913/STATUS.json"
WD = ROOT / "reports/watchdog_tonight_20260914"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(p: Path, default=None):
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def parse_f0_train(log: Path) -> list[dict]:
    out = []
    if not log.is_file():
        return out
    for line in log.read_text(errors="ignore").splitlines():
        if "Global Step" not in line or "train_loss" not in line:
            continue
        m = re.search(r"Global Step\s+(\d+)\s+=>\s+train_loss\s*=\s*([0-9.eE+-]+)", line)
        if m:
            out.append({"step": int(m.group(1)), "loss": float(m.group(2))})
    return out


def downsample(rows: list[dict], key: str = "step", every: int = 1, max_n: int = 800) -> list[dict]:
    if len(rows) <= max_n:
        return rows
    step = max(every, len(rows) // max_n)
    kept = rows[::step]
    if kept[-1] is not rows[-1]:
        kept.append(rows[-1])
    return kept


def collect() -> dict:
    f0_val = load_json(ROOT / "reports/training_logs/F0-CLEAN-V0913-A-S3407/val_loss_history.json", [])
    f0_ms = load_json(ROOT / "reports/training_logs/F0-CLEAN-V0913-A-S3407/F0_MILESTONE.json", {})
    st = load_json(STATUS, {})
    wd = load_json(WD / "status.json", {})
    f2_train = []
    tl = F2 / "train_log.jsonl"
    if tl.is_file():
        for line in tl.read_text().splitlines():
            if line.strip():
                try:
                    f2_train.append(json.loads(line))
                except Exception:
                    pass
    f2_val = []
    vl = F2 / "val_log.jsonl"
    if vl.is_file():
        for line in vl.read_text().splitlines():
            if line.strip():
                try:
                    f2_val.append(json.loads(line))
                except Exception:
                    pass
    phi_auc = []
    phi_loss = []
    curves = PHI / "curves.jsonl"
    if curves.is_file():
        for line in curves.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "val_auc" in r:
                phi_auc.append({"step": r.get("step"), "val_auc": r["val_auc"]})
            elif "loss" in r and "step" in r:
                phi_loss.append({"step": r["step"], "loss": r["loss"]})
    mem_val = load_json(MEM / "val_metrics.json", {})
    mem_test = load_json(MEM / "test_metrics.json", {})
    hb = load_json(F2 / "heartbeat.json", {})
    incidents = []
    inc = WD / "incidents.jsonl"
    if inc.is_file():
        for line in inc.read_text().splitlines():
            if line.strip().startswith("{"):
                try:
                    incidents.append(json.loads(line))
                except Exception:
                    pass

    f0_train = downsample(parse_f0_train(F0 / "fontdiffuser_training.log"), max_n=500)
    f2_train_ds = downsample(f2_train, max_n=600)
    phi_loss_ds = downsample(phi_loss, max_n=400)

    best_phi = max(phi_auc, key=lambda x: x["val_auc"]) if phi_auc else None

    return {
        "generated_at": utcnow(),
        "phase": st.get("phase") or (wd.get("jobs") or {}).get("F2", {}).get("state"),
        "watchdog": wd,
        "f0": {
            "done": (F0 / "DONE.json").is_file(),
            "best": str((F0 / "best").resolve()) if (F0 / "best").exists() else None,
            "best_step": f0_ms.get("selected_step"),
            "best_val": f0_ms.get("selected_loss"),
            "endpoint_val": f0_ms.get("endpoint_loss"),
            "milestone": f0_ms,
            "train": f0_train,
            "val": [{"step": r["step"], "loss": r["loss"]} for r in f0_val],
            "val_aligned": st.get("f0_val_aligned") or [],
        },
        "f2": {
            "done": (F2 / "DONE.json").is_file(),
            "heartbeat": hb,
            "train": [{"step": r["step"], "loss": r.get("loss"), "rsi_gain": r.get("rsi_gain")} for r in f2_train_ds],
            "val": [{"step": r["step"], "val_loss": r.get("val_loss")} for r in f2_val],
            "val_aligned": st.get("f2_val_aligned") or [],
            "max_steps": 80000,
        },
        "e12": {
            "phi_best": best_phi,
            "phi_auc": phi_auc,
            "phi_loss": phi_loss_ds,
            "membership_val": {k: mem_val[k] for k in ("roc_auc", "pr_auc", "brier", "ece", "temperature", "role") if k in mem_val},
            "membership_test": {k: mem_test[k] for k in ("roc_auc", "pr_auc", "brier", "ece", "temperature", "role") if k in mem_test},
        },
        "paths": {
            "morning": "reports/watchdog_tonight_20260914/MORNING.md",
            "status_f0f2": "reports/f0f2_clean_v0913/STATUS.md",
            "status_e12": "reports/e12_b/STATUS.md",
            "incidents": "reports/watchdog_tonight_20260914/incidents.jsonl",
            "contract_f0f2": "reports/EXPERIMENT_F0F2_CLEAN_V0913_20260914.md",
            "contract_e12": "reports/EXPERIMENT_E12B_V0913_20260914.md",
        },
        "incidents": incidents[-20:],
    }


def html_page(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta http-equiv="refresh" content="60"/>
<title>通宵实验看板 · F0/F2-CLEAN + E12-b · 2026-09-14</title>
<style>
:root{{--bg:#eef1f5;--card:#fff;--line:#d5dbe3;--muted:#5c6570;--ink:#1a1a1a;--accent:#1f4a6f;--ok:#2c6e49;--warn:#9a6b12;--clean:#c45c26;--dirty:#7a8490}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}}
header{{background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}}
h1{{margin:0;font-size:1.25rem}} .meta{{color:var(--muted);font-size:12px;margin-top:4px}}
main{{max-width:1180px;margin:16px auto;padding:0 14px 48px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px;margin:14px 0}}
.card h2{{margin:0 0 10px;font-size:1.05rem}}
.row{{display:flex;flex-wrap:wrap;gap:10px}}
.stat{{flex:1 1 150px;border:1px solid var(--line);padding:10px 12px;background:#f7f9fb}}
.stat .k{{font-size:11px;color:var(--muted)}} .stat .v{{font-size:18px;font-variant-numeric:tabular-nums;margin-top:2px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 7px;text-align:left}}
.pill{{display:inline-block;padding:1px 7px;border:1px solid var(--line);font-size:11px}}
.pill.ok{{color:var(--ok);border-color:#b7d4c3}} .pill.run{{color:var(--accent);border-color:#b8c9da}}
canvas{{width:100%;height:260px;background:#fff;border:1px solid var(--line)}}
.mono{{font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap}}
a{{color:var(--accent)}}
.note{{font-size:12px;color:var(--muted);margin-top:8px}}
</style>
</head>
<body>
<header>
  <h1>通宵实验看板 · F0/F2-CLEAN + E12-b</h1>
  <div class="meta" id="hdr">加载中…</div>
  <div class="meta">记录入口：<a href="../watchdog_tonight_20260914/MORNING.md">MORNING.md</a> ·
    <a href="../f0f2_clean_v0913/STATUS.md">F0/F2 STATUS</a> ·
    <a href="../e12_b/STATUS.md">E12-b STATUS</a> ·
    <a href="../watchdog_tonight_20260914/incidents.jsonl">incidents</a>
    · 本页每 60s 自动刷新</div>
</header>
<main>
  <div class="row" id="kpis"></div>

  <section class="card">
    <h2>F0-CLEAN · train loss 与 val16（选 best）</h2>
    <canvas id="f0_train" width="1100" height="260"></canvas>
    <canvas id="f0_val" width="1100" height="260" style="margin-top:10px"></canvas>
    <p class="note" id="f0_note"></p>
    <table><thead><tr><th>step</th><th>dirty val</th><th>clean val</th><th>Δ(c−d)</th></tr></thead>
    <tbody id="f0_tbl"></tbody></table>
  </section>

  <section class="card">
    <h2>F2-CLEAN · train / val（对照脏臂）</h2>
    <canvas id="f2_train" width="1100" height="260"></canvas>
    <canvas id="f2_val" width="1100" height="240" style="margin-top:10px"></canvas>
    <p class="note" id="f2_note"></p>
    <table><thead><tr><th>step</th><th>dirty val</th><th>clean val</th><th>Δ(c−d)</th></tr></thead>
    <tbody id="f2_tbl"></tbody></table>
  </section>

  <section class="card">
    <h2>E12-b · φ val AUC + membership 中间/最终结果</h2>
    <canvas id="phi_auc" width="1100" height="240"></canvas>
    <div class="row" id="e12_stats" style="margin-top:10px"></div>
    <p class="note">φ 选模用 best.pt（val AUC 最大）；membership 温度在 val 上拟合，test 只报告一次。</p>
  </section>

  <section class="card">
    <h2>事故 / 自动拉起</h2>
    <pre class="mono" id="inc"></pre>
  </section>
</main>
<script>
const DATA = {payload};
function num(x,d){{return (x==null||Number.isNaN(x))?'—':Number(x).toFixed(d);}}
function pill(ok, label){{return '<span class="pill '+(ok?'ok':'run')+'">'+label+'</span>';}}

function drawLines(id, series, ylab){{
  const cv=document.getElementById(id); if(!cv) return;
  const ctx=cv.getContext('2d');
  const W=cv.width, H=cv.height, m={{l:56,r:14,t:14,b:34}};
  ctx.clearRect(0,0,W,H);
  const all=[]; series.forEach(s=>s.pts.forEach(p=>all.push(p)));
  if(!all.length){{ctx.fillStyle='#5c6570';ctx.fillText('暂无数据', m.l, H/2); return;}}
  const xs=all.map(p=>p[0]), ys=all.map(p=>p[1]);
  let x0=Math.min(...xs), x1=Math.max(...xs); if(x0===x1) x1=x0+1;
  let y0=Math.min(...ys), y1=Math.max(...ys); const pad=(y1-y0)*0.08||1e-4; y0-=pad; y1+=pad;
  const X=x=>m.l+(x-x0)/(x1-x0)*(W-m.l-m.r);
  const Y=y=>H-m.b-(y-y0)/(y1-y0)*(H-m.t-m.b);
  ctx.strokeStyle='#d5dbe3';ctx.beginPath(); ctx.moveTo(m.l,m.t); ctx.lineTo(m.l,H-m.b); ctx.lineTo(W-m.r,H-m.b); ctx.stroke();
  ctx.fillStyle='#5c6570';ctx.font='11px system-ui';
  ctx.fillText(ylab||'', 8, 12);
  ctx.fillText(String(x0), m.l, H-8); ctx.fillText(String(x1), W-m.r-40, H-8);
  series.forEach(s=>{{
    if(!s.pts.length) return;
    ctx.strokeStyle=s.color; ctx.lineWidth=s.width||1.6; ctx.setLineDash(s.dash||[]);
    ctx.beginPath();
    s.pts.forEach((p,i)=>{{const x=X(p[0]), y=Y(p[1]); if(i) ctx.lineTo(x,y); else ctx.moveTo(x,y);}});
    ctx.stroke(); ctx.setLineDash([]);
  }});
  // legend
  let lx=m.l+8, ly=m.t+8;
  series.forEach(s=>{{
    ctx.fillStyle=s.color; ctx.fillRect(lx, ly, 12, 3);
    ctx.fillStyle='#1a1a1a'; ctx.fillText(s.name, lx+16, ly+4);
    lx += ctx.measureText(s.name).width + 40;
  }});
}}

function render(){{
  const d=DATA;
  const f2s=(d.f2.heartbeat&&d.f2.heartbeat.step)||0;
  document.getElementById('hdr').textContent =
    '生成 '+d.generated_at+' · phase='+(d.phase||'—')+' · F2 '+f2s+'/80000 · 自动刷新 60s';

  const jobs=(d.watchdog&&d.watchdog.jobs)||{{}};
  const kpis=[
    ['F0', d.f0.done?'完成':'进行中', d.f0.done],
    ['F0 best', 'step '+(d.f0.best_step||'—')+' / val '+num(d.f0.best_val,5), true],
    ['F2', d.f2.done?'完成':(f2s+'/80000'), d.f2.done],
    ['F2 val@latest', d.f2.val.length?num(d.f2.val[d.f2.val.length-1].val_loss,6):'—', true],
    ['E12 φ best', d.e12.phi_best?('AUC '+num(d.e12.phi_best.val_auc,4)+' @'+d.e12.phi_best.step):'—', true],
    ['E12 mem test', d.e12.membership_test.roc_auc!=null?('ROC '+num(d.e12.membership_test.roc_auc,4)):'—', true],
  ];
  document.getElementById('kpis').innerHTML=kpis.map(([k,v,ok])=>
    '<div class="stat"><div class="k">'+k+' '+pill(ok, ok&&String(v).includes('完成')||k.includes('best')||k.includes('E12')||k.includes('val')?'ok':'run')+'</div><div class="v">'+v+'</div></div>'
  ).join('');

  drawLines('f0_train', [{{
    name:'F0-CLEAN train loss', color:'#c45c26',
    pts:(d.f0.train||[]).map(r=>[r.step,r.loss])
  }}], 'train loss');
  drawLines('f0_val', [
    {{name:'dirty F0 val', color:'#7a8490', pts:(d.f0.val_aligned||[]).filter(r=>r.dirty_val!=null).map(r=>[r.step,r.dirty_val])}},
    {{name:'clean F0 val', color:'#c45c26', pts:(d.f0.val||[]).map(r=>[r.step,r.loss])}},
  ], 'val16 loss');
  document.getElementById('f0_note').textContent =
    '选模：min val among 5k milestones → best=@'+d.f0.best_step+' (val='+num(d.f0.best_val,6)+')；endpoint 100k val='+num(d.f0.endpoint_val,6)+'。val 面=full dirty val16（与历史 F0 对齐）。';
  document.getElementById('f0_tbl').innerHTML=(d.f0.val_aligned||[]).slice().reverse().slice(0,12).map(r=>
    '<tr><td>'+r.step+'</td><td>'+num(r.dirty_val,6)+'</td><td>'+num(r.clean_val,6)+'</td><td>'+num(r.delta_clean_minus_dirty,6)+'</td></tr>'
  ).join('') || '<tr><td colspan=4>尚无对齐点</td></tr>';

  drawLines('f2_train', [{{
    name:'F2-CLEAN train', color:'#c45c26',
    pts:(d.f2.train||[]).map(r=>[r.step,r.loss])
  }}], 'train loss');
  drawLines('f2_val', [
    {{name:'dirty F2 val', color:'#7a8490', pts:(d.f2.val_aligned||[]).filter(r=>r.dirty_val!=null).map(r=>[r.step,r.dirty_val])}},
    {{name:'clean F2 val', color:'#c45c26', pts:(d.f2.val||[]).map(r=>[r.step,r.val_loss])}},
  ], 'val loss');
  document.getElementById('f2_note').textContent =
    '主对比看 @40k；当前 heartbeat step='+f2s+' loss='+num(d.f2.heartbeat&&d.f2.heartbeat.loss,5)+'。';
  document.getElementById('f2_tbl').innerHTML=(d.f2.val_aligned||[]).slice().reverse().map(r=>
    '<tr><td>'+r.step+'</td><td>'+num(r.dirty_val,6)+'</td><td>'+num(r.clean_val,6)+'</td><td>'+num(r.delta_clean_minus_dirty,6)+'</td></tr>'
  ).join('') || '<tr><td colspan=4>尚无对齐点</td></tr>';

  drawLines('phi_auc', [{{
    name:'φ val AUC', color:'#1f4a6f',
    pts:(d.e12.phi_auc||[]).map(r=>[r.step,r.val_auc])
  }}], 'val AUC');
  const mv=d.e12.membership_val||{{}}, mt=d.e12.membership_test||{{}};
  document.getElementById('e12_stats').innerHTML=[
    ['φ best', d.e12.phi_best?('step '+d.e12.phi_best.step+' · AUC '+num(d.e12.phi_best.val_auc,4)):'—'],
    ['mem val ROC', num(mv.roc_auc,4)],
    ['mem val PR', num(mv.pr_auc,4)],
    ['mem test ROC', num(mt.roc_auc,4)],
    ['mem test PR', num(mt.pr_auc,4)],
    ['temperature', num(mv.temperature,3)],
  ].map(([k,v])=>'<div class="stat"><div class="k">'+k+'</div><div class="v">'+v+'</div></div>').join('');

  document.getElementById('inc').textContent=(d.incidents||[]).map(x=>JSON.stringify(x)).join('\\n')||'无';
}}
render();
</script>
</body>
</html>
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = collect()
    (OUT / "data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "index.html").write_text(html_page(data), encoding="utf-8")
    # symlink-friendly README
    (OUT / "README.md").write_text(
        "# 通宵实验网页看板\n\n"
        "- 打开：本目录 `index.html`（或下方端口）\n"
        "- 刷新数据：`python3 scripts/build_tonight_20260914_dashboard.py`\n"
        "- 文本记录：`../watchdog_tonight_20260914/MORNING.md`\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT / 'index.html'}")
    print(f"phase={data.get('phase')} f2_step={(data.get('f2') or {}).get('heartbeat', {}).get('step')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
