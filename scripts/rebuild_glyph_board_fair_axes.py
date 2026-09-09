#!/usr/bin/env python3
"""Rebuild f03_test16_strat/index.html with fair-axis Mode A/B/C + condition strip.

Preserves existing browse_index.json items (incl. E1). Stages refs/style_ref8/.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/f03_test16_strat"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
REF8 = list("永和书风骨韵天地")
SEED = 3407

METHOD_META = {
    "P1": {
        "label": "官方 P1",
        "shot": 1,
        "ckpt": "official",
        "axes": ["A", "C"],
        "cond": "1-shot Style（优先「永」）",
    },
    "E1_100k": {
        "label": "E1 官方RSI@100k",
        "shot": 1,
        "ckpt": "100k",
        "axes": ["A", "C"],
        "cond": "1-shot + 官方 RSI",
    },
    "F0_100k": {
        "label": "F0 无RSI@100k",
        "shot": 1,
        "ckpt": "100k",
        "axes": ["A", "C"],
        "cond": "1-shot · 无 RSI",
    },
    "F2_75000": {
        "label": "F2 Delta@75k",
        "shot": 8,
        "ckpt": "75k*",
        "axes": ["B", "C"],
        "cond": "ref8 + Δ（无 Support）",
    },
    "F3_80k": {
        "label": "F3 legacy@80k",
        "shot": 8,
        "ckpt": "80k",
        "axes": ["B", "C"],
        "cond": "ref8 + Δ + Support（同 ref8 多通路）",
    },
}

MODES = {
    "A": {
        "title": "A · 1-shot 消融",
        "question": "官方 RSI 有没有用？",
        "methods": ["P1", "E1_100k", "F0_100k"],
        "shot": 1,
        "warn": False,
        "blurb": "公平轴：同一 Content + 单张 Style。P1 未微调；E1 保留官方 RSI；F0 关掉 RSI。",
    },
    "B": {
        "title": "B · 8-ref 增量",
        "question": "Δ 之上再加 Support 有没有增益？",
        "methods": ["F2_75000", "F3_80k"],
        "shot": 8,
        "warn": False,
        "blurb": "公平轴：同一 Content + 固定 ref8。F2=Δ；F3 legacy=Δ+Support。步数 75k vs 80k 仍不完全对齐。",
    },
    "C": {
        "title": "C · 全方法浏览",
        "question": "同屏扫一眼（非总排行榜）",
        "methods": ["P1", "E1_100k", "F0_100k", "F2_75000", "F3_80k"],
        "shot": "mixed",
        "warn": True,
        "blurb": "跨 1-shot 与 8-shot，禁止当作总分榜。下结论请切回 A 或 B。",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def stage_ref8(fonts: list[str]) -> None:
    root = OUT / "refs" / "style_ref8"
    root.mkdir(parents=True, exist_ok=True)
    for stem in fonts:
        d = root / stem
        d.mkdir(parents=True, exist_ok=True)
        for ch in REF8:
            src = DATA / "test" / "StyleImage" / stem / f"{stem}+{cp_of(ch)}.png"
            dst = d / f"{cp_of(ch)}.png"
            if not src.is_file():
                continue
            if dst.is_file() and dst.stat().st_size > 0:
                continue
            Image.open(src).convert("RGB").resize((96, 96)).save(dst)


def _fmt(x, n=4) -> str:
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"


def _mean(d: dict | None, *keys: str):
    cur = d or {}
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _bold_best_cells(rows_vals: list[list], higher: list[bool | None], start_col: int = 0) -> list[list[str]]:
    """Bold best float metric per column. higher: True=max, False=min, None=no compare."""
    if not rows_vals:
        return []
    ncols = len(rows_vals[0])
    best = [None] * ncols
    for c in range(start_col, ncols):
        if c >= len(higher) or higher[c] is None:
            continue
        nums = [float(r[c]) for r in rows_vals if isinstance(r[c], float)]
        if not nums:
            continue
        best[c] = max(nums) if higher[c] else min(nums)

    out = []
    for r in rows_vals:
        cells = []
        for c, v in enumerate(r):
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                s = _fmt(v, 4)
                if best[c] is not None and abs(v - best[c]) < 1e-12:
                    s = f"<b>{s}</b>"
                cells.append(s)
            elif isinstance(v, int) and not isinstance(v, bool):
                cells.append(str(v))
            else:
                cells.append(str(v))
        out.append(cells)
    return out


def metric_tables(payload: dict) -> str:
    metrics = payload.get("metrics") or {}
    methods = payload["methods"]
    mids = [m["id"] for m in methods]
    label_of = {m["id"]: m["label"] for m in methods}
    if not metrics.get("methods"):
        return "<p class='meta'>指标尚未计算。</p>"

    raw = []
    for mid in mids:
        o = metrics["methods"].get(mid, {}).get("overall", {})
        meta = METHOD_META.get(mid, {})
        raw.append(
            [
                label_of.get(mid, mid),
                meta.get("shot", "—"),
                meta.get("ckpt", "—"),
                o.get("n"),
                o.get("L1_mean"),
                o.get("SSIM_mean"),
                o.get("LPIPS_mean"),
            ]
        )
    # bold L1↓ SSIM↑ LPIPS↓ among generators (cols 4,5,6)
    pretty = _bold_best_cells(raw, [None, None, None, None, False, True, False])
    rows = [
        "<table><thead><tr><th>方法</th><th>shot</th><th>ckpt</th><th>n</th>"
        "<th>L1↓</th><th>SSIM↑</th><th>LPIPS↓</th></tr></thead><tbody>"
    ]
    for cells in pretty:
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    rows.append("</tbody></table>")
    rows.append(
        "<p class='cap'>加粗=该列最优（生成方法间）。L1/LPIPS 越低越好；SSIM 越高越好。"
        " SSIM 为整图灰度全局公式（非滑窗），仅作诊断。</p>"
    )

    buckets = payload.get("buckets") or []
    rows.append("<h3>按语种（诊断）</h3><table><thead><tr><th>语种</th>")
    for mid in mids:
        rows.append(f"<th>{label_of.get(mid, mid)} L1↓</th><th>SSIM↑</th>")
    rows.append("</tr></thead><tbody>")
    for b in buckets:
        l1s = []
        ss = []
        for mid in mids:
            bb = metrics["methods"].get(mid, {}).get("by_bucket", {}).get(b, {})
            l1s.append(bb.get("L1_mean"))
            ss.append(bb.get("SSIM_mean"))
        best_l1 = min((x for x in l1s if isinstance(x, float)), default=None)
        best_ss = max((x for x in ss if isinstance(x, float)), default=None)
        cells = [b]
        for l1, s in zip(l1s, ss):
            for v, best in ((l1, best_l1), (s, best_ss)):
                if not isinstance(v, float):
                    cells.append("—")
                else:
                    t = _fmt(v, 4)
                    if best is not None and abs(v - best) < 1e-12:
                        t = f"<b>{t}</b>"
                    cells.append(t)
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def e12_tables(e12: dict, methods: list[dict]) -> str:
    if not e12 or not e12.get("methods"):
        return "<p class='meta'>尚未跑 E12 v5.1 打分。运行 <code>scripts/score_f03_e12_v51.py</code>。</p>"
    mids = [m["id"] for m in methods]
    label_of = {m["id"]: m["label"] for m in methods}
    scorers = e12.get("scorer") or {}
    rows = [
        "<div class='okbox'>协议：query=生成图（或 GT），refs=目标字体 StyleImage <b>ref8</b>；"
        "test16 字体不在 E12 train228 内（held-out）。"
        f" temperature={_fmt(scorers.get('temperature'), 3)}。"
        " 主看 <b>mem_prob (latin+digit)</b>；假名/注音为 membership 外推。"
        " 加粗=该列最优（含 GT 上界行时一并比较）。</div>",
        "<table><thead><tr><th>方法</th><th>shot</th><th>n</th>"
        "<th>φ SC-R↑</th><th>mem_prob↑</th><th>mem_prob latin+digit↑</th>"
        "<th>φ SC-R latin+digit↑</th></tr></thead><tbody>",
    ]
    order = mids + (["GT"] if "GT" in e12["methods"] else [])
    raw = []
    for mid in order:
        a = e12["methods"].get(mid)
        if not a:
            continue
        meta = METHOD_META.get(mid, {})
        lab = label_of.get(mid, "GT（上界）" if mid == "GT" else mid)
        raw.append(
            [
                lab,
                meta.get("shot") if mid != "GT" else "—",
                _mean(a, "phi_sc_r", "n"),
                _mean(a, "phi_sc_r", "mean"),
                _mean(a, "mem_prob", "mean"),
                _mean(a, "mem_prob_latin_digit", "mean"),
                _mean(a, "phi_sc_r_latin_digit", "mean"),
            ]
        )
    pretty = _bold_best_cells(raw, [None, None, None, True, True, True, True])
    for cells in pretty:
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    rows.append("</tbody></table>")

    # generators-only highlight for mem_prob latin+digit
    gen_ids = [m["id"] for m in methods if m["id"] in e12["methods"]]
    if gen_ids:
        best_mid = max(gen_ids, key=lambda mid: _mean(e12["methods"][mid], "mem_prob_latin_digit", "mean") or -1)
        rows.append(
            f"<p class='cap'>生成方法间（不含 GT）：latin+digit mem_prob 最优 = <b>{label_of.get(best_mid, best_mid)}</b>。"
            " φ SC-R 可能高于 GT（已观察到），相对排序仍有效。</p>"
        )

    buckets = sorted({b for mid in mids for b in (e12["methods"].get(mid) or {}).get("by_bucket", {})})
    if buckets:
        rows.append("<h3>按语种 mem_prob↑（全字符）</h3><table><thead><tr><th>语种</th>")
        for mid in mids:
            rows.append(f"<th>{label_of.get(mid, mid)}</th>")
        rows.append("<th>GT</th></tr></thead><tbody>")
        for b in buckets:
            vals = [
                _mean(e12["methods"].get(mid), "by_bucket", b, "mem_prob", "mean")
                for mid in mids + ["GT"]
            ]
            best = max((x for x in vals if isinstance(x, float)), default=None)
            cells = [b]
            for v in vals:
                if not isinstance(v, float):
                    cells.append("—")
                else:
                    t = _fmt(v, 4)
                    if best is not None and abs(v - best) < 1e-12:
                        t = f"<b>{t}</b>"
                    cells.append(t)
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        rows.append("</tbody></table>")
    rows.append(
        "<p class='cap'>来源 <a href='e12_v51_scores.json'>e12_v51_scores.json</a> · "
        "眼测 <a href='http://127.0.0.1:8770/'>:8770</a> · "
        "不与 L1/SSIM 合成总分。</p>"
    )
    return "".join(rows)


def write_html(payload: dict) -> None:
    methods = []
    for m in payload["methods"]:
        mid = m["id"]
        meta = {**METHOD_META.get(mid, {}), **m}
        meta["id"] = mid
        meta["label"] = m.get("label") or METHOD_META.get(mid, {}).get("label", mid)
        methods.append(meta)
    payload = {**payload, "methods": methods, "ref8": REF8, "modes": MODES, "ui": "fair_axes_v1"}

    data_js = {
        "fonts": payload["fonts"],
        "chars": payload["chars"],
        "buckets": payload["buckets"],
        "methods": methods,
        "items": payload["items"],
        "ref8": REF8,
        "modes": MODES,
        "seed": SEED,
    }
    e12 = payload.get("e12_v51") or {}
    e12_path = OUT / "e12_v51_scores.json"
    if not e12 and e12_path.is_file():
        e12 = json.loads(e12_path.read_text(encoding="utf-8"))
    metric_html = metric_tables(payload)
    e12_html = e12_tables(e12, methods)
    html = f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Glyph Board · 公平轴 Mode A/B/C · test16×47</title>
<style>
:root{{--bg:#eef1f5;--card:#fff;--line:#d5dbe3;--muted:#5c6570;--ink:#1a1a1a;--accent:#1f4a6f;--warnbg:#fff4f0;--warnline:#e0b0a0;--okbg:#f0f6f2;--okline:#b7cfc0}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,"Noto Sans SC",sans-serif;background:var(--bg);color:var(--ink)}}
header{{background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}}
h1{{margin:0;font-size:1.2rem}} h2{{font-size:1.05rem;margin:0 0 8px}} h3{{font-size:.95rem;margin:12px 0 6px}}
.meta,.cap{{color:var(--muted);font-size:12px}}
main{{max-width:1320px;margin:16px auto;padding:0 14px 48px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px;margin:14px 0}}
.note{{background:#fff8e8;border:1px solid #e6d7a8;padding:8px 10px;margin:10px 0;font-size:12px}}
.warn{{background:var(--warnbg);border:1px solid var(--warnline);padding:8px 10px;margin:10px 0;font-size:12px;display:none}}
.warn.show{{display:block}}
.okbox{{background:var(--okbg);border:1px solid var(--okline);padding:8px 10px;margin:10px 0;font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left}}
.grid{{overflow-x:auto}}
table.g th,table.g td{{border:1px solid var(--line);padding:3px;text-align:center;vertical-align:bottom;font-size:11px;color:var(--muted)}}
table.g img{{width:72px;height:72px;image-rendering:pixelated;display:block;background:#fff;margin:0 auto}}
table.g td.ch{{font:700 15px/1.2 ui-serif,serif;color:var(--ink)}}
table.g .foot{{font-size:10px;line-height:1.2;margin-top:2px;color:var(--muted)}}
select,button.mode{{padding:6px 10px;border:1px solid var(--line);background:#fff;margin-right:6px;cursor:pointer;font:inherit}}
button.mode[aria-pressed="true"]{{background:var(--ink);color:#fff;border-color:var(--ink)}}
.pill{{display:inline-block;border:1px solid var(--line);padding:1px 8px;font-size:12px;margin-right:6px}}
.cond{{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;margin-top:8px}}
.cond .blk{{border:1px solid var(--line);padding:6px;background:#fafbfc}}
.cond .blk .lab{{font-size:11px;color:var(--muted);margin-bottom:4px}}
.cond .row{{display:flex;gap:4px;flex-wrap:wrap}}
.cond img{{width:56px;height:56px;image-rendering:pixelated;background:#fff;border:1px solid var(--line)}}
.tabs-metric{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}}
.tabs-metric button{{padding:5px 10px;border:1px solid var(--line);background:#fff;cursor:pointer;font:inherit}}
.tabs-metric button[aria-selected="true"]{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.hidden{{display:none}}
a{{color:var(--accent)}}
</style></head><body>
<header>
  <h1>Glyph Board · 按公平轴对比（同一页 Mode A / B / C）</h1>
  <div class="meta">test16 × 47 · DPM++20 CFG7.5 seed{SEED} · 共享：测试集/种子/采样器/Content · 不共享：shot 数、结构条件、ckpt 步数</div>
</header>
<main>
<div class="note">
  <b>怎么读：</b>默认用 Mode A 或 B 下结论；Mode C 仅浏览。<br/>
  像素 L1/SSIM/LPIPS 相对 GT，<b>仅为诊断</b>，不是风格 claim。
  · <a href="PI_BRIEFING_20260909.html">导师汇报</a>
  · <a href="http://127.0.0.1:8768/">Compare Portal</a>
  · <a href="http://127.0.0.1:8770/">E12 v5.1 眼测</a>
  · <a href="timeline_f2.html">F2 时间线</a>
  · <a href="timeline.html">F3 时间线</a>
</div>

<section class="card" id="progress"><h2>出图进度</h2><pre id="prog" class="meta">读取 status.json …</pre></section>

<section class="card">
  <h2>指标（双轨）</h2>
  <div class="tabs-metric">
    <button type="button" data-mpanel="diag" aria-selected="true">Diagnostic · vs GT</button>
    <button type="button" data-mpanel="style" aria-selected="false">Style · E12</button>
  </div>
  <div id="panel-diag">
    <p class="cap">下表含 shot / ckpt 列，避免把 1-shot 与 8-shot 当成同条件总分。</p>
    {metric_html}
    <p class="cap">L1↓ SSIM↑ LPIPS↓ 贴近像素真值；跨文字风格主张看 E12。</p>
  </div>
  <div id="panel-style" class="hidden">
    <p class="cap">生成器侧 E12 v5.1 打分（已并入主评测）。Scorer 自身 φ AUC≈0.901 · membership test ROC≈0.937。</p>
    {e12_html}
  </div>
</section>

<section class="card">
  <h2>对照图 · 公平轴</h2>
  <p>
    <span class="meta">Mode</span>
    <button type="button" class="mode" data-mode="A" aria-pressed="true">A · 1-shot 消融</button>
    <button type="button" class="mode" data-mode="B" aria-pressed="false">B · 8-ref 增量</button>
    <button type="button" class="mode" data-mode="C" aria-pressed="false">C · 全览</button>
  </p>
  <div id="mode-blurb" class="okbox"></div>
  <div id="mode-warn" class="warn"></div>
  <p>
    <label>字体 <select id="font"></select></label>
    <label>语种 <select id="bucket"><option value="">全部</option></select></label>
  </p>
  <div id="cond" class="cond"></div>
  <div class="grid" id="sheet" style="margin-top:12px"></div>
</section>
<p class="cap">生成于 {utc_now()} · ui=fair_axes_v1 · F2@75k 为中间落盘非最终 80k · F1/F3b 待评后进 Mode 槽</p>
</main>
<script>
const DATA = {json.dumps(data_js, ensure_ascii=False)};
const fontSel = document.getElementById('font');
const bucketSel = document.getElementById('bucket');
let mode = 'A';
DATA.fonts.forEach(f => {{ const o=document.createElement('option'); o.value=f; o.textContent=f; fontSel.appendChild(o); }});
const BNAME = {{digit:'数字', latin_upper:'拉丁大写', latin_lower:'拉丁小写', latin_ext:'拉丁扩展', hiragana:'平假名', katakana:'片假名', bopomofo:'注音'}};
DATA.buckets.forEach(b => {{ const o=document.createElement('option'); o.value=b; o.textContent=BNAME[b]||b; bucketSel.appendChild(o); }});
const byId = Object.fromEntries(DATA.methods.map(m => [m.id, m]));

function activeMethods(){{
  const ids = DATA.modes[mode].methods;
  return ids.map(id => byId[id]).filter(Boolean);
}}

function renderCond(){{
  const font = fontSel.value;
  const m = DATA.modes[mode];
  const sample = DATA.items.find(it => it.font===font);
  const content = sample ? sample.content : '';
  const style1 = sample ? sample.style : ('refs/style/'+font+'.png');
  let h = '';
  h += '<div class="blk"><div class="lab">Content（共享 · Noto Regular）</div>';
  if (content) h += '<img src="'+content+'" alt="content"/>';
  h += '<div class="cap">随所选语种字符变化（见下表）</div></div>';
  if (m.shot === 1){{
    h += '<div class="blk"><div class="lab">Style ×1（轴 A 可见输入）</div><img src="'+style1+'" alt="style1"/><div class="cap">优先「永」</div></div>';
  }} else if (m.shot === 8){{
    h += '<div class="blk"><div class="lab">Style ×8 ref8（轴 B 可见输入）</div><div class="row">';
    DATA.ref8.forEach((ch,i) => {{
      const cp = 'u'+ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0');
      h += '<div><img src="refs/style_ref8/'+font+'/'+cp+'.png" alt="'+ch+'" onerror="this.style.opacity=.2"/><div class="cap" style="text-align:center">'+ch+'</div></div>';
    }});
    h += '</div><div class="cap">F3 Support 当前与 ref8 相同字表（多通路，非额外 8 字）</div></div>';
  }} else {{
    h += '<div class="blk"><div class="lab">Style ×1（P1/E1/F0）</div><img src="'+style1+'" alt="style1"/></div>';
    h += '<div class="blk"><div class="lab">Style ×8（F2/F3）</div><div class="row">';
    DATA.ref8.forEach((ch) => {{
      const cp = 'u'+ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0');
      h += '<div><img src="refs/style_ref8/'+font+'/'+cp+'.png" alt="'+ch+'" onerror="this.style.opacity=.2"/><div class="cap" style="text-align:center">'+ch+'</div></div>';
    }});
    h += '</div></div>';
  }}
  document.getElementById('cond').innerHTML = h;
}}

function render(){{
  const font = fontSel.value;
  const bucket = bucketSel.value;
  const mdef = DATA.modes[mode];
  document.getElementById('mode-blurb').textContent = mdef.title + ' — ' + mdef.question + '。' + mdef.blurb;
  const warn = document.getElementById('mode-warn');
  if (mdef.warn){{
    warn.textContent = '⚠ ' + mdef.blurb;
    warn.classList.add('show');
  }} else {{
    warn.classList.remove('show');
  }}
  renderCond();
  const methods = activeMethods();
  const items = DATA.items.filter(it => it.font===font && (!bucket || it.bucket===bucket));
  const heads = ['字','Content','GT'].concat(methods.map(m => m.label));
  let h = '<table class="g"><thead><tr>'+heads.map(x=>'<th>'+x+'</th>').join('')+'</tr>';
  h += '<tr><th></th><th class="cap">共享</th><th class="cap">仅对照</th>';
  for (const m of methods){{
    h += '<th class="foot">'+ (m.shot||'?') +'-shot · '+ (m.ckpt||'') +'<br/>'+ (m.cond||'') +'</th>';
  }}
  h += '</tr></thead><tbody>';
  for (const it of items){{
    h += '<tr><td class="ch">'+it.char+'</td>';
    h += '<td><img src="'+it.content+'"/></td>';
    h += '<td><img src="'+it.gt+'"/></td>';
    for (const m of methods){{
      const src = (it.preds||{{}})[m.id];
      if (src) h += '<td><img src="'+src+'" onerror="this.style.opacity=.25"/></td>';
      else h += '<td><span class="cap">排队</span></td>';
    }}
    h += '</tr>';
  }}
  h += '</tbody></table>';
  document.getElementById('sheet').innerHTML = h;
}}

document.querySelectorAll('button.mode').forEach(btn => {{
  btn.addEventListener('click', () => {{
    mode = btn.dataset.mode;
    document.querySelectorAll('button.mode').forEach(b => b.setAttribute('aria-pressed', b===btn ? 'true':'false'));
    render();
  }});
}});
document.querySelectorAll('.tabs-metric button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tabs-metric button').forEach(b => b.setAttribute('aria-selected', b===btn ? 'true':'false'));
    document.getElementById('panel-diag').classList.toggle('hidden', btn.dataset.mpanel !== 'diag');
    document.getElementById('panel-style').classList.toggle('hidden', btn.dataset.mpanel !== 'style');
  }});
}});
fontSel.onchange = render; bucketSel.onchange = render; render();

async function tick(){{
  try {{
    const s = await (await fetch('status.json?t='+Date.now(), {{cache:'no-store'}})).json();
    const ms = s.methods||{{}};
    let t = '更新 '+ (s.updated_at||'') + ' · 阶段 ' + (s.phase||'') + '\\n';
    for (const [k,v] of Object.entries(ms)){{
      const tot = v.total||752;
      const d = (v.done||0)+(v.skipped||0);
      const pct = tot? Math.round(100*d/tot):0;
      t += k+' '+ (v.phase||'') + ' '+d+'/'+tot+' ('+pct+'%)';
      if (v.eta_s) t += '  ETA '+Math.round(v.eta_s/60)+' min';
      if (v.rate_per_s) t += '  '+v.rate_per_s+'/s';
      t += '\\n';
    }}
    document.getElementById('prog').textContent = t;
  }} catch(e) {{ document.getElementById('prog').textContent = String(e); }}
}}
tick(); setInterval(tick, 8000);
</script>
</body></html>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    idx_path = OUT / "browse_index.json"
    payload = json.loads(idx_path.read_text(encoding="utf-8"))
    fonts = payload["fonts"]
    stage_ref8(fonts)

    # Enrich methods metadata while preserving order/labels from browse_index
    enriched = []
    for m in payload["methods"]:
        mid = m["id"]
        meta = METHOD_META.get(mid, {})
        enriched.append(
            {
                **m,
                "shot": meta.get("shot"),
                "ckpt": meta.get("ckpt"),
                "axes": meta.get("axes"),
                "cond": meta.get("cond"),
            }
        )
    payload["methods"] = enriched
    payload["ref8"] = REF8
    payload["ui"] = "fair_axes_v1"
    payload["generated_at"] = utc_now()
    payload["note"] = (
        "Fair-axis UI: Mode A=1-shot P1/E1/F0; Mode B=8-ref F2/F3; Mode C=all browse. "
        "Style refs for A = refs/style; for B = refs/style_ref8. E12 v5.1 in Style tab."
    )
    e12_path = OUT / "e12_v51_scores.json"
    if e12_path.is_file():
        payload["e12_v51"] = json.loads(e12_path.read_text(encoding="utf-8"))
    mp = OUT / "metrics_summary.json"
    if mp.is_file():
        payload["metrics"] = json.loads(mp.read_text(encoding="utf-8"))
    # Attach style_refs on items (paths only; images staged once per font)
    for it in payload["items"]:
        stem = it["font"]
        it["style_refs_1"] = [it.get("style") or f"refs/style/{stem}.png"]
        it["style_refs_8"] = [f"refs/style_ref8/{stem}/{cp_of(ch)}.png" for ch in REF8]

    idx_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    write_html(payload)
    print("wrote", OUT / "index.html")
    print("staged", OUT / "refs/style_ref8", "fonts", len(fonts))
    if payload.get("e12_v51"):
        print("e12_v51 methods", list(payload["e12_v51"].get("methods", {})))


if __name__ == "__main__":
    main()
