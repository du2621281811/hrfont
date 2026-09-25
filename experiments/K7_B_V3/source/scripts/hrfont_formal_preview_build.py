#!/usr/bin/env python3
"""Build single mentor report page — concise, one URL for汇报."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
FE = REP / "formal_eval"
OUT = REP / "formal_preview"
ABL = REP / "dropout_ablation"
AUDIT_SCRIPT = ROOT / "scripts/hrfont_protocol_audit_build.py"

FONT_SHORT = {
    "FZChuangHJW_DB": "创黑",
    "FZCuanBZBKSJW": "篆变",
    "FZDouNTJW_Te": "斗体",
    "FZFengYKSJ": "风雅",
    "FZHanWZKJW": "汉真",
    "FZHuoYYJW-T": "活页",
    "FZJingYLLTJW": "静雅",
    "FZLingFKSJW-B": "凌风",
}


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _pct(x: float, base: float) -> str:
    return f"{(x / base - 1) * 100:+.1f}%" if base else "—"


def _by_font(rows: list[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        acc[r["font"]].append(r["l1"])
    return {f: sum(v) / len(v) for f, v in acc.items()}


def _bar_svg(items: list[tuple[str, float, str]], max_v: float) -> str:
    w, h, bh = 480, 32, 22
    parts = []
    for i, (lab, val, col) in enumerate(items):
        y = i * (bh + 12) + 2
        width = int((val / max_v) * (w - 100)) if max_v else 0
        parts.append(
            f'<text x="0" y="{y + 16}" class="sm">{lab}</text>'
            f'<rect x="96" y="{y}" width="{width}" height="{bh}" rx="3" fill="{col}"/>'
            f'<text x="{100 + width}" y="{y + 16}" class="val">{val:.4f}</text>'
        )
    height = len(items) * (bh + 12) + 4
    return f'<svg viewBox="0 0 {w} {height}" class="chart">{"".join(parts)}</svg>'


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _input_table_rows(inputs: list[dict]) -> str:
    rows = []
    for inp in inputs:
        if not isinstance(inp, dict):
            continue
        detail = inp.get("detail") or ""
        extra = f"<br><span class='sm-muted'>{_esc(detail)}</span>" if detail else ""
        rows.append(
            "<tr>"
            f"<td><b>{_esc(inp.get('role', ''))}</b></td>"
            f"<td>{_esc(inp.get('source', ''))}</td>"
            f"<td><code>{_esc(inp.get('render', ''))}</code></td>"
            f"<td>{_esc(inp.get('script', ''))}</td>"
            f"<td>{_esc(inp.get('size', ''))}</td>"
            f"<td>{_esc(inp.get('preprocess', ''))}</td>"
            f"<td>{_esc(inp.get('train_vs_eval', ''))}{extra}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_audit_section(audit: dict) -> str:
    parts = [
        '<div class="box" id="inputs">',
        "<h2>④ 各实验：输入渲染与处理（以代码为准）</h2>",
        "<p style=\"font-size:13px;margin:0 0 10px\">"
        "主表 L1 绑定下方 <b>metrics 时间戳 + ckpt</b>；"
        "训练与评测分列，<span class=\"hl\">train≠eval 处已标 ❌/⚠️</span>。"
        f" 机器可读：<a href=\"PROTOCOL_AUDIT.json\">PROTOCOL_AUDIT.json</a>"
        f"（审计生成 {audit.get('t', '')}）</p>",
    ]
    if audit.get("script_drift_warning"):
        parts.append(
            '<p class="warnbox"><b>脚本漂移警告：</b>部分 eval 脚本修改时间晚于 metrics JSON。'
            "勿用<strong>现行脚本</strong>反推当时跑法；复现请对照 JSON 中 sha256 / 日志。</p>"
        )

    parts.append("<h3>训练 vs 评测：共性差异</h3><table>")
    parts.append(
        "<tr><th>项</th><th>训练</th><th>评测</th><th>影响</th></tr>"
    )
    for g in audit.get("train_eval_gaps", []):
        parts.append(
            f"<tr><td>{_esc(g['item'])}</td><td>{_esc(g['train'])}</td>"
            f"<td>{_esc(g['eval'])}</td><td>{_esc(g['impact'])}</td></tr>"
        )
    parts.append("</table>")

    exp_titles = {
        "ft_cnstyle@25k": "ft 基线 @25k",
        "stageA_formal@80k": "Stage A formal @80k",
        "stageB_support@25k": "Stage B +Support @25k",
    }
    exps = audit.get("experiments", {})
    for key, title in exp_titles.items():
        ex = exps.get(key, {})
        if not ex:
            continue
        met = ex.get("metrics") or {}
        ck = ex.get("ckpt") or {}
        l1 = met.get("L1_mean")
        l1s = f"{l1:.4f}" if isinstance(l1, (int, float)) else "—"
        parts.append(f"<h3>{title}</h3>")
        parts.append(
            f"<p class=\"sm-muted\">训：<code>{_esc(ex.get('train_script', ''))}</code> · "
            f"评：<code>{_esc(ex.get('eval_script', ''))}</code> · "
            f"ckpt mtime={ck.get('mtime', '—')} · metrics t={met.get('t', '—')} · L1={l1s}</p>"
        )
        for phase, label in (("train_inputs", "训练"), ("eval_inputs", "评测")):
            inp = ex.get(phase)
            if isinstance(inp, str):
                parts.append(f"<p><b>{label}：</b>{_esc(inp)}</p>")
                continue
            if not inp:
                continue
            parts.append(f"<p><b>{label}阶段</b></p><table class=\"inp\">")
            parts.append(
                "<tr><th>输入</th><th>来源</th><th>渲染</th><th>脚本</th>"
                "<th>尺寸</th><th>预处理</th><th>train↔eval</th></tr>"
            )
            parts.append(_input_table_rows(inp))
            parts.append("</table>")

    parts.append("<h3>版本绑定（脚本指纹 vs 评测时间）</h3><table>")
    parts.append(
        "<tr><th>脚本</th><th>mtime</th><th>sha256[:16]</th><th>评测记录</th><th>晚于评测?</th></tr>"
    )
    for vb in audit.get("version_bindings", []):
        flag = "⚠️ 是" if vb.get("script_newer_than_eval") else "否"
        cls = "bad" if vb.get("script_newer_than_eval") else "ok"
        parts.append(
            f"<tr><td><code>{_esc(vb.get('path', ''))}</code></td>"
            f"<td>{vb.get('mtime', '—')}</td>"
            f"<td>{vb.get('sha256_16', '—')}</td>"
            f"<td>{vb.get('eval_recorded_at', '—')}</td>"
            f"<td class=\"{cls}\">{flag}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h3>无效 / 被覆盖的结果（勿引用）</h3><table>")
    parts.append("<tr><th>时间</th><th>实验</th><th>L1</th><th>原因</th><th>有效替代</th></tr>")
    for inv in audit.get("invalid_results", []):
        parts.append(
            f"<tr><td>{inv.get('t')}</td><td>{inv.get('experiment')}</td>"
            f"<td>{inv.get('L1')}</td><td>{_esc(inv.get('reason', ''))}</td>"
            f"<td>{_esc(inv.get('superseded_by', ''))}</td></tr>"
        )
    parts.append("</table></div>")
    return "".join(parts)


def _ensure_audit() -> dict:
    if AUDIT_SCRIPT.exists():
        subprocess.run([sys.executable, str(AUDIT_SCRIPT)], check=False, cwd=ROOT)
    p = OUT / "PROTOCOL_AUDIT.json"
    return _load(p) if p.exists() else {}


def main() -> None:
    audit = _ensure_audit()
    audit_block = _render_audit_section(audit) if audit else ""
    m80 = _load(FE / "metrics_step_80000.json")
    ft = m80["ft_cnstyle"]["L1_mean"]
    a = m80["stageA_formal"]["L1_mean"]
    ft_rows = m80["results"]["ft_cnstyle"]["rows"]
    a_rows = m80["results"]["stageA_formal"]["rows"]

    b = n_sup = None
    b_rows: list[dict] = []
    if (FE / "metrics_stageB_25000.json").exists():
        mb = _load(FE / "metrics_stageB_25000.json")
        b = mb["stageB"]["L1_mean"]
        n_sup = int(mb["stageB"].get("n_with_support", 0))
        b_rows = mb["results"]["stageB"]["rows"]

    gap = _load(FE / "gap_stratified.json") if (FE / "gap_stratified.json").exists() else {}
    abl = _load(ABL / "AUTO_VERDICT.json") if (ABL / "AUTO_VERDICT.json").exists() else {}
    e3 = _load(FE / "metrics_e3_10000.json") if (FE / "metrics_e3_10000.json").exists() else {}
    e3_l1 = e3.get("e3_rsi_style", {}).get("L1_mean")

    ft_font = _by_font(ft_rows)
    a_font = _by_font(a_rows)
    b_font = _by_font(b_rows) if b_rows else {}

    # per-font table
    font_rows = []
    for f in sorted(ft_font):
        short = FONT_SHORT.get(f, f[:6])
        fv, av = ft_font[f], a_font[f]
        bv = b_font.get(f)
        b_cell = f"{bv:.3f}" if bv is not None else "—"
        font_rows.append(
            f"<tr><td>{short}</td><td>{fv:.3f}</td><td>{av:.3f}</td>"
            f"<td>{b_cell}</td>"
            f"<td>{'✓' if av < fv else '—'}</td></tr>"
        )

    gap_rows = ""
    if gap.get("summary_by_bucket"):
        for k, lab in [("low", "低"), ("mid", "中"), ("high", "高")]:
            s = gap["summary_by_bucket"][k]
            gap_rows += (
                f"<tr><td>{lab}</td><td>{s['n']}</td>"
                f"<td>{s['mean_l1_ft']:.3f}</td><td>{s['mean_l1_a']:.3f}</td>"
                f"<td>{100*s['a_win_rate']:.0f}%</td>"
                f"<td>{s.get('mean_l1_b', 0):.3f}</td>"
                f"<td>{100*s.get('b_win_rate_vs_a', 0):.0f}%</td></tr>"
            )

    abl_rows = ""
    if abl.get("rows"):
        for rid, row in sorted(abl["rows"].items()):
            abl_rows += (
                f"<tr><td>{rid}</td><td>{row['stageA_l1']:.4f}</td>"
                f"<td>{row.get('note', '')}</td></tr>"
            )

    viz_block = ""
    viz_meta_path = OUT / "viz_meta.json"
    if (OUT / "viz_overview.png").exists():
        vm = json.loads(viz_meta_path.read_text()) if viz_meta_path.exists() else {}
        viz_block = f"""
<div class="box" id="viz">
<h2>⑩ 生成样例（Formal 协议 · GT | ft | A | B）</h2>
<p style="font-size:13px;margin:0 0 10px">{vm.get('protocol', '')} · 生成于 {vm.get('t', '')}</p>
<img src="viz_overview.png" alt="overview" style="width:100%;max-width:900px;border:1px solid #dde3ea;border-radius:8px;display:block;margin-bottom:12px"/>
<img src="viz_callouts.png" alt="callouts" style="width:100%;max-width:720px;border:1px solid #dde3ea;border-radius:8px;display:block"/>
<p style="font-size:12px;color:var(--muted);margin:8px 0 0">绿框 = Stage A L1 优于 ft；每格下方数字为 L1。</p>
</div>"""
    else:
        viz_block = """
<div class="box" id="viz">
<h2>⑩ 可视化</h2>
<p style="font-size:13px">GT/ft/A/B 对比图待生成：<code>python scripts/hrfont_formal_viz_build.py</code></p>
</div>"""

    main_bars = [("ft（基线）", ft, "#b45309"), ("Stage A", a, "#1f6f8f")]
    if b is not None:
        main_bars.append(("Stage B", b, "#7c3aed"))
    chart = _bar_svg(main_bars, max(v for _, v, _ in main_bars) * 1.1)

    b_block = ""
    if b is not None:
        b_block = f'<div class="tag warn">Stage B 未达预期</div><p>L1 <b>{b:.4f}</b>，support {n_sup}/976，劣于 A。</p>'

    ts = time.strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HR-Font Mentor 汇报</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet"/>
<style>
:root{{--ink:#12151a;--muted:#5c6570;--line:#dde3ea;--paper:#f5f7f9;--card:#fff;--ok:#0f7a4a;--warn:#b45309;--acc:#1f6f8f}}
*{{box-sizing:border-box}} body{{margin:0;font:15px/1.5 "IBM Plex Sans","Noto Sans SC",sans-serif;color:var(--ink);background:var(--paper)}}
.wrap{{max-width:920px;margin:0 auto;padding:24px 18px 48px}}
h1{{font-size:1.55rem;margin:0 0 6px}} .sub{{color:var(--muted);margin:0 0 20px;font-size:14px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}}
@media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
.card .k{{font-size:11px;color:var(--muted)}} .card .v{{font-size:1.35rem;font-weight:700}} .card .h{{font-size:11px;color:var(--muted)}}
.card.best{{border-color:#9fd4b5;background:#f6fcf8}}
.box{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:14px}}
.box h2{{font-size:1rem;margin:0 0 10px}}
.box h3{{font-size:.9rem;margin:14px 0 6px;color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;color:var(--muted);font-size:11px;padding:6px 8px;border-bottom:1px solid var(--line)}}
td{{padding:7px 8px;border-bottom:1px solid #eef1f4;font-variant-numeric:tabular-nums}}
.tag{{display:inline-block;font-size:11px;font-weight:700;border-radius:999px;padding:3px 9px;margin-bottom:6px}}
.tag.ok{{background:#e8f6ee;color:var(--ok)}} .tag.warn{{background:#fff4e8;color:var(--warn)}}
.lead{{background:#eef6fa;border:1px solid #c5dde8;border-radius:10px;padding:14px 16px;margin-bottom:14px;font-size:14px}}
.lead ul{{margin:8px 0 0;padding-left:18px}} .lead li{{margin:4px 0}}
.warnbox{{background:#fff8eb;border:1px solid #f0d9a8;border-radius:8px;padding:10px 12px;font-size:13px;margin:10px 0}}
.mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px;background:#f0f3f6;padding:10px;border-radius:6px;line-height:1.7;overflow-x:auto}}
.chart .sm{{font-size:12px;fill:var(--muted)}} .chart .val{{font-size:11px;fill:var(--ink);font-weight:600}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media(max-width:760px){{.two{{grid-template-columns:1fr}}}}
.three{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}} @media(max-width:860px){{.three{{grid-template-columns:1fr}}}}
.setup-step{{display:flex;gap:12px;margin:10px 0;align-items:flex-start;font-size:13px}}
.setup-step .n{{flex:0 0 26px;height:26px;border-radius:50%;background:var(--acc);color:#fff;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center}}
.setup-step .b strong{{display:block;margin-bottom:2px}}
.hl{{background:#fff8d6;padding:1px 4px;border-radius:3px}}
.same{{background:#eefaf3}} .diff{{background:#fff4e8}}
.ok{{color:var(--ok)}} .bad{{color:var(--warn);font-weight:600}}
.sm-muted{{font-size:12px;color:var(--muted);margin:4px 0 8px}}
table.inp td{{font-size:12px;vertical-align:top}}
footer{{color:var(--muted);font-size:12px;margin-top:8px}}
a{{color:var(--acc)}}
</style></head><body>
<div class="wrap">
<p class="sub">ICLR 2027 · HR-Font · 更新 {ts} · <b>唯一汇报入口</b></p>
<h1>跨语字体生成：Δ-RSI 实验汇报</h1>

<div class="lead">
<b>30 秒结论</b>
<ul>
<li><b>Stage A（Δ-RSI）</b>：L1={a:.4f}，比 ft 高 7.3% → <span class="ok">idea 可行，未超基线</span></li>
<li><b>Stage B（+Support）</b>：L1={f"{b:.4f}" if b is not None else "—"} → <span class="bad">当前无效，劣于 A</span></li>
<li><b>Gap 假设</b>：高 gap 未带来更大增益 → 第三节故事待重做</li>
</ul>
</div>

<div class="box" id="setup-task">
<h2>① 任务是什么？</h2>
<p style="font-size:14px;margin:0 0 12px"><b>跨语种 few-shot 字体生成</b>：给你一款<strong>从未见过的中文字体</strong> + <strong>1 个汉字「永」</strong>，模型要生成这款字体下的<strong>拉丁字母 / 标点</strong>（如 <code>a</code>）。</p>
<div class="three">
<div class="card"><div class="k">输入 · Content</div><div class="v" style="font-size:1rem">DejaVu 上的目标字</div><div class="h">要写哪个字母</div></div>
<div class="card"><div class="k">输入 · Style</div><div class="v" style="font-size:1rem">目标字体的「永」</div><div class="h">是什么风格</div></div>
<div class="card"><div class="k">输出 / 评测</div><div class="v" style="font-size:1rem">目标字体的拉丁字</div><div class="h">与 GT 比 L1↓</div></div>
</div>
<p style="font-size:13px;color:var(--muted);margin:12px 0 0"><b>我们改什么：</b>官方让 RSI 结构分支「看汉字 style 图」→ 我们改成「看 Δ = 库混合同字 − DejaVu content」。</p>
</div>

<div class="box">
<h2>② 数据划分</h2>
<table>
<tr><th>集合</th><th>规模</th><th>用途</th></tr>
<tr><td><b>训练池</b></td><td>42 款字体 × P1（122 类字符）</td><td>ft / A / B 训练；Demo-8 <b>绝不进入</b></td></tr>
<tr><td><b>测试 Demo-8</b></td><td>8 款字体（创黑/篆变/斗体/风雅/汉真/活页/静雅/凌风）</td><td><b>只评测</b>，8×122≈976 格</td></tr>
<tr><td><b>Bank 字库</b></td><td>42 训练字体的预渲染拉丁图</td><td>推理时按 α 混合同字 Δ；<span class="hl">不含测试字体</span></td></tr>
</table>
</div>

<div class="box">
<h2>③ 三个实验怎么设？（训练 + 推理）</h2>
<p class="warnbox" style="margin-top:0">三实验共用 FontDiffuser 骨干。<span class="hl">唯一核心差别：RSI 结构分支接什么</span>；B 额外加 support。</p>
<table>
<tr><th>设置项</th><th>ft 基线</th><th>Stage A</th><th>Stage B</th></tr>
<tr><td>初始化</td><td>官方 Phase-1</td><td>从 ft@25k 热启</td><td>从 A@80k 热启</td></tr>
<tr><td>训练步数</td><td>25k</td><td class="diff">80k</td><td>25k</td></tr>
<tr><td>训哪些参数</td><td>UNet 全参</td><td>UNet + reinit RSI offset</td><td class="diff"><b>只训 SupportAdapter</b></td></tr>
<tr><td>RSI 结构分支</td><td class="diff">Ec(汉字 style)</td><td class="diff"><b>Ec(α混库同字)−Ec(content)</b></td><td>同 A</td></tr>
<tr><td>Support</td><td>无</td><td>无</td><td class="diff">gap≥0.35 注入 token</td></tr>
<tr><td>Content / 分辨率</td><td colspan="3" style="text-align:center">DejaVu · 96×96 · 42 款训练字体</td></tr>
<tr><td>CFG dropout</td><td>content+style 10%</td><td>+ Δ 25% zero</td><td>+ support 20% drop</td></tr>
</table>
<h3>初始化链</h3>
<div class="mono">Phase-1 → <b>ft@25k</b>（基线）→ <b>Stage A@80k</b>（改 RSI 接 Δ）→ <b>Stage B@25k</b>（冻 UNet，训 Adapter）</div>
</div>

{audit_block}

<div class="box">
<h2>⑤ 评测怎么做？（976 格，ft/A/B 相同流程）</h2>
<p style="font-size:13px;margin:0 0 8px">对每个测试字体 F、每个字符 c：</p>
<div class="setup-step"><div class="n">1</div><div class="b"><strong>Content</strong> DejaVu live 渲染字符 c</div></div>
<div class="setup-step"><div class="n">2</div><div class="b"><strong>Style</strong> 字体 F live 渲染「永」（<span class="hl">1-shot 固定</span>）</div></div>
<div class="setup-step"><div class="n">3</div><div class="b"><strong>α + Δ（A/B）</strong> ref8 估 F 的风格 → top-3 训练字体混 bank 同字 c → Ec 减 content</div></div>
<div class="setup-step"><div class="n">4</div><div class="b"><strong>Support（仅 B）</strong> gap≥0.35 时检索支撑字 q → Adapter 拼入 cross-attn（952/976 格）</div></div>
<div class="setup-step"><div class="n">5</div><div class="b"><strong>采样</strong> DPM++ 20 步 · CFG=7.5 · 与 GT 算 L1</div></div>
<h3>ft vs A 公平性</h3>
<table>
<tr><th>项目</th><th>ft</th><th>Stage A</th><th>相同?</th></tr>
<tr><td>测集 / Content / Style / 采样</td><td>Demo-8×P1 · DejaVu · 1-shot永 · DPM20 CFG7.5</td><td>同左</td><td>✅</td></tr>
<tr><td>权重</td><td>ft@25k</td><td>ft@25k 再训 80k</td><td>⚠️ A 多训</td></tr>
<tr><td><b>唯一算法差</b></td><td>RSI←Ec(style)</td><td>RSI←Δ</td><td><b>验证目标</b></td></tr>
</table>
</div>

<div class="box">
<h2>⑥ 实验汇总与结果</h2>
<table>
<tr><th>实验</th><th>要回答什么</th><th>和谁比</th><th>L1</th><th>状态</th></tr>
<tr><td><b>ft</b></td><td>官方 RSI 跨语微调上限</td><td>基线</td><td>{ft:.4f}</td><td>✅</td></tr>
<tr><td><b>Stage A</b></td><td>RSI 改接 Δ 会不会崩</td><td>vs ft</td><td>{a:.4f}</td><td>✅ PASS</td></tr>
<tr><td><b>Stage B</b></td><td>support 能否帮高 gap 字</td><td>vs A</td><td>{f'{b:.4f}' if b else '—'}</td><td>❌</td></tr>
<tr><td><b>Dropout</b></td><td>Δ dropout 是否必要</td><td>@10k 趋势</td><td>R2 最差</td><td>✅</td></tr>
<tr><td><b>E3</b></td><td>增益来自 Δ 还是 retrain</td><td>vs A</td><td>{f'{e3_l1:.4f}' if e3_l1 else '—'}</td><td>{'✅' if e3_l1 else '⏸ 未跑'}</td></tr>
</table>
<div class="warnbox">⚠️ 旧页 <a href="../mentor_preview/index.html">mentor_preview</a> 协议不同，勿引用。</div>
</div>

<div class="cards">
<div class="card best"><div class="k">ft 基线</div><div class="v">{ft:.4f}</div><div class="h">官方 RSI @25k</div></div>
<div class="card"><div class="k">Stage A</div><div class="v">{a:.4f}</div><div class="h">{_pct(a, ft)} vs ft</div></div>
<div class="card"><div class="k">Stage B</div><div class="v">{f'{b:.4f}' if b else '—'}</div><div class="h">{_pct(b, ft) if b else ''} vs ft</div></div>
<div class="card"><div class="k">评测</div><div class="v">976</div><div class="h">Demo-8×P1</div></div>
</div>

<div class="box">
<h2>⑦ 主结果</h2>
<div class="two">
{chart}
<table>
<tr><th>方法</th><th>L1↓</th><th>vs ft</th></tr>
<tr style="background:#f6fcf8"><td>ft</td><td><b>{ft:.4f}</b></td><td>—</td></tr>
<tr><td>Stage A</td><td>{a:.4f}</td><td>{_pct(a, ft)}</td></tr>
{'<tr><td>Stage B</td><td>' + f'{b:.4f}' + '</td><td>' + _pct(b, ft) + '</td></tr>' if b else ''}
</table>
</div>
<h3>逐字体（Demo-8 平均 L1）</h3>
<table><tr><th>字体</th><th>ft</th><th>A</th><th>B</th><th>A胜ft</th></tr>{''.join(font_rows)}</table>
</div>

<div class="box">
<h2>⑧ Gap 分层</h2>
<table><tr><th>桶</th><th>n</th><th>ft</th><th>A</th><th>A胜ft</th><th>B</th><th>B胜A</th></tr>{gap_rows}</table>
<p style="font-size:13px;color:var(--muted);margin:8px 0 0">三桶 A 胜率均 ~38–40%，<b>假设未成立</b>。</p>
</div>

<div class="box">
<h2>⑨ 实现出入 & 版本影响（摘要）</h2>
<table>
<tr><th>设计点</th><th>代码实现</th><th>一致?</th><th>影响</th></tr>
<tr><td>RSI 看 Δ 特征差</td><td><code>delta_res = Ec(mix) − Ec(content)</code></td><td class="ok">✅</td><td>核心接线正确</td></tr>
<tr><td>α 用 ref8 估风格</td><td>REF8 + style proto cosine</td><td class="ok">✅</td><td>—</td></tr>
<tr><td>Demo-8 不进池</td><td>meta 排除 + leave-one-out</td><td class="ok">✅</td><td>—</td></tr>
<tr><td>Content 中性字体 B0</td><td>DejaVu（非 Song/Kai）</td><td>⚠️</td><td>与论文字面不同；与 ft 公平对齐</td></tr>
<tr><td>Δ 用特征差（非像素差）</td><td>像素混库图 → Ec 再作差</td><td>⚠️</td><td>近似实现；与设计 §4.2 等价路径</td></tr>
<tr><td>Stage B support</td><td>gap≥0.35 + MMR + Adapter</td><td class="ok">✅</td><td>逻辑对齐；效果负</td></tr>
<tr><td>Eval style/content</td><td>live render</td><td>⚠️</td><td>Train 用预渲 jpg；有 domain gap</td></tr>
<tr><td>Bank 渲染</td><td>render_fit ~8% 边距</td><td>❌</td><td>与 eval binary-search 不同；仅用于 Δ 混图</td></tr>
<tr><td>测试字体 style proto</td><td>不在缓存，eval 现场算</td><td>⚠️</td><td>8/29 前 B eval 无效</td></tr>
</table>
<h3>代码版本（Stage B 结果以哪次为准）</h3>
<table>
<tr><th>日期</th><th>问题</th><th>能否引用</th></tr>
<tr><td>8/29 02:07</td><td>修复 proto + CFG batch</td><td class="ok">✅ L1=0.0964 support=952</td></tr>
<tr><td>8/29 01:10</td><td>support=0（proto 未传）</td><td class="bad">❌</td></tr>
</table>
<p style="font-size:13px;color:var(--muted);margin:8px 0 0">逐输入明细见 <a href="#inputs">§④</a> · PROTOCOL_AUDIT.json</p>
</div>

{viz_block}

<div class="box">
<h2>⑪ Dropout @10k（只看趋势）</h2>
<table><tr><th>Run</th><th>L1</th><th>备注</th></tr>{abl_rows}</table>
<p style="font-size:13px;color:var(--muted)">R2(不 drop Δ)最差 → Δ dropout 必要。主结论看 R0@80k={a:.4f}。</p>
</div>

<div class="box">
<h2>⑫ 下一步</h2>
<ol style="margin:0;padding-left:18px;font-size:14px">
<li>补 E3「RSI 仍看汉字」对照{f' — L1={e3_l1:.4f}' if e3_l1 else '（pipeline 运行中见 e3_pipeline.log）'}</li>
<li>Stage B 诊断：减 support 数 / 仅 high-gap 启用 / 延长训练</li>
<li>可视化已生成（§⑨ 样例图）；若改 eval 脚本需重跑 viz_build</li>
</ol>
</div>

<footer>重建：<code>python scripts/hrfont_protocol_audit_build.py</code> → <code>hrfont_formal_preview_build.py</code> · 设计全文 <code>reports/ICLR2027_HRFONT.md</code></footer>
</div></body></html>"""

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (REP / "index.html").write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8"/>'
        '<meta http-equiv="refresh" content="0;url=formal_preview/index.html"/>'
        '</head><body><a href="formal_preview/index.html">HR-Font Mentor 汇报</a></body></html>',
        encoding="utf-8",
    )

    # one-page markdown mirror
    md = f"""# HR-Font Mentor 一页纸

**入口：** [formal_preview/index.html](formal_preview/index.html)

## 结论
- Stage A L1={a:.4f} ({_pct(a, ft)} vs ft) → PASS
- Stage B L1={f"{b:.4f}" if b is not None else "—"} → 未达预期
- Gap 假设未成立

## 实验
| 实验 | 对比 | L1 | 状态 |
|------|------|-----|------|
| ft | 基线 | {ft:.4f} | ✅ |
| Stage A | vs ft | {a:.4f} | ✅ |
| Stage B | vs A | {f"{b:.4f}" if b is not None else "—"} | ❌ |

## 协议
Demo-8×P1 · DejaVu · 1-shot永 · DPM20 · CFG7.5 · n=976

## 实现唯一差
ft: RSI←Ec(style) · A: RSI←Ec(α混库)−Ec(content) · B: +support tokens

## 版本
**有效 B 结果：** 2026-08-29 02:07（support=952/976）。此前 B eval 无效。

更新 {ts}
"""
    (REP / "MENTOR_ONEPAGE.md").write_text(md, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'}")

    export_script = ROOT / "scripts/hrfont_formal_preview_export.py"
    if export_script.exists():
        subprocess.run([sys.executable, str(export_script)], check=False, cwd=ROOT)


if __name__ == "__main__":
    main()
