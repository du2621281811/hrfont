#!/usr/bin/env python3
"""Rebuild mentor_preview/index.html from latest eval JSON + trend data."""
from __future__ import annotations

import json
import re
import statistics as stats
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"
OUT = REP / "mentor_preview"


def short_font(name: str) -> str:
    return name.replace("FZ", "").replace("JW", "")[:14]


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def fmt_delta(x: float) -> str:
    return f"{x:+.3f}"


def win_cls(wins: int, n: int) -> str:
    if wins >= n * 0.75:
        return "pos"
    if wins <= n * 0.375:
        return "neg"
    return ""


def extract_svgs(old_html: str) -> list[str]:
    return re.findall(r"<div class=\"chart-box\"><svg[\s\S]*?</svg></div>", old_html)


def by_char_cells(cells: list[dict]) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        d[c["ch"]].append(c)
    return d


def combined_rows(rows: list[dict]) -> str:
    html = ""
    best_test_step = None
    test_l1s = [(r["step"], r["test_l1"]) for r in rows if "test_l1" in r]
    if test_l1s:
        best_test_step = min(test_l1s, key=lambda x: x[1])[0]
    for r in rows:
        step = r["step"]
        cls = "best" if step == best_test_step else ""
        star = " ★" if step == best_test_step else ""
        tr = fmt3(r["train_loss"]) if "train_loss" in r else "—"
        vl = fmt3(r["val_l1"]) if "val_l1" in r else "—"
        tl = fmt3(r["test_l1"]) if "test_l1" in r else "—"
        vw = pct(r["val_win_rate"]) if "val_win_rate" in r else "—"
        tw = pct(r["test_win_rate"]) if "test_win_rate" in r else "—"
        html += (
            f'<tr class="{cls}"><td>{step:,}{star}</td><td>{tr}</td><td>{vl}</td><td>{tl}</td>'
            f'<td>{vw}</td><td>{tw}</td></tr>\n'
        )
    return html


def main() -> None:
    status = load_json(REP / "STATUS.json") if (REP / "STATUS.json").exists() else {}
    train_step = status.get("step", "?")
    train_loss_now = status.get("loss")

    trend = load_json(OUT / "trend/trend.json")
    ext = load_json(OUT / "extended_demo8_eval.json")
    callouts = load_json(OUT / "viz_callouts.json") if (OUT / "viz_callouts.json").exists() else {}
    viz_meta = load_json(OUT / "viz_test_meta.json") if (OUT / "viz_test_meta.json").exists() else {}
    curves_path = OUT / "train_val_test_curves.json"
    charts_path = OUT / "charts.json"
    curves = load_json(curves_path) if curves_path.exists() else {}
    charts = load_json(charts_path) if charts_path.exists() else {}

    subset_best = min(trend["series"], key=lambda r: r["mean_l1_stageA"])

    results = ext.get("results", {})
    if not results:
        raise SystemExit("extended_demo8_eval.json missing or empty — run hrfont_e2_extended_demo8_eval.py first")
    latest_key = max(results, key=lambda k: int(k))
    r_latest = results[latest_key]
    r70 = results.get("70000", r_latest)
    ft_l1_full = r_latest.get("mean_l1_ft") or r70.get("mean_l1_ft", 0)
    best_test = curves.get("best_test") or {"step": int(latest_key), "l1": r_latest["mean_l1_a"]}
    best_val = curves.get("best_val") or {"step": subset_best["step"], "l1": subset_best["mean_l1_stageA"]}
    splits = curves.get("splits", {})
    combined = curves.get("combined_at_eval_steps", [])

    bust = int(time.time())
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # font table rows @70k and latest
    font_order = ext["fonts"]

    def font_rows(result: dict) -> str:
        rows = []
        for f in font_order:
            fr = result["by_font"][f]
            w, n = fr["wins"], fr["n"]
            cls = win_cls(w, n)
            rows.append(
                f"<tr class=\"{cls}\"><td>{short_font(f)}</td>"
                f"<td>{fmt3(fr['mean_a'])}</td><td>{fmt3(fr['mean_ft'])}</td>"
                f"<td class=\"{'pos' if fr['mean_a'] < fr['mean_ft'] else 'neg'}\">"
                f"{fmt_delta(fr['mean_ft'] - fr['mean_a'])}</td>"
                f"<td>{w}/{n}</td></tr>"
            )
        return "\n".join(rows)

    # char table on full demo @70k
    byc = by_char_cells(r70["cells"])
    char_rows = []
    for ch in ext["chars"]:
        cs = byc[ch]
        ma = stats.mean(c["l1_a"] for c in cs)
        mf = stats.mean(c["l1_ft"] for c in cs)
        wins = sum(1 for c in cs if c["l1_a"] < c["l1_ft"])
        char_rows.append(
            f"<tr><td class=\"ch\">{ch}</td><td>{fmt3(mf)}</td><td>{fmt3(ma)}</td>"
            f"<td class=\"{'pos' if ma < mf else 'neg'}\">{fmt_delta(mf - ma)}</td>"
            f"<td>{wins}/{len(cs)}</td></tr>"
        )

    win_chips = ""
    for c in callouts.get("wins", [])[:4]:
        win_chips += (
            f'<span class="chip win"><b>A更好</b> {short_font(c["font"])} · {c["ch"]} · Δ{c["delta"]:+.2f}</span>'
        )
    lose_chips = ""
    for c in callouts.get("losses", [])[:3]:
        lose_chips += (
            f'<span class="chip lose"><b>A更差</b> {short_font(c["font"])} · {c["ch"]} · Δ{c["delta"]:+.2f}</span>'
        )

    mean_ft_dj = viz_meta.get("mean_l1_ft_dj")
    ft_dj_hint = f" · ft·DJ={fmt3(mean_ft_dj)}" if mean_ft_dj is not None else ""
    viz_step = viz_meta.get("best_ckpt_step", viz_meta.get("best_step", best_test["step"]))
    viz_wins = viz_meta.get("wins", r_latest.get("wins", "?"))
    viz_n = viz_meta.get("n", r_latest.get("n", 64))
    font_cards = ""
    for img_name in viz_meta.get("font_images", []):
        font_key = img_name.replace("font_", "").replace(".png", "")
        font_cards += f"""
    <div class="viz-card">
      <img src="viz_test/{img_name}?t={bust}" alt="{font_key}" loading="lazy"/>
    </div>"""
    if not font_cards:
        font_cards = '<p class="hint">测试图未生成 — 运行 hrfont_e2_rebuild_test_viz.py</p>'

    if charts:
        charts_html = (
            f'<div class="chart-box">{charts["dual"]}</div>\n'
            f'<div class="two" style="margin-top:14px">'
            f'<div class="chart-box">{charts.get("val_l1", "")}</div>'
            f'<div class="chart-box">{charts.get("test_l1", "")}</div>'
            f"</div>"
        )
    else:
        charts_html = '<p class="hint">图表缺失 — 先运行 hrfont_e2_train_val_analysis.py</p>'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HR-Font 中途评测 · Mentor</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet"/>
<style>
:root {{
  --ink:#12151a; --muted:#5c6570; --line:#d8dee6; --paper:#f6f7f9;
  --card:#fff; --good:#0f7a4a; --warn:#b45309; --accent:#1f6f8f;
  --shadow:0 1px 2px rgba(18,21,26,.04), 0 8px 24px rgba(18,21,26,.06);
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font:16px/1.55 "IBM Plex Sans","Noto Sans SC",system-ui,sans-serif; }}
.wrap {{ max-width:1040px; margin:0 auto; padding:32px 22px 72px; }}
.eyebrow {{ color:var(--muted); font-size:13px; letter-spacing:.04em; margin:0 0 8px; }}
h1 {{ font-size:clamp(1.6rem,3vw,2rem); font-weight:700; margin:0 0 10px; letter-spacing:-.02em; line-height:1.2; }}
.lede {{ color:var(--muted); max-width:44rem; margin:0 0 28px; }}
.verdict {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px 22px; box-shadow:var(--shadow); margin-bottom:22px; }}
.verdict .tag {{ display:inline-block; font-size:12px; font-weight:700; color:var(--good); background:#e8f6ee; border-radius:999px; padding:4px 10px; margin-bottom:10px; }}
.verdict p {{ margin:0; font-size:1.05rem; }}
.verdict strong {{ color:var(--good); }}
.note {{ background:#fff8eb; border:1px solid #f0d9a8; border-radius:12px; padding:14px 16px; margin-bottom:18px; font-size:14px; color:#6b4e16; }}
.note b {{ color:#3d2f0a; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:22px; }}
@media (max-width:720px) {{ .metrics {{ grid-template-columns:1fr 1fr; }} }}
.metric {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; box-shadow:var(--shadow); }}
.metric .k {{ color:var(--muted); font-size:12px; font-weight:500; }}
.metric .v {{ font-size:1.55rem; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums; }}
.metric .h {{ color:var(--muted); font-size:12px; margin-top:4px; }}
.metric.best {{ border-color:#9fd4b5; background:linear-gradient(180deg,#f3fbf6,#fff); }}
.metric.warn {{ border-color:#f0d9a8; background:linear-gradient(180deg,#fffaf0,#fff); }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px 22px; box-shadow:var(--shadow); margin-bottom:18px; }}
section h2 {{ font-size:1.05rem; font-weight:700; margin:0 0 6px; }}
section h3 {{ font-size:.95rem; font-weight:600; margin:18px 0 8px; }}
.hint {{ color:var(--muted); font-size:13px; margin:0 0 14px; }}
.charts {{ display:grid; gap:18px; }}
.chart-box {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#fff; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ text-align:left; color:var(--muted); font-weight:600; font-size:12px; padding:8px 10px; border-bottom:1px solid var(--line); }}
td {{ padding:9px 10px; border-bottom:1px solid #eef1f4; font-variant-numeric:tabular-nums; }}
tr.best {{ background:#eefaf3; }}
.pos {{ color:var(--good); font-weight:600; }}
.neg {{ color:var(--warn); font-weight:600; }}
.ch {{ font-family:ui-monospace,Menlo,monospace; font-weight:700; }}
.viz {{ width:100%; border-radius:10px; border:1px solid var(--line); background:#fff; display:block; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:10px 0 0; color:var(--muted); font-size:13px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle; }}
.l-gt i {{ background:#0f7a4a; }} .l-ft i {{ background:#b45309; }} .l-ft-dj i {{ background:#7848a0; }} .l-a i {{ background:#1f6f8f; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 4px; }}
.chip {{ font-size:12px; padding:5px 10px; border-radius:8px; background:#f3f5f7; color:var(--muted); }}
.chip.win {{ background:#e8f6ee; color:var(--good); }}
.chip.lose {{ background:#fff4e8; color:var(--warn); }}
.two {{ display:grid; grid-template-columns:1.1fr .9fr; gap:16px; }}
@media (max-width:860px) {{ .two {{ grid-template-columns:1fr; }} }}
.missing {{ display:grid; grid-template-columns:1fr 1fr; gap:10px 18px; margin:0; padding:0; list-style:none; }}
@media (max-width:640px) {{ .missing {{ grid-template-columns:1fr; }} }}
.missing li {{ background:#f3f5f7; border-radius:10px; padding:12px 14px; font-size:14px; color:var(--muted); }}
.missing b {{ color:var(--ink); display:block; margin-bottom:4px; }}
.viz-legend-bar {{ display:flex; flex-wrap:wrap; gap:16px; align-items:center; padding:12px 14px; background:#f3f5f7; border-radius:10px; margin:0 0 16px; font-size:13px; color:var(--muted); }}
.viz-legend-bar b {{ color:var(--ink); }}
.viz-scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:#fff; padding:8px; margin-bottom:16px; }}
.viz-scroll img {{ display:block; max-width:none; height:auto; }}
.viz-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }}
@media (max-width:860px) {{ .viz-grid {{ grid-template-columns:1fr; }} }}
.viz-card {{ border:1px solid var(--line); border-radius:10px; overflow:hidden; background:#fff; }}
.viz-card img {{ width:100%; display:block; }}
.viz-block {{ margin-top:22px; }}
footer {{ margin-top:10px; color:var(--muted); font-size:12px; }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:.88em; background:#eef1f4; padding:1px 5px; border-radius:4px; }}
</style>
</head>
<body>
<div class="wrap">
  <p class="eyebrow">ICLR 2027 · HR-Font · Stage A 中途评测（Mentor 页）</p>
  <h1>Δ-RSI 接线有没有用？</h1>
  <p class="lede">主表协议：<b>Demo-8 × AaOoRg8S × 25-step × seed 42</b>，对照官方 ft_cnstyle 接线。
  本页是中途探针，不是论文终表。</p>

  <div class="verdict">
    <div class="tag">一句话结论（训练 / 验证 / 测试分开看）</div>
    <p>训练集（42 字体，不含 Demo-8）loss 仍在降（当前 step {train_step} ≈ {train_loss_now or '?'})；
    <strong>测试集</strong>（全 Demo-8，64 格）最佳 L1 <strong>{fmt3(best_test['l1'])}</strong> @ {best_test['step']:,}，
    最新 @ {latest_key} 为 {fmt3(r_latest['mean_l1_a'])}（胜率 {pct(r_latest['win_rate'])}）。
    <strong>验证集</strong>（4 Demo 字体，32 格）最佳 {fmt3(best_val['l1'])} @ {best_val['step']:,}（曾 84% 胜率，<strong>不能当主结论</strong>）。
    train ↓ 与 test L1 已分叉 → 应早停选 test 最佳 ckpt。</p>
  </div>

  <div class="note">
    <b>三套数据：</b>
    <b>训练</b> = {splits.get('train', {}).get('fonts', 42)} 字体 Latin 对 · loss = noise MSE + 0.5×offset（<b>无 VGG</b>）；
    <b>验证</b> = Demo-8 中 {len(splits.get('val', {}).get('fonts', trend['protocol']['fonts']))} 字体 · 生成 L1；
    <b>测试</b> = 全 Demo-8 · 64 格 · 生成 L1（论文主协议）。
    训练 loss 与 L1 不同量纲，不能混为一谈。
  </div>

  <div class="metrics">
    <div class="metric best">
      <div class="k">测试 L1 最佳</div>
      <div class="v">{fmt3(best_test['l1'])}</div>
      <div class="h">Demo-8 64 格 @ {best_test['step']:,}</div>
    </div>
    <div class="metric">
      <div class="k">测试 @最新</div>
      <div class="v">{fmt3(r_latest['mean_l1_a'])}</div>
      <div class="h">step {latest_key} · 胜 {pct(r_latest['win_rate'])}</div>
    </div>
    <div class="metric">
      <div class="k">验证最佳（子集）</div>
      <div class="v">{fmt3(best_val['l1'])}</div>
      <div class="h">32 格 @ {best_val['step']:,}</div>
    </div>
    <div class="metric warn">
      <div class="k">训练 loss 现值</div>
      <div class="v">{train_loss_now if train_loss_now is not None else '—'}</div>
      <div class="h">step {train_step} · 42 字体</div>
    </div>
  </div>

  <section>
    <h2>1. 训练 vs 验证 vs 测试（三条曲线）</h2>
    <p class="hint">灰 = 训练 total loss（1k 步平滑）；蓝 = 验证 L1；绿虚 = 测试 L1。橙虚 = ft 测试基线。</p>
    <div class="charts">
{charts_html}
    </div>
    <h3>各 checkpoint 对齐表</h3>
    <p class="hint">同一训练 step 上的 train loss 与 val/test L1（评测步不一定连续）。★ = 测试 L1 最低。</p>
    <table>
      <thead><tr><th>Step</th><th>Train loss</th><th>Val L1</th><th>Test L1</th><th>Val 胜率</th><th>Test 胜率</th></tr></thead>
      <tbody>
{combined_rows(combined)}
      </tbody>
    </table>
  </section>

  <section>
    <h2>2. Demo-8 测试集全量数值</h2>
    <p class="hint">8 字体 × 8 字 = 64 格 · 同一 seed · 数据：<code>extended_demo8_eval.json</code></p>
    <div class="two">
      <div>
        <h3>@ step 70,000</h3>
        <table>
          <thead><tr><th>字体</th><th>A</th><th>ft</th><th>Δ</th><th>胜</th></tr></thead>
          <tbody>
{font_rows(r70)}
          </tbody>
        </table>
        <p class="hint">汇总：A={fmt3(r70['mean_l1_a'])} · ft={fmt3(r70['mean_l1_ft'])} · {r70['wins']}/{r70['n']}</p>
      </div>
      <div>
        <h3>@ step {latest_key}（最新）</h3>
        <table>
          <thead><tr><th>字体</th><th>A</th><th>ft</th><th>Δ</th><th>胜</th></tr></thead>
          <tbody>
{font_rows(r_latest)}
          </tbody>
        </table>
        <p class="hint">汇总：A={fmt3(r_latest['mean_l1_a'])} · ft={fmt3(r_latest['mean_l1_ft'])} · {r_latest['wins']}/{r_latest['n']}</p>
      </div>
    </div>
    <h3>按字符（Demo-8 平均 @70k）</h3>
    <table>
      <thead><tr><th>字</th><th>ft</th><th>A@70k</th><th>Δ</th><th>胜/8</th></tr></thead>
      <tbody>
{chr(10).join(char_rows)}
      </tbody>
    </table>
  </section>

  <section>
    <h2>3. 测试集视觉对比（全 Demo-8）</h2>
    <p class="hint">8 字体 × 8 字 = 64 格 · @ step {viz_step:,}（测试 L1 最佳）· 同 seed · 绿框=A 胜 ft · 橙框=A 负</p>

    <div class="viz-legend-bar">
      <b>每格四行：</b>
      <span class="l-gt"><i></i>GT 同字体拉丁真值</span>
      <span class="l-ft"><i></i>ft·B0（FZKTJW content，当前 HR 协议）</span>
      <span class="l-ft-dj"><i></i>ft·DJ（DejaVu @96，ft_cnstyle 原渲染）</span>
      <span class="l-a"><i></i>Stage A Δ-RSI（content 同 B0）</span>
      <span>数字 = 该图 L1{ft_dj_hint}</span>
    </div>

    <h3>3a. 全览（8 字体 · 可横向滚动）</h3>
    <div class="viz-scroll">
      <img src="viz_test/demo8_grid_2x4.png?t={bust}" alt="Demo-8 full grid"/>
    </div>

    <h3>3b. 分字体明细（8 张 · 每字体 8 字）</h3>
    <div class="viz-grid">
{font_cards}
    </div>
    <p class="hint">汇总 @ {viz_step:,}：A 胜 ft·B0 <b>{viz_wins}/{viz_n}</b>{ft_dj_hint}</p>
    <div class="chips">{win_chips}</div>
    <div class="chips">{lose_chips}</div>

    <div class="viz-block">
      <h3>3c. 难例放大（HuoYY / FengY / JingY / HanW · 8/R/O/g）</h3>
      <p class="hint">测试集里 historically 难赢 ft 的字体与字符；右侧 ΔL1 = ft − A（正= A 更好）</p>
      <div class="viz-scroll">
        <img src="viz_test/hardcases.png?t={bust}" alt="hard cases"/>
      </div>
    </div>

    <div class="viz-block">
      <h3>3d. 训练步数对比（4 字体 × 4 字 · 70k / 100k / 145k）</h3>
      <p class="hint">同一 seed 下看随 step 字形变化；用于判断何时停训</p>
      <div class="viz-scroll">
        <img src="viz_test/step_compare.png?t={bust}" alt="step compare"/>
      </div>
    </div>
  </section>

  <div class="two">
    <section>
      <h2>4. 验证集 · 4 字体子集分步（附录）</h2>
      <p class="hint">mid-sweep · 32 格 · 验证集子集，不作论文主结论</p>
      <table>
        <thead><tr><th>Step</th><th>A L1</th><th>ft</th><th>Δ</th><th>胜率</th></tr></thead>
        <tbody>
"""
    for r in trend["series"]:
        step = r["step"]
        la = r["mean_l1_stageA"]
        lf = r["mean_l1_ft_cnstyle"]
        wr = r["win_rate"]
        cls = "best" if step == subset_best["step"] else ""
        star = " ★" if step == subset_best["step"] else ""
        html += (
            f'          <tr class="{cls}"><td>{step:,}{star}</td><td>{la:.3f}</td><td>{lf:.3f}</td>'
            f'<td class="pos">{lf - la:+.3f}</td><td>{pct(wr)} ({r["wins"]}/{r["n"]})</td></tr>\n'
        )

    html += f"""        </tbody>
      </table>
    </section>
    <section>
      <h2>5. 分析与风险</h2>
      <ul class="missing">
        <li><b>分叉</b>训练 loss 持续 ↓，验证/测试 L1 在 70k 后 plateau → 典型 early-stop 信号</li>
        <li><b>验证≠测试</b>验证 84% @70k，测试仅 66%；子集不能代替 Demo-8</li>
        <li><b>选 ckpt</b>按测试 L1 最低 @ {best_test['step']:,}，不是 train loss 最低</li>
        <li><b>难例</b>FengY/HuoYY 打平；字符 8 偏弱；字体间 70k→最新有 trade-off</li>
        <li><b>还缺</b>逐步 val diffusion loss、LPIPS/OCR、Stage B</li>
      </ul>
    </section>
  </div>

  <footer>
    数据 <code>train_val_test_curves.json</code> · <code>extended_demo8_eval.json</code> · <code>trend/trend.json</code>
    · 生成 {now} · 训练 step {train_step}
  </footer>
</div>
</body>
</html>
"""

    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} ({len(html)} bytes)")
    print(f"demo8 @70k {r70['wins']}/{r70['n']} @latest {r_latest['wins']}/{r_latest['n']}")


if __name__ == "__main__":
    main()
