#!/usr/bin/env python3
"""Build FD CN→CN vs CN→West protocol + results report for mentor screenshots."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/retrain_v2/fd_protocol_report"
ASSETS = OUT / "assets"

CN2CN_EX = ROOT / "reports/retrain_v2/fd_cn2cn_protocol_examples"
CN2WEST_SHEET = ROOT / "reports/retrain_v2/fd_epoch_gallery/ft_cnstyle/sheet_FZChuangHJW_DB.png"
E10_SUM = ROOT / "reports/retrain_v2/e10_mid_review/summary.json"
CNSTYLE_RES = ROOT / "reports/retrain_v2/FD_CNSTYLE_RESULTS.json"


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir()

    # --- assets ---
    if (CN2CN_EX / "overview_infer.png").exists():
        shutil.copy2(CN2CN_EX / "overview_infer.png", ASSETS / "cn2cn_overview.png")
    ex_one = CN2CN_EX / "assets" / "FZFengYKSJ_一.png"
    if ex_one.exists():
        shutil.copy2(ex_one, ASSETS / "cn2cn_panel_一.png")
    if CN2WEST_SHEET.exists():
        shutil.copy2(CN2WEST_SHEET, ASSETS / "cn2west_sheet.png")

    e10 = load_json(E10_SUM)
    star = next(m for m in e10["methods"] if m.get("star"))
    fd0 = next(m for m in e10["methods"] if m["tag"] == "official")
    cnstyle = load_json(CNSTYLE_RES)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FontDiffuser 协议与结果 · 中→中 vs 中→西</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--ink:#12151a;--muted:#5c6570;--line:#dde3ea;
  --cn2cn:#1f6f6a;--cn2west:#7c4a03;--ok:#0f7a4a;--accent:#2563eb}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,"PingFang SC","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px 18px 64px}}
h1{{font-size:1.55rem;margin:0 0 6px;line-height:1.25}}
.sub{{color:var(--muted);margin:0 0 22px;font-size:14px}}
.section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:0 0 20px}}
.section h2{{margin:0 0 4px;font-size:1.22rem;display:flex;align-items:center;gap:10px}}
.badge{{font-size:11px;font-weight:700;padding:3px 10px;border-radius:999px;color:#fff}}
.badge.cn2cn{{background:var(--cn2cn)}}
.badge.cn2west{{background:var(--cn2west)}}
.tagline{{color:var(--muted);font-size:13px;margin:0 0 14px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:820px){{.grid2{{grid-template-columns:1fr}}}}
.box{{border:1px solid var(--line);border-radius:8px;padding:12px 14px;background:#fafbfc}}
.box h3{{margin:0 0 8px;font-size:14px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:#f0f3f6;color:var(--muted);font-weight:600}}
td code{{font-size:12px;background:#eef2f6;padding:1px 5px;border-radius:3px}}
.hl{{background:#fff8d6;padding:1px 4px;border-radius:3px}}
.result-row{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:14px 0 4px}}
@media(max-width:720px){{.result-row{{grid-template-columns:repeat(2,1fr)}}}}
.kpi{{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fff;text-align:center}}
.kpi .v{{font-size:1.45rem;font-weight:700;margin:4px 0}}
.kpi .l{{font-size:12px;color:var(--muted)}}
.kpi.star{{border-color:var(--ok);background:#f3faf7}}
.kpi.star .v{{color:var(--ok)}}
.fig{{margin-top:14px}}
.fig img{{width:100%;max-width:980px;border:1px solid var(--line);border-radius:8px;display:block}}
.figcap{{font-size:12.5px;color:var(--muted);margin-top:6px}}
.warn{{background:#fff4e6;border:1px solid #f0d8a8;border-radius:8px;padding:10px 12px;font-size:13px;margin:14px 0 0}}
.links{{font-size:13px;margin-top:10px}}
.links a{{color:var(--accent);margin-right:14px}}
.footer{{font-size:12px;color:var(--muted);margin-top:24px}}
</style></head><body><div class="wrap">

<h1>FontDiffuser 实验设置与结果</h1>
<p class="sub">两条独立主线 · 训练/测试图像渲染与处理 · 主结果数字 · 真实示例图（可直接截图汇报）</p>

<!-- ========== CN→CN ========== -->
<div class="section">
<h2><span class="badge cn2cn">E10</span> 中→中（CN→CN）</h2>
<p class="tagline">汉字 content → 目标中文字体 · 253 训练字体 · Demo-8 探针 3×31=93 对</p>

<div class="grid2">
<div class="box">
<h3>训练 · 图像渲染与读入</h3>
<table>
<tr><th>项</th><th>设置</th></tr>
<tr><td>数据包</td><td><code>fontdiffuser_cn2cn_p253</code></td></tr>
<tr><td>Content</td><td><code>R1/content_96</code> · FZKTJW · 逐字搜 size/dx/dy · ink≈0.13 · <span class="hl">96 原生</span></td></tr>
<tr><td>Style</td><td><code>R1/style96/&lt;font&gt;/</code> · 同字体中文池 ~357 字 · <b>随机抽 1</b>（非固定 ref8）</td></tr>
<tr><td>Target/GT</td><td>与 Style 同源 PNG → jpg · 每字体固定 S_f · ink≈0.28</td></tr>
<tr><td>读入</td><td>Resize(96) · ToTensor · Normalize(0.5) · drop 0.1</td></tr>
<tr><td>损失</td><td>noise MSE + perceptual 0.01 + offset 0.5 · 无 SCR</td></tr>
<tr><td>Run</td><td><code>ft_cn2cn_p253</code> → 续训至 abs <b>18000</b></td></tr>
</table>
</div>
<div class="box">
<h3>测试/评测 · 图像渲染与推理</h3>
<table>
<tr><th>项</th><th>设置</th></tr>
<tr><td>Content</td><td><code>R1/content_96</code>（与训练同渲，FZKTJW）</td></tr>
<tr><td>Style</td><td><code>R1/style96</code> · 固定 1-shot「<b>永</b>」</td></tr>
<tr><td>GT</td><td><code>R1/style96/&lt;font&gt;/uXXXX.png</code> 目标汉字真值</td></tr>
<tr><td>测集</td><td>3 字体（风雅/创粗/凌飞）× 31 字（训过14+未训17）</td></tr>
<tr><td>推理</td><td>DPM++ 20 步 · CFG 7.5 · seed 123 · 输出 96×96</td></tr>
<tr><td>主指标</td><td><b>写对 v2</b>↑（内容）· L1↓（风格辅证）</td></tr>
</table>
</div>
</div>

<div class="result-row">
<div class="kpi star"><div class="l">★ E10 @18000 · 写对 v2</div><div class="v">{pct(star["margin"])}</div></div>
<div class="kpi star"><div class="l">★ 内容风险↓</div><div class="v">{pct(star["risk"])}</div></div>
<div class="kpi star"><div class="l">★ L1↓</div><div class="v">{star["l1"]:.3f}</div></div>
<div class="kpi"><div class="l">官方 FD0 · 写对</div><div class="v">{pct(fd0["margin"])}</div></div>
</div>

<div class="fig">
<img src="assets/cn2cn_overview.png" alt="CN2CN infer examples"/>
<p class="figcap">图1 · 推理示例（风雅楷宋）：C=Content(FZKTJW) · S=Style(永) · GT · FD0官方 · E10微调</p>
</div>
<div class="fig">
<img src="assets/cn2cn_panel_一.png" alt="CN2CN protocol panel"/>
<p class="figcap">图2 · 训练/推理/产品协议对照（字「一」）· 金框=Content · 蓝=Style · 绿=GT</p>
</div>
<div class="links">
<a href="../fd_cn2cn_protocol_examples/">详细示例页</a>
<a href="../e10_mid_review/">E10 全 step 曲线</a>
<a href="../mentor_star_compare/">三方法 Mentor 对比</a>
</div>
</div>

<!-- ========== CN→West ========== -->
<div class="section">
<h2><span class="badge cn2west">E4/E5</span> 中→西（CN→Latin/Kana）</h2>
<p class="tagline">中文 style 参考 → 生成拉丁(P1)/假名(P2) · Demo-8 全量 8 字体 · 主表 L1/SSIM</p>

<div class="grid2">
<div class="box">
<h3>训练 · 图像渲染与读入</h3>
<table>
<tr><th>项</th><th>设置</th></tr>
<tr><td>数据包</td><td><code>fontdiffuser</code>（42字）或 <code>fontdiffuser_p253</code></td></tr>
<tr><td>Content</td><td><code>source/</code> · P1=<b>DejaVu Sans</b> · P2=CJK 源字体 · 96 渲</td></tr>
<tr><td>Style</td><td><code>chinese/&lt;font&gt;/</code> 中文池 · <b>随机抽 1</b>（~338 字/字体）</td></tr>
<tr><td>Target/GT</td><td><code>english/&lt;font&gt;/</code> P1∪P2 西文/假名 GT</td></tr>
<tr><td>读入</td><td>Resize(96) · ToTensor · Normalize(0.5) · drop 0.1 · 无 SCR</td></tr>
<tr><td>Run A</td><td><code>ft_cnstyle</code> · 42 字体 · 25k · bs4</td></tr>
<tr><td>Run B</td><td><code>ft_p253_cnstyle</code> · 253 字体 · 12k · bs24</td></tr>
</table>
</div>
<div class="box">
<h3>测试/评测 · 图像渲染与推理</h3>
<table>
<tr><th>项</th><th>设置</th></tr>
<tr><td>Content</td><td>P1：<b>DejaVu Sans</b> @96 二分搜 fit · P2：unified CJK 源 @96</td></tr>
<tr><td>Style</td><td>目标字体 TTF 渲「<b>永</b>」1-shot @96</td></tr>
<tr><td>GT</td><td><code>unified_v1/renders/64/gt_latin|gt_p2</code> → resize 96 打分</td></tr>
<tr><td>测集</td><td>Demo-8 全 8 字体 · P1 拉丁数字 · P2 假名</td></tr>
<tr><td>推理</td><td>DPM++ <b>20</b> 步 · CFG 7.5 · seed 123（lite 探针 15 步）</td></tr>
<tr><td>主指标</td><td><b>L1↓ / SSIM↑</b>（像素，@96 统一 resize）</td></tr>
</table>
</div>
</div>

<div class="result-row">
<div class="kpi star"><div class="l">cnstyle@42 · 25k · P1 SSIM↑</div><div class="v">{cnstyle["fd_cnstyle_42_best25k"]["p1"]["SSIM"]:.3f}</div></div>
<div class="kpi star"><div class="l">cnstyle@42 · P1 L1↓</div><div class="v">{cnstyle["fd_cnstyle_42_best25k"]["p1"]["L1"]:.3f}</div></div>
<div class="kpi"><div class="l">p253 cnstyle@12k · P1 SSIM↑</div><div class="v">{cnstyle["fd_p253_cnstyle_12k"]["p1"]["SSIM"]:.3f}</div></div>
<div class="kpi"><div class="l">p253 cnstyle@12k · P1 L1↓</div><div class="v">{cnstyle["fd_p253_cnstyle_12k"]["p1"]["L1"]:.3f}</div></div>
</div>
<div class="result-row" style="margin-top:0">
<div class="kpi"><div class="l">cnstyle@42 · P2 SSIM↑</div><div class="v">{cnstyle["fd_cnstyle_42_best25k"]["p2"]["SSIM"]:.3f}</div></div>
<div class="kpi"><div class="l">cnstyle@42 · P2 L1↓</div><div class="v">{cnstyle["fd_cnstyle_42_best25k"]["p2"]["L1"]:.3f}</div></div>
<div class="kpi"><div class="l">p253@12k · P2 SSIM↑</div><div class="v">{cnstyle["fd_p253_cnstyle_12k"]["p2"]["SSIM"]:.3f}</div></div>
<div class="kpi"><div class="l">p253@12k · P2 L1↓</div><div class="v">{cnstyle["fd_p253_cnstyle_12k"]["p2"]["L1"]:.3f}</div></div>
</div>

<div class="fig">
<img src="assets/cn2west_sheet.png" alt="CN2West step sheet"/>
<p class="figcap">图3 · 中→西多 step 测试图（创粗黑 · Demo-8 lite 探针 AaBbRrSs0123+あいう）· 行=GT/各 step</p>
</div>
<div class="links">
<a href="../fd_epoch_gallery/">多 step 图库</a>
<a href="../compare_report/">八方法横比</a>
<a href="../CN2WEST_INDEX.html">中→西实验索引</a>
</div>
</div>

<!-- ========== 对比 ========== -->
<div class="section">
<h2>两条主线不可混比</h2>
<table>
<tr><th></th><th>中→中 E10</th><th>中→西 cnstyle</th></tr>
<tr><td><b>任务</b></td><td>汉字 → 汉字</td><td>中文 style → 拉丁/假名</td></tr>
<tr><td><b>Content</b></td><td>FZKTJW R1@96</td><td>DejaVu / CJK 源 @96</td></tr>
<tr><td><b>Style 训练</b></td><td>同字体中文池随机</td><td>同字体中文池随机</td></tr>
<tr><td><b>Style 测试</b></td><td>固定「永」1-shot</td><td>固定「永」1-shot</td></tr>
<tr><td><b>GT</b></td><td>R1 style96 汉字</td><td>unified_v1 西文/假名</td></tr>
<tr><td><b>主指标</b></td><td>写对 v2 + L1</td><td>L1 + SSIM</td></tr>
<tr><td><b>★ checkpoint</b></td><td>abs 18000（cn2cn_p253）</td><td>25k（42字）/ 12k（253字）</td></tr>
</table>
<div class="warn">⚠ 跨语 L1/SSIM 与汉字写对率<strong>不能直接比较</strong>；选型页见 <code>mentor_star_compare</code>（汉字）vs <code>compare_report</code>（跨语）。</div>
</div>

<p style="font-size:12px;color:var(--muted)">生成：python scripts/retrain_v2_fd_protocol_report_build.py · 
<a href="../fd_cn2cn_eval_gallery/">FD Demo-8 全 8 字体评测图</a> ·
<a href="../mentor_star_compare/">mentor 三方法对比</a></p>
</div></body></html>"""

    (OUT / "index.html").write_text(html, encoding="utf-8")
    meta = {
        "cn2cn_star": {
            "write_ok": star["margin"],
            "risk": star["risk"],
            "l1": star["l1"],
            "ckpt": star["ckpt"],
        },
        "cn2west": cnstyle,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
