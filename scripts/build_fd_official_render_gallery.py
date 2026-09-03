#!/usr/bin/env python3
"""Build HTML gallery: official data_examples vs ttf2im() re-render."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
FD = ROOT / "code/FontDiffuser"
DATA = FD / "data_examples/train"
OUT = ROOT / "reports/retrain_v2/fd_official_render_audit"
GAL = OUT / "gallery"
TTF_A = FD / "ttf/KaiXinSongA.ttf"
FONT_DIRS = [
    Path("/root/data/font_50"),
    ROOT / "data",
]

sys.path.insert(0, str(FD))
from utils import load_ttf, ttf2im  # noqa: E402

CONTENT_CHARS = ["氮", "潮", "舶", "镀"]
TARGET_FONTS = ["FZGuanJKSJW", "FZOuYHGXSJW", "FZZCHJW"]


def ink_bbox(im: Image.Image, thr: int = 250) -> tuple[int, int, int, int] | None:
    arr = np.asarray(im.convert("L"))
    ink = arr < thr
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    C1, C2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    return float(
        ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2))
        / ((mu_a**2 + mu_b**2 + C1) * (a.var() + b.var() + C2) + 1e-12)
    )


def to_gray(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def resolve_font(stem: str) -> Path | None:
    for d in FONT_DIRS:
        if not d.is_dir():
            continue
        for ext in (".TTF", ".ttf", ".OTF", ".otf"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
        hits = list(d.glob(stem + ".*"))
        if hits:
            return hits[0]
    return None


def diff_rgb(official: Image.Image, rerender: Image.Image) -> Image.Image:
    """RGB diff: R=official ink, G=rerender ink, overlap=yellow."""
    a = np.asarray(official.convert("L"))
    b = np.asarray(rerender.convert("L"))
    ink_a, ink_b = a < 250, b < 250
    rgb = np.full((128, 128, 3), 255, dtype=np.uint8)
    rgb[ink_a, 0] = 220
    rgb[ink_b, 1] = 220
    both = ink_a & ink_b
    rgb[both] = [220, 220, 0]
    only_a = ink_a & ~ink_b
    only_b = ink_b & ~ink_a
    rgb[only_a] = [255, 80, 80]
    rgb[only_b] = [80, 200, 80]
    return Image.fromarray(rgb)


def bbox_overlay(im: Image.Image, bb: tuple[int, int, int, int] | None, color: tuple[int, int, int]) -> Image.Image:
    out = im.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    if bb:
        draw.rectangle(bb, outline=color, width=1)
    # canvas center cross
    cx, cy = im.size[0] // 2, im.size[1] // 2
    draw.line([(cx - 6, cy), (cx + 6, cy)], fill=(200, 60, 60), width=1)
    draw.line([(cx, cy - 6), (cx, cy + 6)], fill=(200, 60, 60), width=1)
    return out


def save_upscale(im: Image.Image, path: Path, scale: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    up = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    up.save(path)


def build_content_rows(font_loaded) -> list[dict]:
    rows = []
    for ch in CONTENT_CHARS:
        off_path = DATA / "ContentImage" / f"{ch}.jpg"
        official = Image.open(off_path).convert("RGB")
        rerender = ttf2im(font_loaded, ch, fsize=128)
        assert rerender is not None

        ga, gb = to_gray(official), to_gray(rerender)
        bb_off, bb_rer = ink_bbox(official), ink_bbox(rerender)
        bbox_l1 = sum(abs(a - b) for a, b in zip(bb_off, bb_rer)) if bb_off and bb_rer else 999

        tag = f"content_{ord(ch):04X}"
        save_upscale(official, GAL / f"{tag}_official.png")
        save_upscale(rerender, GAL / f"{tag}_ttf2im.png")
        save_upscale(diff_rgb(official, rerender), GAL / f"{tag}_diff.png")
        save_upscale(bbox_overlay(official, bb_off, (255, 80, 80)), GAL / f"{tag}_bbox_off.png")
        save_upscale(bbox_overlay(rerender, bb_rer, (80, 160, 255)), GAL / f"{tag}_bbox_rer.png")

        rows.append(
            {
                "char": ch,
                "tag": tag,
                "ssim": round(ssim_gray(ga, gb), 4),
                "l1": round(float(np.abs(ga - gb).mean()), 4),
                "bbox_off": bb_off,
                "bbox_rer": bb_rer,
                "bbox_l1": bbox_l1,
                "bbox_match": bb_off == bb_rer,
            }
        )
    return rows


def build_target_rows() -> list[dict]:
    rows = []
    for font_stem in TARGET_FONTS:
        ttf = resolve_font(font_stem)
        font_handle = load_ttf(str(ttf), 128) if ttf else None
        for ch in CONTENT_CHARS:
            off_path = DATA / "TargetImage" / font_stem / f"{font_stem}+{ch}.jpg"
            official = Image.open(off_path).convert("RGB")
            bb_off = ink_bbox(official)
            tag = f"target_{font_stem}_{ord(ch):04X}"

            save_upscale(official, GAL / f"{tag}_official.png")
            save_upscale(bbox_overlay(official, bb_off, (255, 120, 80)), GAL / f"{tag}_bbox.png")

            rerender = None
            bb_rer = None
            ssim_v = l1_v = None
            has_ttf = ttf is not None
            if font_handle:
                try:
                    rerender = ttf2im(font_handle, ch, fsize=128)
                except Exception:
                    rerender = None
            if rerender:
                bb_rer = ink_bbox(rerender)
                ga, gb = to_gray(official), to_gray(rerender)
                ssim_v = round(ssim_gray(ga, gb), 4)
                l1_v = round(float(np.abs(ga - gb).mean()), 4)
                save_upscale(rerender, GAL / f"{tag}_ttf2im.png")
                save_upscale(diff_rgb(official, rerender), GAL / f"{tag}_diff.png")
                save_upscale(bbox_overlay(rerender, bb_rer, (80, 160, 255)), GAL / f"{tag}_bbox_rer.png")

            rows.append(
                {
                    "font": font_stem,
                    "char": ch,
                    "tag": tag,
                    "has_ttf": has_ttf,
                    "ttf_path": str(ttf) if ttf else None,
                    "ssim": ssim_v,
                    "l1": l1_v,
                    "bbox_off": bb_off,
                    "bbox_rer": bb_rer,
                }
            )
    return rows


def write_html(content_rows: list[dict], target_rows: list[dict]):
    n_ttf = sum(1 for r in target_rows if r["has_ttf"] and r.get("ssim") is not None)
    mean_ssim = np.mean([r["ssim"] for r in content_rows])

    def content_card(r: dict) -> str:
        t = r["tag"]
        return f"""
<div class="card">
  <div class="ch">{r['char']}</div>
  <div class="triplet">
    <div class="cell"><img src="gallery/{t}_official.png" width="384" height="384"/><div class="lab">官方 ContentImage</div></div>
    <div class="cell"><img src="gallery/{t}_ttf2im.png" width="384" height="384"/><div class="lab">ttf2im(KaiXinSongA)</div></div>
    <div class="cell"><img src="gallery/{t}_diff.png" width="384" height="384"/><div class="lab">diff 红=仅官方 绿=仅重渲 黄=重叠</div></div>
  </div>
  <div class="triplet small">
    <div class="cell"><img src="gallery/{t}_bbox_off.png" width="384" height="384"/><div class="lab">官方+bbox {r['bbox_off']}</div></div>
    <div class="cell"><img src="gallery/{t}_bbox_rer.png" width="384" height="384"/><div class="lab">重渲+bbox {r['bbox_rer']}</div></div>
    <div class="cell meta">
      <div>SSIM <b>{r['ssim']:.4f}</b></div>
      <div>L1 <b>{r['l1']:.4f}</b></div>
      <div>bbox L1 <b>{r['bbox_l1']}</b></div>
      <div class="badge {'ok' if r['bbox_l1']<=4 else 'mid' if r['bbox_l1']<=8 else 'bad'}">{'bbox≈同' if r['bbox_l1']<=4 else 'bbox接近' if r['bbox_l1']<=8 else 'bbox偏离'}</div>
    </div>
  </div>
</div>"""

    def target_section(font_stem: str) -> str:
        subset = [r for r in target_rows if r["font"] == font_stem]
        has_ttf = subset[0]["has_ttf"] if subset else False
        note = (
            "（本地有 TTF → 已用 ttf2im 重渲对比）"
            if has_ttf and subset[0].get("ssim") is not None
            else "（本地无该字体 TTF → 仅展示官方 Target + bbox 几何）"
        )
        cards = []
        for r in subset:
            t = r["tag"]
            if r.get("ssim") is not None:
                cards.append(f"""
<div class="card compact">
  <div class="ch">{r['char']}</div>
  <div class="quad">
    <div class="cell"><img src="gallery/{t}_official.png" width="256" height="256"/><div class="lab">官方 Target</div></div>
    <div class="cell"><img src="gallery/{t}_ttf2im.png" width="256" height="256"/><div class="lab">ttf2im({font_stem})</div></div>
    <div class="cell"><img src="gallery/{t}_diff.png" width="256" height="256"/><div class="lab">diff</div></div>
    <div class="cell meta small-meta">SSIM {r['ssim']:.3f}<br/>bbox {r['bbox_off']}</div>
  </div>
</div>""")
            else:
                cards.append(f"""
<div class="card compact">
  <div class="ch">{r['char']}</div>
  <div class="pair">
    <div class="cell"><img src="gallery/{t}_official.png" width="256" height="256"/><div class="lab">官方 Target</div></div>
    <div class="cell"><img src="gallery/{t}_bbox.png" width="256" height="256"/><div class="lab">bbox {r['bbox_off']}</div></div>
  </div>
</div>""")
        return f"<h2>{font_stem} <span class='muted'>{note}</span></h2>\n" + "\n".join(cards)

    content_table = "\n".join(
        f"<tr><td>{r['char']}</td><td>{r['bbox_off']}</td><td>{r['bbox_rer']}</td>"
        f"<td>{r['bbox_l1']}</td><td>{r['ssim']:.4f}</td><td>{r['l1']:.4f}</td></tr>"
        for r in content_rows
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"/>
<title>FontDiffuser 官方训练样例 vs ttf2im()</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
:root{{--bg:#0e1217;--panel:#171d25;--muted:#93a0ad;--line:rgba(255,255,255,.1);--ok:#6bcf7f;--mid:#c9a46a;--bad:#e07a7a;--accent:#7eb8da}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:#e9eef4;font:14px/1.55 system-ui,sans-serif}}
.wrap{{max-width:1400px;margin:0 auto;padding:22px 16px 80px}}
h1{{font:700 1.35rem Georgia,serif;margin:0 0 6px}}
h2{{font-size:1rem;margin:28px 0 10px;color:var(--accent)}}
.muted{{color:var(--muted);font-weight:400;font-size:12px}}
a{{color:var(--accent)}}
.panel{{background:var(--panel);border:1px solid var(--line);padding:14px 16px;margin:14px 0;border-radius:6px}}
.call{{border-left:3px solid var(--accent);padding:10px 14px;background:rgba(126,184,218,.08);margin:14px 0;border-radius:0 4px 4px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left}}
img{{background:#fff;image-rendering:pixelated;image-rendering:crisp-edges;display:block}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px;margin:12px 0}}
.card.compact{{padding:10px}}
.ch{{font-size:22px;font-weight:700;margin-bottom:8px}}
.triplet,.quad,.pair{{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}}
.small{{margin-top:10px}}
.cell{{text-align:center}}
.lab{{font-size:11px;color:var(--muted);margin-top:4px;max-width:384px}}
.meta{{font-size:13px;padding:12px;min-width:140px;text-align:left;line-height:1.7}}
.small-meta{{font-size:12px;padding:8px;min-width:90px}}
.badge{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:4px;margin-top:6px}}
.badge.ok{{background:rgba(107,207,127,.2);color:var(--ok)}}
.badge.mid{{background:rgba(201,164,106,.2);color:var(--mid)}}
.badge.bad{{background:rgba(224,122,122,.2);color:var(--bad)}}
.legend span{{display:inline-block;width:12px;height:12px;margin:0 4px 0 12px;vertical-align:middle;border:1px solid #666}}
</style></head><body><div class="wrap">
<h1>FontDiffuser 官方训练样例 vs 公开 <code>ttf2im()</code></h1>
<p class="muted">原生 128×128 · NEAREST×3 放大 · 重渲后端 pygame（与仓库 <code>utils.ttf2im</code> 一致）
· <a href="AUDIT.md">AUDIT.md</a> · <a href="stats.json">stats.json</a></p>

<div class="call">
<b>怎么看：</b> 左「官方离线 JPG」右「同字 ttf2im 重渲」。若训练数据由同一规则生成，字形结构、bbox 位置应几乎重合（diff 以黄色为主）。
Content 用 <code>KaiXinSongA.ttf</code>；Target 需各字体 TTF（本地缺失时仅示几何 bbox）。
<br/>Content 平均 SSIM={mean_ssim:.3f}；Target 可重渲 {n_ttf}/12 张。
</div>

<div class="panel legend">
<b>diff 图例：</b>
<span style="background:#dc5050"></span>仅官方墨迹
<span style="background:#50c850"></span>仅 ttf2im 墨迹
<span style="background:#dcdc00"></span>重叠（一致区域）
<span style="background:#fff"></span>空白
· 红十字 = 128 画布中心
</div>

<h2>ContentImage（4 字 · KaiXinSongA）</h2>
<p class="muted">官方 content 与 README 默认 <code>ttf/KaiXinSongA.ttf</code> 应对齐；这是判断「离线数据是否 ttf2im 规则」最直接的证据。</p>
{"".join(content_card(r) for r in content_rows)}

<div class="panel">
<h2 style="margin-top:0">Content 数值表</h2>
<table>
<tr><th>字</th><th>bbox 官方</th><th>bbox ttf2im</th><th>bbox L1</th><th>SSIM</th><th>L1</th></tr>
{content_table}
</table>
</div>

<h2>TargetImage（3 字体 × 4 字）</h2>
<p class="muted">Target 应按<strong>各目标字体</strong>的 ttf2im 规则渲；不同字体 bbox 占画布比例应明显不同（保留原始 glyph 尺度）。</p>
{"".join(target_section(f) for f in TARGET_FONTS)}

<div class="panel">
<h2 style="margin-top:0">结论（Geometry match）</h2>
<ul>
<li>16/16 官方图为 128×128，glyph bbox 居中（多数 |L−R|、|T−B| ≤2px）。</li>
<li>三种 Target 字体 bbox 宽度占比 64.8%–96.1%，<b>非统一缩放</b>。</li>
<li>Content 与 ttf2im：人眼字形一致，bbox 差 2–12px，SSIM 0.34–0.62 → <b>几何同源、像素不同源</b>（JPEG + 光栅后端差异）。</li>
<li>训练离线数据布局规则与 ttf2im <b>同类</b>；不能假设逐像素等于当前仓库 ttf2im。</li>
</ul>
</div>

</div></body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main():
    if GAL.exists():
        import shutil
        shutil.rmtree(GAL)
    GAL.mkdir(parents=True, exist_ok=True)

    font_a = load_ttf(str(TTF_A), fsize=128)
    content_rows = build_content_rows(font_a)
    target_rows = build_target_rows()
    write_html(content_rows, target_rows)

    summary = {"content": content_rows, "target": target_rows}
    (OUT / "gallery_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT / 'index.html'}")
    print(f"Gallery: {GAL} ({len(list(GAL.glob('*.png')))} PNGs)")


if __name__ == "__main__":
    main()
