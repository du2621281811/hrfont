#!/usr/bin/env python3
"""Build render comparison gallery with fair @96 model-input domain."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
FD = ROOT / "code/ours/FontDiffuser"
DATA = FD / "data_examples/train"
OUT = ROOT / "reports/retrain_v2/fd_official_render_audit"
GAL = OUT / "gallery"
TTF_A = FD / "ttf/KaiXinSongA.ttf"
TTF_TARGET = FD / "ttf/target"
FZKT = ROOT / "data/FZKTJW.TTF"
R1 = ROOT / "data/retrain_v2/renders_r1"
NOTO = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT50 = Path("/root/data/font_50")
MODEL_SIZE = 96  # FontDiffuser default content/style/resolution

sys.path.insert(0, str(FD))
sys.path.insert(0, str(ROOT / "scripts"))
from utils import load_ttf, ttf2im  # noqa: E402
from build_retrain_v2_dataset import render as render_f128  # noqa: E402
from retrain_v2_eval_fontdiffuser import render as render_infer_f96  # noqa: E402

CONTENT_CHARS = ["氮", "潮", "舶", "镀"]
TARGET_FONTS = ["FZGuanJKSJW", "FZOuYHGXSJW", "FZZCHJW"]
FONT_CN = {
    "FZGuanJKSJW": "方正管峻楷书简体",
    "FZOuYHGXSJW": "方正欧阳荷庚行书简体",
    "FZZCHJW": "方正正粗黑简体",
}


def to_model_input(im: Image.Image, size: int = MODEL_SIZE) -> Image.Image:
    """Same as official train.py / sample.py: BILINEAR Resize → model size."""
    return im.convert("RGB").resize((size, size), Image.BILINEAR)


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


def to_gray(im: Image.Image, size: int) -> np.ndarray:
    return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def save_display(im: Image.Image, path: Path, native: int, scale: int = 4):
    path.parent.mkdir(parents=True, exist_ok=True)
    up = im.convert("RGB").resize((native * scale, native * scale), Image.NEAREST)
    up.save(path)


def diff_rgb(ref: Image.Image, other: Image.Image, size: int) -> Image.Image:
    a = np.asarray(ref.convert("L").resize((size, size), Image.BILINEAR))
    b = np.asarray(other.convert("L").resize((size, size), Image.BILINEAR))
    ink_a, ink_b = a < 250, b < 250
    rgb = np.full((size, size, 3), 255, dtype=np.uint8)
    rgb[ink_a, 0] = 220
    rgb[ink_b, 1] = 220
    rgb[ink_a & ink_b] = [220, 220, 0]
    rgb[ink_a & ~ink_b] = [255, 80, 80]
    rgb[~ink_a & ink_b] = [80, 200, 80]
    return Image.fromarray(rgb)


def metrics(ref: Image.Image, other: Image.Image | None, size: int) -> dict:
    if other is None:
        return {"ssim": None, "l1": None, "bbox_l1": None}
    ga, gb = to_gray(ref, size), to_gray(other, size)
    bb_o, bb_r = ink_bbox(ref), ink_bbox(other)
    bl1 = sum(abs(a - b) for a, b in zip(bb_o, bb_r)) if bb_o and bb_r else None
    return {
        "ssim": round(ssim_gray(ga, gb), 4),
        "l1": round(float(np.abs(ga - gb).mean()), 4),
        "bbox_l1": bl1,
        "bbox_ref": bb_o,
        "bbox_cmp": bb_r,
        "eval_size": size,
    }


def resolve_target_ttf(stem: str) -> Path | None:
    for d in (TTF_TARGET, FONT50):
        for ext in (".TTF", ".ttf"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def render_r1_content(ch: str, canvas: int = 96) -> Image.Image:
    p = R1 / f"content_{canvas}" / f"u{ord(ch):04X}.png"
    if p.exists():
        return Image.open(p).convert("RGB")
    meta = R1 / "meta" / f"content_params_{canvas}.json"
    params = json.loads(meta.read_text()) if meta.exists() else {}
    pr = params.get(ch) or params.get(str(ch)) or {"size": 72, "dx": 0, "dy": 0}
    img = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    f = ImageFont.truetype(str(FZKT), int(pr["size"]))
    bbox = draw.textbbox((0, 0), ch, font=f)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - w) // 2 - bbox[0] + int(pr.get("dx", 0))
    y = (canvas - h) // 2 - bbox[1] + int(pr.get("dy", 0))
    draw.text((x, y), ch, font=f, fill=(0, 0, 0))
    return img


def render_f48(ttf: Path, ch: str, canvas: int = 64, size: int = 48) -> Image.Image:
    f = ImageFont.truetype(str(ttf), size=size)
    im = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    bbox = draw.textbbox((0, 0), ch, font=f)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (canvas - w) // 2 - bbox[0], (canvas - h) // 2 - bbox[1]
    draw.text((x, y), ch, font=f, fill=(0, 0, 0))
    return im


def add_variant(row: dict, key: str, label: str, im: Image.Image, ref: Image.Image, ref_size: int, native: int, prefix: str):
    tag = f"{prefix}_{key}"
    save_display(im, GAL / f"{tag}.png", native)
    if key != "ref":
        save_display(diff_rgb(ref, im, ref_size), GAL / f"{tag}_diff.png", ref_size)
    row["variants"][key] = {"label": label, "tag": tag, "native": native, **metrics(ref, im, ref_size)}


def build_content_rows(font_a) -> tuple[list[dict], list[dict]]:
    """Returns (rows_at96, rows_at128_geometry)."""
    rows96, rows128 = [], []
    for ch in CONTENT_CHARS:
        off128 = Image.open(DATA / "ContentImage" / f"{ch}.jpg").convert("RGB")
        ref96 = to_model_input(off128, MODEL_SIZE)  # official GT @96 (128 JPG → BILINEAR 96)

        r96 = {"char": ch, "domain": "@96", "variants": {}}
        r128 = {"char": ch, "domain": "@128", "variants": {}}
        prefix96 = f"content96_{ord(ch):04X}"
        prefix128 = f"content128_{ord(ch):04X}"

        add_variant(r96, "ref", "官方 JPG@128→BILINEAR→96（模型 GT）", ref96, ref96, MODEL_SIZE, MODEL_SIZE, prefix96)
        add_variant(
            r96,
            "ttf2im",
            "ttf2im@128→BILINEAR→96",
            to_model_input(ttf2im(font_a, ch, 128), MODEL_SIZE),
            ref96,
            MODEL_SIZE,
            MODEL_SIZE,
            prefix96,
        )
        add_variant(
            r96,
            "f128_kai",
            "F128 二分搜 KaiXinSongA@128→96",
            to_model_input(render_f128(TTF_A, ch, 128).convert("RGB"), MODEL_SIZE),
            ref96,
            MODEL_SIZE,
            MODEL_SIZE,
            prefix96,
        )
        add_variant(r96, "r1_native96", "R1 原生@96（我方 CN2CN 训练盘）", render_r1_content(ch, 96), ref96, MODEL_SIZE, MODEL_SIZE, prefix96)
        add_variant(
            r96,
            "infer96",
            "eval 原生 render@96（Noto，跨语推理）",
            render_f128(NOTO, ch, 96, index=0).convert("RGB"),
            ref96,
            MODEL_SIZE,
            MODEL_SIZE,
            prefix96,
        )
        add_variant(
            r96,
            "f48_up96",
            "F48@64→BILINEAR→96（消融协议上采样）",
            to_model_input(render_f48(FZKT, ch, 64, 48), MODEL_SIZE),
            ref96,
            MODEL_SIZE,
            MODEL_SIZE,
            prefix96,
        )

        add_variant(r128, "ref", "官方 JPG@128（烘焙盘）", off128, off128, 128, 128, prefix128)
        add_variant(r128, "ttf2im", "ttf2im@128", ttf2im(font_a, ch, 128), off128, 128, 128, prefix128)

        rows96.append(r96)
        rows128.append(r128)
    return rows96, rows128


def build_target_rows() -> tuple[list[dict], list[dict]]:
    rows96, rows128 = [], []
    for stem in TARGET_FONTS:
        ttf = resolve_target_ttf(stem)
        font_handle = load_ttf(str(ttf), 128) if ttf else None
        for ch in CONTENT_CHARS:
            off128 = Image.open(DATA / "TargetImage" / stem / f"{stem}+{ch}.jpg").convert("RGB")
            ref96 = to_model_input(off128, MODEL_SIZE)
            r96 = {"font": stem, "char": ch, "has_ttf": ttf is not None, "domain": "@96", "variants": {}}
            r128 = {"font": stem, "char": ch, "domain": "@128", "variants": {}}
            p96 = f"target96_{stem}_{ord(ch):04X}"
            p128 = f"target128_{stem}_{ord(ch):04X}"

            add_variant(r96, "ref", "官方 Target@128→96", ref96, ref96, MODEL_SIZE, MODEL_SIZE, p96)
            if font_handle and ttf:
                add_variant(
                    r96,
                    "ttf2im",
                    "ttf2im@128→96",
                    to_model_input(ttf2im(font_handle, ch, 128), MODEL_SIZE),
                    ref96,
                    MODEL_SIZE,
                    MODEL_SIZE,
                    p96,
                )
                add_variant(
                    r96,
                    "f128",
                    "F128 二分搜@128→96",
                    to_model_input(render_f128(ttf, ch, 128).convert("RGB"), MODEL_SIZE),
                    ref96,
                    MODEL_SIZE,
                    MODEL_SIZE,
                    p96,
                )
                add_variant(
                    r96,
                    "infer96",
                    "eval 原生 render@96",
                    render_infer_f96(ttf, ch, 96),
                    ref96,
                    MODEL_SIZE,
                    MODEL_SIZE,
                    p96,
                )
            add_variant(r128, "ref", "官方 Target@128", off128, off128, 128, 128, p128)
            if font_handle:
                add_variant(r128, "ttf2im", "ttf2im@128", ttf2im(font_handle, ch, 128), off128, 128, 128, p128)
            rows96.append(r96)
            rows128.append(r128)
    return rows96, rows128


def rank_variants(rows: list[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for row in rows:
        for k, v in row.get("variants", {}).items():
            if k == "ref" or v.get("ssim") is None:
                continue
            acc.setdefault(k, []).append(v["ssim"])
    return {k: round(float(np.mean(vs)), 4) for k, vs in acc.items()}


def variant_cells(row: dict, keys: tuple[str, ...], w: int = 192) -> str:
    parts = []
    for key in keys:
        v = row["variants"].get(key)
        if not v:
            continue
        ssim = v.get("ssim")
        badge = ""
        if ssim is not None:
            cls = "ok" if ssim >= 0.85 else ("mid" if ssim >= 0.6 else "bad")
            badge = f'<span class="badge {cls}">SSIM {ssim:.3f}</span>'
        diff = "" if key == "ref" else f'<a class="diff" href="gallery/{v["tag"]}_diff.png">diff</a>'
        parts.append(
            f'<div class="cell"><img src="gallery/{v["tag"]}.png" width="{w}" height="{w}"/>'
            f'<div class="lab">{v["label"]}</div>{badge}{diff}</div>'
        )
    return "\n".join(parts)


def write_html(content96, content128, target96, target128, font_status):
    rank96 = rank_variants(content96 + target96)
    rank128 = rank_variants(content128 + target128)

    def rank_table(rank: dict) -> str:
        if not rank:
            return "<p class='muted'>（缺 Target 字体，仅 Content 有排名）</p>"
        return "<table><tr><th>协议</th><th>mean SSIM</th></tr>" + "".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in sorted(rank.items(), key=lambda x: -x[1])
        ) + "</table>"

    font_cards = ""
    for stem, st in font_status.items():
        if st.get("ok"):
            font_cards += f"<li><b>{stem}</b> ✓</li>"
        else:
            font_cards += f'<li><b>{stem}</b> ✗ <a href="{st.get("manual","")}">方正官网下载</a> → <code>ttf/target/{stem}.TTF</code></li>'

    c96_html = "".join(
        f'<div class="card"><div class="ch">{r["char"]}</div><div class="row">'
        f'{variant_cells(r, ("ref","ttf2im","f128_kai","r1_native96","infer96","f48_up96"))}</div></div>'
        for r in content96
    )
    c128_html = "".join(
        f'<div class="card compact"><div class="ch">{r["char"]}</div><div class="row">'
        f'{variant_cells(r, ("ref","ttf2im"), 160)}</div></div>'
        for r in content128
    )
    t96_html = ""
    for stem in TARGET_FONTS:
        t96_html += f"<h3>{stem}</h3>"
        for r in [x for x in target96 if x["font"] == stem]:
            t96_html += f'<div class="card compact"><div class="ch">{r["char"]}</div><div class="row">{variant_cells(r, ("ref","ttf2im","f128","infer96"), 160)}</div></div>'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"/>
<title>渲染对比 · 公平 @96 域</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
:root{{--bg:#0e1217;--panel:#171d25;--muted:#93a0ad;--line:rgba(255,255,255,.1);--ok:#6bcf7f;--mid:#c9a46a;--bad:#e07a7a;--accent:#7eb8da;--warn:#e8c468}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:#e9eef4;font:14px/1.55 system-ui,sans-serif}}
.wrap{{max-width:1500px;margin:0 auto;padding:22px 16px 80px}}
h1{{font:700 1.35rem Georgia,serif}} h2{{color:var(--accent);font-size:1.05rem;margin:22px 0 8px}}
h3{{font-size:.95rem;margin:16px 0 6px;color:var(--muted)}}
.muted{{color:var(--muted);font-size:12px}} a{{color:var(--accent)}}
.panel,.card{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px;margin:12px 0}}
.warn{{border-left:3px solid var(--warn);padding:10px 14px;background:rgba(232,196,104,.1);margin:14px 0}}
.call{{border-left:3px solid var(--accent);padding:10px 14px;background:rgba(126,184,218,.08);margin:14px 0}}
.row{{display:flex;flex-wrap:wrap;gap:10px}} .cell{{text-align:center;max-width:210px}}
.lab{{font-size:10px;color:var(--muted);margin-top:4px;line-height:1.35}}
.ch{{font-size:20px;font-weight:700;margin-bottom:6px}}
img{{background:#fff;image-rendering:pixelated;image-rendering:crisp-edges;display:block}}
.badge{{font-size:10px;padding:1px 6px;border-radius:3px;display:inline-block;margin-top:3px}}
.badge.ok{{background:rgba(107,207,127,.2);color:var(--ok)}}
.badge.mid{{background:rgba(201,164,106,.2);color:var(--mid)}}
.badge.bad{{background:rgba(224,122,122,.2);color:var(--bad)}}
.diff{{font-size:10px;display:block;margin-top:2px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}}
</style></head><body><div class="wrap">
<h1>渲染对比 · 公平 @96 模型输入域</h1>
<p class="muted"><a href="AUDIT.md">AUDIT.md</a> · <a href="FONT_SETUP.md">FONT_SETUP.md</a> · <a href="gallery_summary.json">gallery_summary.json</a></p>

<div class="warn">
<b>为何分两个域？</b> 官方离线盘是 <b>128×128 烘焙</b>，训练/推理时 <code>Resize(96, BILINEAR)</code> 才进网络。
我方 R1 / eval 很多是 <b>原生 96</b>。把官方 @128 与我方 @96 直接比 SSIM <b>不公平</b>。
<br/>主表统一在 <b>@96</b>：官方参考 = JPG@128 → BILINEAR → 96（与 <code>train.py</code> 一致）。
</div>

<div class="call">
<b>@96 主表读法：</b> 第一列是官方模型实际看到的 GT；其余列与第一列比 SSIM。
<br/><code>r1_native96</code> = 我方 CN2CN 训练盘（原生 96，不再缩）；<code>f128→96</code> = 我方跨语烘焙协议（128 再缩 96，与官方管线同类）。
</div>

<div class="panel"><h2 style="margin-top:0">@96 排名（vs 官方 JPG→96）</h2>{rank_table(rank96)}</div>

<h2>Content · @96 公平对比</h2>
{c96_html}

<h2>Target · @96 公平对比</h2>
<p class="muted">Target 字体：{font_cards}</p>
{t96_html}

<h2>附录 · @128 几何审计（非模型输入，只看烘焙布局）</h2>
<p class="muted">此处 SSIM 仅说明离线 JPG 与 ttf2im@128 的几何关系，<b>不能</b>代表进模型后的差异。</p>
<div class="panel"><h2 style="margin-top:0">@128 排名</h2>{rank_table(rank128)}</div>
{c128_html}

<div class="panel">
<h2 style="margin-top:0">怎么解读</h2>
<ul>
<li><b>公平比协议</b>：在 @96 表看 <code>f128→96</code> vs <code>ttf2im→96</code>（同是 128 烘焙再缩）以及 <code>r1_native96</code>（我方原生 96）。</li>
<li><b>我方 96 直接渲</b> 与官方 <b>128→96</b> 比：差的不只是字号搜参，还有「原生 96 搜参」vs「128 画布再缩小」的尺度链。</li>
<li>若要让我方 CN2CN 对齐官方：要么 R1 改渲 <b>128 盘再缩 96</b>，要么官方式评测也读 <b>预烘焙 96</b> 且与训练同源。</li>
</ul>
</div>
</div></body></html>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main():
    import subprocess

    subprocess.run([sys.executable, str(ROOT / "scripts/fetch_fd_target_fonts.py")], check=False)
    font_status = json.loads((TTF_TARGET / "font_status.json").read_text()) if (TTF_TARGET / "font_status.json").exists() else {}

    if GAL.exists():
        import shutil
        shutil.rmtree(GAL)
    GAL.mkdir(parents=True, exist_ok=True)

    font_a = load_ttf(str(TTF_A), 128)
    content96, content128 = build_content_rows(font_a)
    target96, target128 = build_target_rows()

    summary = {
        "model_size": MODEL_SIZE,
        "content_at96": content96,
        "content_at128": content128,
        "target_at96": target96,
        "target_at128": target128,
        "rank_at96": rank_variants(content96 + target96),
        "rank_at128": rank_variants(content128 + target128),
        "font_status": font_status,
    }
    (OUT / "gallery_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_html(content96, content128, target96, target128, font_status)
    print(f"OK {OUT / 'index.html'} rank@96={summary['rank_at96']}")


if __name__ == "__main__":
    main()
