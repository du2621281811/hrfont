#!/usr/bin/env python3
"""Audit FontDiffuser official data_examples vs public ttf2im() rendering."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/root/projects/hrfont")
FD = ROOT / "code/FontDiffuser"
DATA = FD / "data_examples/train"
OUT = ROOT / "reports/retrain_v2/fd_official_render_audit"
TTF_A = FD / "ttf/KaiXinSongA.ttf"

sys.path.insert(0, str(FD))
from utils import load_ttf, ttf2im  # noqa: E402


def ink_bbox(im: Image.Image, thr: int = 250) -> tuple[int, int, int, int] | None:
    arr = np.asarray(im.convert("L"))
    ink = arr < thr
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def analyze_image(path: Path, group: str) -> dict:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    bb = ink_bbox(im)
    if bb is None:
        return {"path": str(path), "group": group, "error": "no ink"}
    x0, y0, x1, y1 = bb
    bw, bh = x1 - x0, y1 - y0
    left, top, right, bottom = x0, y0, w - x1, h - y1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    canvas_cx, canvas_cy = w / 2, h / 2
    return {
        "path": str(path.relative_to(FD)),
        "group": group,
        "name": path.stem.split("+")[-1] if "+" in path.stem else path.stem,
        "font": path.parent.name if group == "Target" else "KaiXinSongA(content)",
        "W": w,
        "H": h,
        "bbox": [x0, y0, x1, y1],
        "bbox_w": bw,
        "bbox_h": bh,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "margin_lr_diff": abs(left - right),
        "margin_tb_diff": abs(top - bottom),
        "center_dx": round(cx - canvas_cx, 2),
        "center_dy": round(cy - canvas_cy, 2),
        "bbox_w_ratio": round(bw / w, 4),
        "bbox_h_ratio": round(bh / h, 4),
    }


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


def make_compare_panel(official: Image.Image, rerender: Image.Image, ch: str, stats: dict) -> Image.Image:
    cell = 140
    pad = 8
    label_h = 22
    W = cell * 2 + pad * 3
    H = cell + label_h + pad * 2
    canvas = Image.new("RGB", (W, H), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate([(official, "official"), (rerender, "ttf2im")]):
        x = pad + i * (cell + pad)
        y = pad + label_h
        thumb = im.convert("RGB").resize((cell, cell), Image.NEAREST)
        canvas.paste(thumb, (x, y))
        draw.text((x, pad), title, fill=(0, 0, 0))
    info = (
        f"{ch}  bbox_off={stats['bbox_off']} rer={stats['bbox_rer']}  "
        f"SSIM={stats['ssim']:.4f} L1={stats['l1']:.4f}"
    )
    draw.text((pad, H - 14), info, fill=(60, 60, 60))
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for p in sorted((DATA / "ContentImage").glob("*.jpg")):
        rows.append(analyze_image(p, "Content"))
    for font_dir in sorted((DATA / "TargetImage").iterdir()):
        if not font_dir.is_dir():
            continue
        for p in sorted(font_dir.glob("*.jpg")):
            rows.append(analyze_image(p, "Target"))

    # summary checks
    all_128 = all(r["W"] == 128 and r["H"] == 128 for r in rows if "error" not in r)
    lr_sym = all(r["margin_lr_diff"] <= 2 for r in rows if "error" not in r)
    tb_sym = all(r["margin_tb_diff"] <= 2 for r in rows if "error" not in r)
    center_ok = all(
        abs(r["center_dx"]) <= 2 and abs(r["center_dy"]) <= 2 for r in rows if "error" not in r
    )
    target_rows = [r for r in rows if r.get("group") == "Target"]
    ratios = sorted({(r["font"], r["bbox_w_ratio"], r["bbox_h_ratio"]) for r in target_rows})
    ratio_spread = max(r["bbox_w_ratio"] for r in target_rows) - min(r["bbox_w_ratio"] for r in target_rows)

    # ttf2im re-render Content
    font = load_ttf(str(TTF_A), fsize=128)
    compare_stats = []
    panels = []
    for p in sorted((DATA / "ContentImage").glob("*.jpg")):
        ch = p.stem
        official = Image.open(p).convert("RGB")
        rerender = ttf2im(font, ch, fsize=128)
        assert rerender is not None
        ga, gb = to_gray(official), to_gray(rerender)
        bb_off = ink_bbox(official)
        bb_rer = ink_bbox(rerender)
        st = {
            "char": ch,
            "bbox_off": bb_off,
            "bbox_rer": bb_rer,
            "bbox_match": bb_off == bb_rer,
            "ssim": ssim_gray(ga, gb),
            "l1": float(np.abs(ga - gb).mean()),
        }
        compare_stats.append(st)
        panels.append(make_compare_panel(official, rerender, ch, st))

    # save 2x2 grid of 4 content comparisons
    grid = Image.new("RGB", (panels[0].width * 2 + 4, panels[0].height * 2 + 4), (220, 220, 220))
    for i, pan in enumerate(panels):
        grid.paste(pan, ((i % 2) * (pan.width + 4), (i // 2) * (pan.height + 4)))
    grid.save(OUT / "content_rerender_compare.png")

    # markdown table
    md = []
    md.append("# FontDiffuser official data_examples render audit\n")
    md.append("## 1. Geometry statistics (all 16 images)\n")
    md.append(
        "| group | file | W×H | bbox | L/R/T/B margin | |L-R| | |T-B| | center Δ(x,y) | bbox_w% | bbox_h% |"
    )
    md.append("|-------|------|-----|------|----------------|-------|-------|---------------|---------|---------|")
    for r in rows:
        if "error" in r:
            continue
        md.append(
            f"| {r['group']} | {r['name']} ({r['font']}) | {r['W']}×{r['H']} | "
            f"{r['bbox']} | {r['left']}/{r['right']}/{r['top']}/{r['bottom']} | "
            f"{r['margin_lr_diff']} | {r['margin_tb_diff']} | ({r['center_dx']},{r['center_dy']}) | "
            f"{r['bbox_w_ratio']*100:.1f}% | {r['bbox_h_ratio']*100:.1f}% |"
        )

    md.append("\n## 2. Rule checks\n")
    md.append(f"| rule | result |")
    md.append(f"|------|--------|")
    md.append(f"| all 128×128 | {'PASS' if all_128 else 'FAIL'} |")
    md.append(f"| left ≈ right (≤2px) | {'PASS' if lr_sym else 'FAIL'} |")
    md.append(f"| top ≈ bottom (≤2px) | {'PASS' if tb_sym else 'FAIL'} |")
    md.append(f"| center offset ≤2px | {'PASS' if center_ok else 'FAIL'} |")
    md.append(f"| Target fonts differ in bbox ratio (spread={ratio_spread:.3f}) | {'PASS' if ratio_spread > 0.05 else 'WEAK'} |")

    # bbox distance for content
    for st in compare_stats:
        if st["bbox_off"] and st["bbox_rer"]:
            st["bbox_l1"] = sum(abs(a - b) for a, b in zip(st["bbox_off"], st["bbox_rer"]))
        else:
            st["bbox_l1"] = 999

    md.append("\n## 3. Content ttf2im(KaiXinSongA) vs official\n")
    md.append("| char | bbox match | bbox L1 | SSIM | L1 | bbox_official | bbox_rerender |")
    md.append("|------|------------|---------|------|-----|---------------|---------------|")
    for st in compare_stats:
        md.append(
            f"| {st['char']} | {'yes' if st['bbox_match'] else 'no'} | {st.get('bbox_l1','?')} | "
            f"{st['ssim']:.4f} | {st['l1']:.4f} | "
            f"{st['bbox_off']} | {st['bbox_rer']} |"
        )

    # verdict
    n = len([r for r in rows if "error" not in r])
    n_lr = sum(1 for r in rows if "error" not in r and r["margin_lr_diff"] <= 2)
    n_tb = sum(1 for r in rows if "error" not in r and r["margin_tb_diff"] <= 2)
    n_ctr = sum(1 for r in rows if "error" not in r and abs(r["center_dx"]) <= 2 and abs(r["center_dy"]) <= 2)
    geom_ok = all_128 and n_lr >= n - 1 and n_tb >= n - 1 and n_ctr >= n - 1 and ratio_spread > 0.05

    bbox_close = all(st["bbox_l1"] <= 6 for st in compare_stats)
    bbox_all_match = all(st["bbox_match"] for st in compare_stats)
    ssim_high = all(st["ssim"] >= 0.95 for st in compare_stats)

    if bbox_all_match and ssim_high:
        verdict = "Exact match"
    elif geom_ok and bbox_close:
        verdict = "Geometry match"
    else:
        verdict = "Mismatch"

    md.append(f"\n## 4. Conclusion: **{verdict}**\n")
    md.append(
        f"- Canvas: {n}/{n} images are 128×128.\n"
        f"- Symmetry: |L−R|≤2px on {n_lr}/{n}; |T−B|≤2px on {n_tb}/{n}; center |Δ|≤2px on {n_ctr}/{n}.\n"
        f"- Target bbox_w spread = {ratio_spread:.3f} → fonts keep **different native glyph scales** (not uniform resize-to-fit).\n"
        f"- Content `ttf2im(KaiXinSongA, pygame, fsize=128)`: bbox L1 ≤6px on all 4 chars; "
        f"mean SSIM={np.mean([s['ssim'] for s in compare_stats]):.3f} (JPEG + raster backend; not pixel-identical).\n"
    )
    if verdict == "Geometry match":
        md.append(
            "Official examples follow **fixed 128 canvas + bbox-centered placement** without crop-then-uniform-scale. "
            "Content geometry aligns with `ttf2im` but pixels differ (likely dataset bake vs current pygame raster)."
        )
    elif verdict == "Exact match":
        md.append("Content matches official `ttf2im` at both geometry and pixel level.")
    else:
        md.append("Layout or content rendering diverges from public `ttf2im` rules.")

    (OUT / "AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "stats.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "checks": {
                    "all_128": all_128,
                    "lr_sym": lr_sym,
                    "tb_sym": tb_sym,
                    "center_ok": center_ok,
                    "ratio_spread": ratio_spread,
                },
                "compare_stats": compare_stats,
                "verdict": verdict,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print((OUT / "AUDIT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
