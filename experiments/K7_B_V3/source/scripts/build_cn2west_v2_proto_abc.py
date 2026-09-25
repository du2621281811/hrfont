#!/usr/bin/env python3
"""Build proto A / B / C full datasets.

A: per-font fixed size — binary-search max fs where max textbbox across ALL
   chars ≤ inner (canvas-2*margin).  Canvas 96, margin 6, inner 84.  RGB PNG.
B: per-font height-fit — binary-search max fs where max textbbox HEIGHT across
   all chars ≤ inner; width allowed to overflow then shrink.  Canvas 96,
   margin 6, inner 84.  RGB PNG.  (Pillow only.)
C: official pygame render @ fsize=128 on 128×128 canvas, then BILINEAR resize
   to 96×96.  RGB PNG.

Content images use Noto Sans CJK Regular with the same protocol as targets.

Parallel: one worker per font.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
CHARSET = ROOT / "manifests/charset_cn2west_v2_planned.json"
REF_SUMMARY = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2b-hfit/summary.json"
NOTO = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

CANVAS_96 = 96
MARGIN_AB = 6
INNER_AB = CANVAS_96 - 2 * MARGIN_AB   # 84
CANVAS_C = 128

OUT_A = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
OUT_B = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2b-hfit"
OUT_C = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2c-official128"


# ── charset helpers ──────────────────────────────────────────────────────

def target_chars() -> list[str]:
    cs = json.loads(CHARSET.read_text(encoding="utf-8"))
    return list(cs["target_string"])


def style_chars() -> list[str]:
    cs = json.loads(CHARSET.read_text(encoding="utf-8"))
    return list(cs["style_han_338"])


def cp_name(ch: str) -> str:
    return f"u{ord(ch):04X}"


def file_name(stem: str, ch: str) -> str:
    return f"{stem}+{cp_name(ch)}.png"


# ── font-level size search ───────────────────────────────────────────────

def _max_bbox(ttf_path: str, fs: int, chars: list[str]) -> tuple[int, int]:
    """Return (max_w, max_h) of textbbox across all chars at given fs."""
    font = ImageFont.truetype(ttf_path, fs)
    tmp = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(tmp)
    max_w = max_h = 0
    for ch in chars:
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h
    return max_w, max_h


def find_size_A(ttf_path: str, all_chars: list[str], inner: int = INNER_AB) -> int:
    """Proto A: max fs where max(textbbox_w, textbbox_h) ≤ inner for ALL chars."""
    lo, hi, best = 8, 300, 10
    for _ in range(16):
        if lo > hi:
            break
        fs = (lo + hi) // 2
        mw, mh = _max_bbox(ttf_path, fs, all_chars)
        if mw <= inner and mh <= inner:
            best = fs
            lo = fs + 1
        else:
            hi = fs - 1
    return best


def find_size_B(ttf_path: str, all_chars: list[str], inner: int = INNER_AB) -> int:
    """Proto B (hfit): max fs where max textbbox HEIGHT ≤ inner for ALL chars.
    Width may exceed inner — individual glyphs are shrunk at render time."""
    lo, hi, best = 8, 300, 10
    for _ in range(16):
        if lo > hi:
            break
        fs = (lo + hi) // 2
        _, mh = _max_bbox(ttf_path, fs, all_chars)
        if mh <= inner:
            best = fs
            lo = fs + 1
        else:
            hi = fs - 1
    return best


# ── single-glyph render ─────────────────────────────────────────────────

def render_glyph_AB(ttf_path: str, ch: str, fs: int,
                    canvas: int = CANVAS_96, inner: int = INNER_AB) -> Image.Image:
    """Render one glyph centred on canvas.  If textbbox exceeds inner in either
    dimension (possible in B), shrink font size for this glyph only."""
    font = ImageFont.truetype(ttf_path, fs)
    img = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # overflow shrink (mainly for proto B where width can exceed inner)
    cur_fs = fs
    while (w > inner or h > inner) and cur_fs > 10:
        cur_fs -= 1
        font = ImageFont.truetype(ttf_path, cur_fs)
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - w) // 2 - bbox[0]
    y = (canvas - h) // 2 - bbox[1]
    draw.text((x, y), ch, font=font, fill=0)
    return Image.merge("RGB", (img, img, img))


def render_glyph_C(ttf_path: str, ch: str,
                   src_size: int = CANVAS_C, dst_size: int = CANVAS_96) -> Image.Image:
    """Proto C: pygame-style render at fsize=128 on 128×128, then BILINEAR→96.
    We replicate the pygame logic purely in Pillow/FreeType for portability:
    render at fsize=128, centre on 128×128, resize to 96×96."""
    font = ImageFont.truetype(ttf_path, src_size)
    img = Image.new("L", (src_size, src_size), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (src_size - w) // 2 - bbox[0]
    y = (src_size - h) // 2 - bbox[1]
    draw.text((x, y), ch, font=font, fill=0)
    img96 = img.resize((dst_size, dst_size), Image.BILINEAR)
    return Image.merge("RGB", (img96, img96, img96))


# ── per-font job ─────────────────────────────────────────────────────────

def ink_yong(path: Path) -> dict:
    im = np.array(Image.open(path).convert("L"))
    H, W = im.shape
    ys, xs = np.where(im < 250)
    if len(ys) == 0:
        return {"h": 0.0, "w": 0.0, "area": 0.0, "touch": False}
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    touch = bool(y0 == 0 or x0 == 0 or y1 == H - 1 or x1 == W - 1)
    return {
        "h": float((y1 - y0 + 1) / H),
        "w": float((x1 - x0 + 1) / W),
        "area": float((y1 - y0 + 1) * (x1 - x0 + 1) / (H * W)),
        "touch": touch,
    }


def render_one_font(job: dict) -> dict:
    proto = job["proto"]
    split = job["split"]
    stem = job["stem"]
    ttf = job["ttf"]
    out_root = Path(job["out_root"])
    targets = job["targets"]
    styles = job["styles"]

    tdir = out_root / split / "TargetImage" / stem
    sdir = out_root / split / "StyleImage" / stem
    tdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)

    all_chars = targets + styles

    if proto == "C":
        fs = CANVAS_C
        for ch in targets:
            im = render_glyph_C(ttf, ch)
            im.save(tdir / file_name(stem, ch), optimize=True)
        for ch in styles:
            im = render_glyph_C(ttf, ch)
            im.save(sdir / file_name(stem, ch), optimize=True)
    else:
        # A or B
        if proto == "A":
            fs = find_size_A(ttf, all_chars)
        else:
            fs = find_size_B(ttf, all_chars)
        for ch in targets:
            im = render_glyph_AB(ttf, ch, fs)
            im.save(tdir / file_name(stem, ch), optimize=True)
        for ch in styles:
            im = render_glyph_AB(ttf, ch, fs)
            im.save(sdir / file_name(stem, ch), optimize=True)

    yong_p = sdir / file_name(stem, "永")
    yong = ink_yong(yong_p) if yong_p.exists() else {"h": 0, "w": 0, "area": 0, "touch": False}

    # find widest/tallest for metadata
    font_obj = ImageFont.truetype(ttf, fs if proto != "C" else CANVAS_C)
    tmp = Image.new("L", (1, 1))
    dr = ImageDraw.Draw(tmp)
    max_w, max_h, wide_ch, tall_ch = 0, 0, "", ""
    for ch in all_chars:
        bb = dr.textbbox((0, 0), ch, font=font_obj)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if w > max_w:
            max_w, wide_ch = w, ch
        if h > max_h:
            max_h, tall_ch = h, ch

    return {
        "proto": proto, "split": split, "stem": stem, "ttf": ttf,
        "n_target": len(targets), "n_style": len(styles),
        "size": fs, "max_w": max_w, "max_h": max_h,
        "wide": wide_ch, "tall": tall_ch,
        "yong": yong,
    }


def render_content(out_root: Path, proto: str, split: str, chars: list[str]) -> dict:
    cdir = out_root / split / "ContentImage"
    cdir.mkdir(parents=True, exist_ok=True)
    ttf = str(NOTO)
    if proto == "C":
        for ch in chars:
            im = render_glyph_C(ttf, ch)
            im.save(cdir / f"{cp_name(ch)}.png", optimize=True)
        return {"split": split, "n": len(chars), "size": CANVAS_C}
    else:
        all_chars = chars
        if proto == "A":
            fs = find_size_A(ttf, all_chars)
        else:
            fs = find_size_B(ttf, all_chars)
        for ch in chars:
            im = render_glyph_AB(ttf, ch, fs)
            im.save(cdir / f"{cp_name(ch)}.png", optimize=True)
        return {"split": split, "n": len(chars), "size": fs}


def qa_dataset(out_root: Path, fonts: list[dict], targets_n: int, styles_n: int) -> dict:
    errors = []
    counts = {}
    for split in ("train", "val", "test"):
        split_fonts = [f for f in fonts if f["split"] == split]
        t_count = s_count = 0
        for f in split_fonts:
            stem = f["stem"]
            tn = len(list((out_root / split / "TargetImage" / stem).glob("*.png")))
            sn = len(list((out_root / split / "StyleImage" / stem).glob("*.png")))
            t_count += tn
            s_count += sn
            if tn != targets_n or sn != styles_n:
                errors.append({"split": split, "stem": stem, "target": tn, "style": sn})
        cn = len(list((out_root / split / "ContentImage").glob("*.png")))
        counts[split] = {"fonts": len(split_fonts), "target": t_count, "style": s_count, "content": cn}
    return {"ok": len(errors) == 0, "n_error": len(errors), "errors": errors[:20], "counts": counts}


def occupancy(rows: list[dict]) -> dict:
    hs = [r["yong"]["h"] for r in rows if r.get("yong")]
    hs.sort()
    n = len(hs)
    pct = lambda p: hs[max(0, min(n - 1, int(p * n)))] if n else 0.0
    small = [{"split": r["split"], "stem": r["stem"], "yong_h": r["yong"]["h"], "size": r["size"]}
             for r in rows if r["yong"]["h"] < 0.5]
    return {
        "n_fonts": n,
        "yong_h_min": min(hs) if hs else 0,
        "yong_h_median": statistics.median(hs) if hs else 0,
        "yong_h_max": max(hs) if hs else 0,
        "yong_h_p5": pct(0.05), "yong_h_p95": pct(0.95),
        "n_small_lt_50": len(small),
        "pct_small_lt_50": round(100 * len(small) / max(n, 1), 2),
        "n_yong_touch_edge": sum(1 for r in rows if r["yong"].get("touch")),
        "small_stems": small[:10],
    }


def build_proto(proto: str, out_root: Path, workers: int, font_rows: list[dict]) -> dict:
    targets = target_chars()
    styles = style_chars()
    t0 = time.time()
    print(f"\n=== proto {proto} -> {out_root.name} workers={workers} ===", flush=True)

    for split in ("train", "val", "test"):
        render_content(out_root, proto, split, targets)

    jobs = [
        {"proto": proto, "split": r["split"], "stem": r["stem"], "ttf": r["ttf"],
         "out_root": str(out_root), "targets": targets, "styles": styles}
        for r in font_rows
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(render_one_font, j): j["stem"] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            rows.append(fut.result())
            if done % 20 == 0 or done <= 3:
                print(f"  [{proto}] {done}/{len(jobs)}", flush=True)

    rows.sort(key=lambda r: (r["split"], r["stem"]))
    occ = occupancy(rows)
    qa = qa_dataset(out_root, rows, len(targets), len(styles))

    from PIL import __version__ as pv
    protocol_desc = {
        "A": f"per-font fixed size (max textbbox ≤ {INNER_AB}) @ canvas {CANVAS_96}, margin {MARGIN_AB}, Pillow",
        "B": f"per-font height-fit (max textbbox_h ≤ {INNER_AB}, overflow shrink) @ canvas {CANVAS_96}, Pillow",
        "C": f"fsize={CANVAS_C} on {CANVAS_C}×{CANVAS_C} + BILINEAR resize {CANVAS_96}, Pillow (pygame-equiv)",
    }[proto]

    summary = {
        "dataset_id": out_root.name, "proto": proto, "status": "rendered",
        "protocol": protocol_desc,
        "software": {"pillow": pv},
        "content": {"role": "content", "n": len(targets), "font": str(NOTO)},
        "occupancy": occ, "qa": qa, "fonts": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"proto": proto, "qa_ok": qa["ok"],
                       "occ": {k: occ[k] for k in occ if k != "small_stems"},
                       "elapsed_s": summary["elapsed_s"]}, ensure_ascii=False), flush=True)
    return summary


def load_font_rows() -> list[dict]:
    ref = json.loads(REF_SUMMARY.read_text(encoding="utf-8"))
    return [{"split": f["split"], "stem": f["stem"], "ttf": f["ttf"]} for f in ref["fonts"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()
    font_rows = load_font_rows()
    for p, out in [("A", OUT_A), ("B", OUT_B), ("C", OUT_C)]:
        if args.proto in (p, "all"):
            build_proto(p, out, args.workers, font_rows)


if __name__ == "__main__":
    main()
