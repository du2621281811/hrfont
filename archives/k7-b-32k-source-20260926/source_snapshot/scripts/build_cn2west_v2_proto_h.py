#!/usr/bin/env python3
"""Build proto H full dataset (same layout as A/B/C).

H = A, but font-size search uses ink (pixel) bbox instead of logical textbbox.

A: per-font max fs s.t. max textbbox w,h ≤ inner (84)
H: per-font max fs s.t. max ink w,h     ≤ inner (84)

Canvas 96, margin 6, inner 84, one size per font, RGB PNG.
Content: Noto Sans CJK Regular, same H protocol.
Search upper bound 800pt (small em-square fonts need large fs).
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

CANVAS = 96
MARGIN = 6
INNER = CANVAS - 2 * MARGIN  # 84
FS_HI = 800

OUT_H = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2h-inkfit"


def target_chars() -> list[str]:
    return list(json.loads(CHARSET.read_text(encoding="utf-8"))["target_string"])


def style_chars() -> list[str]:
    return list(json.loads(CHARSET.read_text(encoding="utf-8"))["style_han_338"])


def cp_name(ch: str) -> str:
    return f"u{ord(ch):04X}"


def file_name(stem: str, ch: str) -> str:
    return f"{stem}+{cp_name(ch)}.png"


def ink_hw(ttf: str, ch: str, fs: int) -> tuple[int, int]:
    """Ink (non-white pixel) width/height of one glyph at fs. 0,0 if empty."""
    font = ImageFont.truetype(ttf, fs)
    tmp = Image.new("L", (1, 1))
    dr = ImageDraw.Draw(tmp)
    bbox = dr.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return 0, 0
    pad = 16
    im = Image.new("L", (w + pad * 2, h + pad * 2), 255)
    ImageDraw.Draw(im).text((pad - bbox[0], pad - bbox[1]), ch, font=font, fill=0)
    a = np.asarray(im)
    ys, xs = np.where(a < 250)
    if len(ys) == 0:
        return 0, 0
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def max_ink(ttf: str, fs: int, chars: list[str]) -> tuple[int, int]:
    mw = mh = 0
    for ch in chars:
        w, h = ink_hw(ttf, ch, fs)
        if w > mw:
            mw = w
        if h > mh:
            mh = h
    return mw, mh


def find_size_H(ttf: str, all_chars: list[str], inner: int = INNER) -> int:
    """Max fs where max ink w,h of every char ≤ inner.

    Probe at 48pt to find the largest-ink glyphs, binary-search on those,
    then verify the full set (shrink if any overflow).
    """
    probe_fs = 48
    ranked = []
    for ch in all_chars:
        w, h = ink_hw(ttf, ch, probe_fs)
        ranked.append((max(w, h), ch))
    ranked.sort(reverse=True)
    probes = [ch for score, ch in ranked[:12] if score > 0] or all_chars[:8]

    lo, hi, best = 8, FS_HI, 10
    for _ in range(20):
        if lo > hi:
            break
        fs = (lo + hi) // 2
        mw, mh = max_ink(ttf, fs, probes)
        if mw <= inner and mh <= inner:
            best = fs
            lo = fs + 1
        else:
            hi = fs - 1

    mw, mh = max_ink(ttf, best, all_chars)
    if mw <= inner and mh <= inner:
        return best
    lo, hi = 8, best - 1
    safe = 8
    while lo <= hi:
        fs = (lo + hi) // 2
        mw, mh = max_ink(ttf, fs, all_chars)
        if mw <= inner and mh <= inner:
            safe = fs
            lo = fs + 1
        else:
            hi = fs - 1
    return safe


def render_glyph_H(ttf: str, ch: str, fs: int, canvas: int = CANVAS, inner: int = INNER) -> Image.Image:
    """Same centering as A (textbbox), shrink if this glyph's ink exceeds inner."""
    cur = fs
    while cur >= 8:
        iw, ih = ink_hw(ttf, ch, cur)
        if iw <= inner and ih <= inner:
            break
        cur -= 1
    font = ImageFont.truetype(ttf, max(cur, 8))
    img = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((canvas - w) // 2 - bbox[0], (canvas - h) // 2 - bbox[1]), ch, font=font, fill=0)
    return Image.merge("RGB", (img, img, img))


def ink_yong(path: Path) -> dict:
    im = np.array(Image.open(path).convert("L"))
    H, W = im.shape
    ys, xs = np.where(im < 250)
    if len(ys) == 0:
        return {"h": 0.0, "w": 0.0, "area": 0.0, "touch": False}
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return {
        "h": float((y1 - y0 + 1) / H),
        "w": float((x1 - x0 + 1) / W),
        "area": float((y1 - y0 + 1) * (x1 - x0 + 1) / (H * W)),
        "touch": bool(y0 == 0 or x0 == 0 or y1 == H - 1 or x1 == W - 1),
    }


def render_one_font(job: dict) -> dict:
    split, stem, ttf = job["split"], job["stem"], job["ttf"]
    out_root = Path(job["out_root"])
    targets, styles = job["targets"], job["styles"]
    all_chars = targets + styles

    tdir = out_root / split / "TargetImage" / stem
    sdir = out_root / split / "StyleImage" / stem
    tdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)

    fs = find_size_H(ttf, all_chars)
    for ch in targets:
        render_glyph_H(ttf, ch, fs).save(tdir / file_name(stem, ch), optimize=True)
    for ch in styles:
        render_glyph_H(ttf, ch, fs).save(sdir / file_name(stem, ch), optimize=True)

    mw, mh = max_ink(ttf, fs, all_chars)
    yong_p = sdir / file_name(stem, "永")
    yong = ink_yong(yong_p) if yong_p.exists() else {"h": 0, "w": 0, "area": 0, "touch": False}
    return {
        "proto": "H",
        "split": split,
        "stem": stem,
        "ttf": ttf,
        "n_target": len(targets),
        "n_style": len(styles),
        "size": fs,
        "max_ink_w": mw,
        "max_ink_h": mh,
        "yong": yong,
    }


def render_content(out_root: Path, split: str, chars: list[str]) -> dict:
    cdir = out_root / split / "ContentImage"
    cdir.mkdir(parents=True, exist_ok=True)
    ttf = str(NOTO)
    fs = find_size_H(ttf, chars)
    for ch in chars:
        render_glyph_H(ttf, ch, fs).save(cdir / f"{cp_name(ch)}.png", optimize=True)
    return {"split": split, "n": len(chars), "size": fs}


def qa_dataset(out_root: Path, fonts: list[dict], targets_n: int, styles_n: int) -> dict:
    errors, counts = [], {}
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

    def pct(p: float) -> float:
        return hs[max(0, min(n - 1, int(p * n)))] if n else 0.0

    small = [
        {"split": r["split"], "stem": r["stem"], "yong_h": r["yong"]["h"], "size": r["size"]}
        for r in rows
        if r["yong"]["h"] < 0.5
    ]
    return {
        "n_fonts": n,
        "yong_h_min": min(hs) if hs else 0,
        "yong_h_median": statistics.median(hs) if hs else 0,
        "yong_h_max": max(hs) if hs else 0,
        "yong_h_p5": pct(0.05),
        "yong_h_p95": pct(0.95),
        "n_small_lt_50": len(small),
        "pct_small_lt_50": round(100 * len(small) / max(n, 1), 2),
        "n_yong_touch_edge": sum(1 for r in rows if r["yong"].get("touch")),
        "small_stems": small[:10],
    }


def load_font_rows() -> list[dict]:
    ref = json.loads(REF_SUMMARY.read_text(encoding="utf-8"))
    return [{"split": f["split"], "stem": f["stem"], "ttf": f["ttf"]} for f in ref["fonts"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()
    targets, styles = target_chars(), style_chars()
    font_rows = load_font_rows()
    t0 = time.time()
    print(f"=== proto H -> {OUT_H.name} workers={args.workers} ===", flush=True)

    for split in ("train", "val", "test"):
        print(f"  content {split}…", flush=True)
        render_content(OUT_H, split, targets)

    jobs = [
        {"split": r["split"], "stem": r["stem"], "ttf": r["ttf"],
         "out_root": str(OUT_H), "targets": targets, "styles": styles}
        for r in font_rows
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(render_one_font, j): j["stem"] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            rows.append(fut.result())
            if done % 10 == 0 or done <= 3:
                print(f"  [H] {done}/{len(jobs)}", flush=True)

    rows.sort(key=lambda r: (r["split"], r["stem"]))
    occ, qa = occupancy(rows), qa_dataset(OUT_H, rows, len(targets), len(styles))
    try:
        from PIL import __version__ as pv
    except Exception:
        pv = "?"
    summary = {
        "dataset_id": OUT_H.name,
        "proto": "H",
        "status": "rendered",
        "protocol": "H: A-like per-font fixed size, but max ink w,h ≤ 84 @ native 96, margin 6, RGB PNG",
        "software": {"pillow": pv},
        "content": {"font": str(NOTO), "n": len(targets), "note": "same H protocol as Target"},
        "occupancy": occ,
        "qa": qa,
        "fonts": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_H.mkdir(parents=True, exist_ok=True)
    (OUT_H / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "proto": "H", "qa_ok": qa["ok"],
        "occupancy": {k: occ[k] for k in occ if k != "small_stems"},
        "elapsed_s": summary["elapsed_s"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
