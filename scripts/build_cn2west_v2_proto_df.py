#!/usr/bin/env python3
"""Build proto D / F full datasets (same layout as A/B/C).

D: per-glyph binary-search max textbbox @ native 96, margin 8px, RGB PNG
F: per-glyph ink-margin fit @ native 96 (~8% canvas), RGB PNG
Content: Noto Sans CJK Regular, same per-glyph protocol as Target role.

Parallel: one worker per font (deterministic, no shared state).
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
MARGIN_D = 8
MARGIN_F_FRAC = 0.08

OUT_D = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2d-perglyph-max96"
OUT_F = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2f-perglyph-fit96"


def target_chars() -> list[str]:
    cs = json.loads(CHARSET.read_text(encoding="utf-8"))
    s = cs["target_string"]
    return list(s)


def style_chars() -> list[str]:
    cs = json.loads(CHARSET.read_text(encoding="utf-8"))
    return list(cs["style_han_338"])


def cp_name(ch: str) -> str:
    return f"u{ord(ch):04X}"


def file_name(stem: str, ch: str) -> str:
    return f"{stem}+{cp_name(ch)}.png"


def render_D(ttf: str, ch: str, canvas: int = CANVAS, margin: int = MARGIN_D) -> tuple[Image.Image, int]:
    img = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)
    lo, hi, best_fs, best = 8, 800, None, None
    for _ in range(20):
        fs = (lo + hi) // 2
        f = ImageFont.truetype(ttf, fs)
        bbox = draw.textbbox((0, 0), ch, font=f)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= canvas - margin and h <= canvas - margin:
            best, best_fs = (f, bbox), fs
            lo = fs + 1
        else:
            hi = fs - 1
    if best is None:
        best_fs = max(10, canvas // 2)
        f = ImageFont.truetype(ttf, best_fs)
        bbox = draw.textbbox((0, 0), ch, font=f)
    else:
        f, bbox = best
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((canvas - w) // 2 - bbox[0], (canvas - h) // 2 - bbox[1]), ch, font=f, fill=0)
    rgb = Image.merge("RGB", (img, img, img))
    return rgb, int(best_fs)


def render_F(ttf: str, ch: str, canvas: int = CANVAS) -> tuple[Image.Image, int]:
    """Per-glyph ink-margin fit: find fs so min ink margin ≈ target_margin."""
    target_margin = int(round(canvas * MARGIN_F_FRAC))

    def _ink_margin(fs: int) -> tuple[int | None, Image.Image]:
        im = Image.new("RGB", (canvas, canvas), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype(ttf, fs)
        bbox = dr.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= 0 or h <= 0:
            return None, im
        x = (canvas - w) // 2 - bbox[0]
        y = (canvas - h) // 2 - bbox[1]
        dr.text((x, y), ch, fill=(0, 0, 0), font=font)
        a = np.asarray(im.convert("L"))
        ys, xs = np.where(a < 250)
        if len(ys) == 0:
            return None, im
        m = min(int(xs.min()), int(ys.min()), canvas - 1 - int(xs.max()), canvas - 1 - int(ys.max()))
        return m, im

    # Binary search: find largest fs where ink_margin >= target_margin
    lo, hi, best_fs, best_im = 8, 800, None, None
    for _ in range(20):
        if lo > hi:
            break
        fs = (lo + hi) // 2
        m, im = _ink_margin(fs)
        if m is not None and m >= target_margin:
            best_fs, best_im = fs, im
            lo = fs + 1
        else:
            hi = fs - 1

    # Fine-tune: check ±2 around best_fs for closest to target_margin
    if best_fs is not None:
        candidates = []
        for fs in range(max(8, best_fs - 2), best_fs + 3):
            m, im = _ink_margin(fs)
            if m is not None:
                candidates.append((abs(m - target_margin), -fs, fs, im))
        if candidates:
            candidates.sort()
            best_fs, best_im = candidates[0][2], candidates[0][3]

    if best_im is None:
        best_im = Image.new("RGB", (canvas, canvas), (255, 255, 255))
        best_fs = 0
    return best_im, int(best_fs)


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
    renderer = render_D if proto == "D" else render_F

    tdir = out_root / split / "TargetImage" / stem
    sdir = out_root / split / "StyleImage" / stem
    tdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)

    fs_samples = []
    for ch in targets:
        im, fs = renderer(ttf, ch)
        im.save(tdir / file_name(stem, ch), optimize=True)
        fs_samples.append(fs)
    for ch in styles:
        im, fs = renderer(ttf, ch)
        im.save(sdir / file_name(stem, ch), optimize=True)
        fs_samples.append(fs)

    yong_p = sdir / file_name(stem, "永")
    yong = ink_yong(yong_p) if yong_p.exists() else {"h": 0, "w": 0, "area": 0, "touch": False}
    return {
        "proto": proto,
        "split": split,
        "stem": stem,
        "ttf": ttf,
        "n_target": len(targets),
        "n_style": len(styles),
        "yong": yong,
        "fs_median": int(statistics.median(fs_samples)) if fs_samples else 0,
        "fs_min": min(fs_samples) if fs_samples else 0,
        "fs_max": max(fs_samples) if fs_samples else 0,
    }


def render_content(out_root: Path, proto: str, split: str, chars: list[str]) -> dict:
    renderer = render_D if proto == "D" else render_F
    cdir = out_root / split / "ContentImage"
    cdir.mkdir(parents=True, exist_ok=True)
    ttf = str(NOTO)
    fs_list = []
    for ch in chars:
        im, fs = renderer(ttf, ch)
        im.save(cdir / f"{cp_name(ch)}.png", optimize=True)
        fs_list.append(fs)
    return {"split": split, "n": len(chars), "fs_median": int(statistics.median(fs_list))}


def qa_dataset(out_root: Path, fonts: list[dict], targets_n: int, styles_n: int) -> dict:
    errors = []
    counts = {}
    for split in ("train", "val", "test"):
        split_fonts = [f for f in fonts if f["split"] == split]
        t_count = s_count = 0
        for f in split_fonts:
            stem = f["stem"]
            tdir = out_root / split / "TargetImage" / stem
            sdir = out_root / split / "StyleImage" / stem
            tn = len(list(tdir.glob("*.png")))
            sn = len(list(sdir.glob("*.png")))
            t_count += tn
            s_count += sn
            if tn != targets_n or sn != styles_n:
                errors.append({"split": split, "stem": stem, "target": tn, "style": sn})
        cn = len(list((out_root / split / "ContentImage").glob("*.png")))
        counts[split] = {"fonts": len(split_fonts), "target": t_count, "style": s_count, "content": cn}
    return {"ok": len(errors) == 0, "n_error": len(errors), "errors": errors[:20], "counts": counts}


def occupancy(rows: list[dict]) -> dict:
    hs = [r["yong"]["h"] for r in rows if r.get("yong")]
    hs_sorted = sorted(hs)
    n = len(hs)

    def pct(p):
        if n == 0:
            return 0.0
        i = max(0, min(n - 1, int(p * n)))
        return hs_sorted[i]

    small = [{"split": r["split"], "stem": r["stem"], "yong_h": r["yong"]["h"], "fs_median": r["fs_median"]} for r in rows if r["yong"]["h"] < 0.5]
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


def build_proto(proto: str, out_root: Path, workers: int, font_rows: list[dict]) -> dict:
    targets = target_chars()
    styles = style_chars()
    t0 = time.time()
    print(f"\n=== proto {proto} -> {out_root.name} workers={workers} ===", flush=True)

    for split in ("train", "val", "test"):
        render_content(out_root, proto, split, targets)

    jobs = [
        {
            "proto": proto,
            "split": r["split"],
            "stem": r["stem"],
            "ttf": r["ttf"],
            "out_root": str(out_root),
            "targets": targets,
            "styles": styles,
        }
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

    try:
        from PIL import __version__ as pillow_v

        pillow_ver = pillow_v
    except Exception:
        pillow_ver = "?"

    summary = {
        "dataset_id": out_root.name,
        "proto": proto,
        "status": "rendered",
        "protocol": (
            "D: per-glyph binary-search max textbbox @ native 96, margin 8px, RGB PNG"
            if proto == "D"
            else "F: per-glyph ink-margin fit @ native 96 (~8%), RGB PNG"
        ),
        "software": {"pillow": pillow_ver},
        "content": {
            "font": str(NOTO),
            "n": len(targets),
            "note": "per-glyph same protocol as Target",
        },
        "occupancy": occ,
        "qa": qa,
        "fonts": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"proto": proto, "qa_ok": qa["ok"], "occupancy": {k: occ[k] for k in occ if k != "small_stems"}, "elapsed_s": summary["elapsed_s"]}, ensure_ascii=False), flush=True)
    return summary


def load_font_rows() -> list[dict]:
    ref = json.loads(REF_SUMMARY.read_text(encoding="utf-8"))
    return [{"split": f["split"], "stem": f["stem"], "ttf": f["ttf"]} for f in ref["fonts"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", choices=["D", "F", "both"], default="both")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()
    font_rows = load_font_rows()
    if args.proto in ("D", "both"):
        build_proto("D", OUT_D, args.workers, font_rows)
    if args.proto in ("F", "both"):
        build_proto("F", OUT_F, args.workers, font_rows)


if __name__ == "__main__":
    main()
