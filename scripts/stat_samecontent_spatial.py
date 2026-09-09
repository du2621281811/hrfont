#!/usr/bin/env python3
"""Same-content spatial variation on A-protocol 96×96 PNGs.

Measures ink bbox aspect (w/h), coverage, and across-font spread for each
target character. Also compares ContentImage (single content face) vs
TargetImage (train228), and F2/F3 preds vs GT on test16 when present.

Does not require GPU.
"""
from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/projects/hrfont")
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = json.loads((ROOT / "manifests/split_v3_228_16_16.json").read_text())
CHARSET = json.loads((ROOT / "manifests/charset_cn2west_v2_planned.json").read_text())
OUT = ROOT / "reports/samecontent_spatial_variation"
PRED_ROOT = ROOT / "reports/f03_test16_strat/preds"
SEED = 3407


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def script_of(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    name = unicodedata.name(ch, "")
    if "HIRAGANA" in name:
        return "hiragana"
    if "KATAKANA" in name:
        return "katakana"
    if "BOPOMOFO" in name:
        return "bopomofo"
    if "LATIN" in name and ch.isupper():
        return "latin_upper"
    if "LATIN" in name and ch.islower():
        return "latin_lower"
    return "other"


def ink_geom(path: Path, thr: int = 32) -> dict | None:
    arr = np.asarray(Image.open(path).convert("L"))
    ink = arr < (255 - thr)
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    return {
        "w": w,
        "h": h,
        "aspect": w / max(h, 1),  # <1 tall, >1 wide, ~1 square
        "cover": float(ink.mean()),
        "cx": float(xs.mean()) / arr.shape[1],
        "cy": float(ys.mean()) / arr.shape[0],
    }


def summarize(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target_chars = list(CHARSET["target_string"])
    fonts_all = SPLIT["stems"]["train"]
    # Full 228×295 is ~67k PNG opens; use all fonts on probe chars,
    # and a 40-font subsample for the rest (seed 3407).
    rng = np.random.default_rng(3407)
    fonts_sub = sorted(rng.choice(fonts_all, size=min(40, len(fonts_all)), replace=False).tolist())
    fonts = fonts_all
    test_fonts = SPLIT["stems"]["test"]
    probes_set = set(list("1IlijOo0mwMW") + ["あ", "ア", "ㄅ"])

    per_char: dict[str, dict] = {}
    by_script: dict[str, list[float]] = defaultdict(list)
    missing = 0

    for ch in target_chars:
        cp = cp_of(ch)
        cpath = DATA / "train" / "ContentImage" / f"{cp}.png"
        cg = ink_geom(cpath) if cpath.is_file() else None
        aspects, heights, covers = [], [], []
        use_fonts = fonts_all if ch in probes_set else fonts_sub
        i = target_chars.index(ch)
        if i % 20 == 0:
            print(f"char {i}/{len(target_chars)} {ch!r} fonts={len(use_fonts)}", flush=True)
        for font in use_fonts:
            p = DATA / "train" / "TargetImage" / font / f"{font}+{cp}.png"
            g = ink_geom(p) if p.is_file() else None
            if g is None:
                missing += 1
                continue
            aspects.append(g["aspect"])
            heights.append(g["h"])
            covers.append(g["cover"])
        if not aspects:
            continue
        rec = {
            "char": ch,
            "cp": cp,
            "script": script_of(ch),
            "content_aspect": None if cg is None else cg["aspect"],
            "content_h": None if cg is None else cg["h"],
            "target_aspect": summarize(aspects),
            "target_h": summarize(heights),
            "target_cover": summarize(covers),
        }
        rec["aspect_cv"] = rec["target_aspect"]["std"] / max(rec["target_aspect"]["mean"], 1e-6)
        rec["content_vs_target_mean"] = (
            None if cg is None else rec["target_aspect"]["mean"] - cg["aspect"]
        )
        per_char[cp] = rec
        by_script[rec["script"]].extend(aspects)

    script_summary = {k: summarize(v) for k, v in by_script.items()}

    probes = list("1IlijOo0mwMW") + ["あ", "ア", "ㄅ"]
    probe_table = []
    for ch in probes:
        rec = per_char.get(cp_of(ch))
        if rec:
            probe_table.append({
                "char": ch,
                "script": rec["script"],
                "content_aspect": rec["content_aspect"],
                "target_mean": rec["target_aspect"]["mean"],
                "target_p05": rec["target_aspect"]["p05"],
                "target_p95": rec["target_aspect"]["p95"],
                "aspect_cv": rec["aspect_cv"],
                "mean_h": rec["target_h"]["mean"],
            })

    pred_cmp = {}
    for mid in ("F0_100k", "F2_75000", "F2_80000", "F3_80k"):
        rows = []
        for stem in test_fonts:
            for ch in "1IlAOomw":
                cp = cp_of(ch)
                gt = DATA / "test" / "TargetImage" / stem / f"{stem}+{cp}.png"
                pred = PRED_ROOT / mid / "test" / stem / f"test__{stem}__{cp}__s{SEED}.png"
                gg, pg = ink_geom(gt), ink_geom(pred) if pred.is_file() else None
                if gg is None or pg is None:
                    continue
                rows.append({
                    "font": stem,
                    "char": ch,
                    "gt_aspect": gg["aspect"],
                    "pred_aspect": pg["aspect"],
                    "d_aspect": pg["aspect"] - gg["aspect"],
                    "gt_h": gg["h"],
                    "pred_h": pg["h"],
                    "d_h": pg["h"] - gg["h"],
                })
        if rows:
            d_aspect = [r["d_aspect"] for r in rows]
            d_h = [r["d_h"] for r in rows]
            pred_cmp[mid] = {
                "n": len(rows),
                "d_aspect": summarize(d_aspect),
                "d_h": summarize(d_h),
                "examples": rows[:24],
            }

    ranked = sorted(per_char.values(), key=lambda r: r["aspect_cv"], reverse=True)
    payload = {
        "canvas": 96,
        "n_fonts_train": len(fonts_all),
        "n_fonts_subsample": len(fonts_sub),
        "n_fonts_probe_chars": len(fonts_all),
        "n_chars": len(target_chars),
        "missing_png": missing,
        "note": (
            "aspect=ink_bbox_w/h on 96×96 A-protocol PNGs. "
            "<1 tall/narrow, ~1 square, >1 wide. "
            "ContentImage is ONE content face; TargetImage is train228."
        ),
        "script_aspect": script_summary,
        "probes": probe_table,
        "highest_aspect_cv": [
            {"char": r["char"], "cv": r["aspect_cv"], "mean": r["target_aspect"]["mean"],
             "p05": r["target_aspect"]["p05"], "p95": r["target_aspect"]["p95"]}
            for r in ranked[:15]
        ],
        "lowest_aspect_cv": [
            {"char": r["char"], "cv": r["aspect_cv"], "mean": r["target_aspect"]["mean"]}
            for r in ranked[-10:]
        ],
        "pred_vs_gt_test16": pred_cmp,
        "per_char": per_char,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Same-content spatial variation (train228 TargetImage)",
        "",
        "Canvas 96×96, ink bbox aspect = width/height. <1 瘦高，~1 偏方，>1 偏扁。",
        "",
        "## Script 均值",
        "",
        "| script | n | mean aspect | std | p05 | p95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, s in sorted(script_summary.items()):
        lines.append(
            f"| {k} | {s['n']} | {s['mean']:.3f} | {s['std']:.3f} | {s['p05']:.3f} | {s['p95']:.3f} |"
        )
    lines += [
        "",
        "## 探针字（Content vs Target 跨字体）",
        "",
        "| char | content aspect | target mean | p05–p95 | CV | mean h |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in probe_table:
        ca = "—" if r["content_aspect"] is None else f"{r['content_aspect']:.3f}"
        lines.append(
            f"| {r['char']} | {ca} | {r['target_mean']:.3f} | "
            f"{r['target_p05']:.3f}–{r['target_p95']:.3f} | {r['aspect_cv']:.3f} | {r['mean_h']:.1f} |"
        )
    if pred_cmp:
        lines += ["", "## test16 pred − GT aspect（正=比 GT 更扁/更方）", ""]
        for mid, rec in pred_cmp.items():
            d = rec["d_aspect"]
            lines.append(
                f"- **{mid}** n={rec['n']}: mean Δaspect={d['mean']:+.3f} "
                f"(p05={d['p05']:+.3f}, p95={d['p95']:+.3f}); mean Δh={rec['d_h']['mean']:+.1f}px"
            )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'summary.json'} chars={len(per_char)} missing={missing}")
    print((OUT / "README.md").read_text())


if __name__ == "__main__":
    main()
