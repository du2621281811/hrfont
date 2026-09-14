#!/usr/bin/env python3
"""Matched dirty vs clean F0/F2 (+ P1) metrics board on timeline BOARD_CHARS.

Uses existing preds under reports/f03_test16_strat/preds/.
Fairness: only (font,char) keys present for ALL listed methods.
Writes reports/f03_test16_strat/clean_dirty_compare.{json,html}.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/f03_test16_strat"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
E12_SUM = ROOT / "reports/e12_paper/scores_test16_summary.json"
E12_ITEMS = ROOT / "reports/e12_paper/scores_test16_items.json"
SEED = 3407
BOARD_CHARS = list("08AGQaegàěあさアンㄅㄚij")

METHODS = [
    ("P1", "官方 P1（未在本任务上训）", "baseline"),
    ("F0_100k", "F0 脏@100k", "dirty"),
    ("F0C_95000", "F0 净@95k(best)", "clean"),
    ("F0C_100000", "F0 净@100k", "clean"),
    ("F2_40000", "F2 脏@40k", "dirty"),
    ("F2C_40000", "F2 净@40k", "clean"),
    ("F2_80000", "F2 脏@80k", "dirty"),
    ("F2C_80000", "F2 净@80k", "clean"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def script_bucket(ch: str) -> str:
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
    if "LATIN" in name and ch.islower() and ord(ch) > 127:
        return "latin_ext"
    if "LATIN" in name and ch.islower():
        return "latin_lower"
    return "other"


def parse_pred(p: Path) -> tuple[str, str] | None:
    # test__FONT__uXXXX__s3407.png
    m = re.match(r"test__(.+)__(u[0-9A-Fa-f]+)__s\d+\.png$", p.name)
    if not m:
        return None
    cp = m.group(2)
    cp = "u" + cp[1:].upper()
    return m.group(1), cp


def gt_path(font: str, cp: str) -> Path | None:
    for split in ("test", "val", "train"):
        p = DATA / split / "TargetImage" / font / f"{font}+{cp}.png"
        if p.is_file():
            return p
    return None


def to_gray01(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    # lightweight SSIM (same spirit as eval_f03)
    c1, c2 = (0.01 * 1) ** 2, (0.03 * 1) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sa, sb = a.var(), b.var()
    sab = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sa + sb + c2) + 1e-12))


def ink_coverage(a: np.ndarray, thr: float = 0.92) -> float:
    return float((a < thr).mean())


def mean_std(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    arr = np.asarray(xs, dtype=np.float64)
    return {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def index_preds(mid: str) -> dict[tuple[str, str], Path]:
    out = {}
    root = OUT / "preds" / mid
    if not root.is_dir():
        return out
    for p in root.rglob("*.png"):
        parsed = parse_pred(p)
        if not parsed:
            continue
        font, cp = parsed
        # normalize to uXXXX (lowercase u)
        cp = "u" + cp[1:].upper()
        out[(font, cp)] = p
    return out


def score_method(mid: str, keys: list[tuple[str, str]], index: dict, lpips_fn=None, device="cpu") -> list[dict]:
    rows = []
    for font, cp in keys:
        png = index[(font, cp)]
        gtp = gt_path(font, cp)
        if gtp is None:
            continue
        ch = chr(int(cp[1:], 16))
        pred = Image.open(png).convert("RGB").resize((96, 96))
        gt = Image.open(gtp).convert("RGB").resize((96, 96))
        pa, ga = to_gray01(pred), to_gray01(gt)
        rec = {
            "method": mid,
            "font": font,
            "char": ch,
            "cp": cp,
            "bucket": script_bucket(ch),
            "L1": float(np.abs(pa - ga).mean()),
            "SSIM": ssim(pa, ga),
            "coverage": ink_coverage(pa),
            "LPIPS": None,
            "rel": str(png.relative_to(OUT)),
        }
        if lpips_fn is not None:
            import torch

            def to_n11(im: Image.Image):
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                return (torch.from_numpy(arr).permute(2, 0, 1)[None] * 2 - 1).to(device)

            with torch.no_grad():
                rec["LPIPS"] = float(lpips_fn(to_n11(pred), to_n11(gt)).item())
        rows.append(rec)
    return rows


def agg(rows: list[dict]) -> dict:
    out = {
        "n": len(rows),
        "L1_mean": float(np.mean([r["L1"] for r in rows])) if rows else None,
        "SSIM_mean": float(np.mean([r["SSIM"] for r in rows])) if rows else None,
        "coverage_mean": float(np.mean([r["coverage"] for r in rows])) if rows else None,
        "LPIPS_mean": None,
    }
    lp = [r["LPIPS"] for r in rows if r.get("LPIPS") is not None]
    if lp:
        out["LPIPS_mean"] = float(np.mean(lp))
    return out


def load_e12_matched(keys: set[tuple[str, str]]) -> dict:
    """Pull E12 mem_prob for matched keys when item file exists."""
    if not E12_ITEMS.is_file():
        # fall back to summary means only
        if E12_SUM.is_file():
            return {"summary_only": json.loads(E12_SUM.read_text()), "by_method": {}}
        return {"summary_only": None, "by_method": {}}
    items = json.loads(E12_ITEMS.read_text())
    by = defaultdict(list)
    for it in items:
        mid = it.get("method")
        font = it.get("font")
        ch = it.get("char")
        if mid is None or font is None or ch is None:
            continue
        cp = cp_of(ch)
        if (font, cp) not in keys:
            continue
        if "e12_mem_prob" in it:
            by[mid].append(float(it["e12_mem_prob"]))
        elif "mem_prob" in it:
            by[mid].append(float(it["mem_prob"]))
    return {"summary_only": None, "by_method": {m: mean_std(vs) for m, vs in by.items()}}


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}"


def write_html(report: dict, path: Path) -> None:
    rows_html = []
    for mid, label, family in METHODS:
        m = report["methods"].get(mid, {})
        o = m.get("overall", {})
        e12 = report.get("e12_matched", {}).get(mid) or {}
        e12_sum = (report.get("e12_summary") or {}).get("methods", {}).get(mid, {})
        mem = e12.get("mean")
        mem_n = e12.get("n")
        if mem is None:
            mem = (e12_sum.get("latin_digit") or {}).get("mem_prob", {}).get("mean")
            mem_n = (e12_sum.get("latin_digit") or {}).get("mem_prob", {}).get("n")
        rows_html.append(
            "<tr class='{fam}'><td>{lab}</td><td>{fam}</td><td>{n}</td>"
            "<td>{l1}</td><td>{ssim}</td><td>{lpips}</td><td>{mem}</td><td>{memn}</td></tr>".format(
                fam=family,
                lab=label,
                n=o.get("n") or 0,
                l1=fmt(o.get("L1_mean")),
                ssim=fmt(o.get("SSIM_mean")),
                lpips=fmt(o.get("LPIPS_mean")),
                mem=fmt(mem),
                memn=mem_n if mem_n is not None else "—",
            )
        )

    # paired deltas: clean - dirty at matched checkpoints
    pairs = [
        ("F0C_95000", "F0_100k", "F0 净best − 脏100k"),
        ("F0C_100000", "F0_100k", "F0 净100k − 脏100k"),
        ("F2C_40000", "F2_40000", "F2 净40k − 脏40k"),
        ("F2C_80000", "F2_80000", "F2 净80k − 脏80k"),
        ("F2C_40000", "P1", "F2 净40k − 官方P1"),
        ("F2C_80000", "P1", "F2 净80k − 官方P1"),
    ]
    delta_rows = []
    for a, b, name in pairs:
        oa = report["methods"].get(a, {}).get("overall", {})
        ob = report["methods"].get(b, {}).get("overall", {})
        if not oa or not ob:
            continue

        def d(key, better="down"):
            va, vb = oa.get(key), ob.get(key)
            if va is None or vb is None:
                return "—"
            diff = va - vb
            # for SSIM / mem, up is better so flip sign interpretation in note only
            return fmt(diff)

        delta_rows.append(
            f"<tr><td>{name}</td><td>{d('L1_mean')}</td><td>{d('SSIM_mean')}</td><td>{d('LPIPS_mean')}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>干净 vs 脏 · F0/F2 对照（匹配字符）</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#f5f3ef;color:#1a1a1a;line-height:1.45}}
h1{{font-size:1.35rem;margin:0 0 .35rem}}
.meta,.note{{color:#555;font-size:.92rem;max-width:82ch}}
table{{border-collapse:collapse;background:#fff;margin:12px 0}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:right}}
th:first-child,td:first-child,td:nth-child(2){{text-align:left}}
tr.clean{{background:#f3faf3}}
tr.dirty{{background:#fffaf3}}
tr.baseline{{background:#f3f5fa}}
.links a{{margin-right:12px}}
</style></head><body>
<h1>干净 vs 脏 · F0 / F2 对照</h1>
<p class="meta">生成 {report['computed_at']} · 协议 test16 × BOARD_CHARS({len(BOARD_CHARS)}) · 仅统计<strong>所有方法都有预测</strong>的 (font,char) · n={report['n_matched']} · seed {SEED}</p>
<p class="note">L1/SSIM/LPIPS 相对 GT 只是像素诊断，不能单独当「风格成功」。E12 mem_prob（拉丁+数字）更接近家族兼容性。负的 ΔL1 / 正的 ΔSSIM 表示干净侧更好。</p>
<p class="links">
  <a href="timeline_f2_clean.html">时间线看图</a>
  <a href="multi_exp_compare.html">多臂看图</a>
  <a href="../e12_paper/PAPER_TABLES.md">E12 表</a>
</p>
<h2>总表（匹配子集）</h2>
<table>
<thead><tr><th>方法</th><th>臂</th><th>n</th><th>L1↓</th><th>SSIM↑</th><th>LPIPS↓</th><th>E12 mem↑</th><th>E12 n</th></tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
<h2>成对差值（A − B）</h2>
<table>
<thead><tr><th>对比</th><th>ΔL1</th><th>ΔSSIM</th><th>ΔLPIPS</th></tr></thead>
<tbody>
{''.join(delta_rows)}
</tbody></table>
<p class="note">ΔL1&lt;0、ΔSSIM&gt;0、ΔLPIPS&lt;0 → A 更好。正式合同主读点：F2@40k；附录 F2@80k / F0@100k。</p>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    indexes = {mid: index_preds(mid) for mid, _, _ in METHODS}
    # restrict to board cps
    board_cps = {cp_of(ch) for ch in BOARD_CHARS}
    for mid in indexes:
        indexes[mid] = {k: v for k, v in indexes[mid].items() if k[1] in board_cps}

    key_sets = [set(indexes[mid]) for mid, _, _ in METHODS if indexes[mid]]
    if not key_sets:
        raise SystemExit("no preds found")
    matched = sorted(set.intersection(*key_sets))
    if not matched:
        # fall back: keys present for at least clean F0C + dirty F0 + P1 + F2 pairs separately reported
        raise SystemExit("empty intersection across all methods; check pred coverage")

    lpips_fn = None
    if args.lpips:
        import torch
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(args.device).eval()

    methods_out = {}
    all_rows = []
    for mid, label, family in METHODS:
        rows = score_method(mid, matched, indexes[mid], lpips_fn, args.device)
        by_b = defaultdict(list)
        for r in rows:
            by_b[r["bucket"]].append(r)
        methods_out[mid] = {
            "label": label,
            "family": family,
            "overall": agg(rows),
            "by_bucket": {b: agg(rs) for b, rs in sorted(by_b.items())},
        }
        all_rows.extend(rows)

    e12 = load_e12_matched(set(matched))
    e12_summary = json.loads(E12_SUM.read_text()) if E12_SUM.is_file() else None

    report = {
        "computed_at": utc_now(),
        "protocol": "matched BOARD_CHARS across P1/F0/F0C/F2/F2C; diagnostics vs GT",
        "n_matched": len(matched),
        "chars": BOARD_CHARS,
        "fonts": sorted({f for f, _ in matched}),
        "methods": methods_out,
        "e12_matched": e12.get("by_method", {}),
        "e12_summary": e12_summary,
        "caveat": "L1/SSIM/LPIPS are pixel diagnostics. Prefer E12 mem_prob + eye board for style claims. Clean preds currently timeline-subset only.",
    }
    out_json = OUT / "clean_dirty_compare.json"
    out_html = OUT / "clean_dirty_compare.html"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "clean_dirty_compare_items.json").write_text(
        json.dumps(all_rows, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_html(report, out_html)
    print(json.dumps({m: methods_out[m]["overall"] for m, _, _ in METHODS}, indent=2))
    print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
