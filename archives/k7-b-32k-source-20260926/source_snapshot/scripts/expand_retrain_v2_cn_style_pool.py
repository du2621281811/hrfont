#!/usr/bin/env python3
"""Expand train chinese/ to a same-font CN style pool (beyond ref8).

Renders missing glyphs from data/retrain_v2/style_cn_pool.txt into
data/retrain_v2/font/train/chinese/<stem>/ for pipeline_v2 train stems.
Existing PNGs are kept; only missing chars are rendered.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))
from build_retrain_v2_dataset import _render_font_job, char_fname  # noqa: E402
from render_suiti_train_fonts_p253 import resolve_ttf  # noqa: E402

OUT_FONT = ROOT / "data/retrain_v2/font"
TRAIN_STEMS = ROOT / "reports/retrain_v2/suiti_font_screen/pipeline_v2_train_stems.txt"
POOL = ROOT / "data/retrain_v2/style_cn_pool.txt"


def load_pool(path: Path) -> list[str]:
    chars: list[str] = []
    seen: set[str] = set()
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        for ch in ln:
            if "\u4e00" <= ch <= "\u9fff" and ch not in seen:
                seen.add(ch)
                chars.append(ch)
    return chars


def main():
    stems = [s.strip() for s in TRAIN_STEMS.read_text().splitlines() if s.strip()]
    pool = load_pool(POOL)
    print(f"[expand_cn] stems={len(stems)} pool={len(pool)}", flush=True)

    jobs = []
    missing_ttf = []
    for stem in stems:
        ttf = resolve_ttf(stem)
        if ttf is None:
            missing_ttf.append(stem)
            continue
        cn_dir = OUT_FONT / "train" / "chinese" / stem
        cn_dir.mkdir(parents=True, exist_ok=True)
        need = [ch for ch in pool if not (cn_dir / char_fname(ch)).exists()]
        if not need:
            continue
        jobs.append(("train", "chinese", str(ttf), need, str(OUT_FONT)))

    if missing_ttf:
        raise SystemExit(f"missing TTF for: {missing_ttf[:20]} (+{max(0,len(missing_ttf)-20)})")

    print(f"[expand_cn] render_jobs={len(jobs)}", flush=True)
    if not jobs:
        print("[expand_cn] nothing to do", flush=True)
        return

    workers = min(32, max(4, (os.cpu_count() or 8) // 2))
    ok = err = 0
    n_total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_render_font_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r.get("err"):
                err += 1
                print(f"  ERR {r}", flush=True)
            else:
                ok += 1
                n_total += int(r.get("n") or 0)
            if i % 20 == 0 or i == len(futs):
                print(f"  progress {i}/{len(futs)} ok={ok} err={err} glyphs={n_total}", flush=True)

    # summary counts
    counts = []
    for stem in stems:
        d = OUT_FONT / "train" / "chinese" / stem
        counts.append(len(list(d.glob("*.png"))) if d.is_dir() else 0)
    print(
        f"[expand_cn] done ok={ok} err={err} new_glyphs≈{n_total} "
        f"cn_per_font min={min(counts)} max={max(counts)} median={sorted(counts)[len(counts)//2]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
