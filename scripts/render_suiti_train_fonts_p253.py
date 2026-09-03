#!/usr/bin/env python3
"""Render CN style refs + P1∪P2 EN glyphs for pipeline_v2 train/test fonts."""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))
from build_retrain_v2_dataset import STYLE_REFS, _render_font_job, char_fname  # noqa: E402

SUITI = ROOT / "data/suiti_fonts_probe/随体字体集"
FONT50 = Path("/root/data/font_50")
OUT_FONT = ROOT / "data/retrain_v2/font"
TRAIN_STEMS = ROOT / "reports/retrain_v2/suiti_font_screen/pipeline_v2_train_stems.txt"
TEST_STEMS = ROOT / "reports/retrain_v2/suiti_font_screen/pipeline_v2_test_stems.txt"


def resolve_ttf(stem: str) -> Path | None:
    for root in (SUITI, FONT50):
        if not root.exists():
            continue
        for p in list(root.glob(f"{stem}.ttf")) + list(root.glob(f"{stem}.TTF")):
            return p
        # rare: exact stem match among all
        for p in list(root.glob("*.ttf")) + list(root.glob("*.TTF")):
            if p.stem == stem:
                return p
    return None


def needs_render(phase: str, stem: str, n_en_expect: int, n_cn_expect: int = 8) -> bool:
    cn = OUT_FONT / phase / "chinese" / stem
    en = OUT_FONT / phase / "english" / stem
    if not cn.is_dir() or not en.is_dir():
        return True
    if len(list(cn.glob("*.png"))) < n_cn_expect:
        return True
    if len(list(en.glob("*.png"))) < max(10, n_en_expect // 2):
        return True
    return False


def main():
    uni = json.loads((ROOT / "data/unified_v1/meta.json").read_text(encoding="utf-8"))
    english_chars = list(uni["L_p1"]) + list(uni["L_p2"])
    train = [s.strip() for s in TRAIN_STEMS.read_text().splitlines() if s.strip()]
    test = [s.strip() for s in TEST_STEMS.read_text().splitlines() if s.strip()]

    jobs = []
    missing_ttf = []
    for phase, stems in (("train", train), ("test_unknown_style", test)):
        for stem in stems:
            if not needs_render(phase, stem, len(english_chars)):
                continue
            ttf = resolve_ttf(stem)
            if ttf is None:
                missing_ttf.append(stem)
                continue
            jobs.append((phase, "chinese", str(ttf), STYLE_REFS, str(OUT_FONT)))
            jobs.append((phase, "english", str(ttf), english_chars, str(OUT_FONT)))

    print(
        f"[render_p253] train={len(train)} test={len(test)} "
        f"jobs={len(jobs)} missing_ttf={missing_ttf}",
        flush=True,
    )
    if missing_ttf:
        raise SystemExit(f"missing TTF for: {missing_ttf}")
    if not jobs:
        print("[render_p253] nothing to do", flush=True)
        return

    workers = min(32, max(4, (os.cpu_count() or 8) // 2))
    ok = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_render_font_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r.get("err"):
                err += 1
                print(f"[err] {r}", flush=True)
            else:
                ok += 1
            if i % 40 == 0 or i == len(futs):
                print(f"[render_p253] {i}/{len(futs)} ok={ok} err={err}", flush=True)

    # verify train coverage
    thin = []
    for stem in train:
        en = OUT_FONT / "train" / "english" / stem
        cn = OUT_FONT / "train" / "chinese" / stem
        n_en = len(list(en.glob("*.png"))) if en.is_dir() else 0
        n_cn = len(list(cn.glob("*.png"))) if cn.is_dir() else 0
        if n_en < 100 or n_cn < 8:
            thin.append((stem, n_cn, n_en))
    print(f"[render_p253] done thin_or_missing={len(thin)}", flush=True)
    if thin[:20]:
        print("examples:", thin[:20], flush=True)


if __name__ == "__main__":
    main()
