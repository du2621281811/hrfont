#!/usr/bin/env python3
"""Spearman correlation: human ratings vs E12 cosine Style Score.

    python scripts/e12_spearman_human.py \
      --ratings reports/e12_paper/human_ratings.csv \
      --items reports/e12_paper/human_board/items.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/e12_paper"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """O(n^2) Kendall τ-b (ties ignored in concordant sense; fine for n~100)."""
    n = x.size
    if n < 3:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 or dy == 0:
                continue
            if dx == dy:
                conc += 1
            else:
                disc += 1
    denom = conc + disc
    return float((conc - disc) / denom) if denom else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", type=Path, required=True)
    ap.add_argument("--items", type=Path, default=OUT / "human_board" / "items.json")
    ap.add_argument("--attn-fail-threshold", type=float, default=4.0, help="attn score >= this counts as fail")
    ap.add_argument("--max-attn-fail-frac", type=float, default=0.5)
    args = ap.parse_args()

    items = {it["item_id"]: it for it in json.loads(args.items.read_text(encoding="utf-8"))["items"]}
    by_rater: dict[str, dict[str, float]] = defaultdict(dict)
    with args.ratings.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = (row.get("score") or "").strip()
            if not s:
                continue
            by_rater[row["rater"]][row["item_id"]] = float(s)

    rater_reports = {}
    for rater, scores in by_rater.items():
        attn_ids = [i for i, it in items.items() if it.get("attention_check")]
        attn_scored = [scores[i] for i in attn_ids if i in scores]
        fail = sum(1 for s in attn_scored if s >= args.attn_fail_threshold)
        frac = (fail / len(attn_scored)) if attn_scored else 1.0
        rater_reports[rater] = {
            "n_scored": len(scores),
            "n_attn": len(attn_scored),
            "attn_fail": fail,
            "attn_fail_frac": frac,
            "pass": frac <= args.max_attn_fail_frac,
        }

    # mean rating over passing raters, non-attn items with cosine
    passing = [r for r, rep in rater_reports.items() if rep["pass"]]
    xs, ys, ids = [], [], []
    for iid, it in items.items():
        if it.get("attention_check"):
            continue
        cos = it.get("style_cosine")
        if cos is None:
            continue
        vals = [by_rater[r][iid] for r in passing if iid in by_rater[r]]
        if not vals:
            continue
        xs.append(float(np.mean(vals)))
        ys.append(float(cos))
        ids.append(iid)

    # inter-rater on shared non-attn items
    inter = None
    if len(passing) >= 2:
        a, b = passing[0], passing[1]
        shared = [
            iid
            for iid, it in items.items()
            if not it.get("attention_check") and iid in by_rater[a] and iid in by_rater[b]
        ]
        if shared:
            inter = spearman(
                np.asarray([by_rater[a][i] for i in shared], dtype=np.float64),
                np.asarray([by_rater[b][i] for i in shared], dtype=np.float64),
            )

    pending = not xs
    summary = {
        "generated_at": utc_now(),
        "pending": pending,
        "n_items_correlated": len(xs),
        "spearman_human_vs_cosine": spearman(np.asarray(xs), np.asarray(ys)) if xs else None,
        "kendall_tau_human_vs_cosine": kendall_tau(np.asarray(xs), np.asarray(ys)) if xs else None,
        "inter_rater_spearman": inter,
        "raters": rater_reports,
        "passing_raters": passing,
        "note": "Fill human_ratings_template.csv then re-run; empty scores => pending=true",
    }
    out_p = OUT / "human_spearman_summary.json"
    out_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", out_p)


if __name__ == "__main__":
    main()
