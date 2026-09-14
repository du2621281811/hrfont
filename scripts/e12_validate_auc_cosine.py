#!/usr/bin/env python3
"""Same-font vs cross-font AUC using φ cosine only (teacher validation).

    /root/miniforge3/envs/boogu/bin/python scripts/e12_validate_auc_cosine.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts" / "eval_framework"))
from data import (  # noqa: E402
    GlyphDataset,
    as_chars,
    group_table,
    load_manifest,
    pick_negative,
    split_families,
)
from models import load_phi_checkpoint  # noqa: E402
from train_utils import binary_auc  # noqa: E402

CACHE = ROOT / "artifacts/e12/cache_v0913_b"
PHI = ROOT / "runs/e12_phi_s2_b_s3407/best.pt"
OUT = ROOT / "reports/e12_paper"
REF8 = list("永和书风骨韵天地")
QUERY = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
SPLIT_RATIOS = (0.7, 0.15, 0.15)
SPLIT_SEED = 3407


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-per-family", type=int, default=16)
    ap.add_argument("--phi", type=Path, default=PHI)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest = load_manifest(CACHE)
    splits = split_families(manifest, SPLIT_RATIOS, SPLIT_SEED, by_group=True)
    groups = group_table(manifest)
    test_families = list(splits["test"])
    all_families = sorted({x["family"] for x in manifest["fonts"]})

    print(f"[auc-cosine] load φ on {device}; test_families={len(test_families)}", flush=True)
    phi = load_phi_checkpoint(args.phi, 3, 512, str(device)).to(device).eval()
    ds = GlyphDataset(CACHE, all_families, as_chars(REF8) + as_chars(QUERY), 3)
    by = {(r["family"], r["char"]): i for i, r in enumerate(ds.records)}

    rows = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    for f in test_families:
        qs = [c for c in QUERY if (f, c) in by and all((f, r) in by for r in REF8)]
        if not qs:
            continue
        wrong = pick_negative(f, all_families, groups, cross_group=True)
        if wrong is None or not all((wrong, r) in by for r in REF8):
            wrong = pick_negative(
                f,
                [x for x in all_families if all((x, r) in by for r in REF8)],
                groups,
                cross_group=True,
            )
        if wrong is None:
            continue
        step = max(1, len(qs) // args.max_per_family)
        use_q = qs[::step][: args.max_per_family]

        refs_pos = torch.stack([ds[by[(f, r)]][0] for r in REF8]).to(device)
        proto = F.normalize(phi(refs_pos).mean(0), dim=0)

        for ch in use_q:
            if (f, ch) not in by or (wrong, ch) not in by:
                continue
            z_same = F.normalize(phi(ds[by[(f, ch)]][0][None].to(device))[0], dim=0)
            z_cross = F.normalize(phi(ds[by[(wrong, ch)]][0][None].to(device))[0], dim=0)
            # same: query from F_i vs proto F_i; cross: query from F_j vs proto F_i
            cos_same = float((proto * z_same).sum())
            cos_cross = float((proto * z_cross).sum())
            pos_scores.append(cos_same)
            neg_scores.append(cos_cross)
            rows.append(
                {
                    "family": f,
                    "wrong_family": wrong,
                    "char": ch,
                    "cos_same": cos_same,
                    "cos_cross": cos_cross,
                    "style_score_01_same": (cos_same + 1) / 2,
                    "style_score_01_cross": (cos_cross + 1) / 2,
                    "order_ok": cos_same > cos_cross,
                }
            )

    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores
    auc = float(binary_auc(labels, scores)) if pos_scores and neg_scores else float("nan")
    order_acc = float(np.mean([r["order_ok"] for r in rows])) if rows else float("nan")
    summary = {
        "generated_at": utc_now(),
        "protocol": "same-font zh-proto+en-query vs cross-font en-query; cosine only",
        "scorer": {"phi": str(args.phi), "metric": "cosine"},
        "n_pairs": len(rows),
        "n_families": len({r["family"] for r in rows}),
        "mean_cos_same": float(np.mean(pos_scores)) if pos_scores else None,
        "mean_cos_cross": float(np.mean(neg_scores)) if neg_scores else None,
        "gap_same_minus_cross": float(np.mean(pos_scores) - np.mean(neg_scores)) if pos_scores else None,
        "order_accuracy": order_acc,
        "auc_same_vs_cross": auc,
        "test_families": test_families,
    }
    (OUT / "auc_cosine_items.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "auc_cosine_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", OUT / "auc_cosine_summary.json")


if __name__ == "__main__":
    main()
