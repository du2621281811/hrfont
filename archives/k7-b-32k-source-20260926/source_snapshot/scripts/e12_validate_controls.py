#!/usr/bin/env python3
"""E12-b validation: same vs cross-family controls + optional meta helpers.

Uses v0913_clean cache + membership test families (same split as training).

    python scripts/e12_validate_controls.py --device cuda:0
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
from models import MembershipVerifier, load_phi_checkpoint  # noqa: E402
from train_utils import binary_auc  # noqa: E402

CACHE = ROOT / "artifacts/e12/cache_v0913_b"
PHI = ROOT / "runs/e12_phi_s2_b_s3407/best.pt"
MEM = ROOT / "runs/e12_membership_b_s3407/best.pt"
OUT = ROOT / "reports/e12_paper"
REF8 = list("永和书风骨韵天地")
# Match membership training query domain (primary)
QUERY = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
SPLIT_RATIOS = (0.7, 0.15, 0.15)
SPLIT_SEED = 3407


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-per-family", type=int, default=16, help="query chars per family (subsample)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest = load_manifest(CACHE)
    splits = split_families(manifest, SPLIT_RATIOS, SPLIT_SEED, by_group=True)
    groups = group_table(manifest)
    test_families = list(splits["test"])
    all_families = sorted({x["family"] for x in manifest["fonts"]})

    print(f"[controls] load phi+mem on {device}; test_families={len(test_families)}", flush=True)
    phi = load_phi_checkpoint(PHI, 3, 512, str(device)).to(device).eval()
    payload = torch.load(MEM, map_location=device, weights_only=False)
    mem = MembershipVerifier(phi, 512, 256, freeze_encoder=True).to(device)
    mem.load_state_dict(payload["model"])
    mem.eval()
    temp = float(payload.get("temperature", 1.0))

    ds = GlyphDataset(CACHE, all_families, as_chars(REF8) + as_chars(QUERY), 3)
    by = {(r["family"], r["char"]): i for i, r in enumerate(ds.records)}

    # Available query chars per family
    fam_queries: dict[str, list[str]] = {}
    for f in test_families:
        qs = [c for c in QUERY if (f, c) in by and all((f, r) in by for r in REF8)]
        fam_queries[f] = qs

    rows = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    for f in test_families:
        qs = fam_queries[f]
        if not qs:
            continue
        wrong = pick_negative(f, all_families, groups, cross_group=True)
        if wrong is None:
            continue
        # refs must exist for wrong family
        if not all((wrong, r) in by for r in REF8):
            wrong = pick_negative(f, [x for x in all_families if all((x, r) in by for r in REF8)], groups, cross_group=True)
            if wrong is None:
                continue
        # subsample queries
        step = max(1, len(qs) // args.max_per_family)
        use_q = qs[::step][: args.max_per_family]

        refs_pos = torch.stack([ds[by[(f, r)]][0] for r in REF8])  # 8,3,H,W
        refs_neg = torch.stack([ds[by[(wrong, r)]][0] for r in REF8])

        for ch in use_q:
            if (f, ch) not in by:
                continue
            q = ds[by[(f, ch)]][0]
            qb = q.unsqueeze(0).to(device)
            rp = refs_pos.unsqueeze(0).to(device)
            rn = refs_neg.unsqueeze(0).to(device)
            logit_p = float(mem(qb, rp)[0])
            logit_n = float(mem(qb, rn)[0])
            prob_p = float(torch.sigmoid(torch.tensor(logit_p / temp)))
            prob_n = float(torch.sigmoid(torch.tensor(logit_n / temp)))
            pos_scores.append(prob_p)
            neg_scores.append(prob_n)
            rows.append(
                {
                    "family": f,
                    "wrong_family": wrong,
                    "char": ch,
                    "prob_same": prob_p,
                    "prob_cross": prob_n,
                    "logit_same": logit_p,
                    "logit_cross": logit_n,
                    "order_ok": prob_p > prob_n,
                }
            )

    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    scores = pos_scores + neg_scores
    auc = float(binary_auc(labels, scores)) if pos_scores and neg_scores else float("nan")
    order_acc = float(np.mean([r["order_ok"] for r in rows])) if rows else float("nan")
    summary = {
        "generated_at": utc_now(),
        "protocol": "same-family GT query+refs vs same query + cross-family refs",
        "scorer": {"phi": str(PHI), "membership": str(MEM), "temperature": temp},
        "n_pairs": len(rows),
        "n_families": len({r["family"] for r in rows}),
        "mean_prob_same": float(np.mean(pos_scores)) if pos_scores else None,
        "mean_prob_cross": float(np.mean(neg_scores)) if neg_scores else None,
        "gap_same_minus_cross": float(np.mean(pos_scores) - np.mean(neg_scores)) if pos_scores else None,
        "order_accuracy": order_acc,
        "auc_same_vs_cross": auc,
        "test_families": test_families,
    }
    (OUT / "controls_same_vs_cross_items.json").write_text(
        json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "controls_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", OUT / "controls_summary.json")


if __name__ == "__main__":
    main()
