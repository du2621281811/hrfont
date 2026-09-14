#!/usr/bin/env python3
"""E12-b cross-script retrieval with φ cosine (teacher validation).

Gallery: Chinese ref8 mean-pool proto per test family.
Query: Latin glyphs from the same family; rank against all test protos.

    /root/miniforge3/envs/boogu/bin/python scripts/e12_eval_retrieval_cosine.py --device cuda:0
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
from data import GlyphDataset, as_chars, group_table, load_manifest, split_families  # noqa: E402
from models import load_phi_checkpoint  # noqa: E402

CACHE = ROOT / "artifacts/e12/cache_v0913_b"
PHI = ROOT / "runs/e12_phi_s2_b_s3407/best.pt"
OUT = ROOT / "reports/e12_paper"
REF8 = list("永和书风骨韵天地")
QUERY = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
SPLIT_RATIOS = (0.7, 0.15, 0.15)
SPLIT_SEED = 3407


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--phi", type=Path, default=PHI)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    manifest = load_manifest(CACHE)
    splits = split_families(manifest, SPLIT_RATIOS, SPLIT_SEED, by_group=True)
    test_families = sorted(splits["test"])
    print(f"[retrieval] load φ on {device}; n_test={len(test_families)}", flush=True)
    phi = load_phi_checkpoint(args.phi, 3, 512, str(device)).to(device).eval()

    ds = GlyphDataset(CACHE, test_families, as_chars(REF8) + as_chars(QUERY), 3)
    by = {(r["family"], r["char"]): i for i, r in enumerate(ds.records)}

    protos: dict[str, torch.Tensor] = {}
    usable = []
    for f in test_families:
        if not all((f, r) in by for r in REF8):
            continue
        imgs = torch.stack([ds[by[(f, r)]][0] for r in REF8]).to(device)
        z = F.normalize(phi(imgs).mean(0), dim=0)
        protos[f] = z
        usable.append(f)
    if len(usable) < 2:
        raise RuntimeError("need >=2 test families with full ref8")

    fam_list = usable
    proto_mat = torch.stack([protos[f] for f in fam_list])  # N,D
    rows = []
    for f in fam_list:
        qs = [c for c in QUERY if (f, c) in by]
        for ch in qs:
            z = F.normalize(phi(ds[by[(f, ch)]][0][None].to(device))[0], dim=0)
            sims = proto_mat @ z
            order = torch.argsort(sims, descending=True)
            rank = int((order == fam_list.index(f)).nonzero(as_tuple=True)[0]) + 1
            rows.append(
                {
                    "family": f,
                    "char": ch,
                    "sc_r": float(sims[fam_list.index(f)]),
                    "rank": rank,
                    "rank1": rank == 1,
                    "rank5": rank <= 5,
                    "mrr": 1.0 / rank,
                }
            )

    n = len(rows)
    summary = {
        "generated_at": utc_now(),
        "protocol": "cross-script retrieval: zh ref8 proto gallery vs latin queries; cosine; test families only",
        "scorer": {"phi": str(args.phi), "metric": "cosine"},
        "n_queries": n,
        "n_gallery_families": len(fam_list),
        "families": fam_list,
        "R@1": float(np.mean([r["rank1"] for r in rows])) if n else None,
        "R@5": float(np.mean([r["rank5"] for r in rows])) if n else None,
        "MRR": float(np.mean([r["mrr"] for r in rows])) if n else None,
        "mean_sc_r": float(np.mean([r["sc_r"] for r in rows])) if n else None,
    }
    (OUT / "retrieval_items.json").write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "retrieval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote", OUT / "retrieval_summary.json")


if __name__ == "__main__":
    main()
