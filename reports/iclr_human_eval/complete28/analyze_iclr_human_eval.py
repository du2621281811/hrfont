#!/usr/bin/env python3
"""ICLR human-eval tables: mean rank, top-2, pairwise preference, Kendall W, Kendall tau-b.

Drops an annotator only when every attention check places the mismatch candidate in the top 2.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESP = ROOT / "responses.jsonl"
ITEMS = ROOT / "items.json"
METRICS = ROOT / "metric_scores.json"
OUT = ROOT / "stats.json"
METHODS = ["gt", "hrfont", "fcagan", "ftransgan", "garfont", "fontdiffuser"]
LABEL = {
    "gt": "Ground Truth",
    "hrfont": "HR-Font",
    "fcagan": "FCA-GAN",
    "ftransgan": "FTransGAN",
    "garfont": "GAR-Font",
    "fontdiffuser": "FontDiffuser",
}


def load_rows() -> list[dict]:
    if not RESP.is_file():
        return []
    rows = []
    for line in RESP.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def low_quality(rows: list[dict]) -> set[str]:
    by = defaultdict(list)
    for r in rows:
        if r.get("is_attention_check") and r.get("hidden_method") == "attn_mismatch":
            by[r["annotator_id"]].append(int(r["rank"]))
    drop = set()
    for aid, ranks in by.items():
        if ranks and all(rank <= 2 for rank in ranks):
            drop.add(aid)
    return drop


def kendall_w(rank_matrix: np.ndarray) -> float:
    """rank_matrix: raters x candidates, ranks start at 1."""
    m, n = rank_matrix.shape
    if m < 2 or n < 2:
        return float("nan")
    sums = rank_matrix.sum(axis=0)
    s = float(((sums - sums.mean()) ** 2).sum())
    return 12.0 * s / (m * m * (n ** 3 - n))


def tau_b(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    denom = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    if denom == 0:
        return float("nan")
    return (conc - disc) / denom


def ci(samples: list[float]) -> list[float | None]:
    arr = np.asarray([v for v in samples if v == v], dtype=float)
    if arr.size == 0:
        return [None, None]
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def method_stats(formal: list[dict]) -> dict:
    """formal rows are kept annotators, phase formal."""
    by_sample = defaultdict(lambda: defaultdict(list))
    for r in formal:
        by_sample[r["sample_id"]][r["annotator_id"]].append(r)
    mean_rank = {m: [] for m in METHODS}
    top2 = {m: [] for m in METHODS}
    pair = {m: [] for m in METHODS if m != "hrfont"}
    ws = []
    consensus = {}
    for sid, people in by_sample.items():
        if len(people) < 3:
            continue
        mat = []
        acc = {m: [] for m in METHODS}
        for _aid, rows in people.items():
            got = {r["hidden_method"]: int(r["rank"]) for r in rows}
            if not all(m in got for m in METHODS):
                continue
            mat.append([got[m] for m in METHODS])
            for m in METHODS:
                acc[m].append(got[m])
                mean_rank[m].append(got[m])
                top2[m].append(1.0 if got[m] <= 2 else 0.0)
            for m in pair:
                pair[m].append(1.0 if got["hrfont"] < got[m] else 0.0)
        if len(mat) >= 2:
            ws.append(kendall_w(np.asarray(mat, dtype=float)))
        if acc["hrfont"]:
            consensus[sid] = {m: float(np.mean(acc[m])) for m in METHODS}
    summary = {}
    for m in METHODS:
        summary[m] = {
            "mean_rank": float(np.mean(mean_rank[m])) if mean_rank[m] else None,
            "top2": float(np.mean(top2[m])) if top2[m] else None,
            "n": len(mean_rank[m]),
        }
    pref = {m: (float(np.mean(v)) if v else None) for m, v in pair.items()}
    return {
        "methods": summary,
        "pairwise_hr_better": pref,
        "kendall_w": float(np.mean(ws)) if ws else None,
        "consensus": consensus,
        "n_samples_consensus": len(consensus),
    }


def bootstrap(formal: list[dict], n_boot: int = 2000) -> dict:
    samples = sorted({r["sample_id"] for r in formal})
    if len(samples) < 2:
        return {}
    rng = random.Random(3407)
    buckets = {m: {"mean_rank": [], "top2": []} for m in METHODS}
    pref = {m: [] for m in METHODS if m != "hrfont"}
    for _ in range(n_boot):
        draw = [rng.choice(samples) for _ in samples]
        sub = [r for r in formal if r["sample_id"] in set(draw)]
        # weight by multiplicity
        weight = {s: draw.count(s) for s in draw}
        ranks = {m: [] for m in METHODS}
        tops = {m: [] for m in METHODS}
        pairs = {m: [] for m in METHODS if m != "hrfont"}
        grouped = defaultdict(list)
        for r in sub:
            grouped[(r["sample_id"], r["annotator_id"], r["hidden_method"])].append(r)
        seen = defaultdict(dict)
        for (sid, aid, method), rows in grouped.items():
            seen[(sid, aid)][method] = int(rows[-1]["rank"])
        for (sid, _aid), got in seen.items():
            if not all(m in got for m in METHODS):
                continue
            w = weight[sid]
            for m in METHODS:
                ranks[m].extend([got[m]] * w)
                tops[m].extend([1.0 if got[m] <= 2 else 0.0] * w)
            for m in pairs:
                pairs[m].extend([1.0 if got["hrfont"] < got[m] else 0.0] * w)
        for m in METHODS:
            if ranks[m]:
                buckets[m]["mean_rank"].append(float(np.mean(ranks[m])))
                buckets[m]["top2"].append(float(np.mean(tops[m])))
        for m in pairs:
            if pairs[m]:
                pref[m].append(float(np.mean(pairs[m])))
    return {
        "mean_rank_ci": {m: ci(buckets[m]["mean_rank"]) for m in METHODS},
        "top2_ci": {m: ci(buckets[m]["top2"]) for m in METHODS},
        "pairwise_ci": {m: ci(pref[m]) for m in pref},
    }


def metric_tau(consensus: dict, scores: dict) -> dict:
    out = {}
    names = scores.get("metrics") or []
    higher = set(scores.get("higher_better") or [])
    for metric in names:
        xs, ys = [], []
        for sid, human in consensus.items():
            block = (scores.get("samples") or {}).get(sid)
            if not block:
                continue
            for method, hrank in human.items():
                val = (block.get(method) or {}).get(metric)
                if val is None:
                    continue
                xs.append((-hrank))  # higher = better human
                ys.append(val if metric in higher else -val)
        tau = tau_b(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)) if xs else float("nan")
        out[metric] = None if tau != tau else float(tau)
    # bootstrap over samples
    rng = random.Random(3407)
    sids = [s for s in consensus if s in (scores.get("samples") or {})]
    boot = {m: [] for m in names}
    for _ in range(1000):
        if len(sids) < 2:
            break
        draw = [rng.choice(sids) for _ in sids]
        xs, ys = {m: [] for m in names}, {m: [] for m in names}
        for sid in draw:
            human = consensus[sid]
            block = scores["samples"][sid]
            for method, hrank in human.items():
                for metric in names:
                    val = (block.get(method) or {}).get(metric)
                    if val is None:
                        continue
                    xs[metric].append(-hrank)
                    ys[metric].append(val if metric in higher else -val)
        for metric in names:
            if len(xs[metric]) >= 3:
                boot[metric].append(tau_b(np.asarray(xs[metric]), np.asarray(ys[metric])))
    return {m: {"tau_b": out[m], "ci95": ci(boot[m])} for m in names}


def main() -> None:
    rows = load_rows()
    drop = low_quality(rows)
    kept = [r for r in rows if r["annotator_id"] not in drop and r.get("phase") == "formal"]
    stats = method_stats(kept)
    boot = bootstrap(kept)
    scores = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.is_file() else {"metrics": [], "samples": {}}
    tau = metric_tau(stats["consensus"], scores) if stats["consensus"] else {}
    doc = {
        "responses": len(rows),
        "annotators_dropped": sorted(drop),
        "annotators_kept": sorted({r["annotator_id"] for r in kept}),
        **stats,
        "bootstrap95": boot,
        "evaluator_tau": tau,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("dropped", sorted(drop) or "none")
    print("Kendall W", stats["kendall_w"])
    for m in METHODS:
        cell = stats["methods"][m]
        print(f"{LABEL[m]:16} rank {cell['mean_rank']} top2 {cell['top2']} n {cell['n']}")
    for m, v in stats["pairwise_hr_better"].items():
        print(f"HR > {LABEL[m]}", v)


if __name__ == "__main__":
    main()
