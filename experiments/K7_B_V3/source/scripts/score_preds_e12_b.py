#!/usr/bin/env python3
"""Score f03 test16 preds with E12-b (clean protocol).

Also writes meta-ranking summary: GT vs methods vs wrong-family style on GT query.

    python scripts/score_preds_e12_b.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval_framework"))
from models import MembershipVerifier, load_phi_checkpoint  # noqa: E402

OUT_PRED = ROOT / "reports/f03_test16_strat"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
PAPER = ROOT / "reports/e12_paper"
PHI = ROOT / "runs/e12_phi_s2_b_s3407/best.pt"
MEM = ROOT / "runs/e12_membership_b_s3407/best.pt"
FONTS_TSV = ROOT / "manifests/v0913_clean/fonts.tsv"
REF8 = list("永和书风骨韵天地")
SEED = 3407
DEFAULT_METHODS = ["F0_100k", "F0C_100000", "F0C_95000", "F2_40000", "F2C_40000", "F2_80000", "F2C_80000"]
MEM_QUERY_DOMAIN = set(
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
)


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


def load_usability() -> dict[str, str]:
    out = {}
    for line in FONTS_TSV.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] != "stem":
            out[parts[0]] = parts[2]
    return out


def char_allowed(usability: str, ch: str) -> bool:
    """Map fonts.tsv bucket → whether stratified char is in keep."""
    b = script_bucket(ch)
    if usability == "exclude":
        return False
    if usability == "all_scripts":
        return True
    if usability == "no_bopomofo":
        return b != "bopomofo"
    if usability == "han_latin_digit":
        return b in ("digit", "latin_upper", "latin_lower", "latin_ext")
    # unknown → allow latin+digit only (membership domain)
    return ch in MEM_QUERY_DOMAIN


def load_rgb01(path: Path, size: int = 96) -> torch.Tensor:
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None].repeat(3, 1, 1)


def style_path(font: str, ch: str) -> Path:
    return DATA / "test" / "StyleImage" / font / f"{font}+{cp_of(ch)}.png"


def gt_path(font: str, ch: str) -> Path | None:
    p = DATA / "test" / "TargetImage" / font / f"{font}+{cp_of(ch)}.png"
    return p if p.is_file() else None


def pred_path(method: str, font: str, ch: str) -> Path:
    return OUT_PRED / "preds" / method / "test" / font / f"test__{font}__{cp_of(ch)}__s{SEED}.png"


def mean_std(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=0))}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--phi", type=Path, default=PHI)
    ap.add_argument("--mem", type=Path, default=MEM)
    args = ap.parse_args()

    PAPER.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    idx = json.loads((OUT_PRED / "browse_index.json").read_text(encoding="utf-8"))
    fonts, chars = list(idx["fonts"]), list(idx["chars"])
    usability = load_usability()

    print(f"[e12-b] load on {device}", flush=True)
    phi = load_phi_checkpoint(args.phi, 3, 512, str(device)).to(device).eval()
    payload = torch.load(args.mem, map_location=device, weights_only=False)
    mem = MembershipVerifier(phi, 512, 256, freeze_encoder=True).to(device)
    mem.load_state_dict(payload["model"])
    mem.eval()
    temp = float(payload.get("temperature", 1.0))

    refs_by_font: dict[str, torch.Tensor] = {}
    for font in fonts:
        imgs = [load_rgb01(style_path(font, ch)) for ch in REF8]
        refs_by_font[font] = torch.stack(imgs)

    # wrong-family refs: next font in list (cycle)
    wrong_refs: dict[str, torch.Tensor] = {}
    for i, font in enumerate(fonts):
        other = fonts[(i + 1) % len(fonts)]
        wrong_refs[font] = refs_by_font[other]

    rows: list[dict] = []
    targets = ["GT"] + methods

    def score_batch(qs, refs_list, metas):
        q = torch.stack(qs).to(device)
        r = torch.stack(refs_list).to(device)
        zq = phi(q)
        zr = phi(r.reshape(-1, 3, 96, 96)).reshape(q.size(0), 8, -1)
        proto = F.normalize(zr.mean(1), dim=1)
        sc_r = (zq * proto).sum(1)
        logits = mem(q, r)
        probs = torch.sigmoid(logits / temp)
        for i, meta in enumerate(metas):
            rows.append(
                {
                    **meta,
                    "e12_phi_sc_r": float(sc_r[i]),
                    "e12_mem_logit": float(logits[i]),
                    "e12_mem_prob": float(probs[i]),
                    "temperature": temp,
                }
            )

    for method in targets:
        print(f"[e12-b] scoring {method}", flush=True)
        bq, br, bm = [], [], []
        for font in fonts:
            u = usability.get(font, "all_scripts")
            refs = refs_by_font[font]
            for ch in chars:
                if not char_allowed(u, ch):
                    continue
                if method == "GT":
                    p = gt_path(font, ch)
                else:
                    p = pred_path(method, font, ch)
                if p is None or not p.is_file():
                    continue
                bq.append(load_rgb01(p))
                br.append(refs)
                bm.append(
                    {
                        "method": method,
                        "font": font,
                        "char": ch,
                        "cp": cp_of(ch),
                        "bucket": script_bucket(ch),
                        "keep": True,
                        "in_mem_domain": ch in MEM_QUERY_DOMAIN,
                        "refs": "same_family",
                    }
                )
                if len(bq) >= args.batch:
                    score_batch(bq, br, bm)
                    bq, br, bm = [], [], []
        if bq:
            score_batch(bq, br, bm)

    # Wrong-family control on GT queries (meta / construct on test16)
    print("[e12-b] scoring GT + wrong-family refs", flush=True)
    bq, br, bm = [], [], []
    for font in fonts:
        u = usability.get(font, "all_scripts")
        refs = wrong_refs[font]
        for ch in chars:
            if ch not in MEM_QUERY_DOMAIN:
                continue
            if not char_allowed(u, ch):
                continue
            p = gt_path(font, ch)
            if p is None or not p.is_file():
                continue
            bq.append(load_rgb01(p))
            br.append(refs)
            bm.append(
                {
                    "method": "GT_wrong_family",
                    "font": font,
                    "char": ch,
                    "cp": cp_of(ch),
                    "bucket": script_bucket(ch),
                    "keep": True,
                    "in_mem_domain": True,
                    "refs": "cross_family",
                }
            )
            if len(bq) >= args.batch:
                score_batch(bq, br, bm)
                bq, br, bm = [], [], []
    if bq:
        score_batch(bq, br, bm)

    by_m: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(r)

    def agg(rs: list[dict]) -> dict:
        primary = [x for x in rs if x["in_mem_domain"]]
        return {
            "all": {
                "phi_sc_r": mean_std([x["e12_phi_sc_r"] for x in rs]),
                "mem_prob": mean_std([x["e12_mem_prob"] for x in rs]),
            },
            "latin_digit": {
                "phi_sc_r": mean_std([x["e12_phi_sc_r"] for x in primary]),
                "mem_prob": mean_std([x["e12_mem_prob"] for x in primary]),
            },
        }

    method_order = [m for m in targets + ["GT_wrong_family"] if m in by_m]
    summary = {
        "generated_at": utc_now(),
        "scorer": {
            "phi": str(args.phi),
            "membership": str(args.mem),
            "temperature": temp,
            "protocol": "E12-b / v0913_clean; query=pred|GT; refs=font StyleImage ref8; keep via fonts.tsv usability",
            "primary_subset": "latin+digit",
        },
        "methods": {m: agg(by_m[m]) for m in method_order},
        "n_rows": len(rows),
    }

    # Meta-ranking on latin+digit keep cells where all compared methods exist
    rank_methods = ["GT", "F2C_80000", "F2_80000", "F0C_100000", "F0_100k", "GT_wrong_family"]
    rank_methods = [m for m in rank_methods if m in by_m]
    # index by (font,char)
    idx_map: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        if not r["in_mem_domain"]:
            continue
        idx_map[(r["font"], r["char"])][r["method"]] = r["e12_mem_prob"]

    pairs = [
        ("GT", "F2C_80000"),
        ("GT", "F2_80000"),
        ("GT", "F0_100k"),
        ("F2C_80000", "F0_100k"),
        ("F2_80000", "F0_100k"),
        ("GT", "GT_wrong_family"),
        ("F2C_80000", "GT_wrong_family"),
        ("F0_100k", "GT_wrong_family"),
    ]
    pair_acc = {}
    for a, b in pairs:
        if a not in rank_methods or b not in rank_methods:
            continue
        ok = tot = 0
        for key, mp in idx_map.items():
            if a in mp and b in mp:
                tot += 1
                if mp[a] >= mp[b]:
                    ok += 1
        pair_acc[f"{a}>={b}"] = {"n": tot, "acc": (ok / tot) if tot else None, "ok": ok}

    meta = {
        "generated_at": utc_now(),
        "expected": "GT >= gens >= GT_wrong_family (family compatibility)",
        "pairwise_order_accuracy": pair_acc,
        "method_means_latin_digit": {
            m: summary["methods"][m]["latin_digit"]["mem_prob"] for m in method_order
        },
    }

    items_p = PAPER / "scores_test16_items.json"
    sum_p = PAPER / "scores_test16_summary.json"
    meta_p = PAPER / "meta_ranking.json"
    items_p.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    sum_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nmethod | n_ld | mem_prob(latin+digit)↑")
    for m in method_order:
        a = summary["methods"][m]["latin_digit"]["mem_prob"]
        print(f"{m:18s} | {a['n']:4d} | {a['mean']}")
    print("\nmeta pairwise:")
    for k, v in pair_acc.items():
        print(f"  {k}: acc={v['acc']} n={v['n']}")
    print("wrote", sum_p, meta_p)


if __name__ == "__main__":
    main()
