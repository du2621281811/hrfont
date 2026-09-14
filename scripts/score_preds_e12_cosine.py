#!/usr/bin/env python3
"""Score f03 test16 preds with E12-b φ cosine Style Score (teacher metric).

    /root/miniforge3/envs/boogu/bin/python scripts/score_preds_e12_cosine.py --device cuda:0
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

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts" / "eval_framework"))
from models import load_phi_checkpoint  # noqa: E402

OUT_PRED = ROOT / "reports/f03_test16_strat"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
PAPER = ROOT / "reports/e12_paper"
PHI = ROOT / "runs/e12_phi_s2_b_s3407/best.pt"
FONTS_TSV = ROOT / "manifests/v0913_clean/fonts.tsv"
REF8 = list("永和书风骨韵天地")
SEED = 3407
DEFAULT_METHODS = ["F0_100k", "F0C_100000", "F0C_95000", "F2_40000", "F2C_40000", "F2_80000", "F2C_80000"]
LATIN_DIGIT = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"))


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
    b = script_bucket(ch)
    if usability == "exclude":
        return False
    if usability == "all_scripts":
        return True
    if usability == "no_bopomofo":
        return b != "bopomofo"
    if usability == "han_latin_digit":
        return b in ("digit", "latin_upper", "latin_lower", "latin_ext")
    return ch in LATIN_DIGIT


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
    args = ap.parse_args()

    PAPER.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    idx = json.loads((OUT_PRED / "browse_index.json").read_text(encoding="utf-8"))
    fonts, chars = list(idx["fonts"]), list(idx["chars"])
    usability = load_usability()

    print(f"[cosine] load φ on {device}", flush=True)
    phi = load_phi_checkpoint(args.phi, 3, 512, str(device)).to(device).eval()

    refs_by_font: dict[str, torch.Tensor] = {}
    for font in fonts:
        imgs = torch.stack([load_rgb01(style_path(font, ch)) for ch in REF8]).to(device)
        refs_by_font[font] = F.normalize(phi(imgs).mean(0), dim=0).cpu()

    wrong_refs: dict[str, torch.Tensor] = {}
    for i, font in enumerate(fonts):
        other = fonts[(i + 1) % len(fonts)]
        wrong_refs[font] = refs_by_font[other]

    rows: list[dict] = []
    targets = ["GT"] + methods

    def score_batch(qs, protos, metas):
        q = torch.stack(qs).to(device)
        zq = F.normalize(phi(q), dim=1)
        p = torch.stack(protos).to(device)
        cos = (zq * p).sum(1)
        for i, meta in enumerate(metas):
            c = float(cos[i])
            rows.append(
                {
                    **meta,
                    "style_cosine": c,
                    "style_score_01": (c + 1) / 2,
                }
            )

    for method in targets:
        print(f"[cosine] scoring {method}", flush=True)
        bq, bp, bm = [], [], []
        for font in fonts:
            u = usability.get(font, "all_scripts")
            proto = refs_by_font[font]
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
                bp.append(proto)
                bm.append(
                    {
                        "method": method,
                        "font": font,
                        "char": ch,
                        "cp": cp_of(ch),
                        "bucket": script_bucket(ch),
                        "in_latin_digit": ch in LATIN_DIGIT,
                        "refs": "same_family",
                    }
                )
                if len(bq) >= args.batch:
                    score_batch(bq, bp, bm)
                    bq, bp, bm = [], [], []
        if bq:
            score_batch(bq, bp, bm)

    print("[cosine] GT + wrong-family refs", flush=True)
    bq, bp, bm = [], [], []
    for font in fonts:
        u = usability.get(font, "all_scripts")
        proto = wrong_refs[font]
        for ch in chars:
            if ch not in LATIN_DIGIT or not char_allowed(u, ch):
                continue
            p = gt_path(font, ch)
            if p is None or not p.is_file():
                continue
            bq.append(load_rgb01(p))
            bp.append(proto)
            bm.append(
                {
                    "method": "GT_wrong_family",
                    "font": font,
                    "char": ch,
                    "cp": cp_of(ch),
                    "bucket": script_bucket(ch),
                    "in_latin_digit": True,
                    "refs": "cross_family",
                }
            )
            if len(bq) >= args.batch:
                score_batch(bq, bp, bm)
                bq, bp, bm = [], [], []
    if bq:
        score_batch(bq, bp, bm)

    by_m: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(r)

    def agg(rs: list[dict]) -> dict:
        primary = [x for x in rs if x["in_latin_digit"]]
        return {
            "all": {
                "style_cosine": mean_std([x["style_cosine"] for x in rs]),
                "style_score_01": mean_std([x["style_score_01"] for x in rs]),
            },
            "latin_digit": {
                "style_cosine": mean_std([x["style_cosine"] for x in primary]),
                "style_score_01": mean_std([x["style_score_01"] for x in primary]),
            },
        }

    method_order = [m for m in targets + ["GT_wrong_family"] if m in by_m]
    summary = {
        "generated_at": utc_now(),
        "scorer": {
            "phi": str(args.phi),
            "metric": "cosine",
            "style_score_01": "(cosine+1)/2",
            "protocol": "E12-b φ; query=pred|GT; z_zh=mean pool StyleImage ref8; primary=latin+digit",
        },
        "methods": {m: agg(by_m[m]) for m in method_order},
        "n_rows": len(rows),
    }

    # matched subset where clean preds exist
    need = ["GT", "F0_100k", "F0C_100000", "F2_80000", "F2C_80000", "GT_wrong_family"]
    idx_map: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        if not r["in_latin_digit"]:
            continue
        idx_map[(r["font"], r["char"])][r["method"]] = r["style_cosine"]
    matched = [(k, v) for k, v in idx_map.items() if all(m in v for m in need)]
    matched_means = {
        m: float(np.mean([v[m] for _, v in matched])) if matched else None for m in need
    }
    matched_out = {
        "generated_at": utc_now(),
        "n": len(matched),
        "means_style_cosine": matched_means,
        "means_style_score_01": {
            m: ((matched_means[m] + 1) / 2 if matched_means[m] is not None else None) for m in need
        },
        "note": "Cells where clean+dirty preds all exist",
    }

    items_p = PAPER / "scores_test16_cosine_items.json"
    sum_p = PAPER / "scores_test16_cosine_summary.json"
    mat_p = PAPER / "scores_test16_cosine_matched.json"
    items_p.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    sum_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mat_p.write_text(json.dumps(matched_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nmethod | n_ld | style_cosine↑ | style_score_01")
    for m in method_order:
        a = summary["methods"][m]["latin_digit"]["style_cosine"]
        b = summary["methods"][m]["latin_digit"]["style_score_01"]
        print(f"{m:18s} | {a['n']:4d} | {a['mean']} | {b['mean']}")
    print("matched", matched_out)
    print("wrote", sum_p, mat_p)


if __name__ == "__main__":
    main()
