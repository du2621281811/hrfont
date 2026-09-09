#!/usr/bin/env python3
"""Score f03_test16_strat preds with E12 v5.1 (φ SC-R + membership P(same|ref8)).

test16 fonts are outside the E12 train228 pool → held-out fonts for the scorer.
Membership head was trained on Latin+digit queries; kana/bopomofo/ext are reported
separately as extrapolation.

    python scripts/score_f03_e12_v51.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import math
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
from models import MembershipVerifier, load_phi_checkpoint  # noqa: E402

OUT = ROOT / "reports/f03_test16_strat"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
PHI = ROOT / "runs/e12_phi_s2_v51_train228_s3407/best.pt"
MEM = ROOT / "runs/e12_membership_v51_train228_s3407/best.pt"
REF8 = list("永和书风骨韵天地")
SEED = 3407
METHODS = ["P1", "E1_100k", "F0_100k", "F2_75000", "F3_80k"]
# Membership training query domain (primary claim subset)
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


def load_rgb01(path: Path, size: int = 96) -> torch.Tensor:
    """Match GlyphDataset: L/255 → repeat 3ch, (C,H,W)."""
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None].repeat(3, 1, 1)
    return t


def style_path(font: str, ch: str) -> Path:
    return DATA / "test" / "StyleImage" / font / f"{font}+{cp_of(ch)}.png"


def gt_path(font: str, ch: str) -> Path | None:
    p = DATA / "test" / "TargetImage" / font / f"{font}+{cp_of(ch)}.png"
    return p if p.is_file() else None


def pred_path(method: str, font: str, ch: str) -> Path:
    return OUT / "preds" / method / "test" / font / f"test__{font}__{cp_of(ch)}__s{SEED}.png"


def fonts_and_chars() -> tuple[list[str], list[str]]:
    idx = json.loads((OUT / "browse_index.json").read_text(encoding="utf-8"))
    return list(idx["fonts"]), list(idx["chars"])


def mean_std(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=0))}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = torch.device(args.device)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    fonts, chars = fonts_and_chars()

    print(f"[e12] load phi+membership on {device}", flush=True)
    phi = load_phi_checkpoint(PHI, 3, 512, str(device)).to(device).eval()
    payload = torch.load(MEM, map_location=device, weights_only=False)
    mem = MembershipVerifier(phi, 512, 256, freeze_encoder=True).to(device)
    mem.load_state_dict(payload["model"])
    mem.eval()
    mem.encoder.eval()
    temp = float(payload.get("temperature", 1.0))

    # Preload ref8 per font
    refs_by_font: dict[str, torch.Tensor] = {}
    for font in fonts:
        imgs = []
        for ch in REF8:
            p = style_path(font, ch)
            if not p.is_file():
                raise FileNotFoundError(p)
            imgs.append(load_rgb01(p))
        refs_by_font[font] = torch.stack(imgs)  # (8,3,96,96)

    rows: list[dict] = []
    targets = methods + ["GT"]
    for method in targets:
        print(f"[e12] scoring {method} …", flush=True)
        batch_q: list[torch.Tensor] = []
        batch_meta: list[tuple[str, str, str]] = []
        batch_refs: list[torch.Tensor] = []

        def flush():
            nonlocal batch_q, batch_meta, batch_refs
            if not batch_q:
                return
            q = torch.stack(batch_q).to(device)
            r = torch.stack(batch_refs).to(device)  # (B,8,3,H,W)
            zq = phi(q)
            zr = phi(r.reshape(-1, 3, 96, 96)).reshape(q.size(0), 8, -1)
            proto = F.normalize(zr.mean(1), dim=1)
            sc_r = (zq * proto).sum(1)
            logits = mem(q, r)
            probs = torch.sigmoid(logits / temp)
            for i, (font, ch, mid) in enumerate(batch_meta):
                rows.append(
                    {
                        "method": mid,
                        "font": font,
                        "char": ch,
                        "cp": cp_of(ch),
                        "bucket": script_bucket(ch),
                        "in_mem_domain": ch in MEM_QUERY_DOMAIN,
                        "e12_phi_sc_r": float(sc_r[i]),
                        "e12_mem_logit": float(logits[i]),
                        "e12_mem_prob": float(probs[i]),
                        "temperature": temp,
                    }
                )
            batch_q, batch_meta, batch_refs = [], [], []

        for font in fonts:
            refs = refs_by_font[font]
            for ch in chars:
                if method == "GT":
                    p = gt_path(font, ch)
                else:
                    p = pred_path(method, font, ch)
                if p is None or not p.is_file():
                    continue
                batch_q.append(load_rgb01(p))
                batch_refs.append(refs)
                batch_meta.append((font, ch, method))
                if len(batch_q) >= args.batch:
                    flush()
        flush()

    # Aggregate
    by_m: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(r)

    def agg(rs: list[dict]) -> dict:
        out = {
            "phi_sc_r": mean_std([x["e12_phi_sc_r"] for x in rs]),
            "mem_prob": mean_std([x["e12_mem_prob"] for x in rs]),
            "mem_logit": mean_std([x["e12_mem_logit"] for x in rs]),
        }
        primary = [x for x in rs if x["in_mem_domain"]]
        out["mem_prob_latin_digit"] = mean_std([x["e12_mem_prob"] for x in primary])
        out["phi_sc_r_latin_digit"] = mean_std([x["e12_phi_sc_r"] for x in primary])
        by_b = defaultdict(list)
        for x in rs:
            by_b[x["bucket"]].append(x)
        out["by_bucket"] = {
            b: {
                "phi_sc_r": mean_std([x["e12_phi_sc_r"] for x in xs]),
                "mem_prob": mean_std([x["e12_mem_prob"] for x in xs]),
            }
            for b, xs in sorted(by_b.items())
        }
        return out

    summary = {
        "generated_at": utc_now(),
        "scorer": {
            "phi": str(PHI),
            "membership": str(MEM),
            "temperature": temp,
            "ref8": REF8,
            "protocol": "query=pred|GT, refs=target-font StyleImage ref8; held-out test16 fonts vs E12 train228",
            "primary_subset": "latin+digit (membership train query domain)",
            "caveat": "Not weighted into a single score with L1/SSIM. Kana/bopomofo/ext are OOD for membership head.",
        },
        "methods": {m: agg(by_m[m]) for m in targets if m in by_m},
        "n_rows": len(rows),
    }

    items_path = OUT / "e12_v51_items.json"
    summary_path = OUT / "e12_v51_scores.json"
    items_path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Merge into metrics_summary.json
    mp = OUT / "metrics_summary.json"
    if mp.is_file():
        metrics = json.loads(mp.read_text(encoding="utf-8"))
    else:
        metrics = {}
    metrics["e12_v51"] = summary
    metrics["caveat"] = (
        (metrics.get("caveat") or "")
        + " E12 v5.1 style scores in e12_v51 (phi SC-R + membership prob); diagnostic pixels unchanged."
    ).strip()
    mp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Compact table for stdout
    print("\nmethod | n | phi_sc_r↑ | mem_prob↑ | mem_prob(latin+digit)↑")
    for m in targets:
        if m not in summary["methods"]:
            continue
        a = summary["methods"][m]
        print(
            f"{m:12s} | {a['phi_sc_r']['n']:4d} | "
            f"{a['phi_sc_r']['mean']:.4f} | {a['mem_prob']['mean']:.4f} | "
            f"{a['mem_prob_latin_digit']['mean']:.4f}"
        )
    print("wrote", summary_path, items_path)


if __name__ == "__main__":
    main()
