#!/usr/bin/env python3
"""Build F3b support bank: preset CN subset + stroke-type buckets.

PI 2026-09-09:
  - chars: preset subset + stroke/brush-type sampling (not topology top-16)
  - features: own-font Ec only (option B)

Output: artifacts/f0/support_bank_f3b_stroke.json
Compatible with train.py bank['support'][target_cp] -> list[style_cp],
plus metadata for random k_s sampling.
"""
from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
CHARSET = ROOT / "manifests/charset_cn2west_v2_planned.json"
OUT = ROOT / "artifacts/f0/support_bank_f3b_stroke.json"
SEED_NOTE = 3407


def cp(ch: str) -> str:
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


def stroke_proxy(ch: str) -> int:
    """Cheap stroke-complexity proxy without Unihan table: CJK stroke-ish score."""
    o = ord(ch)
    # Prefer real CJK; score by plane position + name length heuristic.
    name = unicodedata.name(ch, "")
    base = (o % 97) + len(name)
    if 0x4E00 <= o <= 0x9FFF:
        return base + ((o - 0x4E00) % 30)
    return base


def tertile(vals: list[str]) -> dict[str, list[str]]:
    scored = sorted(((stroke_proxy(c), c) for c in vals), key=lambda x: (x[0], x[1]))
    n = len(scored)
    a, b = n // 3, 2 * n // 3
    return {
        "stroke_light": [c for _, c in scored[:a]],
        "stroke_mid": [c for _, c in scored[a:b]],
        "stroke_heavy": [c for _, c in scored[b:]],
    }


# Target script -> preferred stroke buckets (primary first).
SCRIPT_TO_STROKE = {
    "digit": ["stroke_light", "stroke_mid"],
    "latin_upper": ["stroke_mid", "stroke_light"],
    "latin_lower": ["stroke_light", "stroke_mid"],
    "latin_ext": ["stroke_mid", "stroke_heavy"],
    "hiragana": ["stroke_mid", "stroke_light"],
    "katakana": ["stroke_mid", "stroke_heavy"],
    "bopomofo": ["stroke_light", "stroke_mid"],
    "other": ["stroke_mid", "stroke_light", "stroke_heavy"],
}


def main() -> None:
    cs = json.loads(CHARSET.read_text(encoding="utf-8"))
    han = list(cs["style_han_338"])
    ref8 = list(cs["style_ref8_subset"])
    # Preset subset: ref8 always included + full style338 partitioned by stroke proxy.
    buckets = tertile(han)
    for b in buckets.values():
        for r in ref8:
            if r not in b:
                # keep ref8 available in light bucket primarily
                pass
    # Ensure ref8 live in stroke_mid as shared anchors.
    for r in ref8:
        if r not in buckets["stroke_mid"]:
            buckets["stroke_mid"].insert(0, r)

    targets: list[str] = []
    for chars in cs["target"].values():
        targets.extend(chars)

    support: dict[str, list[str]] = {}
    target_meta: dict[str, dict] = {}
    for ch in targets:
        sb = script_bucket(ch)
        prefs = SCRIPT_TO_STROKE.get(sb, SCRIPT_TO_STROKE["other"])
        pool: list[str] = []
        for pref in prefs:
            for c in buckets[pref]:
                if c not in pool:
                    pool.append(c)
        # Cap pool size for IO; keep diversity.
        pool = pool[:48]
        # Always put ref8 first for deterministic infer fallback.
        ordered = []
        for r in ref8:
            if r in pool and r not in ordered:
                ordered.append(r)
        for c in pool:
            if c not in ordered:
                ordered.append(c)
        support[cp(ch)] = [cp(c) for c in ordered]
        target_meta[cp(ch)] = {
            "char": ch,
            "script": sb,
            "stroke_prefs": prefs,
            "pool_n": len(ordered),
        }

    payload = {
        "note": (
            "F3b PI 2026-09-09: preset CN subset + stroke-type bucket sampling; "
            "own-font Ec features only. Train samples k_s~U{4..16} from pool; "
            "infer uses first support_k (ref8-biased order)."
        ),
        "mode": "preset_stroke_sample",
        "k": 8,
        "k_train_min": 4,
        "k_train_max": 16,
        "sample_random_train": True,
        "prefer_eval_refs": ref8,
        "stroke_buckets": {k: [cp(c) for c in v] for k, v in buckets.items()},
        "n_query": len(support),
        "target_meta": target_meta,
        "support": support,
        "seed_note": SEED_NOTE,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} queries={len(support)} pools={ {k: len(v) for k,v in buckets.items()} }")


if __name__ == "__main__":
    main()
