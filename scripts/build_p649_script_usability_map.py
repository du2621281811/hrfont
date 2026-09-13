#!/usr/bin/env python3
"""Build overlap260 three-layer review → per-font script usability map for training.

Does not write the old A disk or L1 decisions.json.
Default scope is 原260 (overlap260), the only subset with L1–L3 complete.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REVIEW = ROOT / "data/p649_v2a_review"
LAYERS = ROOT / "data/p649_v2a_layers"
CHARSET_PATH = ROOT / "manifests/charset_cn2west_v2_planned.json"
OUT = LAYERS / "training_map"

BUCKETS = ("all_scripts", "no_bopomofo", "han_latin_digit", "exclude")
SCRIPT_ORDER = (
    "ascii_digits",
    "ascii_letters",
    "latin_ext_letters",
    "hiragana",
    "katakana",
    "bopomofo",
)
USABLE = {
    "all_scripts": list(SCRIPT_ORDER),
    "no_bopomofo": [s for s in SCRIPT_ORDER if s != "bopomofo"],
    "han_latin_digit": ["ascii_digits", "ascii_letters", "latin_ext_letters"],
    "exclude": [],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_of(store: dict, stem: str) -> str:
    rec = store.get(stem) or {}
    return rec.get("review_decision") or ""


def bucket_of(l1: str, l2: str, l3: str) -> str:
    if l1 == "pass":
        if l2 or l3:
            raise ValueError(f"L1 pass must not have L2/L3: {l2!r} {l3!r}")
        return "all_scripts"
    if l1 != "drop":
        raise ValueError(f"L1 incomplete: {l1!r}")
    if l2 == "pass":
        if l3:
            raise ValueError(f"L2 pass must not have L3: {l3!r}")
        return "no_bopomofo"
    if l2 != "drop":
        raise ValueError(f"L2 incomplete: {l2!r}")
    if l3 == "pass":
        return "han_latin_digit"
    if l3 != "drop":
        raise ValueError(f"L3 incomplete: {l3!r}")
    return "exclude"


def expand_chars(charset: dict) -> dict[str, list[str]]:
    out = {k: list(charset["target"][k]) for k in SCRIPT_ORDER}
    out["style_han"] = list(charset["style_han_338"])
    return out


def build() -> dict:
    charset = json.loads(CHARSET_PATH.read_text(encoding="utf-8"))
    chars = expand_chars(charset)
    rank = json.loads((REVIEW / "ink_ratio_rank.json").read_text(encoding="utf-8"))
    l1_path = REVIEW / "decisions.json"
    l2_path = LAYERS / "layer2.json"
    l3_path = LAYERS / "layer3.json"
    l1_doc = json.loads(l1_path.read_text(encoding="utf-8"))
    l2_doc = json.loads(l2_path.read_text(encoding="utf-8"))
    l3_doc = json.loads(l3_path.read_text(encoding="utf-8"))
    l1, l2, l3 = l1_doc["decisions"], l2_doc["decisions"], l3_doc["decisions"]

    overlap = [f for f in rank["fonts"] if f.get("subset") == "overlap260"]
    if len(overlap) != 260:
        raise SystemExit(f"overlap260 n={len(overlap)}, expected 260")

    fonts_out = []
    counts = Counter()
    split_counts: dict[str, Counter] = defaultdict(Counter)
    for font in sorted(overlap, key=lambda x: x["stem"]):
        stem = font["stem"]
        a, b, c = decision_of(l1, stem), decision_of(l2, stem), decision_of(l3, stem)
        bucket = bucket_of(a, b, c)
        usable = list(USABLE[bucket])
        skip = [s for s in SCRIPT_ORDER if s not in usable]
        target_chars = "".join("".join(chars[s]) for s in usable)
        n_target = len(target_chars)
        rec = {
            "stem": stem,
            "split": font["split"],
            "subset": font.get("subset"),
            "family_key": font.get("family_key") or "",
            "style_key": font.get("style_key") or "",
            "mean_ink_ratio": font.get("mean_ink_ratio"),
            "l1": a or None,
            "l2": b or None,
            "l3": c or None,
            "bucket": bucket,
            "usable_scripts": usable,
            "skip_scripts": skip,
            "use_style_han": bucket != "exclude",
            "use_as_train_target": bucket != "exclude" and font["split"] == "train",
            "use_as_eval_gt": bucket != "exclude" and font["split"] in {"val", "test"},
            "n_target_chars": n_target,
            "n_style_han": charset["style_n"] if bucket != "exclude" else 0,
        }
        fonts_out.append(rec)
        counts[bucket] += 1
        split_counts[font["split"]][bucket] += 1

    script_masks = {
        bucket: {
            "usable_scripts": list(USABLE[bucket]),
            "skip_scripts": [s for s in SCRIPT_ORDER if s not in USABLE[bucket]],
            "target_chars": "".join("".join(chars[s]) for s in USABLE[bucket]),
            "n_target_chars": sum(len(chars[s]) for s in USABLE[bucket]),
            "use_style_han": bucket != "exclude",
        }
        for bucket in BUCKETS
    }

    payload = {
        "schema_version": 1,
        "dataset": rank.get("dataset") or l1_doc.get("dataset"),
        "scope": "overlap260",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "l1_decisions": str(l1_path.relative_to(ROOT)),
            "l1_sha256": sha256(l1_path),
            "l1_updated_at": l1_doc.get("updated_at") or "",
            "l2_decisions": str(l2_path.relative_to(ROOT)),
            "l2_sha256": sha256(l2_path),
            "l2_updated_at": l2_doc.get("updated_at") or "",
            "l3_decisions": str(l3_path.relative_to(ROOT)),
            "l3_sha256": sha256(l3_path),
            "l3_updated_at": l3_doc.get("updated_at") or "",
            "rank": "data/p649_v2a_review/ink_ratio_rank.json",
            "charset": str(CHARSET_PATH.relative_to(ROOT)),
        },
        "funnel": {
            "overlap260": 260,
            "l1_pass_all_scripts": counts["all_scripts"],
            "l1_drop": 260 - counts["all_scripts"],
            "l2_pass_no_bopomofo": counts["no_bopomofo"],
            "l2_drop": counts["han_latin_digit"] + counts["exclude"],
            "l3_pass_han_latin_digit": counts["han_latin_digit"],
            "l3_drop_exclude": counts["exclude"],
        },
        "counts": dict(counts),
        "counts_by_split": {sp: dict(split_counts[sp]) for sp in ("train", "val", "test")},
        "bucket_meaning": {
            "all_scripts": "L1 pass. 各语种都能用。训练/评测可用全部 295 个 target + 338 style 汉字。",
            "no_bopomofo": "L2 pass. 只有注音不能用。跳过 37 个 bopomofo，保留拉丁/数字/平假/片假。",
            "han_latin_digit": "L3 pass. 注音+平假+片假不能用。只保留 89 个数字+拉丁+扩展拉丁。",
            "exclude": "L3 drop. 仍不可用。不作为 target GT，也不作为 donor。",
        },
        "script_masks": script_masks,
        "fonts": fonts_out,
        "notes": [
            "Only overlap260 is mapped. new389 is not L1–L3 complete; do not silently train on it.",
            "Keep existing train/val/test splits. Do not move fonts across splits.",
            "Sampler: skip a (font, char) pair if the char's script is in skip_scripts.",
            "Style references stay Chinese StyleImage (338). Exclude fonts drop style_han too.",
            "Do not point training at p649 until this map is the sampler filter.",
        ],
    }
    if counts["all_scripts"] + counts["no_bopomofo"] + counts["han_latin_digit"] + counts["exclude"] != 260:
        raise SystemExit("bucket counts do not sum to 260")
    return payload


def write_outputs(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "script_usability_map.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lookup = {f["stem"]: f["bucket"] for f in payload["fonts"]}
    (OUT / "font_to_bucket.json").write_text(
        json.dumps(lookup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (OUT / "script_usability_map.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "stem",
            "split",
            "family_key",
            "bucket",
            "l1",
            "l2",
            "l3",
            "usable_scripts",
            "skip_scripts",
            "n_target_chars",
            "use_style_han",
            "use_as_train_target",
            "use_as_eval_gt",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for f in payload["fonts"]:
            w.writerow(
                {
                    "stem": f["stem"],
                    "split": f["split"],
                    "family_key": f["family_key"],
                    "bucket": f["bucket"],
                    "l1": f["l1"] or "",
                    "l2": f["l2"] or "",
                    "l3": f["l3"] or "",
                    "usable_scripts": "|".join(f["usable_scripts"]),
                    "skip_scripts": "|".join(f["skip_scripts"]),
                    "n_target_chars": f["n_target_chars"],
                    "use_style_han": int(f["use_style_han"]),
                    "use_as_train_target": int(f["use_as_train_target"]),
                    "use_as_eval_gt": int(f["use_as_eval_gt"]),
                }
            )
    by_bucket = defaultdict(list)
    by_split_bucket = defaultdict(list)
    for f in payload["fonts"]:
        by_bucket[f["bucket"]].append(f["stem"])
        by_split_bucket[(f["split"], f["bucket"])].append(f["stem"])
    for bucket, stems in by_bucket.items():
        (OUT / f"stems_{bucket}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
    for (split, bucket), stems in sorted(by_split_bucket.items()):
        (OUT / f"stems_{split}_{bucket}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")

    (OUT / "README.md").write_text(
        "# 训练映射文件\n\n"
        "正文记录见上一级 [`../REVIEW_RECORD.md`](../REVIEW_RECORD.md)。\n\n"
        "- `script_usability_map.json` / `.csv` / `font_to_bucket.json`\n"
        "- `stems_<bucket>.txt` · `stems_<split>_<bucket>.txt`\n"
        "- 重建：`python3 scripts/build_p649_script_usability_map.py`\n",
        encoding="utf-8",
    )


def main() -> None:
    payload = build()
    write_outputs(payload)
    print(
        f"wrote {OUT} "
        f"all={payload['counts']['all_scripts']} "
        f"no_bpmf={payload['counts']['no_bopomofo']} "
        f"latin={payload['counts']['han_latin_digit']} "
        f"exclude={payload['counts']['exclude']}"
    )


if __name__ == "__main__":
    main()
