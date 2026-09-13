#!/usr/bin/env python3
"""Freeze v0913_clean pair maps from the dirty protocol-A tree.

Other machines only need this map + the existing PNG/cache. Do not re-render.
Rebuild is deterministic: same sources → same TSV bytes → same sha256.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
CHARSET_PATH = ROOT / "manifests/charset_cn2west_v2_planned.json"
SPLIT_PATH = ROOT / "manifests/split_v3_228_16_16.json"
USABILITY_PATH = ROOT / "data/p649_v2a_layers/training_map/script_usability_map.json"
BUCKET_PATH = ROOT / "data/p649_v2a_layers/training_map/font_to_bucket.json"
CONTRACT_PATH = ROOT / "manifests/v0913_clean.json"
OUT = ROOT / "manifests/v0913_clean"

GROUP_P = {"latin": 0.50, "kana": 0.38, "bopomofo": 0.12}
SCRIPT_TO_GROUP = {
    "ascii_digits": "latin",
    "ascii_letters": "latin",
    "latin_ext_letters": "latin",
    "hiragana": "kana",
    "katakana": "kana",
    "bopomofo": "bopomofo",
}
EVAL47 = "0123456789AGMQRWBCOaodpqbegcilnuàéüāěあかさんアカンㄅㄆㄚ"
PAIR_FIELDS = ("split", "font", "char", "cp", "script", "script_group", "bucket")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def expand_charset(charset: dict) -> dict[str, list[str]]:
    return {script: list(chars) for script, chars in charset["target"].items()}


def write_tsv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row[k]) for k in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pairs(fonts: list[dict], chars_by_script: dict[str, list[str]]) -> list[dict]:
    rows = []
    for rec in fonts:
        if rec["bucket"] == "exclude":
            continue
        usable = set(rec["usable_scripts"])
        for script, chars in chars_by_script.items():
            if script not in usable:
                continue
            group = SCRIPT_TO_GROUP[script]
            for ch in chars:
                rows.append({
                    "split": rec["split"],
                    "font": rec["stem"],
                    "char": ch,
                    "cp": cp_of(ch),
                    "script": script,
                    "script_group": group,
                    "bucket": rec["bucket"],
                })
    rows.sort(key=lambda r: (r["split"], r["font"], r["cp"], r["script"]))
    return rows


def build() -> dict:
    charset = json.loads(CHARSET_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    usability = json.loads(USABILITY_PATH.read_text(encoding="utf-8"))
    buckets = json.loads(BUCKET_PATH.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    chars_by_script = expand_charset(charset)
    if list(EVAL47) != list(contract["eval"]["stratified_47"]["chars"]):
        raise SystemExit("EVAL47 drifted from manifests/v0913_clean.json")
    if contract["train_sample"]["group_p"] != GROUP_P:
        raise SystemExit("GROUP_P drifted from manifests/v0913_clean.json")

    fonts = sorted(usability["fonts"], key=lambda r: (r["split"], r["stem"]))
    for rec in fonts:
        if buckets[rec["stem"]] != rec["bucket"]:
            raise SystemExit(f"bucket mismatch {rec['stem']}")
        expected_split_pool = set(split["stems"][rec["split"]])
        if rec["stem"] not in expected_split_pool:
            raise SystemExit(f"{rec['stem']} not in split {rec['split']}")

    pairs = build_pairs(fonts, chars_by_script)
    by_split: dict[str, list[dict]] = defaultdict(list)
    by_split_group: dict[str, Counter] = defaultdict(Counter)
    for row in pairs:
        by_split[row["split"]].append(row)
        by_split_group[row["split"]][row["script_group"]] += 1

    n_train = {g: by_split_group["train"][g] for g in GROUP_P}
    if sum(n_train.values()) != len(by_split["train"]):
        raise SystemExit("train group counts do not cover train pairs")
    pair_weight = {g: GROUP_P[g] / n_train[g] for g in GROUP_P}

    fonts_rows = []
    donor = {g: [] for g in GROUP_P}
    for rec in fonts:
        fonts_rows.append({
            "stem": rec["stem"],
            "split": rec["split"],
            "bucket": rec["bucket"],
            "usable": 0 if rec["bucket"] == "exclude" else 1,
        })
        if rec["split"] == "train" and rec["bucket"] != "exclude":
            usable_groups = {SCRIPT_TO_GROUP[s] for s in rec["usable_scripts"]}
            for group in GROUP_P:
                if group in usable_groups:
                    donor[group].append(rec["stem"])
    for group in donor:
        donor[group] = sorted(donor[group])

    test_keep = {(r["font"], r["cp"]) for r in by_split["test"]}
    eval_rows = []
    for font in split["stems"]["test"]:
        bucket = buckets[font]
        for ch in EVAL47:
            cp = cp_of(ch)
            keep = 1 if (font, cp) in test_keep else 0
            script = next(s for s, chars in chars_by_script.items() if ch in chars)
            eval_rows.append({
                "split": "test",
                "font": font,
                "char": ch,
                "cp": cp,
                "script": script,
                "script_group": SCRIPT_TO_GROUP[script],
                "bucket": bucket,
                "keep": keep,
            })
    if sum(r["keep"] for r in eval_rows) != 704 or len(eval_rows) != 752:
        raise SystemExit(
            f"eval47 keep={sum(r['keep'] for r in eval_rows)} n={len(eval_rows)}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    write_tsv(OUT / "pairs_train.tsv", by_split["train"], PAIR_FIELDS)
    write_tsv(OUT / "pairs_val.tsv", by_split["val"], PAIR_FIELDS)
    write_tsv(OUT / "pairs_test.tsv", by_split["test"], PAIR_FIELDS)
    write_tsv(OUT / "fonts.tsv", fonts_rows, ("stem", "split", "bucket", "usable"))
    write_tsv(
        OUT / "pairs_eval47.tsv",
        eval_rows,
        PAIR_FIELDS + ("keep",),
    )
    (OUT / "donor_train.json").write_text(
        json.dumps(donor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "sample_weights.json").write_text(
        json.dumps(
            {
                "algo": "two_stage_group_then_uniform_pair",
                "seed": 3407,
                "cn2cn_p": 0.0,
                "group_p": GROUP_P,
                "n_train": n_train,
                "pair_weight": pair_weight,
                "note": (
                    "P(pair) = group_p[g] / n_train[g]. "
                    "Materialize-only (no weights) is uniform-over-pairs and is NOT this recipe."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "pairs_train.tsv": sha256_file(OUT / "pairs_train.tsv"),
        "pairs_val.tsv": sha256_file(OUT / "pairs_val.tsv"),
        "pairs_test.tsv": sha256_file(OUT / "pairs_test.tsv"),
        "pairs_eval47.tsv": sha256_file(OUT / "pairs_eval47.tsv"),
        "fonts.tsv": sha256_file(OUT / "fonts.tsv"),
        "donor_train.json": sha256_file(OUT / "donor_train.json"),
        "sample_weights.json": sha256_file(OUT / "sample_weights.json"),
    }
    index = {
        "schema_version": 1,
        "dataset_id": "v0913_clean",
        "status": "frozen",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dirty_png_root": "data/fontdiffuser-p253-t295-s338-cn2west-v2",
        "rebuild": "python3 scripts/build_v0913_clean_map.py",
        "verify": "python3 scripts/verify_v0913_clean_map.py --png-root data/fontdiffuser-p253-t295-s338-cn2west-v2",
        "materialize": (
            "python3 scripts/verify_v0913_clean_map.py "
            "--png-root data/fontdiffuser-p253-t295-s338-cn2west-v2 "
            "--materialize-to data/v0913_clean"
        ),
        "sources": {
            "charset": {"path": str(CHARSET_PATH.relative_to(ROOT)), "sha256": sha256_file(CHARSET_PATH)},
            "split": {"path": str(SPLIT_PATH.relative_to(ROOT)), "sha256": sha256_file(SPLIT_PATH)},
            "usability_map": {"path": str(USABILITY_PATH.relative_to(ROOT)), "sha256": sha256_file(USABILITY_PATH)},
            "font_to_bucket": {"path": str(BUCKET_PATH.relative_to(ROOT)), "sha256": sha256_file(BUCKET_PATH)},
            "contract": {"path": str(CONTRACT_PATH.relative_to(ROOT)), "sha256": sha256_file(CONTRACT_PATH)},
        },
        "path_templates": {
            "target": "{png_root}/{split}/TargetImage/{font}/{font}+{cp}.png",
            "content": "{png_root}/{split}/ContentImage/{cp}.png",
            "style": "{png_root}/{split}/StyleImage/{font}/{font}+{cp}.png",
        },
        "counts": {
            "train": {"all": len(by_split["train"]), **dict(by_split_group["train"])},
            "val": {"all": len(by_split["val"]), **dict(by_split_group["val"])},
            "test": {"all": len(by_split["test"]), **dict(by_split_group["test"])},
            "eval47": {
                "all": len(eval_rows),
                "keep": sum(r["keep"] for r in eval_rows),
                "drop": sum(1 - r["keep"] for r in eval_rows),
            },
            "fonts": {
                "train_effective": sum(1 for r in fonts_rows if r["split"] == "train" and r["usable"]),
                "train_exclude": sum(1 for r in fonts_rows if r["split"] == "train" and not r["usable"]),
                "val": sum(1 for r in fonts_rows if r["split"] == "val"),
                "test": sum(1 for r in fonts_rows if r["split"] == "test"),
            },
        },
        "sample": {
            "name": "v0913_clean.train.s50_k38_b12",
            "group_p": GROUP_P,
            "cn2cn_p": 0.0,
            "seed": 3407,
        },
        "cache": {
            "reuse": True,
            "ec": "artifacts/f0/ec_multiscale_f0/",
            "es": "artifacts/f0/es_spatial_f0/",
        },
        "files": artifacts,
        "do_not": [
            "re-render PNG",
            "rebuild Ec/Es",
            "treat materialize-only as the 50/38/12 recipe",
            "score eval47 rows with keep=0",
        ],
    }
    (OUT / "INDEX.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return index


def main() -> int:
    index = build()
    c = index["counts"]
    print(
        f"v0913_clean frozen  train={c['train']['all']} "
        f"val={c['val']['all']} test={c['test']['all']} "
        f"eval47_keep={c['eval47']['keep']}/{c['eval47']['all']}"
    )
    print(f"INDEX {OUT / 'INDEX.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
