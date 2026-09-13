#!/usr/bin/env python3
"""Load frozen v0913_clean pair maps for training and Δ donor filters."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
DEFAULT_MAP = ROOT / "manifests/v0913_clean"
GROUPS = ("latin", "kana", "bopomofo")


def load_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_phase_pairs(map_dir: str | Path, phase: str) -> tuple[list[dict], list[float] | None]:
    map_dir = Path(map_dir)
    rows = load_tsv(map_dir / f"pairs_{phase}.tsv")
    if not rows:
        raise RuntimeError(f"empty v0913_clean {phase} pairs")
    weights = None
    if phase == "train":
        table = json.loads((map_dir / "sample_weights.json").read_text(encoding="utf-8"))["pair_weight"]
        weights = [float(table[row["script_group"]]) for row in rows]
    return rows, weights


def load_donors(map_dir: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(map_dir, "donor_train.json").read_text(encoding="utf-8"))
    all_fonts = sorted({font for group in GROUPS for font in payload[group]})
    return {**{g: list(payload[g]) for g in GROUPS}, "all": all_fonts}


def load_cp_group(map_dir: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for phase in ("train", "val", "test"):
        for row in load_tsv(Path(map_dir) / f"pairs_{phase}.tsv"):
            out[row["cp"]] = row["script_group"]
    return out


def extra_exclude_indices(train_fonts: list[str], cp: str, donors: dict[str, list[str]], cp_group: dict[str, str]) -> list[int]:
    group = cp_group.get(cp)
    if not group:
        return []
    allowed = set(donors[group])
    return [i for i, font in enumerate(train_fonts) if font not in allowed]
