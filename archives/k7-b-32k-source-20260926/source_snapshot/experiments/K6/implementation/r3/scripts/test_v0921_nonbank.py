#!/usr/bin/env python3
"""Small dependency-light contract tests for the v0921 train-only boundary."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.v0921_nonbank import NonBankFamilyPolicy, V0921Spec


class _CurrentPolicy:
    def exclude(self, font, valid):
        return valid

    def verify(self, font, selected):
        return None

    def snapshot(self):
        return {"source": "current"}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="v0921-contract-") as raw:
        root = Path(raw)
        data = root / "data"
        pairs = root / "pairs.tsv"
        pool = root / "style_pool.json"
        es = root / "es"
        ec = root / "ec"
        data.mkdir(); es.mkdir(); ec.mkdir()
        pairs.write_text("split\tfont\tcp\ntrain\tDemoFont\tu4E00\n", encoding="utf-8")
        pool.write_text(json.dumps({"DemoFont": [f"u{n:04X}" for n in range(0x6C38, 0x6C40)]}), encoding="utf-8")
        spec_path = root / "spec.json"
        spec_path.write_text(json.dumps({
            "schema_version": 1, "dataset_id": "v0921",
            "role": "train_only_nonbank", "bank_inclusion": False,
            "split": "train", "data_root": "data", "pairs": "pairs.tsv",
            "style_pool": "style_pool.json", "es_cache": "es",
            "ec_cache": "ec", "fonts": ["DemoFont"],
        }), encoding="utf-8")
        spec = V0921Spec.load(spec_path)
        spec.assert_target_chars({"u4E00"})
        try:
            spec.validate_runtime({"DemoFont"})
        except RuntimeError as exc:
            assert "bank contamination" in str(exc)
        else:
            raise AssertionError("overlapping v0921/bank fonts were accepted")
        policy = NonBankFamilyPolicy(_CurrentPolicy(), {"DemoFont"})
        assert policy.snapshot()["nonbank_extra_fonts"] == ["DemoFont"]
    print("v0921 non-bank contract tests: PASS")


if __name__ == "__main__":
    main()
