#!/usr/bin/env python3
"""Embed image_index.csv into the standalone, local-file gallery."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "gallery_template.html"
INDEX = ROOT / "image_index.csv"
OUTPUT = ROOT / "index.html"
MODELS = ("K7-B", "A1", "A2", "A3")


def main() -> None:
    records = []
    with INDEX.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            records.append(
                {
                    "split": row["split"],
                    "batch": row["batch"],
                    "font": row["font"],
                    "codepoint": row["codepoint"],
                    "character": row["character"],
                    "script": row["script"],
                    "paths": {model: row["image_" + model] for model in MODELS},
                }
            )

    data = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace(" ", "\\u2028").replace(" ", "\\u2029")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__SAMPLE_DATA__", data)
    if "__SAMPLE_DATA__" in html:
        raise SystemExit("Template placeholder replacement failed")
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Built {OUTPUT.name} with {len(records):,} sample records")


if __name__ == "__main__":
    main()
