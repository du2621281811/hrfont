"""Materialize the frozen K5 train panel as a K6 normal-protocol manifest."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5].parent
K_DIR = ROOT / "repo/experiments/K"
SOURCE = ROOT / "outputs/K5_EVALUATION_20260919/K5-A/train/metrics.json"
TARGET = K_DIR / "K_TRAIN_K5FIXED.json"


def main():
    source = json.loads(SOURCE.read_text())
    test = json.loads((K_DIR / "K_TEST.json").read_text())
    rows = source["rows"]
    assert len(rows) == 2304, len(rows)
    assert all(row["split"] == "train" for row in rows)
    jobs = [
        {
            "index": int(row["index"]),
            "font": row["font"],
            "cp": row["cp"],
            "script": row["script"],
            "split": "train",
            "k": int(row["k"]),
            "refs": row["refs"],
            "seed": int(row["seed"]),
        }
        for row in rows
    ]
    payload = {
        "protocol": test["protocol"],
        "split": "train",
        "cfg": test["cfg"],
        "steps": test["steps"],
        "order": test["order"],
        "seed": test["seed"],
        "clean_sha256": test["clean_sha256"],
        "source": "K5_EVALUATION_20260919/K5-A/train/metrics.json; frozen K5 train panel",
        "jobs": jobs,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {TARGET} with {len(jobs)} jobs")


if __name__ == "__main__":
    main()
