#!/usr/bin/env python3
"""Server-side merge correctness for p649 decisions.json."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_p649_review import TEST_JSON, TEST_JSONL, apply_incoming

DATASET = "fontdiffuser-p649-t295-s338-cn2west-v2a-r0"


def backup() -> tuple[str | None, str | None]:
    a = TEST_JSON.read_text(encoding="utf-8") if TEST_JSON.exists() else None
    b = TEST_JSONL.read_text(encoding="utf-8") if TEST_JSONL.exists() else None
    return a, b


def restore(a: str | None, b: str | None) -> None:
    if a is not None:
        TEST_JSON.write_text(a, encoding="utf-8")
    elif TEST_JSON.exists():
        TEST_JSON.unlink()
    if b is not None:
        TEST_JSONL.write_text(b, encoding="utf-8")
    elif TEST_JSONL.exists():
        TEST_JSONL.unlink()


def load() -> dict:
    return json.loads(TEST_JSON.read_text(encoding="utf-8"))


def rec(stem: str, who: str, decision: str, ts: str) -> dict:
    return {
        stem: {
            "review_decision": decision,
            "reviewer": who,
            "reviewed_at": ts,
            "reason": "",
            "split": "train",
            "subset": "new389",
            "n_glyphs": 633,
        }
    }


def put(who: str, stem: str, decision: str, ts: str, extra: dict | None = None) -> dict:
    payload = {
        "dataset": DATASET,
        "reviewer": who,
        "last_stem": stem,
        "updated_at": ts,
        "reviewers": {who: {"last_stem": stem, "updated_at": ts}},
        "decisions": rec(stem, who, decision, ts),
    }
    if extra:
        payload["decisions"].update(extra)
    return apply_incoming(payload, json_path=TEST_JSON, jsonl_path=TEST_JSONL)


def main() -> None:
    bak = backup()
    fails: list[str] = []
    try:
        TEST_JSON.write_text(
            json.dumps(
                {
                    "dataset": DATASET,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "n": 0,
                    "last_stem": "",
                    "reviewers": {},
                    "decisions": {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        put("alice", "FontA", "pass", "2026-09-11T10:00:00Z")
        put("bob", "FontB", "drop", "2026-09-11T11:00:00Z")
        data = load()
        if set(data["decisions"]) != {"FontA", "FontB"}:
            fails.append(f"both stems {set(data['decisions'])}")
        if data["decisions"]["FontA"]["reviewer"] != "alice":
            fails.append("alice rec lost")
        if data["reviewers"]["alice"]["last_stem"] != "FontA":
            fails.append(f"alice cursor {data['reviewers'].get('alice')}")
        if data["reviewers"]["bob"]["last_stem"] != "FontB":
            fails.append(f"bob cursor {data['reviewers'].get('bob')}")

        # stale offline copy must not overwrite a newer disk record
        put("alice", "FontB", "pass", "2026-09-11T10:30:00Z")
        data = load()
        if data["decisions"]["FontB"]["reviewer"] != "bob":
            fails.append(f"stale overwrite {data['decisions']['FontB']}")

        # newer reconnect write does replace
        put("alice", "FontB", "pass", "2026-09-11T12:00:00Z")
        data = load()
        if data["decisions"]["FontB"]["reviewer"] != "alice":
            fails.append(f"newer did not win {data['decisions']['FontB']}")
        if data["reviewers"]["bob"]["last_stem"] != "FontB":
            fails.append("bob cursor lost after alice wrote another font")

        # reconnect with only local-new stem keeps others
        put("carol", "FontC", "rerender", "2026-09-11T13:00:00Z")
        data = load()
        if set(data["decisions"]) != {"FontA", "FontB", "FontC"}:
            fails.append(f"partial PUT dropped stems {set(data['decisions'])}")
        if data["n"] != 3:
            fails.append(f"n {data['n']}")

        lines = [x for x in TEST_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(lines) != 3:
            fails.append(f"jsonl {len(lines)}")
    finally:
        restore(*bak)

    if fails:
        raise SystemExit("FAIL\n" + "\n".join(fails))
    print("PASS merge: dual-reviewer cursor, stale-offline ignored, newer reconnect wins, partial PUT keeps others")


if __name__ == "__main__":
    main()
