#!/usr/bin/env python3
"""Classify dirty/untracked paths for multi-host git sync.

Prints JSON to stdout. Does not commit. Used by the 2h PM loop.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NEVER_PREFIX = (
    "artifacts/e12/cache_",
    "artifacts/e12/fonts",
    "artifacts/f0/ec_multiscale",
    "artifacts/f0/es_spatial",
    "data/fontdiffuser",
    "data/font/",
    "data/hrfont_bank",
    "data/f03_test16_strat",
    "data/f3_ckpt_dashboard",
    "runs/",
    "logs/",
)
NEVER_SUFFIX = (".pth", ".pt", ".pid", ".lock", ".log", ".zip")
NEVER_NAME = {"STOP", "status.lock", "parallel_runner.lock"}

HOLD_PREFIX = (
    "code/variants/cn2west_f123_rsi/FontDiffuser/src/",
    "code/variants/cn2west_f123_rsi/FontDiffuser/train.py",
    "code/variants/cn2west_f123_rsi/FontDiffuser/configs/",
    "scripts/launch_cn2west_f123.py",
    "scripts/queue_f2p_f3bp.py",
    "reports/training_logs/",
    "reports/f123_dashboard/status.json",
    "reports/compare_portal/manifest.json",
    "reports/watchdog_",
    "reports/f03_test16_strat/logs/",
)

MUST_PREFIX = (
    "scripts/",
    "docs/",
    "provenance/",
    "manifests/",
    ".cursor/rules/",
    "reports/f03_test16_strat/",
    "reports/PI_BRIEFING",
    "reports/compare_portal/PI_BRIEFING",
    "reports/artifacts_sync/",
    "PROJECT.md",
    "COLLABORATOR_GUIDE.md",
    "README.md",
    "code/README.md",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def classify(path: str) -> str:
    p = path.replace("\\", "/").strip().strip('"')
    if p.startswith("./"):
        p = p[2:]
    name = Path(p).name
    if name in NEVER_NAME or p.endswith(NEVER_SUFFIX):
        return "never"
    if name == ".gitignore" or p in {".gitignore", "gitignore"}:
        return "must_sync"
    if p.startswith(".cursor/"):
        return "must_sync"
    if "f2p" in p.lower() or "f2_pattn" in p.lower() or "probe_style_domain" in p:
        return "hold_wip"
    if any(p.startswith(x) or p == x.rstrip("/") for x in NEVER_PREFIX):
        return "never"
    if any(p.startswith(x) for x in HOLD_PREFIX):
        return "hold_wip"
    if any(p.startswith(x) or p == x.rstrip("/") for x in MUST_PREFIX):
        if p.startswith("scripts/launch_cn2west_f123.py") or p.startswith("scripts/queue_f2p"):
            return "hold_wip"
        return "must_sync"
    if p.startswith("code/"):
        return "hold_wip"
    return "hold_wip"


def porcelain_paths() -> list[str]:
    out = git("status", "--porcelain", "--untracked-files=normal")
    paths = []
    for line in out.splitlines():
        if not line.strip():
            continue
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw.strip())
    return paths


def main() -> int:
    git("fetch", "origin", "main")
    behind = int(git("rev-list", "--count", "HEAD..origin/main") or "0")
    ahead = int(git("rev-list", "--count", "origin/main..HEAD") or "0")
    buckets = {"must_sync": [], "hold_wip": [], "never": []}
    for p in porcelain_paths():
        buckets[classify(p)].append(p)
    if buckets["must_sync"] and behind:
        action = "pull_rebase_then_commit_push"
    elif buckets["must_sync"]:
        action = "commit_push_must_sync"
    elif behind:
        action = "pull_rebase_only"
    elif ahead:
        action = "push_only"
    else:
        action = "noop"

    rec = {
        "action": action,
        "behind": behind,
        "ahead": ahead,
        "head": git("rev-parse", "--short", "HEAD"),
        "origin_main": git("rev-parse", "--short", "origin/main"),
        **buckets,
        "note": "Auto-loop may commit+push must_sync only. hold_wip needs a human.",
    }
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    log = ROOT / "reports" / "pm_sync_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
