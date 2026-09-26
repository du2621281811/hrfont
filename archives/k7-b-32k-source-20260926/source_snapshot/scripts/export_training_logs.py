#!/usr/bin/env python3
"""Export git-trackable training logs out of the gitignored runs/ and logs/ trees.

`runs/` holds checkpoints and is gitignored, but the loss curves and provenance files
inside it are what collaborators actually need to review. This copies only small text
artifacts into `reports/training_logs/`, which is tracked.

Run: python scripts/export_training_logs.py
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
RUNS = ROOT / "runs"
OUT = ROOT / "reports/training_logs"

# Small text artifacts worth tracking, per run dir.
COPY_FILES = (
    "launch_meta.json", "p1_load_allowlist.json", "parent_load_manifest.json",
    "parity_gate.json", "curves.jsonl", "train_log.jsonl", "val_log.jsonl",
    "val_metrics.json", "self_tests.json", "config.resolved.yaml", "DONE.json",
    "STOP_PROVENANCE.json",
)
RUN_GLOBS = ("F0-*", "F1-*", "F2-*", "F3-*", "e12_*")

LOSS_LINE = re.compile(r"Global Step (\d+) => train_loss = ([0-9.eE+-]+)")


def export_fontdiffuser_loss(run: Path, dest: Path) -> int:
    """`fontdiffuser_training.log` is gitignored by extension; re-emit it as jsonl."""
    src = run / "fontdiffuser_training.log"
    if not src.is_file():
        return 0
    rows = [{"step": int(m.group(1)), "train_loss": float(m.group(2))}
            for m in LOSS_LINE.finditer(src.read_text(errors="ignore"))]
    if not rows:
        return 0
    with (dest / "train_loss.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for pattern in RUN_GLOBS:
        for run in sorted(RUNS.glob(pattern)):
            if not run.is_dir():
                continue
            dest = OUT / run.name
            dest.mkdir(parents=True, exist_ok=True)
            copied = []
            for name in COPY_FILES:
                src = run / name
                if src.is_file():
                    shutil.copy2(src, dest / name)
                    copied.append(name)
            n_loss = export_fontdiffuser_loss(run, dest)
            if n_loss:
                copied.append(f"train_loss.jsonl({n_loss})")
            milestones = sorted(p.name for p in run.glob("global_step_*"))
            if milestones:
                (dest / "milestones.json").write_text(
                    json.dumps({"milestones": milestones}, indent=2) + "\n", encoding="utf-8")
                copied.append("milestones.json")
            if not copied:
                dest.rmdir()
                continue
            index.append({"run": run.name, "files": copied})
            print(f"{run.name}: {', '.join(copied)}")

    (OUT / "INDEX.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "runs/ (gitignored: checkpoints and *.log stay local)",
        "runs": index,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nexported {len(index)} runs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
