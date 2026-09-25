#!/usr/bin/env python3
"""Project-management preflight before training or formal evaluation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def git_porcelain(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def check(name: str, ok: bool, detail: str, failures: list[str]) -> None:
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}")
    if not ok:
        failures.append(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []

    dirty = git_porcelain(PROJECT)
    check(
        "root_git_clean",
        args.allow_dirty or not dirty,
        "clean" if not dirty else f"dirty ({len(dirty.splitlines())} paths)",
        failures,
    )
    print(f"      HEAD={git_head(PROJECT)}")

    official = PROJECT / "code/official/FontDiffuser"
    ours = PROJECT / "code/ours/FontDiffuser"
    check("official_tree", official.is_dir(), str(official.relative_to(PROJECT)), failures)
    check("ours_legacy_tree", ours.is_dir(), str(ours.relative_to(PROJECT)), failures)

    marker_o = official / "ZZZ_READ_ME_THIS_IS_OFFICIAL_UPSTREAM.txt"
    marker_u = ours / "ZZZ_READ_ME_THIS_IS_OURS_PATCHED_NOT_OFFICIAL.txt"
    check("official_marker", marker_o.is_file(), "present" if marker_o.is_file() else "missing", failures)
    check("ours_marker", marker_u.is_file(), "present" if marker_u.is_file() else "missing", failures)

    for required in (
        PROJECT / "PROJECT.md",
        PROJECT / "docs/PROJECT_MANAGEMENT.md",
        PROJECT / "provenance/REGISTRY.md",
        PROJECT / "code/variants/README.md",
    ):
        check(
            f"doc:{required.relative_to(PROJECT)}",
            required.is_file(),
            "present" if required.is_file() else "missing",
            failures,
        )

    if args.variant:
        variant_root = PROJECT / "code/variants" / args.variant / "FontDiffuser"
        check(
            "variant_tree",
            variant_root.is_dir(),
            str(variant_root.relative_to(PROJECT)) if variant_root.is_dir() else f"missing {args.variant}",
            failures,
        )
        if args.variant in {"official", "ours", "ours-legacy"}:
            check("variant_name", False, "reserved name", failures)

    if args.experiment_id:
        run_dir = PROJECT / "runs" / args.experiment_id
        prov = PROJECT / "provenance/runs" / f"{args.experiment_id}.json"
        if run_dir.exists() or prov.exists():
            check(
                "experiment_id_unique",
                False,
                f"already exists (run={run_dir.exists()} provenance={prov.exists()})",
                failures,
            )
        else:
            check("experiment_id_unique", True, args.experiment_id, failures)

        registry = (PROJECT / "provenance/REGISTRY.md").read_text(encoding="utf-8")
        if args.experiment_id in registry and "planned" not in registry.lower():
            # soft note only when ID already listed as completed
            pass
        print(f"      registry mentions id: {args.experiment_id in registry}")

    print()
    if failures:
        print(f"preflight FAILED ({len(failures)}): {', '.join(failures)}")
        print("Fix before training. See docs/PROJECT_MANAGEMENT.md")
        return 1
    print("preflight PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
