#!/usr/bin/env python3
"""Create compact, machine-verifiable dataset and experiment provenance records."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shown(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: Path) -> dict:
    root = root.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_digest = sha256_file(path)
        tree_digest.update(f"{relative}\0{size}\0{file_digest}\n".encode())
        total_bytes += size
    return {
        "root": shown(root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
    }


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_state(repo: Path, allow_dirty: bool) -> dict:
    repo = repo.resolve()
    commit = run_git(repo, "rev-parse", "HEAD")
    status = run_git(repo, "status", "--porcelain")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise SystemExit(f"refusing dirty Git tree: {repo}")
    patch = ""
    if dirty:
        patch = run_git(repo, "diff", "--binary", "HEAD")
    remotes = run_git(repo, "remote", "-v").splitlines()
    return {
        "path": shown(repo),
        "commit": commit,
        "branch": run_git(repo, "branch", "--show-current"),
        "dirty": dirty,
        "dirty_paths": [line[3:] for line in status.splitlines()],
        "dirty_patch_sha256": hashlib.sha256(patch.encode()).hexdigest() if patch else None,
        "remotes": remotes,
    }


def dependency_versions() -> dict:
    versions = {}
    for package in ("torch", "torchvision", "accelerate", "diffusers", "numpy", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def path_record(path: Path) -> dict:
    path = path.resolve()
    if path.is_file():
        return {
            "path": shown(path),
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        return {"kind": "directory", **tree_fingerprint(path)}
    raise SystemExit(f"missing provenance input: {path}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def dataset_command(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")
    payload = {
        "schema_version": 1,
        "kind": "dataset",
        "dataset_id": args.id,
        "created_at": utc_now(),
        **tree_fingerprint(root),
        "source_note": args.note,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_command(args: argparse.Namespace) -> None:
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    records = {}
    for item in args.record_hash:
        label, separator, digest = item.partition("=")
        if not separator or len(digest) != 64:
            raise SystemExit(f"--record-hash must be LABEL=64_HEX: {item}")
        records[label] = digest
    payload = {
        "schema_version": 1,
        "kind": "experiment",
        "experiment_id": args.id,
        "created_at": utc_now(),
        "provenance_certainty": args.certainty,
        "code_recoverable": args.code_recoverable,
        "git": [git_state(repo, args.allow_dirty) for repo in args.git_repo],
        "dataset": {
            "manifest": shown(args.dataset_manifest),
            "dataset_id": dataset["dataset_id"],
            "tree_sha256": dataset["tree_sha256"],
            "manifest_sha256": sha256_file(args.dataset_manifest),
        },
        "command": args.command,
        "code_files": [path_record(path) for path in args.code_file],
        "recorded_historical_hashes": records,
        "initialization": [path_record(path) for path in args.init],
        "artifacts": [path_record(path) for path in args.artifact],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": dependency_versions(),
        },
        "note": args.note,
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--id", required=True)
    dataset.add_argument("--root", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    dataset.add_argument("--note", default="")
    dataset.set_defaults(func=dataset_command)

    run = subparsers.add_parser("run")
    run.add_argument("--id", required=True)
    run.add_argument("--dataset-manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--git-repo", type=Path, action="append", default=[])
    run.add_argument("--code-file", type=Path, action="append", default=[])
    run.add_argument("--record-hash", action="append", default=[])
    run.add_argument("--init", type=Path, action="append", default=[])
    run.add_argument("--artifact", type=Path, action="append", default=[])
    run.add_argument("--command", default="")
    run.add_argument("--note", default="")
    run.add_argument("--certainty", choices=("exact", "retro_partial"), default="exact")
    run.add_argument("--code-recoverable", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--allow-dirty", action="store_true")
    run.set_defaults(func=run_command)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
