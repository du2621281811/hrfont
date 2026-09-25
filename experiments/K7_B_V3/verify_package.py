#!/usr/bin/env python3
"""Read-only integrity audit of the shared K7-B V3 package."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path, expected, failures):
    actual = digest(path) if path.is_file() else "MISSING"
    if actual != expected:
        failures.append(f"{path.relative_to(ROOT)}: expected {expected}, got {actual}")


def main():
    failures = []
    source = read("source/K7_CODE_IDENTITY.json")["files"]
    for name, expected in source.items():
        check(ROOT / "source" / name, expected, failures)
    print(f"Current source identity: {len(source)} files, {len(failures)} mismatches")

    config = read("provenance/config_50k.json")
    checks = {
        "contracts/CALIBRATION.json": config["calibration_sha256"],
        "contracts/teacher/teacher_stats.pt": config["teacher_stats_sha256"],
        "contracts/v0921_spec.json": config["data_identity"]["v0921"]["spec_sha256"],
        "contracts/v0921/pairs_train.tsv": config["data_identity"]["v0921"]["pairs_sha256"],
        "contracts/v0921/style_pool.json": config["data_identity"]["v0921"]["style_pool_sha256"],
        "source/experiments/K4/detail_manifest.json": config["detail_sha256"],
        "source/experiments/K4/aliases.json": config["data_identity"]["aliases_sha256"],
        # The historical config key is named family_groups_sha256, but runtime hashes weight_groups.json.
        "source/experiments/K4/weight_groups.json": config["data_identity"]["family_groups_sha256"],
    }
    for name, expected in config["data_identity"]["data"].items():
        checks[f"contracts/v2/{name}"] = expected
    for name, expected in checks.items():
        check(ROOT / name, expected, failures)
    continuation = read("provenance/CONTINUATION_MANIFEST.json")
    check(ROOT / "ops/train_k7_b_50k_overlay.py", continuation["overlay_sha256"], failures)
    check(ROOT / "source/experiments/K6/AUTHORIZATION_K7_B_50K.json", continuation["authorization_sha256"], failures)
    print(f"Frozen contracts: {len(checks) + 2} SHA-256 checks")

    historical = config["identity"]["files"]
    drift = [(name, expected, source.get(name)) for name, expected in historical.items()
             if source.get(name) != expected]
    print(f"Historical train/source drift: {len(drift)} files (disclosed, not package-integrity failures)")
    for name, old, current in drift:
        print(f"  {name}\n    frozen={old}\n    current={current}")
    if failures:
        print("INTEGRITY FAILURES:")
        for failure in failures:
            print("  " + failure)
        raise SystemExit(1)
    print("Package integrity OK; historical drift is a separate reproducibility limitation.")


if __name__ == "__main__":
    main()
