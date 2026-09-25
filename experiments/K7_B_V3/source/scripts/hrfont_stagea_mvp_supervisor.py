#!/usr/bin/env python3
"""Network-independent watchdog: resume both arms, evaluate, then compare."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PYTHON = Path("/root/miniforge3/envs/boogu/bin/python")
BASE = ROOT / "runs/stagea_mvp"
STATUS_PATH = BASE / "supervisor_status.json"
INTERVAL = 60
MILESTONES = (500, 2500, 5000, 7500, 10000)
ARMS = {
    "control": {"gpu": 3, "out": BASE / "control_seed20260902"},
    "delta": {"gpu": 2, "out": BASE / "delta_seed20260902"},
}
CODE_FILES = [
    ROOT / "scripts/hrfont_delta_feature.py",
    ROOT / "scripts/hrfont_stagea_mvp_train.py",
    ROOT / "scripts/hrfont_stagea_mvp_eval.py",
    ROOT / "scripts/hrfont_stagea_mvp_compare.py",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


LOCKED_HASHES = {str(path.relative_to(ROOT)): digest(path) for path in CODE_FILES}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def code_is_locked() -> bool:
    return all(digest(ROOT / relative) == expected for relative, expected in LOCKED_HASHES.items())


def process_running(arm: str) -> bool:
    marker = f"hrfont_stagea_mvp_train.py --arm {arm}"
    result = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def start_training(arm: str) -> None:
    spec = ARMS[arm]
    log_path = BASE / f"{arm}_runtime.log"
    command = [
        str(PYTHON),
        str(ROOT / "scripts/hrfont_stagea_mvp_train.py"),
        "--arm",
        arm,
        "--gpu",
        str(spec["gpu"]),
        "--max-steps",
        "10000",
        "--output-dir",
        str(spec["out"]),
        "--save-every",
        "500",
    ]
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def run_eval(arm: str) -> bool:
    spec = ARMS[arm]
    final_path = spec["out"] / "final.pt"
    metrics_path = spec["out"] / "metrics.json"
    if metrics_path.exists():
        return True
    command = [
        str(PYTHON),
        str(ROOT / "scripts/hrfont_stagea_mvp_eval.py"),
        "--arm",
        arm,
        "--checkpoint",
        str(final_path),
        "--gpu",
        str(spec["gpu"]),
        "--output",
        str(metrics_path),
    ]
    log_path = BASE / f"{arm}_eval_runtime.log"
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return result.returncode == 0 and metrics_path.exists()


def run_compare() -> bool:
    output = BASE / "comparison.json"
    if output.exists():
        return True
    command = [
        str(PYTHON),
        str(ROOT / "scripts/hrfont_stagea_mvp_compare.py"),
        "--control",
        str(ARMS["control"]["out"] / "metrics.json"),
        "--delta",
        str(ARMS["delta"]["out"] / "metrics.json"),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, cwd=ROOT)
    return result.returncode == 0 and output.exists()


def main() -> None:
    history: list[dict] = []
    recorded: set[tuple[str, int]] = set()
    while True:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        code_ok = code_is_locked()
        arm_states = {}
        for arm, spec in ARMS.items():
            status = load_json(spec["out"] / "status.json")
            step = int(status.get("step", 0))
            completed = (spec["out"] / "final.pt").exists() and step >= 10000
            running = process_running(arm)
            arm_states[arm] = {
                "step": step,
                "completed": completed,
                "running": running,
                "loss": status.get("loss"),
                "diff_loss": status.get("diff_loss"),
                "updated_at": status.get("updated_at"),
            }
            for milestone in MILESTONES:
                key = (arm, milestone)
                if step >= milestone and key not in recorded:
                    history.append(
                        {
                            "arm": arm,
                            "milestone": milestone,
                            "observed_step": step,
                            "loss": status.get("loss"),
                            "diff_loss": status.get("diff_loss"),
                            "t": now,
                        }
                    )
                    recorded.add(key)
            if not completed and not running and code_ok:
                start_training(arm)
                arm_states[arm]["restart_requested"] = True

        phase = "training"
        error = None
        if not code_ok:
            phase = "blocked_code_drift"
            error = "MVP code changed after supervisor lock; refusing restart/eval"
        elif all(state["completed"] for state in arm_states.values()):
            phase = "evaluating"
            evaluations_ok = run_eval("control") and run_eval("delta")
            if evaluations_ok:
                phase = "comparing"
                if run_compare():
                    phase = "completed"
            else:
                error = "evaluation failed; retrying next interval"

        write_status(
            {
                "phase": phase,
                "updated_at": now,
                "network_dependency": "none",
                "code_sha256": LOCKED_HASHES,
                "arms": arm_states,
                "milestone_history": history,
                "error": error,
            }
        )
        if phase in ("completed", "blocked_code_drift"):
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status(
            {
                "phase": "supervisor_crashed",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": f"{type(exc).__name__}: {exc}",
                "code_sha256": LOCKED_HASHES,
            }
        )
        raise
