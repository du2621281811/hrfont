#!/usr/bin/env python3
"""Paired statistical decision for completed Stage A MVP metrics."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    args = parser.parse_args()

    control = json.loads(args.control.read_text(encoding="utf-8"))
    delta = json.loads(args.delta.read_text(encoding="utf-8"))
    c_rows = {(row["font"], row["char"]): row for row in control["rows"]}
    d_rows = {(row["font"], row["char"]): row for row in delta["rows"]}
    keys = sorted(set(c_rows) & set(d_rows))
    if not keys:
        raise RuntimeError("no paired evaluation rows")

    differences = np.asarray(
        [d_rows[key]["l1"] - c_rows[key]["l1"] for key in keys],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260902)
    bootstrap_means = np.empty(args.bootstrap, dtype=np.float64)
    for start in range(0, args.bootstrap, 1000):
        count = min(1000, args.bootstrap - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        bootstrap_means[start : start + count] = differences[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

    control_mean = float(np.mean([c_rows[key]["l1"] for key in keys]))
    delta_mean = float(np.mean([d_rows[key]["l1"] for key in keys]))
    improvement = control_mean - delta_mean
    win_rate = float(np.mean(differences < 0))
    numeric_pass = improvement >= 0.002 and ci_high < 0 and win_rate > 0.5
    clearly_worse = ci_low > 0
    verdict = "GO_NUMERIC" if numeric_pass else ("NO_GO" if clearly_worse else "INCONCLUSIVE")

    by_font = {}
    for font in sorted({font for font, _ in keys}):
        font_keys = [key for key in keys if key[0] == font]
        values = np.asarray(
            [d_rows[key]["l1"] - c_rows[key]["l1"] for key in font_keys]
        )
        by_font[font] = {
            "n": len(font_keys),
            "control_L1": float(np.mean([c_rows[key]["l1"] for key in font_keys])),
            "delta_L1": float(np.mean([d_rows[key]["l1"] for key in font_keys])),
            "delta_minus_control": float(values.mean()),
            "delta_win_rate": float(np.mean(values < 0)),
        }

    payload = {
        "experiment": "A-MVP-PAIRED-COMPARISON",
        "verdict": verdict,
        "numeric_pass": numeric_pass,
        "write_accuracy_check": "pending_not_available_for_full_latin_P1",
        "n_pairs": len(keys),
        "control_L1": control_mean,
        "delta_L1": delta_mean,
        "control_minus_delta": improvement,
        "delta_minus_control_ci95": [float(ci_low), float(ci_high)],
        "delta_win_rate": win_rate,
        "criteria": {
            "L1_improvement_at_least_0.002": bool(improvement >= 0.002),
            "paired_ci_below_zero": bool(ci_high < 0),
            "delta_win_rate_above_0.5": bool(win_rate > 0.5),
        },
        "by_font": by_font,
        "control_metrics": str(args.control),
        "delta_metrics": str(args.delta),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({k: v for k, v in payload.items() if k != "by_font"}, indent=2))


if __name__ == "__main__":
    main()
