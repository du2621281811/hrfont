#!/usr/bin/env python3
"""Pre-registered Q0/Q1 selector; consumes val16 metric JSON, never test16."""
import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("q0", type=Path); p.add_argument("q1", type=Path)
    p.add_argument("--delta-style", type=float, default=0.0)
    a = p.parse_args()
    rows = [json.loads(x.read_text()) for x in (a.q0, a.q1)]
    for row in rows:
        assert row.get("split") == "val16" and row.get("step") == 20000
        assert row.get("identity_noninferior") and row.get("quality_noninferior")
    gain = rows[1]["sc_gap"] - rows[0]["sc_gap"]
    winner = "q1_roleswap" if gain > a.delta_style and rows[1].get("ci_low_vs_q0", 0) > 0 else "q0_inherited"
    print(json.dumps({"winner": winner, "q1_sc_gap_gain": gain}, indent=2))

if __name__ == "__main__": main()
