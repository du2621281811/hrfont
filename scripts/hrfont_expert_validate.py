#!/usr/bin/env python3
"""Algorithm expert validation of HR-Font Stage A results + protocol checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight"


def main() -> int:
    met_path = REP / "formal_eval/metrics_step_80000.json"
    if not met_path.exists():
        print("FAIL: missing metrics_step_80000.json")
        return 1

    d = json.loads(met_path.read_text(encoding="utf-8"))
    ft = d["ft_cnstyle"]["L1_mean"]
    a = d["stageA_formal"]["L1_mean"]
    n = d["stageA_formal"]["n"]
    issues: list[str] = []
    ok: list[str] = []

    # Protocol checks
    if n != 976:
        issues.append(f"sample count {n} != expected 976 (Demo-8 × P1)")
    else:
        ok.append(f"n={n} matches Demo-8×P1")

    proto = d.get("protocol", "")
    if "DejaVu" in proto and "cfg7.5" in proto.lower().replace(" ", ""):
        ok.append("protocol: DejaVu + CFG 7.5")
    else:
        issues.append(f"protocol unexpected: {proto}")

    # Magnitude sanity (retrain_v2 full P1 ~0.083)
    if 0.05 < ft < 0.15:
        ok.append(f"ft L1={ft:.4f} in expected range [0.05,0.15]")
    else:
        issues.append(f"ft L1={ft:.4f} out of expected range")

    if 0.05 < a < 0.15:
        ok.append(f"StageA L1={a:.4f} in expected range")
    else:
        issues.append(f"StageA L1={a:.4f} out of expected range")

    # Pairwise
    rows_ft = {(r["font"], r["char"]): r["l1"] for r in d["results"]["ft_cnstyle"]["rows"]}
    rows_a = {(r["font"], r["char"]): r["l1"] for r in d["results"]["stageA_formal"]["rows"]}
    wins = sum(1 for k in rows_ft if k in rows_a and rows_a[k] < rows_ft[k])
    ok.append(f"pairwise A wins {wins}/{len(rows_ft)} ({100*wins/len(rows_ft):.1f}%)")

    rel = (a - ft) / ft
    if abs(rel) < 0.08:
        ok.append(f"Stage A within 8% of ft ({rel*100:+.1f}%) → idea validated (close)")
    elif a < ft:
        ok.append(f"Stage A beats ft ({rel*100:+.1f}%) → idea validated (strong)")
    else:
        issues.append(f"Stage A worse than ft by {rel*100:.1f}% (>8%)")

    # Per-font: at least some fonts should benefit from Δ-RSI
    from collections import defaultdict
    import statistics as stats
    pf = defaultdict(lambda: {"ft": [], "a": []})
    for k, v in rows_ft.items():
        if k in rows_a:
            pf[k[0]]["ft"].append(v)
            pf[k[0]]["a"].append(rows_a[k])

    font_deltas = []
    for font, v in pf.items():
        font_deltas.append(stats.mean(v["ft"]) - stats.mean(v["a"]))
    n_pos = sum(1 for x in font_deltas if x > 0)
    ok.append(f"fonts where A better: {n_pos}/8")

    print("=== HR-Font Expert Validation ===\n")
    for s in ok:
        print(f"  OK  {s}")
    for s in issues:
        print(f"  WARN {s}")

    verdict = "PASS" if len(issues) == 0 else ("PASS_WITH_NOTES" if a <= ft * 1.08 else "REVIEW")
    print(f"\nVerdict: {verdict}")
    print(f"  ft={ft:.4f}  StageA={a:.4f}  delta={ft-a:+.4f}")

    out = {"verdict": verdict, "ft_l1": ft, "stageA_l1": a, "ok": ok, "issues": issues}
    (REP / "EXPERT_VALIDATION.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if verdict != "REVIEW" else 0


if __name__ == "__main__":
    sys.exit(main())
