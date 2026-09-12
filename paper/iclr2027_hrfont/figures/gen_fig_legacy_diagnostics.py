#!/usr/bin/env python3
"""Plot completed legacy metrics directly from the versioned JSON report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_style import COLORS, apply_style, save_all


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
METRICS = REPO / "reports" / "f03_test16_strat" / "metrics_summary.json"


def main():
    apply_style()
    with METRICS.open("r", encoding="utf-8") as f:
        data = json.load(f)

    keys = ["F0_100k", "F1_80000", "F2_80000", "F3_80k"]
    labels = ["F0", "F1", "F2", "F3"]
    colors = ["#9AA4B2", COLORS["sky"], COLORS["vermillion"], COLORS["purple"]]
    phi = [data["e12_v51"]["methods"][k]["phi_sc_r"]["mean"] for k in keys]
    mem = [data["e12_v51"]["methods"][k]["mem_prob"]["mean"] for k in keys]
    l1 = [data["methods"][k]["overall"]["L1_mean"] for k in keys]
    lpips = [data["methods"][k]["overall"]["LPIPS_mean"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.25), gridspec_kw={"wspace": .42})
    x = np.arange(len(keys))
    width = .34

    axes[0].bar(x - width / 2, phi, width, color=colors, edgecolor="white", linewidth=.5)
    axes[0].bar(x + width / 2, mem, width, color=colors, alpha=.45,
                edgecolor=COLORS["ink"], linewidth=.45, hatch="//")
    axes[0].set_ylim(.45, .74)
    axes[0].set_ylabel(r"Compatibility proxy $\uparrow$")
    axes[0].legend(["$\\phi$ SC-R", "membership"], frameon=False, fontsize=7, loc="upper left")

    axes[1].bar(x, l1, .58, color=colors, edgecolor="white", linewidth=.5)
    axes[1].set_ylim(.066, .073)
    axes[1].set_ylabel(r"L1 $\downarrow$")

    axes[2].bar(x, lpips, .58, color=colors, edgecolor="white", linewidth=.5)
    axes[2].set_ylim(.138, .161)
    axes[2].set_ylabel(r"LPIPS $\downarrow$")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=.18, linewidth=.5)
        ax.set_axisbelow(True)

    fig.text(.5, -.02, "Current-system diagnostics on 752 glyphs; metrics capture complementary properties.",
             ha="center", va="top", fontsize=7, color=COLORS["muted"])
    save_all(fig, HERE / "fig_legacy_diagnostics")


if __name__ == "__main__":
    main()
