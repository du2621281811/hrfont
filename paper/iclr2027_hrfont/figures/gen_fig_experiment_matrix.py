#!/usr/bin/env python3
"""Generate a claim-oriented experiment matrix as editable vector art."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_style import COLORS, apply_style, save_all


HERE = Path(__file__).resolve().parent


def node(ax, x, y, label, subtitle, color, selected=False):
    patch = FancyBboxPatch(
        (x, y), 0.17, 0.15,
        boxstyle="round,pad=0.01,rounding_size=0.018",
        facecolor="#FFFFFF" if not selected else "#FBE9E7",
        edgecolor=color, linewidth=1.4 if selected else 0.9,
    )
    ax.add_patch(patch)
    ax.text(x + 0.085, y + 0.102, label, ha="center", va="center",
            fontsize=7.3, fontweight="bold", color=color)
    ax.text(x + 0.085, y + 0.045, subtitle, ha="center", va="center",
            fontsize=7, color=COLORS["ink"], linespacing=1.1)


def connect(ax, x1, y1, x2, y2, label):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=0.9,
                                 color=COLORS["line"]))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.025, label,
            ha="center", va="bottom", fontsize=7, color=COLORS["muted"])


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.92, "Round D  |  hold R1 fixed; change only the variation prior",
            fontsize=7.5, fontweight="bold", color=COLORS["orange"])
    xs = [0.03, 0.275, 0.52, 0.765]
    node(ax, xs[0], 0.69, "D0", "no Delta", COLORS["muted"])
    node(ax, xs[1], 0.69, "D1", "old feature\nmean", COLORS["orange"])
    node(ax, xs[2], 0.69, "D2", "geometry\nmean", COLORS["blue"])
    node(ax, xs[3], 0.69, "D3", "geometry\nSet-Delta", COLORS["vermillion"], selected=True)
    connect(ax, xs[0] + .17, .765, xs[1], .765, "Delta effect")
    connect(ax, xs[1] + .17, .765, xs[2], .765, "remove appearance")
    connect(ax, xs[2] + .17, .765, xs[3], .765, "retain candidates")

    ax.text(0.02, 0.58, "Round R  |  hold selected D* fixed; change only reference evidence",
            fontsize=7.5, fontweight="bold", color=COLORS["green"])
    node(ax, 0.15, 0.35, "R1", "per-ref\nglobal", COLORS["muted"])
    node(ax, 0.415, 0.35, "R2", "learned\nlocal K/V", COLORS["green"])
    node(ax, 0.68, 0.35, "R3", "graphics K\nlearned V", COLORS["purple"], selected=True)
    connect(ax, .32, .425, .415, .425, "local evidence")
    connect(ax, .585, .425, .68, .425, "graphics key")

    ax.add_patch(FancyBboxPatch((0.03, 0.05), 0.92, 0.18,
                                boxstyle="round,pad=0.012,rounding_size=0.018",
                                facecolor=COLORS["panel"], edgecolor="#D0D5DD", linewidth=.8))
    ax.text(0.05, 0.19, "Focused controls that demonstrate each advantage",
            fontsize=7, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(0.05, 0.125,
            "D3 vs D4: anchor-relative residual     |     D3 vs D6: target-character alignment\n"
            "R3 vs R4 / FSFont-style K/V: graphics semantics     |     D7: injection contribution",
            fontsize=7, color=COLORS["ink"], va="top", linespacing=1.35)
    ax.text(0.95, 0.01, "20k interim → paired continuation to 40k → formal evaluation",
            ha="right", va="bottom", fontsize=7, color=COLORS["muted"], fontstyle="italic")

    fig.subplots_adjust(0, 0, 1, 1)
    save_all(fig, HERE / "fig_experiment_matrix")


if __name__ == "__main__":
    main()
