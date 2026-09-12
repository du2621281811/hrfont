#!/usr/bin/env python3
"""Generate an explicit, editable qualitative-results placeholder grid."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_style import COLORS, apply_style, save_all


HERE = Path(__file__).resolve().parent


def main():
    apply_style()
    columns = ["Content", "Ref set", "FTransGAN", "FCAGAN", "FontDiff.", "CF-Font", "HR-Font / Ours", "GT"]
    rows = ["Ref-observable", "Bank-supported", "Failure case"]
    fig, ax = plt.subplots(figsize=(5.5, 2.35))
    ax.set_xlim(-1.12, len(columns))
    ax.set_ylim(0, len(rows) + .55)
    ax.axis("off")

    for c, label in enumerate(columns):
        ax.text(c + .5, len(rows) + .30, label, ha="center", va="center",
                fontsize=7, fontweight="bold",
                color=COLORS["vermillion"] if "Ours" in label else COLORS["ink"])
    for r, row_label in enumerate(rows):
        y = len(rows) - 1 - r
        ax.text(-1.06, y + .52, row_label, ha="left", va="center",
                fontsize=7, color=COLORS["muted"])
        for c in range(len(columns)):
            fill = "#FBE9E7" if c == 6 else "#F2F4F7"
            edge = COLORS["vermillion"] if c == 6 else "#D0D5DD"
            ax.add_patch(Rectangle((c + .08, y + .16), .84, .72,
                                   facecolor=fill, edgecolor=edge, linewidth=.8))
            ax.text(c + .5, y + .52, "RESULT\nPENDING", ha="center", va="center",
                    fontsize=7, color=COLORS["muted"], linespacing=1.2)

    ax.text(len(columns) / 2, .03,
            "Populate only with fixed content/ref/noise/sampler and a preregistered row-selection rule.",
            ha="center", va="bottom", fontsize=7, color=COLORS["muted"], fontstyle="italic")
    fig.subplots_adjust(0, 0, 1, 1)
    save_all(fig, HERE / "fig_qualitative_template")


if __name__ == "__main__":
    main()
