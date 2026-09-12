#!/usr/bin/env python3
"""Generate the editable HR-Font method overview as PDF/SVG/PNG."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_style import COLORS, apply_style, save_all


HERE = Path(__file__).resolve().parent


def box(ax, xy, wh, title, lines, face, edge, title_color=None, dashed=False):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.045, title, ha="left", va="top",
            fontsize=8, fontweight="bold", color=title_color or COLORS["ink"])
    ax.text(x + 0.02, y + h - 0.11, lines, ha="left", va="top",
            fontsize=7, linespacing=1.25, color=COLORS["ink"])


def arrow(ax, start, end, color, label=None, dashed=False, curve=0.0):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8,
        linewidth=1.0, color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(a)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, label, ha="center", va="bottom",
                fontsize=7, color=color)


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.02, 0.67), (0.15, 0.22), "Neutral content", "glyph identity\nfixed anchor $B_0(c)$",
        "#EAF2F8", COLORS["blue"])
    box(ax, (0.02, 0.38), (0.15, 0.20), "Target refs", "ordered ref1 ... ref8\nobserved CN glyphs",
        "#E8F5F0", COLORS["green"])
    box(ax, (0.02, 0.10), (0.15, 0.18), "Font bank", "train fonts\nsame target char $c$",
        "#FFF4E5", COLORS["orange"])

    box(ax, (0.23, 0.57), (0.23, 0.32), "Set-Delta proposals",
        "retrieve Top-$K$ donors\nTSDF/gradient/mask residuals\nretain donor axis\n$\{d_{j,c}\}_{j=1}^{K}$",
        "#FFF4E5", COLORS["orange"])
    box(ax, (0.23, 0.12), (0.23, 0.32), "Graphics-Ref evidence",
        "per-ref global tokens\nprimitive keys: radius, angle,\nendpoint, junction\nlearned Es12 values",
        "#E8F5F0", COLORS["green"])

    box(ax, (0.53, 0.39), (0.22, 0.38), "Role-separated fusion",
        r"location-dependent donor attention" "\n" r"$\alpha$: proposal bias only" "\n"
        r"local ref attention" "\n" r"real mask + synchronized CFG",
        "#F3EFFA", COLORS["purple"])
    box(ax, (0.80, 0.39), (0.18, 0.38), "Diffusion realization",
        "inherited FontDiffuser\nwarp geometry first\nvalue path: ablation\noutput $\hat y_c$",
        "#FBE9E7", COLORS["vermillion"])

    arrow(ax, (0.17, 0.78), (0.23, 0.78), COLORS["blue"])
    arrow(ax, (0.17, 0.19), (0.23, 0.66), COLORS["orange"], curve=-0.08)
    arrow(ax, (0.17, 0.48), (0.23, 0.70), COLORS["green"], curve=0.10)
    arrow(ax, (0.17, 0.46), (0.23, 0.28), COLORS["green"], curve=-0.08)
    arrow(ax, (0.17, 0.75), (0.53, 0.60), COLORS["blue"], curve=0.08)
    arrow(ax, (0.46, 0.72), (0.53, 0.64), COLORS["orange"])
    arrow(ax, (0.46, 0.28), (0.53, 0.48), COLORS["green"])
    arrow(ax, (0.75, 0.58), (0.80, 0.58), COLORS["purple"])

    ax.text(0.345, 0.52, "bank-supported", ha="center", va="center",
            fontsize=7, color=COLORS["orange"], fontstyle="italic")
    ax.text(0.345, 0.07, "reference-observed", ha="center", va="center",
            fontsize=7, color=COLORS["green"], fontstyle="italic")
    ax.text(0.89, 0.31, "a coherent cross-script family",
            ha="center", va="center", fontsize=7, color=COLORS["muted"])

    fig.subplots_adjust(0, 0, 1, 1)
    save_all(fig, HERE / "fig_method_overview")


if __name__ == "__main__":
    main()
