#!/usr/bin/env python3
"""Compose mentor-facing figures for the weekly briefing."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from PIL import Image
import numpy as np

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/weekly_20260905"
DATA = ROOT / "data"
DASH = ROOT / "reports/e1_ft_v2_dashboard"
CJK_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

PROTOS = {
    "A": "fontdiffuser-p253-t295-s338-cn2west-v2",
    "B": "fontdiffuser-p253-t295-s338-cn2west-v2b-hfit",
    "C": "fontdiffuser-p253-t295-s338-cn2west-v2c-official128",
    "D": "fontdiffuser-p253-t295-s338-cn2west-v2d-perglyph-max96",
    "F": "fontdiffuser-p253-t295-s338-cn2west-v2f-perglyph-fit96",
    "H": "fontdiffuser-p253-t295-s338-cn2west-v2h-inkfit",
}
PROTO_NOTE = {
    "A": "统一字号\n无缩放  ←选用",
    "B": "按高度适配",
    "C": "128→96 缩放",
    "D": "逐字最大号",
    "F": "逐字墨迹适配",
    "H": "墨迹搜号\n←已弃用",
}


def cjk(size: float, weight: str = "normal") -> FontProperties:
    return FontProperties(fname=CJK_PATH, size=size, weight=weight)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def glyph(proto: str, split: str, font: str, role: str, cp: str) -> Path:
    root = DATA / PROTOS[proto] / split / role / font
    return root / f"{font}+{cp}.png"


def fig_protocols() -> None:
    font = "FZBenMXYTJW_Italic"
    rows = [
        ("StyleImage", "u6C38", "风格参考「永」"),
        ("TargetImage", "u0041", "目标西文「A」"),
        ("TargetImage", "u0061", "目标西文「a」"),
    ]
    keys = list(PROTOS)
    fig, axes = plt.subplots(len(rows), len(keys), figsize=(11.2, 6.2))
    fig.patch.set_facecolor("white")
    for i, (role, cp, rlab) in enumerate(rows):
        for j, proto in enumerate(keys):
            ax = axes[i, j]
            img = load_rgb(glyph(proto, "train", font, role, cp))
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
            if i == 0:
                title = f"{proto}"
                ax.set_title(title, fontproperties=cjk(11, "bold"), pad=4)
                ax.text(
                    0.5, -0.08, PROTO_NOTE[proto], transform=ax.transAxes,
                    ha="center", va="top", fontproperties=cjk(7.5), color="#444",
                )
            if j == 0:
                ax.set_ylabel(rlab, fontproperties=cjk(10))
    fig.suptitle("同一字体、同一字符 · 六种渲染协议", fontproperties=cjk(13, "bold"), y=0.995)
    fig.text(
        0.5, 0.01,
        "训练与评测已冻结协议 A（逐字体统一字号、不缩放）。H 人工检查后弃用。",
        ha="center", fontproperties=cjk(9), color="#333",
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.96))
    fig.savefig(OUT / "fig1_six_protocols.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, w, h, text, facecolor, edge="#333", fs=9, lw=1.1):
    patch = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=edge, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
        fontproperties=cjk(fs), color="#1a1a1a", wrap=True,
    )
    return patch


def _arrow(ax, start, end):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2,
                        mutation_scale=11),
    )


def fig_method() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    fig.patch.set_facecolor("white")
    titles = ["官方 FontDiffuser：参考字结构", "本方法：目标字残差 Δ"]
    faces = ("#f4f6f8", "#eef6f1")
    for ax, title, bg in zip(axes, titles, faces):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontproperties=cjk(12, "bold"), pad=8)
        ax.add_patch(mpatches.Rectangle((0.15, 0.2), 9.7, 9.4, facecolor=bg, edgecolor="none"))

    ax = axes[0]
    _box(ax, (0.5, 7.6), 4.0, 1.5, "中文风格参考\n（同时送入 Es 与 Ec）", "#fff", fs=8.5)
    _box(ax, (5.5, 7.6), 4.0, 1.5, "中性内容图\n（西文字 c）", "#fff")
    _arrow(ax, (2.5, 7.6), (2.5, 6.35))
    _arrow(ax, (7.5, 7.6), (7.5, 6.35))
    _box(ax, (0.5, 4.7), 4.0, 1.5, "Ec(参考字)\n→ 结构支路 RSI", "#fff4df", edge="#9a6700")
    _box(ax, (5.5, 4.7), 4.0, 1.5, "Ec(c)\n→ 身份支路", "#fff")
    _box(ax, (0.5, 2.6), 9.0, 1.4, "Es(参考字) → 风格注意力", "#fff")
    _arrow(ax, (2.5, 4.7), (3.2, 4.1))
    _arrow(ax, (7.5, 4.7), (6.8, 4.1))
    _arrow(ax, (5.0, 2.6), (5.0, 2.15))
    _box(ax, (2.2, 0.5), 5.6, 1.5, "UNet 生成西文 c\nRSI 学习参考字与 c 的空间错位", "#fff4df", edge="#9a6700", fs=8.5)
    ax.text(5, 9.55, "官方即允许异字；跨语系对应更弱是本文假设", ha="center",
            fontproperties=cjk(7.7), color="#7a5200")

    ax = axes[1]
    _box(ax, (0.4, 7.6), 3.0, 1.5, "中文参考 R\n（风格）", "#fff")
    _box(ax, (3.6, 7.6), 3.0, 1.5, "库字体上的\n同一个 c", "#fff")
    _box(ax, (6.8, 7.6), 2.8, 1.5, "中性 c\n（Noto）", "#fff")
    _arrow(ax, (1.9, 7.6), (1.9, 6.35))
    _arrow(ax, (5.1, 7.6), (4.4, 6.35))
    _arrow(ax, (8.2, 7.6), (6.8, 6.35))
    _box(ax, (0.4, 4.7), 3.0, 1.5, "Es(R)\n风格 9 token", "#fff")
    _box(ax, (3.6, 4.7), 6.0, 1.5, "加权混合 − 中性编码  =  Δ\n同字、相对变化", "#d9efe3", edge="#2f5d3a", fs=9)
    _arrow(ax, (1.9, 4.7), (3.2, 3.55))
    _arrow(ax, (6.6, 4.7), (6.6, 3.55))
    _box(ax, (0.4, 2.5), 4.4, 1.4, "风格注意力\n（官方支路，不动）", "#fff", fs=8.5)
    _box(ax, (5.1, 2.5), 4.5, 1.4, "RSI 结构源改为 Δ\n（模块本身不改）", "#d9efe3", edge="#2f5d3a", fs=8.5)
    _arrow(ax, (2.6, 2.5), (4.2, 2.1))
    _arrow(ax, (7.3, 2.5), (5.8, 2.1))
    _box(ax, (2.2, 0.45), 5.6, 1.5, "UNet 生成西文 c\n结构与身份指向同一个字", "#d9efe3", edge="#2f5d3a")
    ax.text(5, 9.55, "改接线，不改 RSI 内部", ha="center",
            fontproperties=cjk(8.5), color="#2f5d3a")

    fig.tight_layout()
    fig.savefig(OUT / "fig2_method.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_e1_examples() -> None:
    samples = [
        ("test_FZBuGTJW_u0041.png", "A"),
        ("test_FZChuangHJW_DB_u0061.png", "a"),
        ("test_FZCuanBZBKSJW_u0047.png", "G"),
        ("test_FZBuGTJW_u0061.png", "a"),
    ]
    cols = ["内容（中性字）", "风格参考", "真值", "E1 生成"]
    folders = [
        DASH / "refs/content",
        DASH / "refs/style",
        DASH / "refs/gt",
        DASH / "preds/step100000",
    ]
    fig, axes = plt.subplots(len(samples), 4, figsize=(8.8, 7.4))
    fig.patch.set_facecolor("white")
    for i, (name, _) in enumerate(samples):
        for j, folder in enumerate(folders):
            ax = axes[i, j]
            ax.imshow(load_rgb(folder / name), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
            if i == 0:
                ax.set_title(cols[j], fontproperties=cjk(11, "bold"), pad=6)
    fig.suptitle("基线 E1（协议 A）满训后的生成示例", fontproperties=cjk(13, "bold"), y=0.995)
    fig.text(
        0.5, 0.012,
        "未见过的测试字体。本周方法实验从此检查点续训，不改数据协议。",
        ha="center", fontproperties=cjk(9), color="#333",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT / "fig3_e1_examples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_protocols()
    fig_method()
    fig_e1_examples()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
