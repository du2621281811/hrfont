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
                spine.set_color("#2f5d3a" if proto == "A" else "#8b1e1e" if proto == "H" else "#cccccc")
                spine.set_linewidth(2.0 if proto in ("A", "H") else 1.0)
            if i == 0:
                title = "A（采用）" if proto == "A" else "H（弃用）" if proto == "H" else proto
                title_color = "#2f5d3a" if proto == "A" else "#8b1e1e" if proto == "H" else "#222"
                ax.set_title(title, fontproperties=cjk(10.5, "bold"), pad=4, color=title_color)
                ax.text(
                    0.5, -0.08, PROTO_NOTE[proto], transform=ax.transAxes,
                    ha="center", va="top", fontproperties=cjk(7.5), color="#444",
                )
            if j == 0:
                ax.set_ylabel(rlab, fontproperties=cjk(10))
    fig.suptitle("同一字体、各行同一字符 · 六种渲染协议", fontproperties=cjk(13, "bold"), y=0.995)
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
    fig, ax = plt.subplots(figsize=(13.4, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11.5)
    ax.axis("off")

    fig.suptitle(
        "E2b vs E2：逐项对照（预期唯一变量 = RSI 结构源）",
        fontproperties=cjk(15, "bold"), y=0.985,
    )
    ax.text(
        7, 10.82,
        "两组都从 E1@100k 初始化；冻结 Es/Ec；n-shot 风格条件、UNet、RSI 内部与 loss 相同",
        ha="center", va="center", fontproperties=cjk(9), color="#444",
    )

    label_x, left_x, right_x = 0.25, 2.45, 8.25
    box_w, box_h = 5.15, 1.25
    row_y = [8.95, 7.25, 5.55, 3.85, 2.15]
    row_labels = [
        ("风格条件", "共享"),
        ("内容条件", "共享"),
        ("RSI 结构源", "唯一预期差异"),
        ("RSI 内部", "共享"),
        ("训练设置", "共享"),
    ]

    ax.add_patch(mpatches.FancyBboxPatch(
        (left_x, 9.95), box_w, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#fff4df", edgecolor="#9a6700", linewidth=1.2,
    ))
    ax.text(left_x + box_w / 2, 10.26, "E2b · official-RSI control",
            ha="center", va="center", fontproperties=cjk(11, "bold"), color="#7a5200")
    ax.add_patch(mpatches.FancyBboxPatch(
        (right_x, 9.95), box_w, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#eaf5ee", edgecolor="#2f5d3a", linewidth=1.2,
    ))
    ax.text(right_x + box_w / 2, 10.26, "E2 · Δ-RSI",
            ha="center", va="center", fontproperties=cjk(11, "bold"), color="#2f5d3a")

    for y, (name, tag) in zip(row_y, row_labels):
        ax.text(label_x, y + 0.73, name, ha="left", va="center",
                fontproperties=cjk(10, "bold"), color="#222")
        tag_color = "#8b1e1e" if "差异" in tag else "#666"
        ax.text(label_x, y + 0.30, tag, ha="left", va="center",
                fontproperties=cjk(7.6), color=tag_color)

    shared_face = "#f5f6f7"
    left_diff, right_diff = "#fff4df", "#eaf5ee"
    left_edge, right_edge = "#9a6700", "#2f5d3a"

    shared_rows = [
        "R={r1,…,rn},  n∈{1,…,8}\nmean Es(R) → 9 style tokens",
        "Noto c → Ec(c) → MCA\n提供目标字符身份与多尺度内容",
        None,
        "Q=结构条件，K/V=UNet skip\nCrossAttn → 18-channel offset → DCN(skip)",
        "E1@100k · freeze Es/Ec · 80k\nLdiff + 0.01 Lperc + 0.5 Loffset",
    ]
    for idx, (y, text) in enumerate(zip(row_y, shared_rows)):
        if idx == 2:
            continue
        _box(ax, (left_x, y), box_w, box_h, text, shared_face, edge="#777", fs=9.3, lw=1.0)
        _box(ax, (right_x, y), box_w, box_h, text, shared_face, edge="#777", fs=9.3, lw=1.0)

    _box(
        ax, (left_x, row_y[2]), box_w, box_h,
        "Ec(R[0])\n绝对参考字结构（R[0] 为首个参考字）",
        left_diff, edge=left_edge, fs=9.5, lw=1.5,
    )
    _box(
        ax, (right_x, row_y[2]), box_w, box_h,
        "Δc = ΣTop10 αs Ec(Bs,c) − Ec(Noto c)\n同一目标字符的相对残差",
        right_diff, edge=right_edge, fs=9.2, lw=1.5,
    )

    warning = (
        "当前已启动 run 仍有第二差异：E2 对结构源做 25% drop，而 E2b 未 drop。"
        "因此当前结果仅作 pilot；正式 matched 对照需统一 source_drop 后重跑。"
    )
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 0.45), 13.1, 1.05, boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor="#fdecea", edgecolor="#8b1e1e", linewidth=1.3,
    ))
    ax.text(6.85, 0.98, warning, ha="center", va="center",
            fontproperties=cjk(9.2, "bold"), color="#8b1e1e", wrap=True)

    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.965))
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
        axes[i, 0].set_ylabel(f"测试例 {i + 1} · {samples[i][1]}", fontproperties=cjk(8.5))
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
