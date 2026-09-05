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
        ("StyleImage", "u6C38", "中文参考「永」"),
        ("TargetImage", "u0041", "拉丁字母「A」"),
        ("TargetImage", "u3042", "平假名「あ」"),
        ("TargetImage", "u30A2", "片假名「ア」"),
        ("TargetImage", "u3105", "注音符号「ㄅ」"),
    ]
    keys = list(PROTOS)
    fig, axes = plt.subplots(len(rows), len(keys), figsize=(11.2, 9.0))
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
    fig.suptitle("六种渲染协议在不同文字系统上的效果", fontproperties=cjk(13, "bold"), y=0.995)
    fig.text(
        0.5, 0.01,
        "每行固定同一字体与字符，仅改变渲染协议。A 保留相对字号且不插值，作为唯一训练与评测协议。",
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


def _arrow(ax, x1, y1, x2, y2, color="#555") -> None:
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.35,
                        mutation_scale=12, connectionstyle="arc3,rad=0"),
    )


def fig_fd_flow() -> None:
    """Compact FD data-flow: three lanes, one merge, one change point."""
    fig, ax = plt.subplots(figsize=(12.2, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    fig.suptitle(
        "FontDiffuser 数据流：三路汇入；只改结构源",
        fontproperties=cjk(14, "bold"), y=0.98,
    )

    # lane headers
    headers = [
        (0.4, 3.5, "#4a6fa5", "内容 · 定字"),
        (4.2, 3.5, "#9a6700", "风格 · 定风格"),
        (8.0, 3.8, "#8b1e1e", "结构 · 唯一切换"),
    ]
    for x, w, color, title in headers:
        ax.text(x + w / 2, 7.85, title, ha="center", va="center",
                fontproperties=cjk(11, "bold"), color=color)

    # inputs / structure choices (same visual row)
    _box(ax, (0.4, 6.45), 3.5, 1.0, "目标字 c\n（中性字体）", "#eef3f8", edge="#4a6fa5", fs=10)
    _box(ax, (4.2, 6.45), 3.5, 1.0, "汉字参考 R\n（1–8 张）", "#fff4df", edge="#9a6700", fs=10)

    ax.add_patch(mpatches.FancyBboxPatch(
        (8.0, 6.35), 3.8, 1.2, boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor="#fffafa", edgecolor="#8b1e1e", linewidth=1.5, linestyle="--",
    ))
    _box(ax, (8.1, 6.5), 1.75, 0.9, "E2b\nEc(R[0])", "#fff4df", edge="#9a6700", fs=9, lw=1.25)
    ax.text(9.9, 6.95, "或", ha="center", va="center",
            fontproperties=cjk(9, "bold"), color="#8b1e1e")
    _box(ax, (10.05, 6.5), 1.65, 0.9, "E2\nΔc", "#eaf5ee", edge="#2f5d3a", fs=9, lw=1.25)

    # mid modules
    _box(ax, (0.4, 4.85), 3.5, 0.95, "Ec → MCA", "#eef3f8", edge="#4a6fa5", fs=11, lw=1.2)
    _box(ax, (4.2, 4.85), 3.5, 0.95, "Es → 风格注意力", "#fff4df", edge="#9a6700", fs=11, lw=1.2)
    _box(ax, (8.0, 4.85), 3.8, 0.95, "结构条件 → RSI 的 Q", "#f8f0f0", edge="#8b1e1e", fs=10.5, lw=1.25)

    for x in (2.15, 5.95, 9.9):
        _arrow(ax, x, 6.45, x, 5.8)

    # merge rail
    for x in (2.15, 5.95, 9.9):
        ax.plot([x, x], [4.85, 4.35], color="#555", lw=1.3)
    ax.plot([2.15, 9.9], [4.35, 4.35], color="#555", lw=1.3)
    ax.annotate(
        "", xy=(6.1, 4.05), xytext=(6.1, 4.35),
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.35, mutation_scale=12),
    )

    _box(
        ax, (0.4, 2.15), 11.4, 1.9,
        "UNet（不改）　MCA + 风格注意力 + x_t,t\n"
        "RSI（不改）　Q=结构，K/V=skip → offset → DCN",
        "#f5f6f7", edge="#444", fs=11, lw=1.3,
    )
    _arrow(ax, 6.1, 2.15, 6.1, 1.65)
    _box(
        ax, (0.4, 0.55), 11.4, 1.1,
        "输出 noise_pred　·　损失 MSE + 感知 + mean|δ|（不改）",
        "#eef3f8", edge="#4a6fa5", fs=10.5, lw=1.2,
    )
    ax.text(
        6.1, 0.2,
        "读法：左两路与官方相同；右路在 Ec(R[0]) 与 Δc 之间二选一。",
        ha="center", va="center", fontproperties=cjk(8.5), color="#555",
    )

    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.96))
    fig.savefig(OUT / "fig2_fd_flow.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


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
        "两组从同一基线开始；风格条件、内容条件、生成网络、RSI 内部和训练目标均相同",
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
    ax.text(left_x + box_w / 2, 10.26, "E2b · 官方 RSI 对照",
            ha="center", va="center", fontproperties=cjk(11, "bold"), color="#7a5200")
    ax.add_patch(mpatches.FancyBboxPatch(
        (right_x, 9.95), box_w, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="#eaf5ee", edgecolor="#2f5d3a", linewidth=1.2,
    ))
    ax.text(right_x + box_w / 2, 10.26, "E2 · 目标字残差 Δ",
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
        "1–8 张中文参考 R\nEs 特征按空间位置平均（9 个位置）",
        "中性字体上的目标字 c → Ec(c) → 内容支路\n提供字符身份与多尺度内容",
        None,
        "结构条件与 UNet 特征交互\n预测位移 → 形变内容特征（RSI 本身不改）",
        "同一 E1@100k 起点 · 冻结 Es/Ec · 80k\n相同数据、生成网络与损失",
    ]
    for idx, (y, text) in enumerate(zip(row_y, shared_rows)):
        if idx == 2:
            continue
        _box(ax, (left_x, y), box_w, box_h, text, shared_face, edge="#777", fs=9.3, lw=1.0)
        _box(ax, (right_x, y), box_w, box_h, text, shared_face, edge="#777", fs=9.3, lw=1.0)

    _box(
        ax, (left_x, row_y[2]), box_w, box_h,
        "Ec(R[0])\n使用首个中文参考字的绝对结构",
        left_diff, edge=left_edge, fs=9.5, lw=1.5,
    )
    _box(
        ax, (right_x, row_y[2]), box_w, box_h,
        "Δc = ΣTop10 αs Ec(Bs,c) − Ec(Noto c)\n使用同一目标字符相对中性字体的变化",
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
    fig_fd_flow()
    fig_method()
    fig_e1_examples()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
