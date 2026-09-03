#!/usr/bin/env python3
"""Build catalog.json for charset picker — aligned with font cmap script taxonomy."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FC = Path("/root/projects/font_crosslingual")
META = json.loads((ROOT / "data/meta_chars.json").read_text(encoding="utf-8"))
TRAIN = [
    s.strip()
    for s in (ROOT / "manifests/pipeline_v2_train_stems_253.txt").read_text().splitlines()
    if s.strip()
]
TEST = [Path(f).stem for f in META["demo_fonts"]]
OUT = ROOT / "reports/charset_picker"
CN_SAMPLE = FC / "data/retrain_v2/font/train/chinese/FZBaiZBYTJW"
COVERAGE_PATH = OUT / "font_coverage_253.json"

# Same taxonomy as the cmap pie analysis
GROUPS = [
    {
        "id": "han",
        "label": "汉字 Han",
        "role": "style",
        "hint": "字库主体（约 84% cmap）。数据集里作 Style，不进 Target。",
        "dataset_selectable": True,
    },
    {
        "id": "latin_ascii",
        "label": "拉丁 ASCII",
        "role": "target",
        "hint": "可打印 ASCII（空格/标点/数字/字母）。",
        "dataset_selectable": True,
    },
    {
        "id": "latin_ext",
        "label": "拉丁扩展",
        "role": "target",
        "hint": "重音 + 拼音调号（字库里通常只有几十个，不是西欧全集）。",
        "dataset_selectable": True,
    },
    {
        "id": "hiragana",
        "label": "平假名",
        "role": "target",
        "hint": "字库假名块有 ~83；数据集目前只渲了 P2 的 15 个。",
        "dataset_selectable": True,
    },
    {
        "id": "katakana",
        "label": "片假名",
        "role": "target",
        "hint": "字库假名块有 ~86；数据集目前只渲了 P2 的 15 个。",
        "dataset_selectable": True,
    },
    {
        "id": "bopomofo",
        "label": "注音",
        "role": "target",
        "hint": "字库有 37；数据集目前只渲了 ㄅ–ㄐ 共 12。",
        "dataset_selectable": True,
    },
    {
        "id": "hangul",
        "label": "韩文 Hangul",
        "role": "none",
        "hint": "本 253 字库全部没有。不可进数据集。",
        "dataset_selectable": False,
    },
    {
        "id": "cjk_punct_fw",
        "label": "CJK 标点/全角",
        "role": "none",
        "hint": "字库都有，但当前中→西协议未纳入 Target/Style。",
        "dataset_selectable": False,
    },
    {
        "id": "symbols",
        "label": "符号",
        "role": "none",
        "hint": "字库都有少量；当前协议未纳入。",
        "dataset_selectable": False,
    },
]


def char_fname(ch: str) -> str:
    if "A" <= ch <= "Z":
        return f"{ch}+.png"
    if "a" <= ch <= "z" or ("0" <= ch <= "9"):
        return f"{ch}.png"
    return f"u{ord(ch):04X}.png"


def group_of_target(ch: str) -> str:
    o = ord(ch)
    if 0x3040 <= o <= 0x309F:
        return "hiragana"
    if 0x30A0 <= o <= 0x30FF:
        return "katakana"
    if 0x3100 <= o <= 0x312F:
        return "bopomofo"
    if o < 0x80:
        return "latin_ascii"
    return "latin_ext"


def main() -> None:
    if not COVERAGE_PATH.is_file():
        raise SystemExit(
            f"missing {COVERAGE_PATH}; generate font_coverage_253.json first"
        )
    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    cov_by_stem = {r["stem"]: r for r in coverage["fonts"]}

    p1 = list(META["L_p1"])
    p2 = list(META["L_p2"])
    ref8 = list(META["ref8"])
    han_chars = [
        chr(int(p.stem[1:], 16))
        for p in sorted(CN_SAMPLE.glob("u*.png"), key=lambda p: p.stem)
    ]

    chars = []
    for ch in p1 + p2:
        chars.append(
            {
                "ch": ch,
                "cp": f"U+{ord(ch):04X}",
                "role": "target",
                "group": group_of_target(ch),
                "file": char_fname(ch),
                "in_default_p1p2": True,
                "note": "no_gt_often" if ch == " " else "",
            }
        )
    for ch in han_chars:
        chars.append(
            {
                "ch": ch,
                "cp": f"U+{ord(ch):04X}",
                "role": "style",
                "group": "han",
                "file": f"u{ord(ch):04X}.png",
                "in_default_ref8": ch in ref8,
                "note": "ref8" if ch in ref8 else "",
            }
        )

    fonts = []
    for stem in TRAIN:
        cov = cov_by_stem.get(stem)
        fonts.append(
            {
                "stem": stem,
                "split": "train",
                "default_include": True,
                "en_dir": f"glyphs/train_en/{stem}",
                "cn_dir": f"glyphs/train_cn/{stem}",
                "cmap_total": cov["total"] if cov else None,
                "counts": cov["counts"] if cov else None,
                "has": cov["has"] if cov else None,
            }
        )
    for stem in TEST:
        fonts.append(
            {
                "stem": stem,
                "split": "test",
                "default_include": False,
                "en_dir": f"glyphs/test_en/{stem}",
                "cn_dir": f"glyphs/test_cn/{stem}",
                "cmap_total": None,
                "counts": None,
                "has": None,
            }
        )

    presets = {
        "current_protocol": {
            "label": "当前协议 · Target P1∪P2 + Style ref8",
            "target_cps": [f"U+{ord(c):04X}" for c in p1 + p2 if c != " "],
            "style_cps": [f"U+{ord(c):04X}" for c in ref8],
            "fonts": "train_default",
        },
        "ascii_only": {
            "label": "仅拉丁 ASCII Target",
            "target_cps": [f"U+{ord(c):04X}" for c in p1 if c != " " and ord(c) < 0x80],
            "style_cps": [f"U+{ord(c):04X}" for c in ref8],
            "fonts": "train_default",
        },
        "latin_all_p1": {
            "label": "P1 全拉丁（ASCII+扩展）",
            "target_cps": [f"U+{ord(c):04X}" for c in p1 if c != " "],
            "style_cps": [f"U+{ord(c):04X}" for c in ref8],
            "fonts": "train_default",
        },
        "kana_zhuyin_p2": {
            "label": "仅 P2 假名/注音",
            "target_cps": [f"U+{ord(c):04X}" for c in p2],
            "style_cps": [f"U+{ord(c):04X}" for c in ref8],
            "fonts": "train_default",
        },
        "style_pool338": {
            "label": "Style 用满已渲 338 汉字",
            "target_cps": [f"U+{ord(c):04X}" for c in p1 + p2 if c != " "],
            "style_cps": [f"U+{ord(c):04X}" for c in han_chars],
            "fonts": "train_default",
        },
    }

    # Bank-level coverage table for UI header
    bank_coverage = []
    for g in GROUPS:
        s = coverage["summary"].get(g["id"])
        if not s:
            continue
        bank_coverage.append(
            {
                "id": g["id"],
                "label": g["label"],
                "fonts_with": s["fonts_with"],
                "fonts_without": s["fonts_without"],
                "pct_fonts_with": s["pct_fonts_with"],
                "count_median": s["count_median"],
                "count_min": s["count_min"],
                "count_max": s["count_max"],
                "threshold_has": s["threshold_has"],
                "dataset_selectable": g["dataset_selectable"],
                "role": g["role"],
            }
        )

    catalog = {
        "schema_version": 2,
        "title": "HR-Font 字库语种 · 有无统计 · 数据集勾选",
        "taxonomy_note": (
            "分类与「每款字库一般含什么」cmap 实测一致。"
            "「字库有无」看 cmap；「能否勾选」看是否已渲进数据集。"
        ),
        "counts": {
            "train_fonts": len(TRAIN),
            "test_fonts": len(TEST),
            "target_chars_rendered": len(p1) + len(p2),
            "style_han_rendered": len(han_chars),
        },
        "notes": [
            "两层不要混：① 字库 cmap 里有什么；② 数据集已渲、可勾选进训练的是什么。",
            "253 训练字体：除韩文外，汉字/ASCII/拉丁扩展/假名/注音/标点/符号均 253/253 有。",
            "韩文 Hangul：253/253 都没有 → 本盘不能做韩文 Target。",
            "数据集已渲：Target=P1∪P2（164）；Style 汉字=338 池。假名/注音字库更多，但未全部渲染。",
            "空格常无有效 GT，默认不勾选。",
        ],
        "groups": GROUPS,
        "bank_coverage": bank_coverage,
        "coverage_thresholds": coverage.get("summary")
        and {k: v["threshold_has"] for k, v in coverage["summary"].items()},
        "presets": presets,
        "fonts": fonts,
        "chars": chars,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "catalog.json"
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} bank_rows={len(bank_coverage)} fonts={len(fonts)} chars={len(chars)}")


if __name__ == "__main__":
    main()
