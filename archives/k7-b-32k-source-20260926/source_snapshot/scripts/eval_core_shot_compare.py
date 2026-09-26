#!/usr/bin/env python3
"""Core-model 1-shot vs 8-shot compare on frozen Demo-8 × curated chars.

Reuses existing f03_test16_strat preds when the protocol matches.
Generates only missing F-arm true-1-shot (Es+Δ from 永) and F3b 8/1-shot.

Demo-8 = original frozen test8 (not the 8 fonts later promoted from train).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))

import eval_f03_test16_strat as E  # noqa: E402

PRED = E.OUT / "preds"
BOARD = E.OUT / "core_shot_board.html"
MANIFEST = E.OUT / "core_shot_board.json"
REF8 = list("永和书风骨韵天地")
SEED = 3407

# Original test8 kept when v3 promoted 8 extra train fonts into test16.
DEMO8 = [
    "FZChuangHJW_DB",
    "FZCuanBZBKSJW",
    "FZDouNTJW_Te",
    "FZFengYKSJ",
    "FZHanWZKJW",
    "FZHuoYYJW-T",
    "FZJingYLLTJW",
    "FZLingFKSJW-B",
]

# Stratified 47 (already on disk for P1/E1/F0/F1/F2/F3 8-shot) plus extra
# style-heavy glyphs that usually show slant / weight / terminals / counters.
CHARS = [
    {"ch": "0", "bucket": "digit", "why": "字腔、字重"},
    {"ch": "1", "bucket": "digit", "why": "衬线/字重；与 l 易混"},
    {"ch": "2", "bucket": "digit", "why": "开尾、曲线"},
    {"ch": "3", "bucket": "digit", "why": "开尾、对比"},
    {"ch": "4", "bucket": "digit", "why": "开口/闭口"},
    {"ch": "5", "bucket": "digit", "why": "横折、字重"},
    {"ch": "6", "bucket": "digit", "why": "闭口字腔"},
    {"ch": "7", "bucket": "digit", "why": "横、斜势"},
    {"ch": "8", "bucket": "digit", "why": "粗细对比、双腔"},
    {"ch": "9", "bucket": "digit", "why": "字腔、收笔"},
    {"ch": "A", "bucket": "latin_upper", "why": "斜势、顶点"},
    {"ch": "B", "bucket": "latin_upper", "why": "双腔、字重"},
    {"ch": "C", "bucket": "latin_upper", "why": "开口、对比"},
    {"ch": "G", "bucket": "latin_upper", "why": "字距刺、字重"},
    {"ch": "K", "bucket": "latin_upper", "why": "对角、连接"},
    {"ch": "M", "bucket": "latin_upper", "why": "字宽、斜笔"},
    {"ch": "O", "bucket": "latin_upper", "why": "圆/方、字腔"},
    {"ch": "Q", "bucket": "latin_upper", "why": "尾巴，风格最明显"},
    {"ch": "R", "bucket": "latin_upper", "why": "腿部收笔"},
    {"ch": "S", "bucket": "latin_upper", "why": "双向曲线"},
    {"ch": "W", "bucket": "latin_upper", "why": "字宽、斜笔"},
    {"ch": "Y", "bucket": "latin_upper", "why": "分叉、端点"},
    {"ch": "Z", "bucket": "latin_upper", "why": "对角、字重"},
    {"ch": "a", "bucket": "latin_lower", "why": "字腔、单/双层"},
    {"ch": "b", "bucket": "latin_lower", "why": "升部、字腔"},
    {"ch": "c", "bucket": "latin_lower", "why": "开口"},
    {"ch": "d", "bucket": "latin_lower", "why": "升部、字腔"},
    {"ch": "e", "bucket": "latin_lower", "why": "横、字腔"},
    {"ch": "f", "bucket": "latin_lower", "why": "升部、横"},
    {"ch": "g", "bucket": "latin_lower", "why": "闭口/开口 g"},
    {"ch": "i", "bucket": "latin_lower", "why": "点、字身"},
    {"ch": "k", "bucket": "latin_lower", "why": "对角、连接"},
    {"ch": "l", "bucket": "latin_lower", "why": "身份探针，对 1"},
    {"ch": "n", "bucket": "latin_lower", "why": "衬线、字宽"},
    {"ch": "o", "bucket": "latin_lower", "why": "圆/方"},
    {"ch": "p", "bucket": "latin_lower", "why": "降部、字腔"},
    {"ch": "q", "bucket": "latin_lower", "why": "降部、字腔"},
    {"ch": "r", "bucket": "latin_lower", "why": "肩、短笔画"},
    {"ch": "s", "bucket": "latin_lower", "why": "双向曲线"},
    {"ch": "u", "bucket": "latin_lower", "why": "字宽、字腔"},
    {"ch": "y", "bucket": "latin_lower", "why": "降部、斜笔"},
    {"ch": "à", "bucket": "latin_ext", "why": "开音符"},
    {"ch": "é", "bucket": "latin_ext", "why": "闭音符 + e"},
    {"ch": "ü", "bucket": "latin_ext", "why": "分音符"},
    {"ch": "ā", "bucket": "latin_ext", "why": "长音符"},
    {"ch": "ě", "bucket": "latin_ext", "why": "上声符"},
    {"ch": "あ", "bucket": "hiragana", "why": "曲线、飞白"},
    {"ch": "か", "bucket": "hiragana", "why": "折、点"},
    {"ch": "き", "bucket": "hiragana", "why": "多横、连接"},
    {"ch": "さ", "bucket": "hiragana", "why": "断笔/连笔"},
    {"ch": "の", "bucket": "hiragana", "why": "环、收笔"},
    {"ch": "は", "bucket": "hiragana", "why": "竖、钩"},
    {"ch": "め", "bucket": "hiragana", "why": "环、内点"},
    {"ch": "ん", "bucket": "hiragana", "why": "端点/收笔"},
    {"ch": "ア", "bucket": "katakana", "why": "折角、字重"},
    {"ch": "カ", "bucket": "katakana", "why": "折、点"},
    {"ch": "シ", "bucket": "katakana", "why": "短斜笔"},
    {"ch": "ツ", "bucket": "katakana", "why": "短点方向"},
    {"ch": "ト", "bucket": "katakana", "why": "折角"},
    {"ch": "ン", "bucket": "katakana", "why": "短笔画端点"},
    {"ch": "ㄅ", "bucket": "bopomofo", "why": "钩、对比"},
    {"ch": "ㄆ", "bucket": "bopomofo", "why": "横、开口"},
    {"ch": "ㄇ", "bucket": "bopomofo", "why": "门形、字宽"},
    {"ch": "ㄋ", "bucket": "bopomofo", "why": "折、收笔"},
    {"ch": "ㄚ", "bucket": "bopomofo", "why": "开口、斜势"},
    {"ch": "ㄨ", "bucket": "bopomofo", "why": "叉、斜笔"},
    {"ch": "ㄩ", "bucket": "bopomofo", "why": "竖弯"},
]

# 1-shot-only models: native single Style image (永).
REUSE_S1 = [
    ("P1", "原版 FD", "s1"),
    ("E1_100k", "E1@100k", "s1"),
    ("F0_100k", "F0@100k", "s1"),
    ("F1_80000", "F1@80k", "s1"),
]
# Few-shot models: reuse 8-shot; generate true 1-shot (Es+Δ = 永).
REUSE_S8 = [
    ("F2_80000", "F2@80k", "s8"),
    ("F2P_40000", "F2-P@40k", "s8"),
    ("F2RL_40000", "F2-RL128@40k", "s8"),
    ("F2PRL_40000", "F2-PRL@40k", "s8"),
    ("F3_80k", "F3@80k", "s8"),
    ("F3bP_40000", "F3b-P@40k", "s8"),
]
GENERATE = [
    ("F2_80000_k1", "F2@80k", "s1"),
    ("F2P_40000_k1", "F2-P@40k", "s1"),
    ("F2RL_40000", "F2-RL128@40k", "s8"),
    ("F2RL_40000_k1", "F2-RL128@40k", "s1"),
    ("F2PRL_40000", "F2-PRL@40k", "s8"),
    ("F2PRL_40000_k1", "F2-PRL@40k", "s1"),
    ("F3_80k_k1", "F3@80k", "s1"),
    ("F3b_80000", "F3b@80k", "s8"),
    ("F3b_80000_k1", "F3b@80k", "s1"),
    ("F3bP_40000_k1", "F3b-P@40k", "s1"),
]

COL_S1 = [
    ("P1", "原版 FD"),
    ("E1_100k", "E1@100k"),
    ("F0_100k", "F0@100k"),
    ("F1_80000", "F1@80k"),
    ("F2_80000_k1", "F2@80k"),
    ("F2P_40000_k1", "F2-P@40k"),
    ("F2RL_40000_k1", "F2-RL@40k"),
    ("F2PRL_40000_k1", "F2-PRL@40k"),
    ("F3_80k_k1", "F3@80k"),
    ("F3b_80000_k1", "F3b@80k"),
    ("F3bP_40000_k1", "F3b-P@40k"),
]
COL_S8 = [
    ("F2_80000", "F2@80k"),
    ("F2P_40000", "F2-P@40k"),
    ("F2RL_40000", "F2-RL@40k"),
    ("F2PRL_40000", "F2-PRL@40k"),
    ("F3_80k", "F3@80k"),
    ("F3b_80000", "F3b@80k"),
    ("F3bP_40000", "F3b-P@40k"),
]
BOARD_METHODS = [m for m, _ in COL_S1] + [m for m, _ in COL_S8]
PROTO_KEEP = E.OUT / "PROTOCOL.json.keep"

FAMILIES = {
    "cond": {
        "label": "条件",
        "ids": ["content", "style1", "ref8", "gt"],
    },
    "baseline": {
        "label": "基线",
        "ids": ["P1", "E1_100k", "F0_100k", "F1_80000"],
    },
    "f2": {
        "label": "F2 族",
        "ids": [
            "F2_80000_k1",
            "F2P_40000_k1",
            "F2RL_40000_k1",
            "F2PRL_40000_k1",
            "F2_80000",
            "F2P_40000",
            "F2RL_40000",
            "F2PRL_40000",
        ],
    },
    "f3": {
        "label": "F3 族",
        "ids": [
            "F3_80k_k1",
            "F3b_80000_k1",
            "F3bP_40000_k1",
            "F3_80k",
            "F3b_80000",
            "F3bP_40000",
        ],
    },
}


def fam_of(mid: str) -> str:
    for name, spec in FAMILIES.items():
        if mid in spec["ids"]:
            return name
    return ""


def utc() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def pred_rel(mid: str, font: str, ch: str) -> str:
    return f"preds/{mid}/test/{font}/test__{font}__{cp_of(ch)}__s{SEED}.png"


def pred_abs(mid: str, font: str, ch: str) -> Path:
    return PRED / mid / "test" / font / f"test__{font}__{cp_of(ch)}__s{SEED}.png"


def apply_subset() -> None:
    E.STRATIFIED = [row["ch"] for row in CHARS]
    E.fonts = lambda: list(DEMO8)  # type: ignore[method-assign]


def stage_condition_images() -> None:
    data = E.DATA
    content_dir = E.OUT / "refs" / "content"
    gt_dir = E.OUT / "refs" / "gt"
    content_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for row in CHARS:
        ch, cp = row["ch"], cp_of(row["ch"])
        dst = content_dir / f"{cp}.png"
        if not dst.is_file():
            src = None
            for sp in ("test", "val", "train"):
                p = data / sp / "ContentImage" / f"{cp}.png"
                if p.is_file():
                    src = p
                    break
            if src is not None:
                Image.open(src).convert("RGB").save(dst)
        for font in DEMO8:
            gdst = gt_dir / f"{font}+{cp}.png"
            if gdst.is_file():
                continue
            gsrc = data / "test" / "TargetImage" / font / f"{font}+{cp}.png"
            if gsrc.is_file():
                Image.open(gsrc).convert("RGB").save(gdst)


def restore_protocol() -> None:
    proto = E.OUT / "PROTOCOL.json"
    if PROTO_KEEP.is_file():
        proto.write_text(PROTO_KEEP.read_text(encoding="utf-8"), encoding="utf-8")


def cmd_generate(args: argparse.Namespace) -> None:
    apply_subset()
    stage_condition_images()
    proto = E.OUT / "PROTOCOL.json"
    if proto.is_file() and not PROTO_KEEP.is_file():
        PROTO_KEEP.write_text(proto.read_text(encoding="utf-8"), encoding="utf-8")
    methods = args.methods
    if not methods:
        cov = {m["id"]: m for m in coverage()["methods"]}
        methods = [m for m in BOARD_METHODS if cov.get(m, {}).get("miss", 1)]
    try:
        for mid in methods:
            if mid not in E.METHODS:
                raise SystemExit(f"unknown method {mid}")
            spec = E.METHODS[mid]
            ns = argparse.Namespace(
                method=mid,
                device=args.device,
                shard="0/1",
                overwrite=args.overwrite,
            )
            print(
                f"== generate {mid} kind={spec['kind']} device={args.device} "
                f"fonts={len(DEMO8)} chars={len(CHARS)}"
            )
            E.cmd_generate(ns)
    finally:
        restore_protocol()


def coverage() -> dict:
    rows = []
    all_mids = list(dict.fromkeys(BOARD_METHODS))
    for mid in all_mids:
        ok = miss = 0
        missing = []
        for font in DEMO8:
            for row in CHARS:
                p = pred_abs(mid, font, row["ch"])
                if p.is_file():
                    ok += 1
                else:
                    miss += 1
                    if len(missing) < 6:
                        missing.append(f"{font}:{row['ch']}")
        rows.append({"id": mid, "ok": ok, "miss": miss, "total": ok + miss, "missing_ex": missing})
    return {"fonts": DEMO8, "chars": [r["ch"] for r in CHARS], "methods": rows}


def cmd_status(_: argparse.Namespace) -> None:
    print(json.dumps(coverage(), ensure_ascii=False, indent=2))


def build_manifest() -> dict:
    cov = coverage()
    items = []
    for font in DEMO8:
        for row in CHARS:
            ch = row["ch"]
            items.append(
                {
                    "font": font,
                    "char": ch,
                    "cp": cp_of(ch),
                    "bucket": row["bucket"],
                    "why": row["why"],
                    "content": f"refs/content/{cp_of(ch)}.png",
                    "gt": f"refs/gt/{font}+{cp_of(ch)}.png",
                    "style1": f"refs/style/{font}.png",
                    "style8": [f"refs/style_ref8/{font}/{cp_of(c)}.png" for c in REF8],
                    "preds": {
                        mid: pred_rel(mid, font, ch)
                        for mid, _ in COL_S1 + COL_S8
                    },
                }
            )
    payload = {
        "generated_at": utc(),
        "seed": SEED,
        "sampler": "dpmsolver++ 20 CFG7.5 order2",
        "demo8": DEMO8,
        "demo8_note": "v3 冻结的原 test8；不含后来从 train 升上来的 8 套",
        "ref8": "".join(REF8),
        "style1": "永",
        "chars": CHARS,
        "col_s1": [{"id": a, "label": b, "fam": fam_of(a)} for a, b in COL_S1],
        "col_s8": [{"id": a, "label": b, "fam": fam_of(a)} for a, b in COL_S8],
        "families": FAMILIES,
        "protocol": {
            "content": "同一张 Noto 协议 A ContentImage",
            "s1": "P1/E1/F0/F1 用 Style=永；F 臂 true-1-shot 为 Es+Δ 都用 永",
            "s8": "F2/F2-P/F2-RL128/F2-PRL/F3/F3b/F3b-P 用 ref8=永和书风骨韵天地（Es+Δ；RL/PRL 另加 local L）",
            "oneshot_only": ["P1", "E1_100k", "F0_100k", "F1_80000"],
            "not_reused": "F2/F3 历史 *_s1 仍是 Es=永 且 Δ=ref8，本表 1-shot 列不用它们",
        },
        "coverage": cov,
        "items": items,
        "n": len(items),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def write_html(payload: dict) -> None:
    html = f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>核心模型 1-shot / 8-shot 对照 · Demo-8</title>
<style>
:root {{
  --bg:#14120f; --card:#1c1914; --line:#3a342c; --fg:#e4dcc8;
  --muted:#8d8473; --acc:#b7c98a; --warn:#c8a24b; --head:#221e18;
  --base:#c8a24b; --base-bg:#2f2614; --base-head:#3d3118;
  --f2:#6aa8a0; --f2-bg:#141c1b; --f2-head:#1a2a28;
  --f3:#9b8fd4; --f3-bg:#1b1724; --f3-head:#262033;
  --cond-bg:#1a1814;
}}
* {{ box-sizing:border-box; }}
html, body {{ height:100%; }}
body {{
  margin:0; display:flex; flex-direction:column; overflow:hidden;
  font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--fg);
}}
header {{
  flex:0 0 auto; z-index:30; background:var(--card);
  border-bottom:1px solid var(--line); padding:10px 16px 8px;
}}
h1 {{ font-size:1.05rem; margin:0 0 4px; font-weight:650; }}
.meta {{ color:var(--muted); font-size:12px; max-width:110ch; line-height:1.45; }}
.bar {{ display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; margin-top:8px; font-size:13px; }}
.bar label {{ cursor:pointer; color:var(--fg); }}
select, button {{ background:var(--head); color:var(--fg); border:1px solid var(--line); padding:4px 8px; font:inherit; }}
button:hover {{ border-color:var(--acc); color:var(--acc); }}
.legs {{ display:flex; gap:10px; align-items:center; margin-left:6px; }}
.leg {{ display:inline-flex; align-items:center; gap:5px; color:var(--muted); font-size:12px; }}
.leg i {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
.leg.baseline i {{ background:var(--base); }}
.leg.f2 i {{ background:var(--f2); }}
.leg.f3 i {{ background:var(--f3); }}
main {{ flex:1; min-height:0; display:flex; flex-direction:column; padding:8px 16px 12px; }}
.note {{ flex:0 0 auto; color:var(--muted); font-size:12px; margin:0 0 8px; max-width:120ch; }}
.scroll {{
  flex:1; min-height:0; overflow:auto; border:1px solid var(--line); background:#17140f;
}}
table {{ border-collapse:separate; border-spacing:0; font-size:11px; }}
th, td {{
  border-right:1px solid var(--line); border-bottom:1px solid var(--line);
  padding:3px 4px; text-align:center; vertical-align:middle;
}}
thead th {{
  position:sticky; top:0; z-index:6; background:var(--head);
  white-space:nowrap; padding:6px 5px 7px;
}}
thead th.stub {{ left:0; z-index:9; background:var(--head); }}
th .shot {{ display:block; font-size:10px; font-weight:600; letter-spacing:.02em; opacity:.85; }}
th.fam-baseline, td.fam-baseline {{ background:var(--base-bg); }}
thead th.fam-baseline {{ background:var(--base-head); color:#f0d078; box-shadow:inset 0 -3px 0 var(--base); }}
th.fam-f2, td.fam-f2 {{ background:var(--f2-bg); }}
thead th.fam-f2 {{ background:var(--f2-head); color:#b7e4de; box-shadow:inset 0 -3px 0 var(--f2); }}
th.fam-f3, td.fam-f3 {{ background:var(--f3-bg); }}
thead th.fam-f3 {{ background:var(--f3-head); color:#d4c8ff; box-shadow:inset 0 -3px 0 var(--f3); }}
th.fam-cond, td.fam-cond {{ background:var(--cond-bg); }}
thead th.fam-cond {{ color:var(--muted); }}
td.chr {{
  position:sticky; left:0; z-index:3; background:var(--card);
  font-size:16px; font-weight:700; min-width:2.4em;
  box-shadow:1px 0 0 var(--line);
}}
.why {{ color:var(--muted); font-size:10px; font-weight:400; max-width:9em; }}
img.g {{ width:56px; height:56px; image-rendering:pixelated; background:#fff; display:block; }}
.refs {{ display:flex; gap:1px; }}
.refs img {{ width:22px; height:22px; image-rendering:pixelated; background:#fff; }}
.miss {{ color:#666; font-size:18px; }}
code {{ color:var(--warn); }}
td.hl-cell {{ outline:2px solid #f0d078; outline-offset:-2px; }}
td.fam-baseline.hl-cell {{ outline-color:var(--base); background:#5a4518; }}
td.fam-f2.hl-cell {{ outline-color:var(--f2); background:#244844; }}
td.fam-f3.hl-cell {{ outline-color:var(--f3); background:#3a3060; }}
td.fam-cond.hl-cell {{ outline-color:#d0c4a0; background:#3a3428; }}
th.hl-col {{ filter:brightness(1.25); }}
</style></head><body>
<header>
<h1>核心模型对照 · Demo-8 × {len(CHARS)} 字 · 1-shot / 8-shot</h1>
<p class="meta">同一 Content、同一套 Style 图、seed 3407、DPM++20 CFG7.5。
原版 FD / E1 / F0 / F1 只做 1-shot（永）。
F2 / F2-P / F2-RL128 / F2-PRL / F3 / F3b / F3b-P 同时做 1-shot（Es+Δ 都用永）和 8-shot（ref8）。
GT 只作 positive control，不是天花板。</p>
<div class="bar">
<label>视图 <select id="viewSel"><option value="font">按字体（字为行）</option><option value="char">按字（字体为行）</option></select></label>
<label>字体 <select id="fontSel"></select></label>
<label>字 <select id="charSel"></select></label>
<label><input type="checkbox" id="s1" checked> 1-shot</label>
<label><input type="checkbox" id="s8" checked> 8-shot</label>
<span class="legs">
  <span class="leg baseline"><i></i>基线 P1/E1/F0/F1</span>
  <span class="leg f2"><i></i>F2 族（含 RL128）</span>
  <span class="leg f3"><i></i>F3 族</span>
</span>
<span id="bk"></span>
<button id="allBk">全选桶</button>
<button type="button" id="noneBk">清空桶</button>
</div>
</header>
<main>
<p class="note">悬停一格会点亮<strong>同行同族</strong>（F3 含 F3 / F3b / F3b-P 的 1-shot 与 8-shot）。表头随表格滚动钉住。
身份字看 <b>l vs 1</b>；风格字优先看 <b>Q / K / g / き / シ / ㄅ</b>。</p>
<div class="scroll"><table id="sheet"></table></div>
</main>
<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};
const fontSel = document.getElementById('fontSel');
DATA.demo8.forEach(f => {{
  const o = document.createElement('option'); o.value=f; o.textContent=f; fontSel.appendChild(o);
}});
const charSel = document.getElementById('charSel');
DATA.chars.forEach(c => {{
  const o = document.createElement('option'); o.value=c.ch; o.textContent=c.ch+' · '+c.why; charSel.appendChild(o);
}});
const viewSel = document.getElementById('viewSel');
viewSel.onchange = render;
charSel.onchange = render;
const buckets = [...new Set(DATA.chars.map(c => c.bucket))];
const bkBox = document.getElementById('bk');
buckets.forEach(b => {{
  const l = document.createElement('label');
  l.innerHTML = `<input type="checkbox" class="bk" value="${{b}}" checked> ${{b}}`;
  bkBox.appendChild(l);
}});
function onBk() {{ return new Set([...document.querySelectorAll('.bk:checked')].map(x => x.value)); }}
document.getElementById('allBk').onclick = () => {{ document.querySelectorAll('.bk').forEach(x => x.checked=true); render(); }};
document.getElementById('noneBk').onclick = () => {{ document.querySelectorAll('.bk').forEach(x => x.checked=false); render(); }};
document.getElementById('s1').onchange = render;
document.getElementById('s8').onchange = render;
fontSel.onchange = render;
document.querySelectorAll('.bk').forEach(x => x.addEventListener('change', render));

function img(src) {{
  return `<img class="g" loading="lazy" src="${{src}}" onerror="this.parentNode.innerHTML='<span class=miss>∅</span>'">`;
}}
function cell(src, fam, id, label) {{
  return `<td class="fam-${{fam}}" data-fam="${{fam}}" data-id="${{id}}" title="${{label}}">${{img(src)}}</td>`;
}}
function colTh(c, shot) {{
  return `<th class="fam-${{c.fam}}" data-fam="${{c.fam}}" data-id="${{c.id}}" title="${{c.label}} · ${{shot}}"><span class="shot">${{shot}}</span>${{c.label}}</th>`;
}}
function condTh(id, label) {{
  return `<th class="fam-cond" data-fam="cond" data-id="${{id}}">${{label}}</th>`;
}}

const sheet = document.getElementById('sheet');

function clearHl() {{
  sheet.querySelectorAll('.hl-cell,.hl-col').forEach(el => el.classList.remove('hl-cell','hl-col'));
}}
function paintFam(fam, row) {{
  clearHl();
  if (!fam) return;
  sheet.querySelectorAll('[data-fam="' + fam + '"]').forEach(n => {{
    if (n.tagName === 'TH') n.classList.add('hl-col');
    else if (!row || n.parentElement === row) n.classList.add('hl-cell');
  }});
}}
sheet.addEventListener('pointerover', e => {{
  const el = e.target.closest('[data-fam]');
  if (!el) return;
  const row = el.closest('tbody tr');
  paintFam(el.dataset.fam, row);
}});
sheet.addEventListener('pointerleave', clearHl);

function render() {{
  const font = fontSel.value;
  const ch = charSel.value;
  const byFont = viewSel.value === 'font';
  const show1 = document.getElementById('s1').checked;
  const show8 = document.getElementById('s8').checked;
  const bk = onBk();
  const s1 = DATA.col_s1, s8 = DATA.col_s8;
  const stub = byFont ? '字' : '字体';
  let head = `<tr><th class="stub">${{stub}}</th>`;
  head += condTh('content','Content') + condTh('style1','永') + condTh('ref8','ref8') + condTh('gt','GT');
  if (show1) s1.forEach(c => head += colTh(c, '1-shot'));
  if (show8) s8.forEach(c => head += colTh(c, '8-shot'));
  head += `</tr>`;
  const rows = DATA.items.filter(it => byFont
    ? (it.font===font && bk.has(it.bucket))
    : (it.char===ch));
  let body = '';
  for (const it of rows) {{
    const ref8 = `<div class="refs">${{it.style8.map(p => `<img src="${{p}}" loading="lazy">`).join('')}}</div>`;
    const lab = byFont ? `${{it.char}}<div class="why">${{it.why}}</div>` : it.font;
    body += `<tr><td class="chr">${{lab}}</td>`;
    body += cell(it.content, 'cond', 'content', 'Content');
    body += cell(it.style1, 'cond', 'style1', '永');
    body += `<td class="fam-cond" data-fam="cond" data-id="ref8" title="ref8">${{ref8}}</td>`;
    body += cell(it.gt, 'cond', 'gt', 'GT');
    if (show1) s1.forEach(c => body += cell(it.preds[c.id], c.fam, c.id, c.label + ' · 1-shot'));
    if (show8) s8.forEach(c => body += cell(it.preds[c.id], c.fam, c.id, c.label + ' · 8-shot'));
    body += `</tr>`;
  }}
  sheet.innerHTML = `<thead>${{head}}</thead><tbody>${{body}}</tbody>`;
}}
render();
</script>
</body></html>
"""
    BOARD.write_text(html, encoding="utf-8")
    print("wrote", BOARD, "items", payload["n"])


def cmd_board(_: argparse.Namespace) -> None:
    stage_condition_images()
    payload = build_manifest()
    write_html(payload)
    miss = [m for m in payload["coverage"]["methods"] if m["miss"]]
    if miss:
        print("INCOMPLETE:")
        for m in miss:
            print(f"  {m['id']}: {m['ok']}/{m['total']} missing={m['missing_ex']}")
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--method", dest="methods", action="append")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--overwrite", action="store_true")
    g.set_defaults(func=cmd_generate)
    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)
    b = sub.add_parser("board")
    b.set_defaults(func=cmd_board)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
