#!/usr/bin/env python3
"""G-series 1-shot / 8-shot board on v0913 test16×stratified47, vs dirty F2/F2-RL.

Protocol matches reports/f03_test16_strat (same 16 test fonts + STRATIFIED chars).
G0b/G0c are image-conditioned (native 1-style); only oneshot columns are generated.
F123 arms use artifacts/g0 caches. Dirty F2/F2-RL preds are hard-linked from f03.

Writes reports/g_v0913_shot/{preds,index.html,metrics.json,PROTOCOL.json}.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))
import eval_f03_test16_strat as E  # noqa: E402

OUT = ROOT / "reports/g_v0913_shot"
E.OUT = OUT

F123 = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
F0V = ROOT / "code/variants/cn2west_f0_rsifree/FontDiffuser"
G0_ES = ROOT / "artifacts/g0/es_spatial"
G0_EC = ROOT / "artifacts/g0/ec_multiscale"
G0_LOCAL = ROOT / "artifacts/g0/es_local"
F03_PREDS = ROOT / "reports/f03_test16_strat/preds"

# Display / generate order for the board.
BOARD_COLS = [
    ("GT", "GT", "对照"),
    ("G0b_s1", "G0b@10k", "1-shot"),
    ("G0c_s1", "G0c", "1-shot"),
    ("G1_s1", "G1", "1-shot"),
    ("G1_s8", "G1", "8-shot"),
    ("G2_s1", "G2", "1-shot"),
    ("G2_s8", "G2", "8-shot"),
    ("G2RL_s1", "G2-RL", "1-shot"),
    ("G2RL_s8", "G2-RL", "8-shot"),
    ("pilot_s1", "pilot@best1k", "1-shot"),
    ("pilot_s8", "pilot@best1k", "8-shot"),
    ("F2_80000_k1", "dirty F2@80k", "1-shot"),
    ("F2_80000", "dirty F2@80k", "8-shot"),
    ("F2RL_40000_k1", "dirty F2-RL@40k", "1-shot"),
    ("F2RL_40000", "dirty F2-RL@40k", "8-shot"),
]

CURATED = {
    "0", "2", "8", "A", "G", "Q", "R", "a", "g", "e", "à", "ě", "あ", "さ", "ア", "ㄅ", "ㄚ",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pick_g0c_ckpt() -> Path:
    run = ROOT / "runs/G0c-F0-V0913-BS256-A-S3407"
    done = run / "DONE.json"
    if done.is_file():
        return run / "global_step_20000"
    steps = sorted(
        (p for p in run.glob("global_step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not steps:
        raise FileNotFoundError(f"no G0c ckpt under {run}")
    return steps[-1]


def build_methods() -> dict:
    g0c_ckpt = pick_g0c_ckpt()
    g0c_step = g0c_ckpt.name.split("_")[-1]
    return {
        "G0b_s1": {
            "label": "G0b@10k",
            "kind": "image",
            "variant": F0V,
            "ckpt": ROOT / "runs/G0b-F0-V0913-BS256-A-S3407/global_step_10000",
            "style_oneshot": True,
        },
        "G0c_s1": {
            "label": f"G0c@{g0c_step}",
            "kind": "image",
            "variant": F0V,
            "ckpt": g0c_ckpt,
            "style_oneshot": True,
        },
        "G1_s1": {
            "label": "G1@10k 1-shot",
            "kind": "f1",
            "variant": F123,
            "ckpt": ROOT / "runs/G1-F1-V0913-A-S3407/global_step_10000",
            "style_oneshot": True,
            "es_cache": G0_ES,
            "ec_cache": G0_EC,
        },
        "G1_s8": {
            "label": "G1@10k 8-shot",
            "kind": "f1",
            "variant": F123,
            "ckpt": ROOT / "runs/G1-F1-V0913-A-S3407/global_step_10000",
            "style_oneshot": False,
            "es_cache": G0_ES,
            "ec_cache": G0_EC,
        },
        "G2_s1": {
            "label": "G2@10k 1-shot",
            "kind": "f2",
            "variant": F123,
            "ckpt": ROOT / "runs/G2-F2-V0913-A-S3407/global_step_10000",
            "style_oneshot": True,
            "delta_oneshot": True,
            "es_cache": G0_ES,
            "ec_cache": G0_EC,
        },
        "G2_s8": {
            "label": "G2@10k 8-shot",
            "kind": "f2",
            "variant": F123,
            "ckpt": ROOT / "runs/G2-F2-V0913-A-S3407/global_step_10000",
            "style_oneshot": False,
            "es_cache": G0_ES,
            "ec_cache": G0_EC,
        },
        "G2RL_s1": {
            "label": "G2-RL@10k 1-shot",
            "kind": "f2",
            "variant": F123,
            "ckpt": ROOT / "runs/G2-RL-V0913-A-S3407/global_step_10000",
            "style_oneshot": True,
            "delta_oneshot": True,
            "style_rl128": True,
            "es_cache": G0_ES,
            "ec_cache": G0_EC,
            "es_local_cache": G0_LOCAL,
        },
        "G2RL_s8": {
            "label": "G2-RL@10k 8-shot",
            "kind": "f2",
            "variant": F123,
            "ckpt": ROOT / "runs/G2-RL-V0913-A-S3407/global_step_10000",
            "style_oneshot": False,
            "style_rl128": True,
            "es_cache": G0_ES,
            "ec_cache": G0_EC,
            "es_local_cache": G0_LOCAL,
        },
        "pilot_s1": {
            "label": "pilot@best1k 1-shot",
            "kind": "f2",
            "variant": F123,
            "ckpt": ROOT / "runs/G-RL-pilot-V0913-A-S3407/global_step_1000",
            "style_oneshot": True,
            "delta_oneshot": True,
            "style_rl128": True,
            "es_cache": G0_ES,
            "ec_cache": G0_EC,
            "es_local_cache": G0_LOCAL,
        },
        "pilot_s8": {
            "label": "pilot@best1k 8-shot",
            "kind": "f2",
            "variant": F123,
            "ckpt": ROOT / "runs/G-RL-pilot-V0913-A-S3407/global_step_1000",
            "style_oneshot": False,
            "style_rl128": True,
            "es_cache": G0_ES,
            "ec_cache": G0_EC,
            "es_local_cache": G0_LOCAL,
        },
        # Dirty baselines: reuse f03 preds (linked). Prefer *_k1 = true 1-shot
        # (Es+Δ both 永). Do NOT use *_s1 Mode D (Es=永, Δ=ref8).
        "F2_80000": {"label": "dirty F2@80k 8-shot", "kind": "reuse", "reuse_from": "F2_80000"},
        "F2_80000_k1": {"label": "dirty F2@80k 1-shot", "kind": "reuse", "reuse_from": "F2_80000_k1"},
        "F2RL_40000": {"label": "dirty F2-RL@40k 8-shot", "kind": "reuse", "reuse_from": "F2RL_40000"},
        "F2RL_40000_k1": {"label": "dirty F2-RL@40k 1-shot", "kind": "reuse", "reuse_from": "F2RL_40000_k1"},
    }


def link_tree(src: Path, dst: Path) -> int:
    n = 0
    if not src.is_dir():
        print(f"WARN missing reuse source {src}", flush=True)
        return 0
    for png in src.rglob("*.png"):
        rel = png.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        meta_src = png.with_suffix(".json")
        meta_dst = out.with_suffix(".json")
        for s, d in ((png, out), (meta_src, meta_dst)):
            if not s.is_file():
                continue
            if d.exists() or d.is_symlink():
                continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
            n += 1
    return n


def generate_one(mid: str, device: str, overwrite: bool) -> None:
    spec = E.METHODS[mid]
    stems = E.fonts()
    kind = spec["kind"]
    if kind == "reuse":
        n = link_tree(F03_PREDS / spec["reuse_from"], OUT / "preds" / mid)
        print(json.dumps({"method": mid, "phase": "reuse", "linked_files": n}), flush=True)
        return
    if kind == "image":
        E.generate_image_method(mid, device, stems, overwrite)
    elif kind == "f1":
        E.generate_f1(device, stems, overwrite, mid=mid)
    elif kind == "f2":
        E.generate_f2(device, stems, overwrite, mid=mid)
    else:
        raise SystemExit(f"unknown kind {kind} for {mid}")


def compute_metrics(mids: list[str]) -> dict:
    rows = []
    for mid in mids:
        for png in (OUT / "preds" / mid).rglob("*.png"):
            meta_p = png.with_suffix(".json")
            if meta_p.is_file():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
            else:
                # reuse trees sometimes miss sidecar; parse path
                parts = png.parts
                # .../preds/MID/test/FONT/test__FONT__CP__s3407.png
                try:
                    font = parts[-2]
                    name = png.stem  # test__FONT__uXXXX__s3407
                    cp = [x for x in name.split("__") if x.startswith("u")][0]
                    ch = chr(int(cp[1:], 16))
                    meta = {"method": mid, "font": font, "char": ch, "cp": cp, "bucket": E.script_bucket(ch)}
                except Exception:
                    continue
            gtp = E.gt_path(meta["font"], meta["char"])
            if gtp is None:
                continue
            pred = Image.open(png).convert("RGB")
            gt = Image.open(gtp).convert("RGB")
            pa, ga = E.to_gray01(pred), E.to_gray01(gt)
            rows.append(
                {
                    **meta,
                    "method": mid,
                    "L1": float(abs(pa - ga).mean()),
                    "SSIM": E.ssim(pa, ga),
                    "coverage": E.ink_coverage(pa),
                    "rel": str(png.relative_to(OUT)),
                }
            )
    by_m: dict[str, list] = {m: [] for m in mids}
    for r in rows:
        by_m.setdefault(r["method"], []).append(r)
    methods = {}
    for mid in mids:
        label = E.METHODS.get(mid, {}).get("label", mid)
        methods[mid] = {"label": label, "overall": E.agg(by_m.get(mid, []))}
    report = {
        "computed_at": utc_now(),
        "protocol": "v0913 test16 × stratified47 · DPM++20 CFG7.5 seed3407 · G caches for G* ; dirty F2/F2RL linked",
        "fonts": E.fonts(),
        "chars": E.STRATIFIED,
        "methods": methods,
        "n_items": len(rows),
        "caveat": "L1/SSIM vs GT are diagnostics only. Dirty F2/F2-RL use old F0 caches; G* use G0b caches.",
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


# Related-column families for hover highlight (same row).
COL_FAMILIES = {
    "REF1": "ref",
    "REF8": "ref",
    "GT": "gt",
    "G0b_s1": "g0",
    "G0c_s1": "g0",
    "G1_s1": "g1",
    "G1_s8": "g1",
    "G2_s1": "g2",
    "G2_s8": "g2",
    "G2RL_s1": "g2",
    "G2RL_s8": "g2",
    "pilot_s1": "g2",
    "pilot_s8": "g2",
    "F2_80000_k1": "dirty_f2",
    "F2_80000": "dirty_f2",
    "F2RL_40000_k1": "dirty_f2rl",
    "F2RL_40000": "dirty_f2rl",
}

# Filter chips: model group id → column mids.
# GT defaults unchecked (blind compare); Ref defaults on.
MODEL_FILTERS = [
    ("ref", "Ref", ["REF1", "REF8"]),
    ("gt", "GT", ["GT"]),
    ("g0", "G0b/G0c", ["G0b_s1", "G0c_s1"]),
    ("g1", "G1", ["G1_s1", "G1_s8"]),
    ("g2", "G2", ["G2_s1", "G2_s8"]),
    ("g2rl", "G2-RL", ["G2RL_s1", "G2RL_s8"]),
    ("pilot", "pilot", ["pilot_s1", "pilot_s8"]),
    ("df2", "dirty F2", ["F2_80000_k1", "F2_80000"]),
    ("df2rl", "dirty F2-RL", ["F2RL_40000_k1", "F2RL_40000"]),
]

FILTER_DEFAULT_OFF = {"gt"}


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def _ensure_ref_assets(stem: str) -> tuple[str | None, list[str]]:
    """Materialize 1-shot (永) + 8-shot style refs under reports/.../refs/."""
    style_dir = E.DATA / "test" / "StyleImage" / stem
    ref1_rel = None
    ch1 = "永"
    src1 = style_dir / f"{stem}+{E.cp_of(ch1)}.png"
    if src1.is_file():
        dst1 = OUT / "refs" / stem / f"{stem}+{E.cp_of(ch1)}.png"
        _link_or_copy(src1, dst1)
        ref1_rel = f"refs/{stem}/{stem}+{E.cp_of(ch1)}.png"
    ref8_rels = []
    for ch in E.REF8:
        src = style_dir / f"{stem}+{E.cp_of(ch)}.png"
        if not src.is_file():
            continue
        dst = OUT / "refs" / stem / f"{stem}+{E.cp_of(ch)}.png"
        _link_or_copy(src, dst)
        ref8_rels.append(f"refs/{stem}/{stem}+{E.cp_of(ch)}.png")
    return ref1_rel, ref8_rels


def write_html(report: dict) -> None:
    fonts = E.fonts()
    # Prefer curated chars that exist in STRATIFIED; fall back to all.
    chars = [c for c in E.STRATIFIED if c in CURATED] or list(E.STRATIFIED)

    # Display columns: Ref1 / Ref8 / GT / methods...
    display_cols = [
        ("REF1", "Ref1", "永"),
        ("REF8", "Ref8", "8字"),
        *BOARD_COLS,
    ]

    def col_attrs(mid: str, lab: str, shot: str) -> str:
        fam = COL_FAMILIES.get(mid, "other")
        if mid in {"REF1", "REF8", "GT"}:
            shot_key = "na"
        elif shot.startswith("1"):
            shot_key = "1"
        elif "8" in shot:
            shot_key = "8"
        else:
            shot_key = "na"
        model_key = next((fid for fid, _, mids in MODEL_FILTERS if mid in mids), fam)
        return (
            f"data-col='{mid}' data-family='{fam}' data-model='{model_key}' "
            f"data-shot='{shot_key}' data-label='{lab}'"
        )

    head = "".join(
        f"<th class='col-head' {col_attrs(mid, lab, shot)}>"
        f"<div class='ml'>{lab}</div><div class='ms'>{shot}</div></th>"
        for mid, lab, shot in display_cols
    )
    # summary table
    sum_rows = []
    for mid, lab, shot in BOARD_COLS:
        if mid == "GT":
            continue
        o = report["methods"].get(mid, {}).get("overall", {})
        n = o.get("n") or 0
        l1 = o.get("L1_mean")
        ss = o.get("SSIM_mean")
        fam = COL_FAMILIES.get(mid, "other")
        shot_key = "1" if shot.startswith("1") else "8"
        model_key = next((fid for fid, _, mids in MODEL_FILTERS if mid in mids), fam)
        attrs = f"data-model='{model_key}' data-shot='{shot_key}' data-family='{fam}'"
        sum_rows.append(
            f"<tr class='sum-row' {attrs}><td>{lab}</td><td>{shot}</td><td>{n}</td>"
            f"<td>{l1:.4f}</td><td>{ss:.4f}</td></tr>"
            if l1 is not None
            else f"<tr class='sum-row' {attrs}><td>{lab}</td><td>{shot}</td><td>{n}</td><td>—</td><td>—</td></tr>"
        )

    model_chips = "".join(
        "<label class='chip'>"
        f"<input type='checkbox' class='flt-model' value='{fid}'"
        f"{'' if fid in FILTER_DEFAULT_OFF else ' checked'}> {lab}</label>"
        for fid, lab, _ in MODEL_FILTERS
    )
    sections = []
    for stem in fonts:
        ref1_rel, ref8_rels = _ensure_ref_assets(stem)
        blocks = [
            f"<h2 id='{stem}'>{stem}</h2>",
            "<div class='tbl-wrap'><table class='board'>",
            f"<thead><tr><th class='sticky-char'>char</th>{head}</tr></thead><tbody>",
        ]
        for ch in chars:
            tds = [f"<td class='meta sticky-char'>{ch}</td>"]
            # Ref1
            if ref1_rel:
                tds.append(
                    f"<td class='cell refcell' {col_attrs('REF1', 'Ref1', '永')}>"
                    f"<img src='{ref1_rel}' alt='ref1'></td>"
                )
            else:
                tds.append(f"<td class='cell miss' {col_attrs('REF1', 'Ref1', '永')}>—</td>")
            # Ref8 strip
            if ref8_rels:
                imgs = "".join(f"<img class='r8' src='{r}' alt='r'>" for r in ref8_rels)
                tds.append(
                    f"<td class='cell refcell ref8' {col_attrs('REF8', 'Ref8', '8字')}>"
                    f"<div class='r8wrap'>{imgs}</div></td>"
                )
            else:
                tds.append(f"<td class='cell miss' {col_attrs('REF8', 'Ref8', '8字')}>—</td>")
            # GT
            gtp = E.gt_path(stem, ch)
            mid_gt, lab_gt, shot_gt = BOARD_COLS[0]
            if gtp and gtp.is_file():
                gdst = OUT / "gt" / stem / f"{stem}+{E.cp_of(ch)}.png"
                _link_or_copy(gtp, gdst)
                tds.append(
                    f"<td class='cell' {col_attrs(mid_gt, lab_gt, shot_gt)}>"
                    f"<img src='gt/{stem}/{stem}+{E.cp_of(ch)}.png'></td>"
                )
            else:
                tds.append(f"<td class='cell' {col_attrs(mid_gt, lab_gt, shot_gt)}></td>")
            for mid, lab, shot in BOARD_COLS[1:]:
                rel = f"preds/{mid}/test/{stem}/test__{stem}__{E.cp_of(ch)}__s{E.SEED}.png"
                p = OUT / rel
                attrs = col_attrs(mid, lab, shot)
                if p.is_file():
                    tds.append(f"<td class='cell' {attrs}><img src='{rel}' alt='{mid}'></td>")
                else:
                    tds.append(f"<td class='cell miss' {attrs}>—</td>")
            blocks.append("<tr>" + "".join(tds) + "</tr>")
        blocks.append("</tbody></table></div>")
        sections.append("\n".join(blocks))

    nav = " · ".join(f"<a href='#{f}'>{f}</a>" for f in fonts)
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>G v0913 1/8-shot · vs dirty F2 / F2-RL</title>
<style>
:root{{--bg:#111;--line:#333;--accent:#5af;--hl:#1e3a4f;--hl-strong:#2a5570;--dim:.22;--ctrl-h:120px}}
*{{box-sizing:border-box}}
html{{overflow-x:auto}}
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;padding:20px;padding-top:calc(var(--ctrl-h) + 10px);background:var(--bg);color:#eee}}
.ctrl{{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(17,17,17,.97);border-bottom:1px solid #345;
  padding:10px 16px 12px;backdrop-filter:blur(6px)}}
.ctrl h1{{font-size:16px;margin:0 0 8px;font-weight:600}}
.ctrl .row{{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:4px 0;font-size:13px}}
.chip{{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border:1px solid #456;border-radius:4px;background:#16202a;cursor:pointer;user-select:none}}
.chip:has(input:not(:checked)){{opacity:.45;border-color:#333}}
.chip.gt-chip{{border-color:#a65}}
.chip input{{margin:0}}
.hint{{color:#888;font-size:12px}}
/* IMPORTANT: do NOT set overflow-x on .tbl-wrap — it breaks position:sticky for thead */
.tbl-wrap{{margin:8px 0 18px;border:1px solid #2a2a2a;display:inline-block;min-width:100%}}
table.board{{border-collapse:separate;border-spacing:0;margin:0;background:#111}}
table.board th,table.board td{{border:1px solid var(--line);padding:4px;vertical-align:middle;text-align:center;background:#111}}
table.board thead th{{position:sticky;top:var(--ctrl-h);z-index:20;background:#1c2834;box-shadow:0 2px 0 #456}}
table.board .sticky-char{{position:sticky;left:0;z-index:15;background:#161616;min-width:36px}}
table.board thead .sticky-char{{z-index:25;left:0;top:var(--ctrl-h);background:#1c2834}}
table.board img{{width:72px;height:72px;image-rendering:pixelated;background:#1a1a1a;display:block;margin:0 auto}}
/* Ref8: must beat `table.board img` specificity or 72px tiles overlap in the grid */
table.board td.ref8{{min-width:92px;max-width:100px;padding:3px}}
table.board .r8wrap{{display:grid;grid-template-columns:repeat(4,20px);grid-template-rows:repeat(2,20px);gap:2px;justify-content:center;align-content:center;width:86px;margin:0 auto;overflow:hidden}}
table.board .r8wrap img.r8,table.board img.r8{{width:20px;height:20px;max-width:20px;max-height:20px;margin:0;padding:0;object-fit:contain}}
.meta{{font-size:12px;color:#aaa}} .ml{{font-size:11px;white-space:nowrap}} .ms{{font-size:10px;color:#9ab}}
.miss{{color:#666;font-size:12px}} h2{{font-size:16px;margin:28px 0 8px}}
.warn{{color:#fd6}} .note{{color:#aaa;font-size:13px;line-height:1.45}} a{{color:#9cf}}
.sum{{border-collapse:collapse;margin:10px 0}}
.sum td,.sum th{{font-size:12px;padding:6px 8px;border:1px solid #333}}
body.hide-shot-1 [data-shot="1"],
body.hide-shot-8 [data-shot="8"]{{display:none !important}}
body.hide-model-ref [data-model="ref"],
body.hide-model-gt [data-model="gt"],
body.hide-model-g0 [data-model="g0"],
body.hide-model-g1 [data-model="g1"],
body.hide-model-g2 [data-model="g2"],
body.hide-model-g2rl [data-model="g2rl"],
body.hide-model-pilot [data-model="pilot"],
body.hide-model-df2 [data-model="df2"],
body.hide-model-df2rl [data-model="df2rl"]{{display:none !important}}
body.hl-on table.board tbody tr.is-hover td.cell{{opacity:var(--dim);transition:opacity .08s,background .08s,outline .08s}}
body.hl-on table.board tbody tr.is-hover td.cell.rel{{opacity:1;background:var(--hl)}}
body.hl-on table.board tbody tr.is-hover td.cell.rel-self{{opacity:1;background:var(--hl-strong);outline:2px solid var(--accent);outline-offset:-2px}}
body.hl-on table.board thead th.rel{{background:#24364a}}
body.hl-on table.board thead th.rel-self{{background:#2f5a78;outline:2px solid var(--accent);outline-offset:-2px}}
#hover-tag{{display:none;margin-left:auto;color:#9cf;font-size:12px}}
#hover-tag.on{{display:inline}}
</style>
<div class="ctrl" id="ctrl">
  <h1>G 系 v0913 · one / few-shot 看板</h1>
  <div class="row"><span class="hint">列</span>{model_chips}
    <button type="button" id="btn-all" class="chip">全选</button>
    <button type="button" id="btn-none" class="chip">清空</button>
    <button type="button" id="btn-hide-gt" class="chip gt-chip">隐藏 GT</button>
    <span id="hover-tag"></span>
  </div>
  <div class="row"><span class="hint">Shot</span>
    <label class="chip"><input type="checkbox" class="flt-shot" value="1" checked> 1-shot</label>
    <label class="chip"><input type="checkbox" class="flt-shot" value="8" checked> 8-shot</label>
    <span class="hint">表头 sticky（滚动保持可见）。Ref1=永；Ref8=永和书风骨韵天地。GT 默认隐藏，勾选后显示。</span>
  </div>
</div>
<p class="note">{report['protocol']}<br>computed {report['computed_at']}</p>
<p class="note">评测协议：<b>test16 × stratified47</b>（与 F 系 <code>f03_test16_strat</code> 相同 16 个 test 字体 + 同一套分层字符；本页展示 curated 子集）。
训练侧 G 用 <code>v0913_clean</code> 过滤对；看板 PNG 仍来自同一 <code>fontdiffuser-…-v2</code> test 根目录。dirty F2/F2-RL 直接复用 f03 预测。</p>
<p class="warn">{report['caveat']}<br>
G0b/G0c 为单 style 图条件，仅 1-shot。dirty F2/F2-RL 对照列用 true 1-shot（*_k1：Es+Δ=永），不是 Mode D（*_s1：Es=永、Δ=ref8）。</p>
<p class="note">字体导航：{nav}</p>
<h2>像素诊断（相对 GT，非正式风格结论）</h2>
<table class="sum"><thead><tr><th>方法</th><th>shot</th><th>n</th><th>L1↓</th><th>SSIM↑</th></tr></thead>
<tbody>{''.join(sum_rows)}</tbody></table>
{''.join(sections)}
<script>
(function(){{
  const body = document.body;
  const tag = document.getElementById('hover-tag');
  const ctrl = document.getElementById('ctrl');
  function syncCtrlH(){{
    const h = Math.ceil(ctrl.getBoundingClientRect().height);
    document.documentElement.style.setProperty('--ctrl-h', h + 'px');
  }}
  syncCtrlH();
  window.addEventListener('resize', syncCtrlH);

  function applyFilters(){{
    document.querySelectorAll('.flt-model').forEach(inp => {{
      body.classList.toggle('hide-model-' + inp.value, !inp.checked);
    }});
    const s1 = document.querySelector('.flt-shot[value="1"]');
    const s8 = document.querySelector('.flt-shot[value="8"]');
    body.classList.toggle('hide-shot-1', !(s1 && s1.checked));
    body.classList.toggle('hide-shot-8', !(s8 && s8.checked));
    syncCtrlH();
  }}
  document.querySelectorAll('.flt-model,.flt-shot').forEach(el => el.addEventListener('change', applyFilters));
  document.getElementById('btn-all').onclick = () => {{
    document.querySelectorAll('.flt-model').forEach(i => i.checked = true); applyFilters();
  }};
  document.getElementById('btn-none').onclick = () => {{
    document.querySelectorAll('.flt-model').forEach(i => i.checked = false); applyFilters();
  }};
  document.getElementById('btn-hide-gt').onclick = () => {{
    const gt = document.querySelector('.flt-model[value="gt"]');
    if (gt) {{ gt.checked = false; applyFilters(); }}
  }};
  applyFilters();

  function clearHover(){{
    body.classList.remove('hl-on');
    document.querySelectorAll('.is-hover,.rel,.rel-self').forEach(el => el.classList.remove('is-hover','rel','rel-self'));
    tag.classList.remove('on'); tag.textContent = '';
  }}
  function onEnter(cell){{
    const fam = cell.dataset.family;
    const col = cell.dataset.col;
    const tr = cell.closest('tr');
    if (!tr || !fam) return;
    clearHover();
    body.classList.add('hl-on');
    tr.classList.add('is-hover');
    tr.querySelectorAll('td.cell').forEach(td => {{
      if (td.dataset.family === fam) td.classList.add('rel');
      if (td.dataset.col === col) td.classList.add('rel-self');
    }});
    document.querySelectorAll('table.board thead th.col-head').forEach(th => {{
      if (th.dataset.family === fam) th.classList.add('rel');
      if (th.dataset.col === col) th.classList.add('rel-self');
    }});
    const lab = cell.dataset.label || col;
    const shot = cell.dataset.shot === '1' ? '1-shot' : (cell.dataset.shot === '8' ? '8-shot' : '');
    tag.textContent = '悬停：' + lab + (shot ? ' · ' + shot : '') + ' · 系列 ' + fam;
    tag.classList.add('on');
  }}
  document.querySelectorAll('table.board td.cell').forEach(td => {{
    td.addEventListener('mouseenter', () => onEnter(td));
    td.addEventListener('mouseleave', clearHover);
  }});
}})();
</script>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (ROOT / "G_V0913_SHOT.html").write_text(
        html.replace("preds/", "reports/g_v0913_shot/preds/")
        .replace("gt/", "reports/g_v0913_shot/gt/")
        .replace("refs/", "reports/g_v0913_shot/refs/")
        .replace("href='#", "href='reports/g_v0913_shot/index.html#"),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--methods",
        default="",
        help="comma ids; default = all generate + reuse",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--skip-reuse", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "preds").mkdir(exist_ok=True)

    methods = build_methods()
    E.METHODS.update(methods)

    protocol = {
        "created_at": utc_now(),
        "split": "test",
        "fonts": E.fonts(),
        "chars": E.STRATIFIED,
        "seed": E.SEED,
        "g0c_ckpt": str(methods["G0c_s1"]["ckpt"]),
        "methods": {
            k: {
                "label": v["label"],
                "kind": v["kind"],
                "ckpt": str(v.get("ckpt", v.get("reuse_from", ""))),
                "style_oneshot": v.get("style_oneshot"),
            }
            for k, v in methods.items()
        },
    }
    (OUT / "PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    want = [x.strip() for x in args.methods.split(",") if x.strip()]
    if not want:
        want = list(methods.keys())
    if args.skip_reuse:
        want = [m for m in want if methods[m]["kind"] != "reuse"]

    if not args.html_only:
        for mid in want:
            if mid not in methods:
                raise SystemExit(f"unknown method {mid}")
            t0 = time.time()
            print(f"===== START {mid} =====", flush=True)
            generate_one(mid, args.device, args.overwrite)
            print(f"===== DONE {mid} in {time.time()-t0:.1f}s =====", flush=True)

    report = compute_metrics(list(methods.keys()))
    write_html(report)
    print("WROTE", OUT / "index.html", flush=True)
    print(json.dumps({m: report["methods"][m]["overall"] for m in report["methods"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
