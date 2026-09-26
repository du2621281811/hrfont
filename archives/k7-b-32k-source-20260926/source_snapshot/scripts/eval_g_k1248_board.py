#!/usr/bin/env python3
"""HTML board for reports/g_v0913_shot_k1248 (old Ref8 · 1/2/4/8-shot).

Mirrors the layout/filters of eval_g_v0913_shot_board.py, without dirty F2 columns.
Preds already exist; this script is html (+ optional pixel diagnostics) only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))
import eval_f03_test16_strat as E  # noqa: E402

OUT = ROOT / "reports/g_v0913_shot_k1248"
OLD = ROOT / "reports/g_v0913_shot"
E.OUT = OUT

REF8 = "永和书风骨韵天地"
TRAIN5 = ["FZPTYJW", "FZDuHJW_Cu", "FZShuLTJW-H", "FZYiMSJW-T", "FZFeiSJW-EL"]
VAL5 = ["FZYouHK_511M", "FZLTHProGBK_H", "FZJunYTJW-H", "FZBangSKKXJW", "FZKANGJW"]
SHOTS = (1, 2, 4, 8)

CURATED = {
    "0", "2", "8", "A", "G", "Q", "R", "a", "g", "e", "à", "ě", "あ", "さ", "ア", "ㄅ", "ㄚ",
}

BOARD_COLS: list[tuple[str, str, str]] = [
    ("GT", "GT", "对照"),
    ("G0b_s1", "G0b@10k", "1-shot"),
    ("G0c_s1", "G0c@20k", "1-shot"),
]
for name, lab in [
    ("G1", "G1"),
    ("G2", "G2"),
    ("G2RL", "G2-RL"),
    ("pilot", "pilot@1k"),
    ("pilot8", "pilot8@best"),
    ("TCG2", "TC-G2"),
    ("TCG2RL", "TC-G2RL"),
]:
    for k in SHOTS:
        if name.startswith("G0"):
            continue
        BOARD_COLS.append((f"{name}_s{k}", lab, f"{k}-shot"))

COL_FAMILIES = {
    "REF1": "ref",
    "REF8": "ref",
    "GT": "gt",
    "G0b_s1": "g0",
    "G0c_s1": "g0",
}
for mid, lab, shot in BOARD_COLS[1:]:
    if mid.startswith("G0"):
        COL_FAMILIES[mid] = "g0"
    elif mid.startswith("G1"):
        COL_FAMILIES[mid] = "g1"
    elif mid.startswith("G2RL"):
        COL_FAMILIES[mid] = "g2rl"
    elif mid.startswith("G2"):
        COL_FAMILIES[mid] = "g2"
    elif mid.startswith("pilot8"):
        COL_FAMILIES[mid] = "pilot8"
    elif mid.startswith("pilot"):
        COL_FAMILIES[mid] = "pilot"
    elif mid.startswith("TCG2RL"):
        COL_FAMILIES[mid] = "tcg2rl"
    elif mid.startswith("TCG2"):
        COL_FAMILIES[mid] = "tcg2"

MODEL_FILTERS = [
    ("ref", "Ref", ["REF1", "REF8"]),
    ("gt", "GT", ["GT"]),
    ("g0", "G0b/G0c", ["G0b_s1", "G0c_s1"]),
    ("g1", "G1", [f"G1_s{k}" for k in SHOTS]),
    ("g2", "G2", [f"G2_s{k}" for k in SHOTS]),
    ("g2rl", "G2-RL", [f"G2RL_s{k}" for k in SHOTS]),
    ("pilot", "pilot", [f"pilot_s{k}" for k in SHOTS]),
    ("pilot8", "pilot8", [f"pilot8_s{k}" for k in SHOTS]),
    ("tcg2", "TC-G2", [f"TCG2_s{k}" for k in SHOTS]),
    ("tcg2rl", "TC-G2RL", [f"TCG2RL_s{k}" for k in SHOTS]),
]
FILTER_DEFAULT_OFF = {"gt", "pilot", "pilot8"}  # keep board readable by default


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def method_ids() -> list[str]:
    return [m for m, _, _ in BOARD_COLS if m != "GT"]


def split_fonts() -> list[tuple[str, str]]:
    test16 = json.loads((ROOT / "manifests/split_v3_228_16_16.json").read_text())["stems"]["test"]
    out: list[tuple[str, str]] = []
    for s in test16:
        out.append(("test", s))
    for s in TRAIN5:
        out.append(("train", s))
    for s in VAL5:
        out.append(("val", s))
    return out


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _ensure_ref_assets(stem: str, split: str) -> tuple[str | None, list[str]]:
    style_dir = E.DATA / split / "StyleImage" / stem
    ref1_rel = None
    ch1 = "永"
    src1 = style_dir / f"{stem}+{E.cp_of(ch1)}.png"
    if src1.is_file():
        dst1 = OUT / "refs" / split / stem / f"{stem}+{E.cp_of(ch1)}.png"
        _link_or_copy(src1, dst1)
        ref1_rel = f"refs/{split}/{stem}/{stem}+{E.cp_of(ch1)}.png"
    ref8_rels = []
    for ch in REF8:
        src = style_dir / f"{stem}+{E.cp_of(ch)}.png"
        if not src.is_file():
            continue
        dst = OUT / "refs" / split / stem / f"{stem}+{E.cp_of(ch)}.png"
        _link_or_copy(src, dst)
        ref8_rels.append(f"refs/{split}/{stem}/{stem}+{E.cp_of(ch)}.png")
    return ref1_rel, ref8_rels


def pred_rel(mid: str, split: str, stem: str, ch: str) -> str:
    return f"preds/{mid}/{split}/{stem}/{split}__{stem}__{E.cp_of(ch)}__s{E.SEED}.png"


def compute_metrics(mids: list[str], only_split: str = "test") -> dict:
    rows = []
    for mid in mids:
        root = OUT / "preds" / mid / only_split
        if not root.is_dir():
            continue
        for png in root.rglob("*.png"):
            meta_p = png.with_suffix(".json")
            if meta_p.is_file():
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                font = meta.get("font") or png.parts[-2]
                ch = meta.get("char")
                if not ch:
                    try:
                        cp = [x for x in png.stem.split("__") if x.startswith("u")][0]
                        ch = chr(int(cp[1:], 16))
                    except Exception:
                        continue
            else:
                try:
                    font = png.parts[-2]
                    cp = [x for x in png.stem.split("__") if x.startswith("u")][0]
                    ch = chr(int(cp[1:], 16))
                except Exception:
                    continue
            E.configure_split(only_split, out=OUT)
            gtp = E.gt_path(font, ch)
            if gtp is None or not gtp.is_file():
                continue
            pred = Image.open(png).convert("RGB")
            gt = Image.open(gtp).convert("RGB")
            pa, ga = E.to_gray01(pred), E.to_gray01(gt)
            rows.append(
                {
                    "method": mid,
                    "font": font,
                    "char": ch,
                    "L1": float(abs(pa - ga).mean()),
                    "SSIM": E.ssim(pa, ga),
                }
            )
    by_m: dict[str, list] = {m: [] for m in mids}
    for r in rows:
        by_m.setdefault(r["method"], []).append(r)
    methods = {mid: {"label": mid, "overall": E.agg(by_m.get(mid, []))} for mid in mids}
    report = {
        "computed_at": utc_now(),
        "protocol": (
            f"old Ref8={REF8} · test16(+train5/val5 board) × stratified47 · "
            "DPM++20 CFG7.5 seed3407 · metrics on test only"
        ),
        "ref8": REF8,
        "shot_sets": {str(k): REF8[:k] for k in SHOTS},
        "fonts_test": json.loads((ROOT / "manifests/split_v3_228_16_16.json").read_text())["stems"]["test"],
        "chars": E.STRATIFIED,
        "methods": methods,
        "n_items": len(rows),
        "caveat": "L1/SSIM vs GT are diagnostics only, not style verdicts. No dirty F2 on this board.",
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def write_html(report: dict) -> None:
    chars = [c for c in E.STRATIFIED if c in CURATED] or list(E.STRATIFIED)
    display_cols = [
        ("REF1", "Ref1", "永"),
        ("REF8", "Ref8", "8字"),
        *BOARD_COLS,
    ]

    def shot_key(shot: str) -> str:
        for k in SHOTS:
            if shot.startswith(str(k)):
                return str(k)
        return "na"

    def col_attrs(mid: str, lab: str, shot: str) -> str:
        fam = COL_FAMILIES.get(mid, "other")
        sk = "na" if mid in {"REF1", "REF8", "GT"} else shot_key(shot)
        model_key = next((fid for fid, _, mids in MODEL_FILTERS if mid in mids), fam)
        return (
            f"data-col='{mid}' data-family='{fam}' data-model='{model_key}' "
            f"data-shot='{sk}' data-label='{lab}'"
        )

    head = "".join(
        f"<th class='col-head' {col_attrs(mid, lab, shot)}>"
        f"<div class='ml'>{lab}</div><div class='ms'>{shot}</div></th>"
        for mid, lab, shot in display_cols
    )

    sum_rows = []
    for mid, lab, shot in BOARD_COLS:
        if mid == "GT":
            continue
        o = report["methods"].get(mid, {}).get("overall", {})
        n = o.get("n") or 0
        l1 = o.get("L1_mean")
        ss = o.get("SSIM_mean")
        fam = COL_FAMILIES.get(mid, "other")
        sk = shot_key(shot)
        model_key = next((fid for fid, _, mids in MODEL_FILTERS if mid in mids), fam)
        attrs = f"data-model='{model_key}' data-shot='{sk}' data-family='{fam}'"
        if l1 is not None:
            sum_rows.append(
                f"<tr class='sum-row' {attrs}><td>{lab}</td><td>{shot}</td><td>{n}</td>"
                f"<td>{l1:.4f}</td><td>{ss:.4f}</td></tr>"
            )
        else:
            sum_rows.append(
                f"<tr class='sum-row' {attrs}><td>{lab}</td><td>{shot}</td><td>{n}</td><td>—</td><td>—</td></tr>"
            )

    model_chips = "".join(
        "<label class='chip'>"
        f"<input type='checkbox' class='flt-model' value='{fid}'"
        f"{'' if fid in FILTER_DEFAULT_OFF else ' checked'}> {lab}</label>"
        for fid, lab, _ in MODEL_FILTERS
    )
    shot_chips = "".join(
        f"<label class='chip'><input type='checkbox' class='flt-shot' value='{k}' checked> {k}-shot</label>"
        for k in SHOTS
    )

    sections = []
    nav_bits = []
    for split, stem in split_fonts():
        aid = f"{split}__{stem}"
        nav_bits.append(f"<a href='#{aid}'>{split}/{stem}</a>")
        ref1_rel, ref8_rels = _ensure_ref_assets(stem, split)
        E.configure_split(split, out=OUT)
        blocks = [
            f"<h2 id='{aid}'>{split} · {stem}</h2>",
            "<div class='tbl-wrap'><table class='board'>",
            f"<thead><tr><th class='sticky-char'>char</th>{head}</tr></thead><tbody>",
        ]
        for ch in chars:
            tds = [f"<td class='meta sticky-char'>{ch}</td>"]
            if ref1_rel:
                tds.append(
                    f"<td class='cell refcell' {col_attrs('REF1', 'Ref1', '永')}>"
                    f"<img src='{ref1_rel}' alt='ref1'></td>"
                )
            else:
                tds.append(f"<td class='cell miss' {col_attrs('REF1', 'Ref1', '永')}>—</td>")
            if ref8_rels:
                imgs = "".join(f"<img class='r8' src='{r}' alt='r'>" for r in ref8_rels)
                tds.append(
                    f"<td class='cell refcell ref8' {col_attrs('REF8', 'Ref8', '8字')}>"
                    f"<div class='r8wrap'>{imgs}</div></td>"
                )
            else:
                tds.append(f"<td class='cell miss' {col_attrs('REF8', 'Ref8', '8字')}>—</td>")

            gtp = E.gt_path(stem, ch)
            if gtp and gtp.is_file():
                gdst = OUT / "gt" / split / stem / f"{stem}+{E.cp_of(ch)}.png"
                _link_or_copy(gtp, gdst)
                tds.append(
                    f"<td class='cell' {col_attrs('GT', 'GT', '对照')}>"
                    f"<img src='gt/{split}/{stem}/{stem}+{E.cp_of(ch)}.png'></td>"
                )
            else:
                tds.append(f"<td class='cell' {col_attrs('GT', 'GT', '对照')}></td>")

            for mid, lab, shot in BOARD_COLS[1:]:
                rel = pred_rel(mid, split, stem, ch)
                p = OUT / rel
                attrs = col_attrs(mid, lab, shot)
                if p.is_file():
                    tds.append(f"<td class='cell' {attrs}><img src='{rel}' alt='{mid}'></td>")
                else:
                    tds.append(f"<td class='cell miss' {attrs}>—</td>")
            blocks.append("<tr>" + "".join(tds) + "</tr>")
        blocks.append("</tbody></table></div>")
        sections.append("\n".join(blocks))

    hide_shot_css = "\n".join(
        f'body.hide-shot-{k} [data-shot="{k}"]{{display:none !important}}' for k in SHOTS
    )
    hide_model_css = "\n".join(
        f'body.hide-model-{fid} [data-model="{fid}"]{{display:none !important}}'
        for fid, _, _ in MODEL_FILTERS
    )

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>G v0913 k1248 · old Ref8</title>
<style>
:root{{--bg:#111;--line:#333;--accent:#5af;--hl:#1e3a4f;--hl-strong:#2a5570;--dim:.22;--ctrl-h:140px}}
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
.tbl-wrap{{margin:8px 0 18px;border:1px solid #2a2a2a;display:inline-block;min-width:100%}}
table.board{{border-collapse:separate;border-spacing:0;margin:0;background:#111}}
table.board th,table.board td{{border:1px solid var(--line);padding:4px;vertical-align:middle;text-align:center;background:#111}}
table.board thead th{{position:sticky;top:var(--ctrl-h);z-index:20;background:#1c2834;box-shadow:0 2px 0 #456}}
table.board .sticky-char{{position:sticky;left:0;z-index:15;background:#161616;min-width:36px}}
table.board thead .sticky-char{{z-index:25;left:0;top:var(--ctrl-h);background:#1c2834}}
table.board img{{width:72px;height:72px;image-rendering:pixelated;background:#1a1a1a;display:block;margin:0 auto}}
table.board td.ref8{{min-width:92px;max-width:100px;padding:3px}}
table.board .r8wrap{{display:grid;grid-template-columns:repeat(4,20px);grid-template-rows:repeat(2,20px);gap:2px;justify-content:center;align-content:center;width:86px;margin:0 auto;overflow:hidden}}
table.board .r8wrap img.r8,table.board img.r8{{width:20px;height:20px;max-width:20px;max-height:20px;margin:0;padding:0;object-fit:contain}}
.meta{{font-size:12px;color:#aaa}} .ml{{font-size:11px;white-space:nowrap}} .ms{{font-size:10px;color:#9ab}}
.miss{{color:#666;font-size:12px}} h2{{font-size:16px;margin:28px 0 8px}}
.warn{{color:#fd6}} .note{{color:#aaa;font-size:13px;line-height:1.45}} a{{color:#9cf}}
.sum{{border-collapse:collapse;margin:10px 0}}
.sum td,.sum th{{font-size:12px;padding:6px 8px;border:1px solid #333}}
{hide_shot_css}
{hide_model_css}
body.hl-on table.board tbody tr.is-hover td.cell{{opacity:var(--dim);transition:opacity .08s,background .08s,outline .08s}}
body.hl-on table.board tbody tr.is-hover td.cell.rel{{opacity:1;background:var(--hl)}}
body.hl-on table.board tbody tr.is-hover td.cell.rel-self{{opacity:1;background:var(--hl-strong);outline:2px solid var(--accent);outline-offset:-2px}}
body.hl-on table.board thead th.rel{{background:#24364a}}
body.hl-on table.board thead th.rel-self{{background:#2f5a78;outline:2px solid var(--accent);outline-offset:-2px}}
#hover-tag{{display:none;margin-left:auto;color:#9cf;font-size:12px}}
#hover-tag.on{{display:inline}}
</style>
<div class="ctrl" id="ctrl">
  <h1>G 系 v0913 · k=1/2/4/8 · old Ref8（永和书风骨韵天地）</h1>
  <div class="row"><span class="hint">列</span>{model_chips}
    <button type="button" id="btn-all" class="chip">全选</button>
    <button type="button" id="btn-none" class="chip">清空</button>
    <button type="button" id="btn-hide-gt" class="chip gt-chip">隐藏 GT</button>
    <span id="hover-tag"></span>
  </div>
  <div class="row"><span class="hint">Shot</span>{shot_chips}
    <span class="hint">默认隐藏 GT / pilot / pilot8。Ref1=永；Ref8=永和书风骨韵天地；k-shot 用前缀。无 dirty F2。</span>
  </div>
</div>
<p class="note">{report['protocol']}<br>computed {report['computed_at']}</p>
<p class="note">字体：test16 + train5 + val5；字符展示 curated 子集（全量 preds 在目录内）。旧板：
<a href="/g_shot/">/g_shot/</a></p>
<p class="warn">{report['caveat']}</p>
<p class="note">字体导航：{' · '.join(nav_bits)}</p>
<h2>像素诊断（test split vs GT，非正式风格结论）</h2>
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
    document.querySelectorAll('.flt-shot').forEach(inp => {{
      body.classList.toggle('hide-shot-' + inp.value, !inp.checked);
    }});
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
    const shot = cell.dataset.shot !== 'na' ? (cell.dataset.shot + '-shot') : '';
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-metrics", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mids = method_ids()
    if args.skip_metrics:
        report = {
            "computed_at": utc_now(),
            "protocol": f"old Ref8={REF8} · metrics deferred",
            "methods": {m: {"label": m, "overall": {"n": 0}} for m in mids},
            "caveat": "L1/SSIM deferred; images are ready.",
        }
    else:
        print("computing test metrics…", flush=True)
        report = compute_metrics(mids, only_split="test")
    write_html(report)
    print("WROTE", OUT / "index.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
