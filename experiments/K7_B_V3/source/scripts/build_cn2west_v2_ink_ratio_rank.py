#!/usr/bin/env python3
"""Build formal R0 ink_ratio_rank + pass/drop/rerender review UI (protocol A).

Primary metric (collaborator): mean_ink_ratio = mean(ink_bbox_area / 96²) over 633 chars.
Secondary: mean_ink_pixel_ratio = mean(|S| / 96²).

Outputs under data/cn2west_v2_abc_review/proto_A_ink/:
  ink_ratio_rank.json / .csv
  review.html
  ink_review_decisions.jsonl  (empty starter; append-only in review)
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/projects/hrfont")
DS = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
CHARSET = json.loads((ROOT / "manifests/charset_cn2west_v2_planned.json").read_text(encoding="utf-8"))
OUT = ROOT / "data/cn2west_v2_abc_review/proto_A_ink"
A_SUMMARY = DS / "summary.json"
B_SCREEN = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2b-hfit/qa/screen/screen_report.json"
INK_REPORT = OUT / "ink_report.json"
CANVAS = 96
T_INK = 250
# PI-facing draft absolute threshold (font-level mean bbox area)
ABS_MEAN_BBOX = 0.20


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def all_cps() -> set[str]:
    cps: set[str] = set()
    for ch in CHARSET["style_han_338"]:
        cps.add(cp_of(ch))
    for chars in CHARSET["target"].values():
        for ch in chars:
            cps.add(cp_of(ch))
    for ch in CHARSET["target_string"]:
        cps.add(cp_of(ch))
    return cps


def scan_font(job: dict) -> dict:
    """Per-font ink bbox area, pixel fill, touch_edge over Target+Style PNGs."""
    stem, split = job["stem"], job["split"]
    root = Path(job["root"])
    cps = set(job["cps"])
    bboxes: list[float] = []
    fills: list[float] = []
    n_empty = 0
    touch_edge = False

    for sub in ("TargetImage", "StyleImage"):
        d = root / split / sub / stem
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            if "+" not in p.name:
                continue
            cp = p.name.split("+", 1)[1].removesuffix(".png")
            if cp not in cps:
                continue
            try:
                a = np.asarray(Image.open(p).convert("L"))
            except Exception:
                continue
            ys, xs = np.where(a < T_INK)
            if len(ys) == 0:
                n_empty += 1
                bboxes.append(0.0)
                fills.append(0.0)
                continue
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            h = (y1 - y0 + 1) / CANVAS
            w = (x1 - x0 + 1) / CANVAS
            bboxes.append(float(h * w))
            fills.append(float(np.count_nonzero(a < T_INK)) / (CANVAS * CANVAS))
            if y0 <= 0 or x0 <= 0 or y1 >= CANVAS - 1 or x1 >= CANVAS - 1:
                touch_edge = True

    arr_b = np.array(bboxes, dtype=np.float64) if bboxes else np.array([0.0])
    arr_f = np.array(fills, dtype=np.float64) if fills else np.array([0.0])
    return {
        "stem": stem,
        "split": split,
        "n": len(bboxes),
        "n_empty": n_empty,
        "mean_ink_ratio": float(arr_b.mean()),
        "ink_ratio_p5": float(np.percentile(arr_b, 5)),
        "ink_ratio_p50": float(np.percentile(arr_b, 50)),
        "ink_ratio_p95": float(np.percentile(arr_b, 95)),
        "mean_ink_pixel_ratio": float(arr_f.mean()),
        "touch_edge": touch_edge,
    }


def load_a_meta() -> dict[str, dict]:
    sa = json.loads(A_SUMMARY.read_text(encoding="utf-8"))
    out = {}
    for f in sa["fonts"]:
        out[f["stem"]] = {
            "fs_A": f.get("size"),
            "tall_ch": f.get("tall_ch"),
            "max_h": f.get("max_h"),
            "wide_ch": f.get("wide_ch"),
            "max_w": f.get("max_w"),
            "split": f.get("split"),
        }
    return out


def load_b_screen() -> dict[str, dict]:
    if not B_SCREEN.exists():
        return {}
    sc = json.loads(B_SCREEN.read_text(encoding="utf-8"))
    out = {}
    for f in sc.get("fonts", []):
        out[f["stem"]] = {
            "auto_screen_severity": f.get("severity"),
            "auto_screen_reasons": f.get("reasons") or [],
        }
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rows(font_scans: list[dict], a_meta: dict, b_screen: dict) -> list[dict]:
    rows = []
    for sc in font_scans:
        stem = sc["stem"]
        am = a_meta.get(stem, {})
        bs = b_screen.get(stem, {})
        mean_r = sc["mean_ink_ratio"]
        auto_flags = []
        if mean_r < ABS_MEAN_BBOX:
            auto_flags.append(f"mean_bbox<{ABS_MEAN_BBOX}")
        if sc["touch_edge"]:
            auto_flags.append("touch_edge")
        if sc["n_empty"] > 0:
            auto_flags.append("n_empty>0")
        sev = bs.get("auto_screen_severity")
        if sev in ("drop", "review"):
            auto_flags.append(f"b_screen:{sev}")

        suggested = ""
        if mean_r < ABS_MEAN_BBOX:
            suggested = "drop"
        elif sc["touch_edge"] or sc["n_empty"] > 0:
            suggested = "rerender"
        elif sev == "drop":
            suggested = "drop"
        elif sev == "review":
            suggested = "review"

        rows.append({
            "stem": stem,
            "split": sc.get("split") or am.get("split"),
            "mean_ink_ratio": mean_r,
            "ink_ratio_p5": sc["ink_ratio_p5"],
            "ink_ratio_p50": sc["ink_ratio_p50"],
            "ink_ratio_p95": sc["ink_ratio_p95"],
            "mean_ink_pixel_ratio": sc["mean_ink_pixel_ratio"],
            "fs_A": am.get("fs_A"),
            "tall_ch": am.get("tall_ch"),
            "max_h": am.get("max_h"),
            "wide_ch": am.get("wide_ch"),
            "max_w": am.get("max_w"),
            "touch_edge": sc["touch_edge"],
            "n_empty": sc["n_empty"],
            "n_glyphs": sc["n"],
            "auto_screen_severity": sev,
            "auto_screen_reasons": bs.get("auto_screen_reasons") or [],
            "auto_flags": auto_flags,
            "suggested_action": suggested,
            "review_candidate": bool(auto_flags),
            "review_decision": "",
            "reviewer": "",
            "reviewed_at": "",
        })

    rows.sort(key=lambda r: (r["mean_ink_ratio"], r["stem"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "rank", "stem", "split", "mean_ink_ratio", "ink_ratio_p5", "ink_ratio_p50",
        "ink_ratio_p95", "mean_ink_pixel_ratio", "fs_A", "tall_ch", "max_h",
        "wide_ch", "max_w", "touch_edge", "n_empty", "auto_screen_severity",
        "auto_screen_reasons", "auto_flags", "suggested_action", "review_candidate",
        "review_decision", "reviewer", "reviewed_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["auto_screen_reasons"] = "|".join(row.get("auto_screen_reasons") or [])
            row["auto_flags"] = "|".join(row.get("auto_flags") or [])
            w.writerow(row)


REVIEW_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>R0 · A ink_ratio 审查</title>
<style>
:root{--bg:#eef1f5;--panel:#fff;--line:#d5dbe3;--muted:#5c6570;--warn:#8a4b08;--bad:#8b1a1a}
*{box-sizing:border-box} html,body{height:100%;margin:0}
body{font:13px/1.45 system-ui,sans-serif;background:var(--bg);color:#1a1a1a;display:flex;flex-direction:column}
header{flex:0 0 auto;background:#fff;border-bottom:1px solid var(--line);padding:10px 14px}
h1{margin:0;font-size:1.1rem} .meta{color:var(--muted);font-size:12px;margin-top:2px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
input,select,button{padding:5px 9px;border:1px solid var(--line);border-radius:4px;font:inherit;background:#fff}
.layout{flex:1;min-height:0;display:grid;grid-template-columns:380px 1fr}
aside{background:var(--panel);border-right:1px solid var(--line);overflow:auto}
#fontList{list-style:none;margin:0;padding:0}
#fontList li{padding:8px 10px;border-bottom:1px solid #eef1f5;cursor:pointer}
#fontList li:hover{background:#f5f7fa} #fontList li.active{background:#e8eef5}
#fontList li.cand{border-left:3px solid #c47a12}
.stem{font-weight:600;font-family:ui-monospace,monospace;font-size:12px}
.sub{color:var(--muted);font-size:11px;margin-top:2px}
main{overflow:auto;padding:14px}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;margin-left:4px}
.badge.cand{background:#fff3cd;color:var(--warn)}
.badge.ok{background:#e8f5e9;color:#1b5e20}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(72px,1fr));gap:8px;margin-top:10px}
.cell{background:#fff;border:1px solid var(--line);padding:4px;text-align:center}
.cell img{width:64px;height:64px;image-rendering:pixelated;background:#fff}
.cell .lab{font-size:11px;color:var(--muted);margin-top:2px}
.dec{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0;padding:10px;background:#fff;border:1px solid var(--line)}
.dec label{cursor:pointer}
pre{background:#fff;border:1px solid var(--line);padding:10px;overflow:auto;font-size:11px}
.note{color:var(--muted);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>R0 · 协议 A · ink_ratio 人工审查（pass / drop / rerender）</h1>
  <div class="meta" id="meta"></div>
  <div class="toolbar">
    <label>显示
      <select id="filter">
        <option value="candidates">审查候选（自动标记）</option>
        <option value="abs020">mean_bbox &lt; 20%</option>
        <option value="all">全部字体</option>
        <option value="decided">已决策</option>
        <option value="undecided">未决策</option>
      </select>
    </label>
    <input id="q" type="search" placeholder="搜索 stem…" style="min-width:160px"/>
    <button type="button" id="btnExport">导出 decisions JSONL</button>
    <button type="button" id="btnSaveLocal">写入本机 localStorage</button>
    <span class="note" id="stats"></span>
  </div>
</header>
<div class="layout">
  <aside><ul id="fontList"></ul></aside>
  <main id="main"><p class="note">加载 ink_ratio_rank.json…</p></main>
</div>
<script>
const PROBES = ["永","和","书","Aa","Gg","Ww","Mm","0","1","i","l","ㄅ","あ","ア"];
let DATA = null;
let decisions = {}; // stem -> {review_decision, reviewer, reviewed_at, reason}

function imgUrl(stem, split, ch){
  const cp = "u" + ch.codePointAt(0).toString(16).toUpperCase().padStart(4,"0");
  const role = (ch === "永" || /[\u4e00-\u9fff]/.test(ch) && !"AaGgWwMm01il".includes(ch))
    ? ((ch.length===1 && ch.charCodeAt(0)>0x3000) ? "StyleImage" : "TargetImage")
    : "TargetImage";
  // Prefer Style for CJK style set, Target for latin/digits; both exist for 永 under Style+Target pools
  const sub = /[\u4e00-\u9fff\u3040-\u30ff\u3100-\u312f]/.test(ch) ? "StyleImage" : "TargetImage";
  // 永 is in style_han; ASCII in target
  const folder = /[\u4e00-\u9fff]/.test(ch) || /[\u3040-\u30ff\u3100-\u312f]/.test(ch) ? "StyleImage" : "TargetImage";
  // Actually target set includes CJK too in this dataset layout: TargetImage has target chars, StyleImage has style_han.
  // Use Target for Latin/digits; Style for 永 and kana/bopo probes that are style; fallback both.
  let use = folder;
  if ("永和书".includes(ch)) use = "StyleImage";
  else if (/[A-Za-z0-9]/.test(ch)) use = "TargetImage";
  else use = "StyleImage";
  return `../../fontdiffuser-p253-t295-s338-cn2west-v2/${split}/${use}/${stem}/${stem}+${cp}.png`;
}

function loadLocal(){
  try { decisions = JSON.parse(localStorage.getItem("hrfont_ink_review_v1")||"{}") || {}; }
  catch(e){ decisions = {}; }
}

function saveLocal(){
  localStorage.setItem("hrfont_ink_review_v1", JSON.stringify(decisions));
  alert("已写入 localStorage（导出 JSONL 才算正式回写）");
}

function filtered(){
  const mode = document.getElementById("filter").value;
  const q = document.getElementById("q").value.trim().toLowerCase();
  return DATA.fonts.filter(f => {
    if (q && !f.stem.toLowerCase().includes(q)) return false;
    const d = (decisions[f.stem]||{}).review_decision || f.review_decision || "";
    if (mode === "candidates") return f.review_candidate;
    if (mode === "abs020") return f.mean_ink_ratio < 0.20;
    if (mode === "decided") return !!d;
    if (mode === "undecided") return !d;
    return true;
  });
}

function renderList(){
  const ul = document.getElementById("fontList");
  const rows = filtered();
  document.getElementById("stats").textContent =
    `显示 ${rows.length} / ${DATA.fonts.length} · 候选 ${DATA.summary.n_review_candidates} · mean_bbox<20% ${DATA.summary.n_mean_bbox_lt_020}`;
  ul.innerHTML = rows.map(f => {
    const d = (decisions[f.stem]||{}).review_decision || "";
    const badge = f.review_candidate ? `<span class="badge cand">候选</span>` : `<span class="badge ok">ok</span>`;
    return `<li class="${f.review_candidate?"cand":""}" data-stem="${f.stem}">
      <div class="stem">#${f.rank} ${f.stem}${badge}${d?` [${d}]`:""}</div>
      <div class="sub">${f.split} · mean_bbox ${(100*f.mean_ink_ratio).toFixed(1)}% · fill ${(100*f.mean_ink_pixel_ratio).toFixed(2)}% · ${f.suggested_action||"—"}</div>
    </li>`;
  }).join("");
  ul.querySelectorAll("li").forEach(li => li.onclick = () => showFont(li.dataset.stem));
  if (rows[0]) showFont(rows[0].stem);
}

function showFont(stem){
  const f = DATA.fonts.find(x => x.stem === stem);
  if (!f) return;
  document.querySelectorAll("#fontList li").forEach(li => li.classList.toggle("active", li.dataset.stem===stem));
  const d = decisions[stem] || {};
  const cur = d.review_decision || "";
  const main = document.getElementById("main");
  const flags = (f.auto_flags||[]).join(", ") || "—";
  main.innerHTML = `
    <h2 style="margin:0 0 6px;font-family:ui-monospace,monospace">${f.stem}</h2>
    <div class="note">rank=${f.rank} · split=${f.split} · fs_A=${f.fs_A} · tall=${f.tall_ch}/${f.max_h} · wide=${f.wide_ch}/${f.max_w}</div>
    <div class="note">mean_ink_ratio(bbox)=<b>${(100*f.mean_ink_ratio).toFixed(2)}%</b>
      · p5/p50/p95=${(100*f.ink_ratio_p5).toFixed(1)}/${(100*f.ink_ratio_p50).toFixed(1)}/${(100*f.ink_ratio_p95).toFixed(1)}%
      · pixel_fill=${(100*f.mean_ink_pixel_ratio).toFixed(3)}%
      · touch=${f.touch_edge} · empty=${f.n_empty}
      · B-screen=${f.auto_screen_severity||"—"}</div>
    <div class="note">auto_flags: ${flags} · suggested: <b>${f.suggested_action||"—"}</b></div>
    <div class="dec">
      <span>决策：</span>
      ${["pass","drop","rerender"].map(v =>
        `<label><input type="radio" name="dec" value="${v}" ${cur===v?"checked":""}/> ${v}</label>`
      ).join("")}
      <input id="reviewer" placeholder="reviewer" value="${d.reviewer||""}" style="width:120px"/>
      <input id="reason" placeholder="reason code / 备注" value="${d.reason||""}" style="min-width:220px"/>
      <button type="button" id="btnApply">应用</button>
    </div>
    <div class="grid" id="probeGrid"></div>
    <p class="note">探针图读 A 盘 PNG；缺图则裂图（cmap/空）。</p>`;
  const grid = document.getElementById("probeGrid");
  grid.innerHTML = PROBES.map(ch => {
    const url = imgUrl(stem, f.split, ch);
    return `<div class="cell"><img src="${url}" alt="${ch}" loading="lazy"/><div class="lab">${ch}</div></div>`;
  }).join("");
  document.getElementById("btnApply").onclick = () => {
    const sel = document.querySelector('input[name="dec"]:checked');
    if (!sel) { alert("请选择 pass/drop/rerender"); return; }
    decisions[stem] = {
      review_decision: sel.value,
      reviewer: document.getElementById("reviewer").value.trim() || "anonymous",
      reviewed_at: new Date().toISOString(),
      reason: document.getElementById("reason").value.trim(),
      mean_ink_ratio: f.mean_ink_ratio,
      suggested_action: f.suggested_action,
    };
    saveLocalSilent();
    renderList();
    showFont(stem);
  };
}

function saveLocalSilent(){
  localStorage.setItem("hrfont_ink_review_v1", JSON.stringify(decisions));
}

function exportJsonl(){
  const lines = Object.entries(decisions).map(([stem, d]) => JSON.stringify({
    dataset: DATA.dataset,
    dataset_sha256_hint: DATA.hashes?.ink_ratio_rank_json || "",
    stem,
    ...d,
  }));
  const blob = new Blob([lines.join("\n")+(lines.length?"\n":"")], {type:"text/plain"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "ink_review_decisions_export.jsonl";
  a.click();
}

async function main(){
  loadLocal();
  const r = await fetch("./ink_ratio_rank.json");
  DATA = await r.json();
  document.getElementById("meta").textContent =
    `${DATA.dataset} · T=${DATA.ink_threshold} · 主指标=${DATA.primary_metric} · 草案阈值 mean_bbox<${DATA.draft_thresholds.absolute_mean_bbox} · generated ${DATA.generated_at}`;
  document.getElementById("filter").onchange = renderList;
  document.getElementById("q").oninput = renderList;
  document.getElementById("btnExport").onclick = exportJsonl;
  document.getElementById("btnSaveLocal").onclick = saveLocal;
  renderList();
}
main();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--from-ink-report", action="store_true",
                    help="Reuse ink_report.json bbox stats; still scan for touch/fill")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    a_meta = load_a_meta()
    b_screen = load_b_screen()
    cps = sorted(all_cps())

    jobs = []
    for stem, meta in sorted(a_meta.items()):
        jobs.append({
            "stem": stem,
            "split": meta["split"],
            "root": str(DS),
            "cps": cps,
        })

    print(f"scanning {len(jobs)} fonts × {len(cps)} cps, workers={args.workers}", flush=True)
    font_scans: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(scan_font, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            font_scans.append(fut.result())
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    rows = build_rows(font_scans, a_meta, b_screen)
    mean_arr = np.array([r["mean_ink_ratio"] for r in rows])
    train_means = np.array([r["mean_ink_ratio"] for r in rows if r["split"] == "train"])
    calib_p5 = float(np.percentile(train_means, 5)) if len(train_means) else None

    n_abs = sum(1 for r in rows if r["mean_ink_ratio"] < ABS_MEAN_BBOX)
    n_cand = sum(1 for r in rows if r["review_candidate"])
    n_touch = sum(1 for r in rows if r["touch_edge"])
    n_empty = sum(1 for r in rows if r["n_empty"] > 0)

    payload = {
        "dataset": DS.name,
        "proto": "A",
        "primary_metric": "mean_ink_ratio = mean(ink_bbox_h * ink_bbox_w / 96^2)",
        "secondary_metric": "mean_ink_pixel_ratio = mean(|S|/96^2)",
        "ink_threshold": T_INK,
        "charset_n": len(cps),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "draft_thresholds": {
            "absolute_mean_bbox": ABS_MEAN_BBOX,
            "calibration_split": "train",
            "calibration_p5_mean_bbox": calib_p5,
            "note": "未冻结；待 PI 确认。绝对阈值 mean_bbox<0.20 为当前草案。",
        },
        "summary": {
            "n_fonts": len(rows),
            "n_mean_bbox_lt_020": n_abs,
            "n_review_candidates": n_cand,
            "n_touch_edge": n_touch,
            "n_with_empty": n_empty,
            "global_mean_of_font_means": float(mean_arr.mean()),
            "global_p5_of_font_means": float(np.percentile(mean_arr, 5)),
            "global_median_of_font_means": float(np.median(mean_arr)),
        },
        "fonts": rows,
        "hashes": {},
    }

    json_path = OUT / "ink_ratio_rank.json"
    csv_path = OUT / "ink_ratio_rank.csv"
    review_path = OUT / "review.html"
    decisions_path = OUT / "ink_review_decisions.jsonl"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(rows, csv_path)
    review_path.write_text(REVIEW_HTML, encoding="utf-8")
    if not decisions_path.exists():
        decisions_path.write_text("", encoding="utf-8")

    payload["hashes"] = {
        "ink_ratio_rank_json": sha256_file(json_path),
        "ink_ratio_rank_csv": sha256_file(csv_path),
    }
    # rewrite with hashes
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload["hashes"]["ink_ratio_rank_json"] = sha256_file(json_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("wrote", json_path)
    print("wrote", csv_path)
    print("wrote", review_path)
    print("summary", payload["summary"])
    print("mean_bbox<0.20 fonts:")
    for r in rows:
        if r["mean_ink_ratio"] < ABS_MEAN_BBOX:
            print(f"  #{r['rank']} {r['stem']} {r['split']} {r['mean_ink_ratio']:.4f} suggested={r['suggested_action']}")


if __name__ == "__main__":
    main()
