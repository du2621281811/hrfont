#!/usr/bin/env python3
"""Build an offline HTML board to eye-check E12 φ_s2 T2 judgments.

Same protocol as formal T2 (ref8 prototype vs hash-picked cross-typeface wrong),
but sampled characters + shuffled A/B so a human can judge before revealing scores.
Default: CPU, seed3407 / cache_v4 / best.pt — do not touch training GPUs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval_framework"))
from data import (  # noqa: E402
    GlyphDataset,
    as_chars,
    group_table,
    load_manifest,
    pick_negative,
    split_families,
)
from models import load_phi_checkpoint  # noqa: E402

REF8 = ["永", "和", "书", "风", "骨", "韵", "天", "地"]
# Mix of easy/hard Latin + digits for a quick pass (~11 per family).
PROBE_CHARS = ["A", "B", "M", "R", "a", "e", "g", "n", "0", "1", "8"]


def _b64_png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _copy_glyph(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "artifacts/e12/cache_v4"))
    ap.add_argument("--phi", default=str(ROOT / "runs/e12_phi_s2_v4_s3407/best.pt"))
    ap.add_argument("--split-seed", type=int, default=3407)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "reports/e12_eye_probe_v4_s3407"))
    ap.add_argument("--chars", default=",".join(PROBE_CHARS))
    args = ap.parse_args()

    cache = Path(args.cache)
    out = Path(args.out)
    img_root = out / "img"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    img_root.mkdir()

    manifest = load_manifest(cache)
    splits = split_families(manifest, (0.7, 0.15, 0.15), args.split_seed, by_group=True)
    groups = group_table(manifest)
    test_fams = splits["test"]
    all_f = sorted({x["family"] for x in manifest["fonts"]})
    chars = as_chars([c.strip() for c in args.chars.split(",") if c.strip()])
    need = REF8 + chars

    ds = GlyphDataset(cache, all_f, need, channels=3)
    by = {(r["family"], r["char"]): i for i, r in enumerate(ds.records)}
    path_by = {(r["family"], r["char"]): cache / r["path"] for r in ds.records}

    device = torch.device(args.device)
    print(f"[eye-probe] loading phi on {device} …", flush=True)
    phi = load_phi_checkpoint(args.phi, 3, 512, device).to(device).eval()

    def embed_many(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], torch.Tensor]:
        """Batch embed (family, char) pairs; returns CPU normalized vectors."""
        out_map: dict[tuple[str, str], torch.Tensor] = {}
        bs = 64
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            xs = torch.stack([ds[by[p]][0] for p in chunk]).to(device)
            with torch.no_grad():
                zs = phi(xs).cpu()
            for p, z in zip(chunk, zs):
                out_map[p] = z
            print(f"[eye-probe] embedded {min(i+bs, len(pairs))}/{len(pairs)}", flush=True)
        return out_map

    # Only test families + their T2 negatives (not full pool).
    pair_fams: list[tuple[str, str]] = []
    for fam in test_fams:
        wrong = pick_negative(fam, all_f, groups, cross_group=True)
        if wrong is None:
            continue
        pair_fams.append((fam, wrong))
    need_fams = sorted({f for pair in pair_fams for f in pair})
    emb_keys = [(f, c) for f in need_fams for c in REF8 + chars]
    emb = embed_many(emb_keys)

    prototypes = {}
    for f in need_fams:
        z = torch.stack([emb[(f, c)] for c in REF8]).mean(0)
        prototypes[f] = torch.nn.functional.normalize(z, dim=0)

    trials = []
    fam_stats = []
    rng_global = random.Random(args.split_seed)

    for fam, wrong in pair_fams:
        # Copy refs
        ref_rels = []
        for c in REF8:
            rel = f"ref/{fam}/u{ord(c):06X}.png"
            _copy_glyph(path_by[(fam, c)], img_root / rel)
            ref_rels.append(rel)

        rows = []
        for ch in chars:
            z = emb[(fam, ch)]
            s_ok = float(z @ prototypes[fam])
            s_bad = float(z @ prototypes[wrong])
            e12_pick = "correct" if s_ok >= s_bad else "wrong"
            margin = s_ok - s_bad

            left_is_correct = bool(rng_global.randrange(2))
            if left_is_correct:
                left_src, right_src = path_by[(fam, ch)], path_by[(wrong, ch)]
                left_role, right_role = "correct", "wrong"
            else:
                left_src, right_src = path_by[(wrong, ch)], path_by[(fam, ch)]
                left_role, right_role = "wrong", "correct"

            slug = hashlib.sha256(f"{fam}|{ch}|{args.split_seed}".encode()).hexdigest()[:10]
            left_rel = f"trials/{slug}_L.png"
            right_rel = f"trials/{slug}_R.png"
            _copy_glyph(left_src, img_root / left_rel)
            _copy_glyph(right_src, img_root / right_rel)

            row = {
                "id": slug,
                "char": ch,
                "bucket": "digit" if ch.isdigit() else ("upper" if ch.isupper() else "lower"),
                "left": left_rel,
                "right": right_rel,
                "left_role": left_role,
                "right_role": right_role,
                "answer_side": "L" if left_role == "correct" else "R",
                "e12_pick_side": ("L" if left_role == e12_pick else "R"),
                "e12_pick": e12_pick,
                "score_correct": round(s_ok, 4),
                "score_wrong": round(s_bad, 4),
                "margin": round(margin, 4),
                "e12_correct_vs_gt": e12_pick == "correct",
            }
            rows.append(row)
            trials.append({"family": fam, "wrong": wrong, **row})
        fam_stats.append(
            {
                "family": fam,
                "wrong": wrong,
                "group": groups.get(fam),
                "wrong_group": groups.get(wrong),
                "ref": ref_rels,
                "n": len(rows),
                "e12_acc": round(sum(1 for r in rows if r["e12_correct_vs_gt"]) / max(1, len(rows)), 3),
                "mean_margin": round(sum(r["margin"] for r in rows) / max(1, len(rows)), 4),
                "rows": rows,
            }
        )

    hard = sorted(trials, key=lambda t: abs(t["margin"]))[:12]
    e12_wrong = [t for t in trials if not t["e12_correct_vs_gt"]]
    payload = {
        "title": "E12 人眼探针 · φ_s2 T2（v4 / seed3407）",
        "protocol": {
            "ref8": REF8,
            "probe_chars": chars,
            "split_seed": args.split_seed,
            "cache": str(cache),
            "phi": str(args.phi),
            "negatives": "cross_typeface hash pick (same as formal T2)",
            "formal_t2_auc": 0.7906,
            "note": "先选左右哪张更像上方中文风格；再点「揭晓」。这是审计 scorer，不是比 F0/F3。",
        },
        "families": fam_stats,
        "hard_smallest_margin": hard,
        "e12_mistakes": e12_wrong,
        "summary": {
            "n_trials": len(trials),
            "e12_acc_on_probe": round(sum(1 for t in trials if t["e12_correct_vs_gt"]) / max(1, len(trials)), 3),
            "n_e12_mistakes": len(e12_wrong),
        },
    }
    (out / "probe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "index.html").write_text(_html(payload), encoding="utf-8")
    print(json.dumps({"out": str(out), **payload["summary"], "families": [f["family"] for f in fam_stats]}, ensure_ascii=False))


def _html(data: dict) -> str:
    # Keep CSS simple; scores hidden until reveal.
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{data["title"]}</title>
<style>
:root{{--bg:#f2f4f6;--card:#fff;--ink:#1a2330;--muted:#5c6773;--line:#d7dde4;--ok:#1b7f4a;--bad:#a33;--accent:#245b78}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,"Noto Sans SC",sans-serif}}
header,main,footer{{max-width:980px;margin:auto;padding:18px 16px}}
h1{{font-size:22px;margin:0 0 8px}}
h2{{font-size:17px;margin:28px 0 10px}}
.muted{{color:var(--muted);font-size:13px}}
.note{{background:var(--card);border-left:4px solid var(--accent);padding:10px 12px;margin:12px 0}}
.stat{{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}}
.stat span{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 10px;font-size:13px}}
.fam{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}}
.refs{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 12px}}
.refs img{{width:56px;height:56px;background:#fff;border:1px solid var(--line)}}
.trial{{border-top:1px solid var(--line);padding:12px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:420px}}
.opt{{border:2px solid var(--line);border-radius:8px;padding:8px;cursor:pointer;text-align:center;background:#fafbfc}}
.opt.selected{{border-color:var(--accent)}}
.opt.reveal-ok{{border-color:var(--ok);background:#eef8f1}}
.opt.reveal-bad{{border-color:var(--bad);background:#fdeeee}}
.opt img{{width:96px;height:96px;display:block;margin:0 auto}}
.btn{{appearance:none;border:1px solid var(--line);background:#eef2f5;border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit}}
.btn:hover{{border-color:var(--accent)}}
.score{{display:none;margin-top:8px;font-size:13px;color:var(--muted)}}
.score.show{{display:block}}
.tag{{display:inline-block;font-size:11px;padding:1px 6px;border-radius:4px;background:#eef2f5;margin-left:6px}}
.tag.ok{{background:#e5f6ec;color:var(--ok)}}
.tag.bad{{background:#fde8e8;color:var(--bad)}}
</style>
</head>
<body>
<header>
  <p class="muted">离线探针 · 先判断再揭晓 · 对应正式 T2 协议</p>
  <h1>{data["title"]}</h1>
  <div class="note">{data["protocol"]["note"]}<br/>
  正式门 T2 AUC≈{data["protocol"]["formal_t2_auc"]}（门限 0.90，未过）。本页探针字符：{", ".join(data["protocol"]["probe_chars"])}。</div>
  <div class="stat">
    <span>题量 <b>{data["summary"]["n_trials"]}</b></span>
    <span>E12 在本探针上相对 GT 正确率 <b>{data["summary"]["e12_acc_on_probe"]}</b></span>
    <span>E12 选错 <b>{data["summary"]["n_e12_mistakes"]}</b> 题</span>
    <span>你的作答正确率 <b id="humanAcc">—</b></span>
    <span>你与 E12 一致率 <b id="agreeAcc">—</b></span>
  </div>
  <p class="muted">用法：看上方 8 个中文 → 点左/右哪张拉丁更像同款风格 →「揭晓本题」。全部做完后看页顶统计。</p>
</header>
<main id="main"></main>
<footer class="muted">seed {data["protocol"]["split_seed"]} · cache_v4 · φ={data["protocol"]["phi"]}</footer>
<script>
const DATA = {json.dumps(data, ensure_ascii=False)};
const answers = {{}}; // id -> 'L'|'R'

function render() {{
  const main = document.getElementById('main');
  let html = '';
  DATA.families.forEach((fam, fi) => {{
    html += `<section class="fam"><h2>${{fi+1}}. ${{fam.family}} <span class="muted">vs wrong: ${{fam.wrong}}</span></h2>`;
    html += `<p class="muted">E12 本族探针正确率 ${{fam.e12_acc}} · 平均 margin ${{fam.mean_margin}}</p>`;
    html += `<div class="refs">` + fam.ref.map((r,i)=>`<div><img src="img/${{r}}" alt=""/><div class="muted" style="text-align:center;font-size:11px">${{DATA.protocol.ref8[i]}}</div></div>`).join('') + `</div>`;
    fam.rows.forEach(row => {{
      html += `<div class="trial" data-id="${{row.id}}">
        <div><b>字符 ${{row.char}}</b><span class="tag">${{row.bucket}}</span></div>
        <div class="pair">
          <div class="opt" data-side="L" onclick="pick('${{row.id}}','L')"><div class="muted">A</div><img src="img/${{row.left}}" alt=""/></div>
          <div class="opt" data-side="R" onclick="pick('${{row.id}}','R')"><div class="muted">B</div><img src="img/${{row.right}}" alt=""/></div>
        </div>
        <button class="btn" onclick="reveal('${{row.id}}')">揭晓本题</button>
        <div class="score" id="score-${{row.id}}"></div>
      </div>`;
    }});
    html += `</section>`;
  }});

  if (DATA.e12_mistakes.length) {{
    html += `<h2>E12 在本探针上选错的题（可对照）</h2><p class="muted">margin 为 cos(correct)−cos(wrong)；负值=模型更偏向 wrong 族。</p><ul>`;
    DATA.e12_mistakes.forEach(t => {{
      html += `<li><code>${{t.family}}</code> / ${{t.char}} · margin ${{t.margin}} · scores ${{t.score_correct}} vs ${{t.score_wrong}}</li>`;
    }});
    html += `</ul>`;
  }}
  main.innerHTML = html;
}}

function pick(id, side) {{
  answers[id] = side;
  const trial = document.querySelector(`.trial[data-id="${{id}}"]`);
  trial.querySelectorAll('.opt').forEach(el => el.classList.toggle('selected', el.dataset.side===side));
  updateStats();
}}

function findRow(id) {{
  for (const f of DATA.families) for (const r of f.rows) if (r.id===id) return r;
  return null;
}}

function reveal(id) {{
  const row = findRow(id);
  const trial = document.querySelector(`.trial[data-id="${{id}}"]`);
  trial.querySelectorAll('.opt').forEach(el => {{
    const role = el.dataset.side==='L' ? row.left_role : row.right_role;
    el.classList.add(role==='correct' ? 'reveal-ok' : 'reveal-bad');
  }});
  const box = document.getElementById('score-'+id);
  const human = answers[id];
  const humanOk = human ? (human===row.answer_side) : null;
  const agree = human ? (human===row.e12_pick_side) : null;
  box.classList.add('show');
  box.innerHTML = `正确答案：${{row.answer_side==='L'?'A':'B'}}（同族） · E12 选：${{row.e12_pick_side==='L'?'A':'B'}}（${{row.e12_correct_vs_gt?'对':'错'}}）<br/>
    cos同族=${{row.score_correct}} · cos错族=${{row.score_wrong}} · margin=${{row.margin}}<br/>
    你的选择：${{human? (human==='L'?'A':'B') : '未选'}} ${{humanOk===null?'':(humanOk?'✓':'✗')}} · 与 E12 ${{agree===null?'—':(agree?'一致':'不一致')}}`;
  updateStats();
}}

function updateStats() {{
  let nH=0, okH=0, nA=0, okA=0;
  for (const id of Object.keys(answers)) {{
    const row = findRow(id);
    if (!row) continue;
    nH++; if (answers[id]===row.answer_side) okH++;
    nA++; if (answers[id]===row.e12_pick_side) okA++;
  }}
  document.getElementById('humanAcc').textContent = nH ? (okH/nH).toFixed(3)+` (${{okH}}/${{nH}})` : '—';
  document.getElementById('agreeAcc').textContent = nA ? (okA/nA).toFixed(3)+` (${{okA}}/${{nA}})` : '—';
}}

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
