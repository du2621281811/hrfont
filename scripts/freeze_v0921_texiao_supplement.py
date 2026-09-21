#!/usr/bin/env python3
"""Freeze texiao pass-2 human audit picks as supplement dataset v0921."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "manifests" / "v0921_texiao_supplement"
CONTRACT = ROOT / "manifests" / "v0921_texiao_supplement.json"
LIVE = Path("/root/texiao_fonts/live")
CHARSET = json.loads((ROOT / "manifests/charset_cn2west_v2_planned.json").read_text(encoding="utf-8"))
TARGET = CHARSET["target"]
TARGET_N = CHARSET["target_n"]
SCRIPT_TO_BUCKETS = {
    "latin": ["ascii_digits", "ascii_letters"],
    "latin_ext": ["latin_ext_letters"],
    "hiragana": ["hiragana"],
    "katakana": ["katakana"],
    "zhuyin": ["bopomofo"],
}
SK = list(SCRIPT_TO_BUCKETS)


def sha1_12(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def resolve_font(it: dict) -> Path | None:
    clean = it["clean"]
    cands = []
    if it.get("path"):
        cands.append(Path(it["path"]))
    for ext in (".TTF", ".ttf", ".OTF", ".otf", ".ttc", ".TTC"):
        cands.append(LIVE / f"{clean}{ext}")
    for p in cands:
        if p and p.is_file():
            return p
    return None


def font_cmap(path: Path) -> set[int]:
    from fontTools.ttLib import TTFont

    font = TTFont(str(path), fontNumber=0, lazy=True)
    try:
        cmap = set()
        for t in font["cmap"].tables:
            cmap.update(t.cmap.keys())
        return cmap
    finally:
        font.close()


def coverage_for(cmap: set[int], scripts: list[str]) -> tuple[dict, dict, str]:
    coverage: dict = {}
    missing: dict = {}
    ok = True
    for sk in scripts:
        for bucket in SCRIPT_TO_BUCKETS.get(sk, []):
            chars = TARGET[bucket]
            hit = sum(1 for ch in chars if ord(ch) in cmap)
            n = TARGET_N[bucket]
            full = hit == n
            coverage[bucket] = {"hit": hit, "n": n, "full": full}
            if not full:
                ok = False
                miss = "".join(ch for ch in chars if ord(ch) not in cmap)
                missing[bucket] = miss
    return coverage, missing, ("ok" if ok else "incomplete_selected_target")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from",
        dest="src",
        default=str(ROOT / "reports/texiao_fonts_screen/picks_pass2_export.json"),
    )
    args = ap.parse_args()
    src = Path(args.src)
    if not src.is_file():
        print(f"FAIL missing export: {src}", flush=True)
        return 1

    raw = json.loads(src.read_text(encoding="utf-8"))
    train_in = raw.get("train") or []
    if not train_in:
        print("FAIL empty train", flush=True)
        return 1

    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    incomplete = []
    missing_no_font = []
    script_counter: Counter[str] = Counter()
    for it in train_in:
        clean = it["clean"]
        scripts = [s for s in (it.get("scripts") or []) if s in SK]
        # keep latin if only digit was implied historically — UI uses latin for digits+letters
        scripts = [s for s in SK if s in scripts]
        path = resolve_font(it)
        if not path:
            missing_no_font.append(clean)
            continue
        try:
            cmap = font_cmap(path)
            coverage, missing_chars, qa = coverage_for(cmap, scripts)
        except Exception as e:
            print(f"WARN cmap {clean}: {e}", flush=True)
            coverage, missing_chars, qa = {}, {}, "cmap_error"
        for s in scripts:
            script_counter[s] += 1
        row = {
            "clean": clean,
            "disp": it.get("disp") or clean,
            "family": it.get("family") or clean,
            "scripts": scripts,
            "west_score": it.get("west_score"),
            "font_file": path.name,
            "font_sha1_12": sha1_12(path),
            "font_bytes": path.stat().st_size,
            "font_dir_hint": "texiao_fonts/live/",
            "path": str(path),
            "audit_a_fs": it.get("audit_a_fs"),
            "cover_probe": it.get("cover") or {},
            "qa_status": qa,
            "coverage": coverage,
        }
        if missing_chars:
            row["missing_chars"] = missing_chars
            incomplete.append(clean)
        rows.append(row)

    rows.sort(key=lambda r: (r["family"], r["clean"]))
    n = len(rows)
    n_ok = sum(1 for r in rows if r["qa_status"] == "ok")
    n_inc = n - n_ok
    fams = {r["family"] for r in rows}

    fonts_all = {
        "dataset_id": "v0921_texiao_supplement",
        "version": "v0921",
        "title": f"v0921 特效字体补充（二筛正式 {n}）",
        "created_at": "2026-09-21",
        "updated_at": now,
        "n_fonts": n,
        "n_families": len(fams),
        "n_ok": n_ok,
        "n_incomplete": n_inc,
        "source_export": str(src),
        "source_export_utc": raw.get("utc"),
        "pool": "texiao_fonts",
        "pass": "human_audit_pass2",
        "target_buckets": TARGET_N,
        "target_total": CHARSET["target_total"],
        "script_to_buckets": SCRIPT_TO_BUCKETS,
        "qa": {
            "policy": "keep_all_pass2; document missing_chars for incomplete selected targets",
            "cmap_checked_on_selected_scripts": True,
            "protocol": "A",
        },
        "incomplete_cleans": incomplete,
        "missing_font_files": missing_no_font,
        "script_counts": dict(script_counter),
        "train": rows,
        "important": {
            "official_n": n,
            "incomplete_documented_n": n_inc,
            "skip_missing_chars_when_training": True,
            "preview_probes_are_not_full_295": True,
            "not_merged_into_v0913_png_pairs": True,
        },
    }

    stems = [r["clean"] for r in rows]
    (OUT / "fonts_all.json").write_text(json.dumps(fonts_all, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # alias with count in name for collaborators
    (OUT / f"fonts_all_{n}.json").write_text(json.dumps(fonts_all, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "stems_all.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
    (OUT / f"stems_all_{n}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")

    miss_fonts = [
        {
            "clean": r["clean"],
            "disp": r["disp"],
            "scripts": r["scripts"],
            "font_file": r["font_file"],
            "missing_chars": r.get("missing_chars") or {},
            "coverage": {k: v for k, v in (r.get("coverage") or {}).items() if not v.get("full")},
        }
        for r in rows
        if r.get("missing_chars")
    ]
    miss_doc = {
        "dataset_id": "v0921_texiao_supplement",
        "n": len(miss_fonts),
        "note": "这些字体仍在正式包内；训练/渲染时请跳过 missing_chars 中的码点。",
        "fonts": miss_fonts,
    }
    (OUT / "MISSING_CHARS.json").write_text(json.dumps(miss_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# v0921 缺字说明",
        "",
        f"正式规模 **{n}**；其中勾选语种 target 不全 **{n_inc}** 套（仍收录）。",
        "",
        "| clean | scripts | 缺字 bucket | 缺字 |",
        "|-------|---------|-------------|------|",
    ]
    for r in miss_fonts:
        for bucket, chars in (r.get("missing_chars") or {}).items():
            lines.append(
                f"| `{r['clean']}` | {','.join(r['scripts'])} | `{bucket}` | `{chars}` |"
            )
    if not miss_fonts:
        lines.append("| — | — | — | （无） |")
    (OUT / "MISSING_CHARS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    qa = {
        "dataset_id": "v0921_texiao_supplement",
        "built_at": now,
        "n_export": raw.get("n"),
        "n_resolved": n,
        "n_ok": n_ok,
        "n_incomplete": n_inc,
        "n_missing_font_file": len(missing_no_font),
        "script_counts": dict(script_counter),
        "missing_font_files": missing_no_font,
    }
    (OUT / "QA_REPORT.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "QA_REPORT.md").write_text(
        "\n".join(
            [
                "# v0921 QA",
                "",
                f"- export n: **{raw.get('n')}**",
                f"- resolved fonts: **{n}**",
                f"- ok (selected target full): **{n_ok}**",
                f"- incomplete (documented): **{n_inc}**",
                f"- missing font file: **{len(missing_no_font)}**",
                f"- scripts: `{dict(script_counter)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    readme = f"""# v0921 特效字体补充数据（合作者说明）

**版本**：`v0921_texiao_supplement` · 冻结日 2026-09-21

## 重要（先读）

1. **正式规模是 {n} 套**（特效池人工一筛 + 协议 A 二筛）。
2. **每套字体带可用语种**：字段 `scripts`（`latin` / `latin_ext` / `hiragana` / `katakana` / `zhuyin`）。
3. **勾选语种 target 不全仍收录**；缺字见 `missing_chars` / [`MISSING_CHARS.md`](MISSING_CHARS.md)。训练时跳过缺字码点。
4. **审核页探针 ≠ 训练 target**（UI 约 8 字预览；训练字表 cn2west **295**）。
5. **本包是字体名单 + 语种/缺字元数据**，尚未并入 `v0913_clean` 的 PNG pair；TTF **不进 git**。

| 计数 | 值 |
|------|-----|
| 正式字体 | **{n}** |
| 勾选语种 target 全满 | {n_ok} |
| Ext/语种不全但已写明 | {n_inc} |
| 源导出 utc | `{raw.get('utc')}` |

## 正式文件

| 用途 | 路径 |
|------|------|
| **合同** | [`../v0921_texiao_supplement.json`](../v0921_texiao_supplement.json) |
| **全量清单** | [`fonts_all_{n}.json`](fonts_all_{n}.json) |
| stem 列表 | [`stems_all_{n}.txt`](stems_all_{n}.txt) |
| 缺字说明 | [`MISSING_CHARS.md`](MISSING_CHARS.md) |

## 与其它波次

- `v0913_clean`：主线可用性 pair。
- `v0917_west_style`：方正西文风格补充（232）。
- `v0921_texiao_supplement`：特效字体池人工二筛补充（本包）。

## TTF

逻辑目录：`/root/texiao_fonts/live/<font_file>`（主机 `/home/data/male/texiao_fonts`）。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "SYNC.md").write_text(readme, encoding="utf-8")
    (OUT / "DOWNLOAD.md").write_text(
        "\n".join(
            [
                "# v0921 TTF 获取",
                "",
                "- 源：特效字体 live 目录 `/root/texiao_fonts/live/`",
                "- stem 列表：`stems_all.txt`",
                "- 打包可复用 `reports/texiao_fonts_screen/pass1_pack_*` 流程，按本版 stems 重打。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # families rollup
    fam_map: dict[str, list[str]] = {}
    for r in rows:
        fam_map.setdefault(r["family"], []).append(r["clean"])
    (OUT / "families.json").write_text(
        json.dumps({"n_families": len(fam_map), "families": fam_map}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # contract + INDEX
    files_meta = {}
    for p in sorted(OUT.iterdir()):
        if p.is_file():
            files_meta[p.name] = {
                "path": f"manifests/v0921_texiao_supplement/{p.name}",
                "bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }
    contract = {
        "schema_version": 2,
        "dataset_id": "v0921_texiao_supplement",
        "kind": "texiao_supplement_font_manifest",
        "status": "frozen",
        "created_at": "2026-09-21",
        "updated_at": now,
        "note": f"正式规模={n}。特效池人工一筛+协议A二筛。每套含 scripts；不全仍收录，缺字见 missing_chars。说明见 manifests/v0921_texiao_supplement/README.md。",
        "primary_manifest": f"manifests/v0921_texiao_supplement/fonts_all_{n}.json",
        "missing_chars": "manifests/v0921_texiao_supplement/MISSING_CHARS.md",
        "stems": f"manifests/v0921_texiao_supplement/stems_all_{n}.txt",
        "font_dir_hint": "texiao_fonts/live/",
        "source_export": str(src),
        "counts": {"all": n, "ok_full_selected_target": n_ok, "incomplete_documented": n_inc},
        "target_total": CHARSET["target_total"],
        "target_buckets": TARGET_N,
        "script_to_buckets": SCRIPT_TO_BUCKETS,
        "download": "manifests/v0921_texiao_supplement/DOWNLOAD.md",
        "title": f"v0921 特效字体补充（{n} 正式；缺字写明）",
        "readme": "manifests/v0921_texiao_supplement/README.md",
        "important": {
            "official_n": n,
            "incomplete_documented_n": n_inc,
            "primary_manifest": f"manifests/v0921_texiao_supplement/fonts_all_{n}.json",
            "skip_missing_chars_when_training": True,
            "preview_probes_are_not_full_295": True,
        },
    }
    CONTRACT.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files_meta["v0921_texiao_supplement.json"] = {
        "path": "manifests/v0921_texiao_supplement.json",
        "bytes": CONTRACT.stat().st_size,
        "sha256": file_sha256(CONTRACT),
    }
    index = {
        "schema_version": 2,
        "dataset_id": "v0921_texiao_supplement",
        "status": "frozen",
        "built_at": now,
        "policy": f"{n}_all_with_missing_chars_documented",
        "counts": contract["counts"],
        "files": files_meta,
        "download": contract["download"],
        "readme": contract["readme"],
        "important": {"official_n": n, "incomplete_documented_n": n_inc},
    }
    (OUT / "INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # also keep a copy next to screen reports
    report_dir = ROOT / "reports" / "texiao_fonts_screen"
    (report_dir / "picks_v0921_export.json").write_text(
        json.dumps(
            {
                "title": "texiao_fonts_picks_audit",
                "dataset_id": "v0921_texiao_supplement",
                "utc": raw.get("utc") or now,
                "n": n,
                "render": "protocol_A",
                "train": [
                    {
                        "clean": r["clean"],
                        "disp": r["disp"],
                        "family": r["family"],
                        "scripts": r["scripts"],
                        "path": r["path"],
                        "west_score": r["west_score"],
                        "cover": r.get("cover_probe") or {},
                        "audit_a_fs": r.get("audit_a_fs"),
                    }
                    for r in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"DONE v0921 n={n} ok={n_ok} incomplete={n_inc} missing_file={len(missing_no_font)} out={OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
