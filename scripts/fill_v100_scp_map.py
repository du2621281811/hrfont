#!/usr/bin/env python3
"""Fill manifests/v100_scp_map.json on the *source* host (3090).

Run from /root/projects/hrfont. Writes sizes, existence, hostname, git HEAD,
and stem→TTF paths. Does not copy any bytes. Commit only the JSON.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
MAP_PATH = ROOT / "manifests/v100_scp_map.json"
REF_SUMMARY = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2b-hfit/summary.json"
SUITI = ROOT / "data/suiti_fonts_probe/随体字体集"
FONT50 = Path("/root/data/font_50")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def bytes_of(path: Path) -> int | None:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        total = 0
        for p in path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
            elif p.is_symlink() and p.exists() and p.is_file():
                total += p.stat().st_size
        return total
    return None


def resolve_ttf(stem: str, hinted: str | None) -> str:
    candidates: list[Path] = []
    if hinted:
        candidates.append(Path(hinted))
    for root in (SUITI, FONT50, ROOT / "data"):
        if not root.is_dir():
            continue
        for ext in (".TTF", ".ttf", ".otf", ".OTF"):
            candidates.append(root / f"{stem}{ext}")
        for p in root.glob(f"{stem}.*"):
            if p.suffix.lower() in (".ttf", ".otf"):
                candidates.append(p)
    for p in candidates:
        if p.is_file():
            return str(p)
    return hinted or ""


def load_stems() -> list[dict]:
    rows = []
    for split in ("train", "val", "test"):
        p = ROOT / f"manifests/pipeline_v3_{split}_stems.txt"
        for line in p.read_text(encoding="utf-8").splitlines():
            stem = line.strip()
            if stem:
                rows.append({"split": split, "stem": stem, "src_ttf": "", "exists": None})
    return rows


def main() -> int:
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    src = data.setdefault("hosts", {}).setdefault("src_3090", {})
    src["hostname"] = socket.gethostname()
    src["filled_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    src["git_head"] = git_head()
    src["project_root"] = str(ROOT)
    data["filled_by"] = os.environ.get("USER") or os.environ.get("LOGNAME") or "root"
    data["filled_at"] = src["filled_at"]
    data["src_git_head"] = src["git_head"]

    ttf_by_stem: dict[str, str] = {}
    if REF_SUMMARY.is_file():
        ref = json.loads(REF_SUMMARY.read_text(encoding="utf-8"))
        for f in ref.get("fonts") or []:
            ttf_by_stem[f.get("stem", "")] = f.get("ttf") or ""
        data.setdefault("ttf_render_optional", {})["ref_summary_exists"] = True
        data["ttf_render_optional"]["ref_summary_src"] = str(REF_SUMMARY)
        data["ttf_render_optional"]["pillow"] = (ref.get("software") or {}).get("pillow")
    else:
        data.setdefault("ttf_render_optional", {})["ref_summary_exists"] = False

    for item in data.get("payloads") or []:
        rel = item.get("src_rel") or ""
        path = ROOT / rel
        exists = path.exists()
        size = bytes_of(path) if exists else None
        item["exists_on_src"] = exists
        item["bytes_on_src"] = size
        item["src_abs"] = str(path)

    opt = data.setdefault("ttf_render_optional", {})
    stems = opt.get("stems") or load_stems()
    missing = []
    for row in stems:
        hinted = ttf_by_stem.get(row["stem"]) or row.get("src_ttf")
        path = resolve_ttf(row["stem"], hinted)
        row["src_ttf"] = path
        row["exists"] = bool(path) and Path(path).is_file()
        if not row["exists"]:
            missing.append(row["stem"])
    opt["stems"] = stems
    opt["n_stems"] = len(stems)
    opt["n_ttf_ok"] = len(stems) - len(missing)
    opt["n_ttf_missing"] = len(missing)
    opt["missing_stems"] = missing[:40]
    opt["font_search_roots_exist"] = {
        str(FONT50): FONT50.is_dir(),
        str(SUITI): SUITI.is_dir(),
    }

    MAP_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MAP_PATH}")
    print(f"  src hostname={src.get('hostname')} HEAD={src.get('git_head', '')[:12]}")
    for item in data.get("payloads") or []:
        sz = item.get("bytes_on_src")
        human = f"{sz/1e9:.2f}G" if isinstance(sz, int) else "-"
        print(f"  [{('OK' if item.get('exists_on_src') else 'MISS'):4}] {human:>8}  {item.get('id')}  {item.get('src_rel')}")
    print(f"  TTF {opt.get('n_ttf_ok')}/{opt.get('n_stems')} found; missing={opt.get('n_ttf_missing')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
