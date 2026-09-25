#!/usr/bin/env python3
"""Place FontDiffuser TargetImage fonts under code/ours/FontDiffuser/ttf/target/.

Official mapping (data_examples):
  FZGuanJKSJW   → 方正管峻楷书简体
  FZOuYHGXSJW   → 方正欧阳荷庚行书简体
  FZZCHJW       → 方正正粗黑简体

Search order: ttf/target/ (already present) → suiti → font_50 → symlink.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
FD = ROOT / "code/ours/FontDiffuser"
OUT = FD / "ttf/target"
SUITI = ROOT / "data/suiti_fonts_probe/随体字体集"
FONT50 = Path("/root/data/font_50")

FONTS = {
    "FZGuanJKSJW": {
        "cn": "方正管峻楷书简体",
        "foundertype_id": 680,
        "url": "https://www.foundertype.com/index.php/FontInfo/index/id/680",
    },
    "FZOuYHGXSJW": {
        "cn": "方正欧阳荷庚行书简体",
        "foundertype_id": 5764,
        "url": "https://www.foundertype.com/index.php/FontInfo/ductdetail/id/5764",
    },
    "FZZCHJW": {
        "cn": "方正正粗黑简体",
        "foundertype_id": 6131,
        "url": "https://www.foundertype.com/index.php/FontInfo/index/id/6131",
    },
}


def find_ttf(stem: str) -> Path | None:
    for d in (OUT, SUITI, FONT50, ROOT / "data"):
        if not d.is_dir():
            continue
        for ext in (".TTF", ".ttf", ".otf", ".OTF"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
        for p in list(d.glob(f"{stem}.*")):
            if p.suffix.lower() in (".ttf", ".otf"):
                return p
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = {}
    for stem, meta in FONTS.items():
        dst = OUT / f"{stem}.TTF"
        if dst.exists():
            status[stem] = {"ok": True, "path": str(dst), "source": "existing"}
            continue
        src = find_ttf(stem)
        if src:
            shutil.copy2(src, dst)
            status[stem] = {"ok": True, "path": str(dst), "source": str(src)}
        else:
            status[stem] = {
                "ok": False,
                "cn": meta["cn"],
                "manual": meta["url"],
                "hint": f"Copy {stem}.TTF to {dst}",
            }
    (OUT / "font_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for v in status.values() if v.get("ok"))
    print(f"[fonts] {ok}/{len(FONTS)} ready in {OUT}")
    for stem, v in status.items():
        print(f"  {stem}: {'OK ' + v.get('source','') if v.get('ok') else 'MISSING → ' + v.get('manual','')}")
    return 0 if ok == len(FONTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
