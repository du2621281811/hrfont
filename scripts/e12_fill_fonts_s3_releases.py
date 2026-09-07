#!/usr/bin/env python3
"""Fill missing fonts_s3 files from GitHub releases (no proxy). Does not touch F3."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
FONTS = ROOT / "artifacts" / "e12" / "fonts_s3"
UA = {"User-Agent": "hrfont-e12-fetch/1.0"}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# slug -> (repo, preferred filename substring)
NEED = [
    ("genyo_min", "ButTaiwan/genyo-font", "GenYoMin2TC-H"),
    ("gensen_rounded", "ButTaiwan/gensen-font", "GenSenRounded2-H"),
    ("genwan_min", "ButTaiwan/genwan-font", "GenWanMin2"),
    ("genryu_font", "ButTaiwan/genryu-font", "GenRyuMin2-R"),
    ("genyog_font", "ButTaiwan/genyog-font", "GenYoGothic2TC-H"),
    ("evilsung", "ButTaiwan/evilsung", "EvilSung-Regular"),
    ("iansui", "ButTaiwan/iansui", "Iansui-Regular"),
    ("yozai", "lxgw/yozai-font", "Yozai-Regular"),
    ("tangyuan", "lxgw/TangYuan-font", "TangYuan"),
]


def get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with OPENER.open(req, timeout=timeout) as resp:
        return resp.read()


def assets(repo: str) -> list[dict]:
    data = json.loads(get(f"https://api.github.com/repos/{repo}/releases/latest", timeout=60))
    return list(data.get("assets") or [])


def pick(items: list[dict], needle: str) -> dict | None:
    needle = needle.lower()
    scored = []
    for a in items:
        name = a["name"]
        low = name.lower()
        if not low.endswith((".ttf", ".otf", ".ttc", ".zip")):
            continue
        pen = 0
        if needle.lower() not in low and needle.replace("-", "").lower() not in low.replace("-", ""):
            pen += 20
        if any(x in low for x in ("italic", "oblique", "mono", "hanonly")):
            pen += 50
        if "regular" in low:
            pen -= 5
        scored.append((pen, -a.get("size", 0), a))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2] if scored else None


def save_font(slug: str, name: str, data: bytes) -> Path:
    dest = FONTS / slug
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.iterdir():
        if old.suffix.lower() in {".ttf", ".otf", ".ttc"}:
            old.unlink()
    out = dest / Path(name).name
    out.write_bytes(data)
    return out


def from_zip(slug: str, needle: str, data: bytes) -> Path | None:
    import io
    import zipfile
    needle_l = needle.lower()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        cands = [n for n in zf.namelist()
                 if n.lower().endswith((".ttf", ".otf", ".ttc")) and "__MACOSX" not in n]
        if not cands:
            return None
        def score(n: str) -> tuple:
            low = Path(n).name.lower()
            pen = 0 if needle_l in low else 10
            if "italic" in low or "oblique" in low:
                pen += 50
            if "regular" in low:
                pen -= 5
            return (pen, -len(n))
        cands.sort(key=score)
        name = Path(cands[0]).name
        return save_font(slug, name, zf.read(cands[0]))


def main() -> int:
    ok = 0
    for slug, repo, needle in NEED:
        print(f"... {slug}", flush=True)
        try:
            items = assets(repo)
            a = pick(items, needle)
            if a is None:
                print(f"FAIL {slug}: no asset")
                continue
            blob = get(a["browser_download_url"], timeout=300)
            if a["name"].lower().endswith(".zip"):
                path = from_zip(slug, needle, blob)
            else:
                path = save_font(slug, a["name"], blob)
            if path is None:
                print(f"FAIL {slug}: zip empty")
                continue
            print(f"OK   {slug} ({path.name}, {path.stat().st_size // 1024} KiB)", flush=True)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {slug}: {type(exc).__name__}: {exc}", flush=True)
    n = len(list(FONTS.rglob("*.ttf")) + list(FONTS.rglob("*.otf")) + list(FONTS.rglob("*.ttc")))
    print(f"downloaded_now={ok} fonts_s3_total={n}")
    return 0 if n >= 25 else 1


if __name__ == "__main__":
    raise SystemExit(main())
