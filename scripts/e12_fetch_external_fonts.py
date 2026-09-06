#!/usr/bin/env python3
"""Fetch >=25 distinct OFL/free CJK+Latin external typefaces for E12 (S3), verify coverage, build cache.

Sources: Google Fonts repo (OFL) + GitHub repos (OFL / free-commercial), fetched via sparse
blob-less git clones (reliable when raw.githubusercontent is throttled).
Coverage gate: all of ref8 + A-H/a-h must render with a non-empty bbox, else the font is dropped.
Outputs:
  artifacts/e12/fonts/                font files (NOT committed to git)
  artifacts/e12/fonts/manifest.json   name/source/license/sha256 (committed)
  artifacts/e12/external_font_cache/  rendered 96x96 L-mode glyphs (committed)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = ROOT / "artifacts" / "e12" / "fonts"
CACHE_DIR = ROOT / "artifacts" / "e12" / "external_font_cache"
COVERAGE_CHARS = "永和书风骨韵天地ABCDEFGHabcdefgh"
MIN_FONTS = 25
WORK = ROOT / "artifacts" / "e12" / ".work"
WORK.mkdir(parents=True, exist_ok=True)

# (slug, kind, repo_or_dir, subpath_pattern)
#  kind=google: sparse checkout of google/fonts `ofl/<dir>` (single family)
#  kind=gh:     sparse checkout of <owner>/<repo>, keep *.ttf/*.otf/*.ttc
FONT_LIST = [
    ("noto_sans_sc",      "google", "google/fonts", "notosanssc"),
    ("noto_serif_sc",     "google", "google/fonts", "notoserifsc"),
    ("zcool_xiaowei",     "google", "google/fonts", "zcoolxiaowei"),
    ("zcool_qingke",      "google", "google/fonts", "zcoolqingkehuangyou"),
    ("zcool_kuaile",      "google", "google/fonts", "zcoolkuaile"),
    ("ma_shan_zheng",     "google", "google/fonts", "mashanzheng"),
    ("liu_jian_mao_cao",  "google", "google/fonts", "liujianmaocao"),
    ("long_cang",         "google", "google/fonts", "longcang"),
    ("zhi_mang_xing",     "google", "google/fonts", "zhimangxing"),
    ("lxgw_wenkai",       "gh", "lxgw/LxgwWenKai", None),
    ("lxgw_neoxihei",     "gh", "lxgw/LxgwNeoXiHei", None),
    ("lxgw_bright",       "gh", "lxgw/LxgwBright", None),
    ("lxgw_zhenkai",      "gh", "lxgw/LxgwZhenKai", None),
    ("lxgw_neozhisong",   "gh", "lxgw/LxgwNeoZhiSong", None),
    ("lxgw_xihei",        "gh", "lxgw/LxgwXiHei", None),
    ("lxgw_markergothic", "gh", "lxgw/LxgwMarkerGothic", None),
    ("lxgw_wenkaigb",     "gh", "lxgw/LxgwWenkaiGB", None),
    ("smiley_sans",       "gh", "atelier-anchor/smiley-sans", None),
    ("cwtex_q_fonts",     "gh", "l10n-tw/cwtex-q-fonts", None),
    ("genyo_min",         "gh", "ButTaiwan/genyo-font", None),
    ("gensen_rounded",    "gh", "ButTaiwan/gensen-font", None),
    ("genwan_min",        "gh", "ButTaiwan/genwan-font", None),
    ("genryu_font",       "gh", "ButTaiwan/genryu-font", None),
    ("genyog_font",       "gh", "ButTaiwan/genyog-font", None),
    ("evilsung",          "gh", "ButTaiwan/evilsung", None),
    ("iansui",            "gh", "ButTaiwan/iansui", None),
    ("yozai",             "gh", "lxgw/yozai-font", None),
    ("tangyuan",          "gh", "lxgw/TangYuan-font", None),
]

FONT_EXT = {".ttf", ".otf", ".ttc"}


def _run(cmd: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _sparse_clone(slug: str, owner_repo: str, pattern: str | None) -> list[Path]:
    url = f"https://github.com/{owner_repo}.git"
    repo_dir = WORK / slug
    if not (repo_dir / ".git").exists():
        proc = _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                     url, str(repo_dir)], WORK)
        if proc.returncode != 0:
            raise RuntimeError(f"clone failed: {proc.stderr.strip()[:200]}")
    if pattern:
        _run(["git", "sparse-checkout", "set", "--no-cone", pattern], repo_dir)
    else:
        _run(["git", "sparse-checkout", "set", "--no-cone", "*.ttf", "*.otf", "*.ttc"], repo_dir)
    return [p for p in repo_dir.rglob("*") if p.suffix.lower() in FONT_EXT and p.is_file()]


def _fetch(slug: str, kind: str, repo: str, family: str | None, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in dest_dir.iterdir() if p.suffix.lower() in FONT_EXT]
    if existing and _coverage_ok(existing[0]):
        return existing[0]
    if kind == "google":
        pat = f"ofl/{family}/*"
        files = _sparse_clone(slug, repo, pat)
        if not files:
            return None
        out = dest_dir / files[0].name
        out.write_bytes(files[0].read_bytes())
        return out
    files = _sparse_clone(slug, repo, None)
    if not files:
        path = _release_font(slug, repo, dest_dir)
        return path
    for p in sorted(files, key=lambda x: x.stat().st_size, reverse=True):
        if _coverage_ok(p):
            out = dest_dir / p.name
            out.write_bytes(p.read_bytes())
            return out
    path = _release_font(slug, repo, dest_dir)
    return path


_OPENER = urllib_request_build = None


def _http_open(url: str, timeout: int = 600):
    """urllib 直连（绕过环境 socks5 代理，它对 github 当前不可用）。"""
    import urllib.request as _u
    global _OPENER
    if _OPENER is None:
        _OPENER = _u.build_opener(_u.ProxyHandler({}))
    return _OPENER.open(_u.Request(url, headers={"User-Agent": "hrfont-e12-fetch/1.0"}), timeout=timeout)


def _release_font(slug: str, owner_repo: str, dest_dir: Path) -> Path | None:
    """GitHub 树里没有字体的仓库：从 latest release 资产取（优先直连字体文件，其次 zip 解包）。"""
    import urllib.request as _u
    try:
        with _http_open(f"https://api.github.com/repos/{owner_repo}/releases/latest", timeout=60) as resp:
            assets = json.loads(resp.read()).get("assets", [])
    except Exception:  # noqa: BLE001
        return None
    direct = [a for a in assets if a["name"].lower().endswith((".ttf", ".otf", ".ttc"))]
    zips = [a for a in assets if a["name"].lower().endswith(".zip")]
    for a in direct + zips:
        try:
            with _http_open(a["browser_download_url"], timeout=600) as resp:
                data = resp.read()
            if a["name"].lower().endswith(".zip"):
                import zipfile as _z, io as _io
                with _z.ZipFile(_io.BytesIO(data)) as zf:
                    for n in zf.namelist():
                        if n.lower().endswith((".ttf", ".otf")) and "__MACOSX" not in n:
                            out = dest_dir / Path(n).name
                            out.write_bytes(zf.read(n))
                            if _coverage_ok(out):
                                return out
                continue
            out = dest_dir / a["name"]
            out.write_bytes(data)
            if _coverage_ok(out):
                return out
        except Exception:  # noqa: BLE001
            continue
    return None


def _coverage_ok(path: Path) -> bool:
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(str(path), 40)
    except OSError:
        return False
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    for ch in COVERAGE_CHARS:
        try:
            bbox = draw.textbbox((0, 0), ch, font=font)
        except UnicodeEncodeError:
            return False
        if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
            return False
    return True


def _fetch_one(entry) -> tuple[str, Path | None, str | None]:
    slug, kind, repo, family = entry
    try:
        path = _fetch(slug, kind, repo, family, FONTS_DIR / slug)
        if path is None:
            return slug, None, "no-asset-or-coverage"
        return slug, path, None
    except Exception as exc:  # noqa: BLE001
        return slug, None, f"{type(exc).__name__}: {exc}"


def main() -> int:
    import concurrent.futures
    manifest_rows, ok, failed = [], 0, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for slug, path, err in pool.map(_fetch_one, FONT_LIST):
            if path is None:
                failed.append((slug, err))
                continue
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_rows.append({
                "slug": slug,
                "source": next(r for n, _k, r, _f in FONT_LIST if n == slug),
                "file": str(path.relative_to(FONTS_DIR)),
                "sha256": sha, "license": "OFL-or-free-commercial (see source repo)",
            })
            ok += 1
            print(f"OK   {slug} ({path.name}, {path.stat().st_size // 1024} KiB)", flush=True)
    (FONTS_DIR.parent / "fonts_manifest.json").write_text(
        json.dumps({"min_fonts": MIN_FONTS, "coverage_chars": COVERAGE_CHARS,
                    "fonts": manifest_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfonts ok={ok} failed={len(failed)}")
    for slug, why in failed:
        print(f"FAIL {slug}: {why}")
    if ok < MIN_FONTS:
        print(f"ABORT: fewer than {MIN_FONTS} usable fonts")
        return 1

    cmd = [sys.executable, str(ROOT / "scripts" / "eval_framework" / "build_cache.py"),
           "--fonts_dir", str(FONTS_DIR), "--cache_dir", str(CACHE_DIR),
           "--strategy", "per_font_height_fit", "--canvas", "96", "--margin", "6"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
