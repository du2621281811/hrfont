#!/usr/bin/env python3
"""Smoke-check hidden L2/L3 page on :19000 without touching L1 decisions.json."""
from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

L1 = Path("/root/projects/hrfont/data/p649_v2a_review/decisions.json")
BOARD = "http://127.0.0.1:19000"
PUBLIC = "http://172.19.45.13:19000"
LAYERS = "/p649_v2a_layers/review.html"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def http(url: str, data: bytes | None = None, method: str | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, method=method or ("PUT" if data is not None else "GET"))
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> int:
    fails: list[str] = []
    before = sha(L1)
    code, body = http(BOARD + LAYERS)
    if code != 200:
        fails.append(f"127.0.0.1 layers GET {code}")
    text = body.decode("utf-8", "replace")
    if "分层可用性" not in text:
        fails.append("layers html missing title")
    if "p649_v2a_layers" not in http(BOARD + "/")[1].decode("utf-8", "replace"):
        pass
    else:
        fails.append("board index unexpectedly links layers")
    pub_code, _ = http(PUBLIC + LAYERS)
    if pub_code != 200:
        fails.append(f"172.19.45.13 layers GET {pub_code}")
    portal_code, _ = http("http://127.0.0.1:19001" + LAYERS)
    if portal_code != 404:
        fails.append(f"19001 layers should 404, got {portal_code}")
    forbid_code, _ = http(BOARD + "/p649_v2a_layers/decisions.json", b"{}")
    if forbid_code not in {403, 404}:
        fails.append(f"layers decisions PUT should be blocked, got {forbid_code}")
    after = sha(L1)
    if after != before:
        fails.append("L1 decisions.json hash changed")
    if fails:
        print("FAIL")
        for f in fails:
            print(" -", f)
        return 1
    print("OK hidden layers on :19000; L1 hash unchanged; 19001 still 404")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
