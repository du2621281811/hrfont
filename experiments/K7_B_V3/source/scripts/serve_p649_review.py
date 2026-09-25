#!/usr/bin/env python3
"""Serve hrfont/data and persist p649 review decisions to disk.

GET/PUT  /p649_v2a_review/decisions.json
Also writes decisions.jsonl next to it (resume / 交卷).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/root/projects/hrfont/data")
REVIEW = ROOT / "p649_v2a_review"
DECISIONS_JSON = REVIEW / "decisions.json"
DECISIONS_JSONL = REVIEW / "decisions.jsonl"
TEST_JSON = REVIEW / "decisions.test.json"
TEST_JSONL = REVIEW / "decisions.test.jsonl"
PUT_STORES = {
    "/p649_v2a_review/decisions.json": (DECISIONS_JSON, DECISIONS_JSONL),
    "/p649_v2a_review/decisions.test.json": (TEST_JSON, TEST_JSONL),
}


def merge_decisions(old: dict, incoming: dict) -> dict:
    out = dict(old)
    for stem, rec in incoming.items():
        if not isinstance(rec, dict):
            continue
        cur = out.get(stem) or {}
        if (rec.get("reviewed_at") or "") >= (cur.get("reviewed_at") or ""):
            out[stem] = rec
    return out


def merge_reviewers(old: dict, incoming: dict) -> dict:
    out = dict(old or {})
    for name, rec in (incoming or {}).items():
        name = str(name or "").strip()
        if not name or not isinstance(rec, dict):
            continue
        cur = out.get(name) or {}
        if (rec.get("updated_at") or "") >= (cur.get("updated_at") or ""):
            merged = dict(cur)
            merged.update(rec)
            out[name] = merged
    return out


def load_prev(path: Path | None = None) -> dict:
    path = path or DECISIONS_JSON
    if not path.is_file():
        return {"decisions": {}, "reviewers": {}}
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"decisions": {}, "reviewers": {}}
    if not isinstance(prev, dict):
        return {"decisions": {}, "reviewers": {}}
    return prev


def apply_incoming(
    data: dict,
    *,
    json_path: Path | None = None,
    jsonl_path: Path | None = None,
) -> dict:
    if not isinstance(data, dict):
        raise ValueError("json object required")
    json_path = json_path or DECISIONS_JSON
    jsonl_path = jsonl_path or DECISIONS_JSONL
    prev = load_prev(json_path)
    incoming = data.get("decisions", data)
    if not isinstance(incoming, dict):
        raise ValueError("decisions must be an object")
    old_dec = prev.get("decisions") if isinstance(prev.get("decisions"), dict) else {}
    # A PUT of the wrapper object without a decisions key would treat dataset/n as stems.
    if "decisions" not in data and any(k in data for k in ("dataset", "reviewers", "reviewer", "n")):
        incoming = {}
    merged = merge_decisions(old_dec, incoming)
    reviewers = merge_reviewers(prev.get("reviewers") or {}, data.get("reviewers") or {})
    who = str(data.get("reviewer") or "").strip()
    now = datetime.now(timezone.utc).isoformat()
    if who:
        n_who = sum(
            1
            for rec in merged.values()
            if isinstance(rec, dict) and str(rec.get("reviewer") or "").strip() == who
        )
        reviewers = merge_reviewers(
            reviewers,
            {
                who: {
                    "last_stem": data.get("last_stem") or "",
                    "updated_at": data.get("updated_at") or now,
                    "n": n_who,
                }
            },
        )
    payload = {
        "dataset": data.get("dataset") or prev.get("dataset") or "fontdiffuser-p649-t295-s338-cn2west-v2a-r0",
        "updated_at": now,
        "n": len(merged),
        "reviewer": who or prev.get("reviewer") or "",
        "last_stem": data.get("last_stem") if who else (prev.get("last_stem") or ""),
        "reviewers": reviewers,
        "decisions": merged,
    }
    write_payload(payload, json_path=json_path, jsonl_path=jsonl_path)
    return payload


def write_payload(
    payload: dict,
    *,
    json_path: Path | None = None,
    jsonl_path: Path | None = None,
) -> None:
    json_path = json_path or DECISIONS_JSON
    jsonl_path = jsonl_path or DECISIONS_JSONL
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = []
    for stem, rec in sorted((payload.get("decisions") or {}).items()):
        if isinstance(rec, dict):
            lines.append(json.dumps({"stem": stem, **rec}, ensure_ascii=False))
    jsonl_path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        if args and str(args[0]).startswith("PUT "):
            super().log_message(fmt, *args)
            return
        path = str(args[0]) if args else ""
        if "decisions.json" in path or "review.html" in path:
            super().log_message(fmt, *args)

    def end_headers(self) -> None:
        if self.path.split("?")[0] in set(PUT_STORES) | {"/p649_v2a_review/review.html"}:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        if self.path.split("?")[0] not in PUT_STORES:
            self.send_error(403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_PUT(self) -> None:
        store = PUT_STORES.get(self.path.split("?")[0])
        if store is None:
            self.send_error(403, "PUT only allowed for decisions.json")
            return
        json_path, jsonl_path = store
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw.decode("utf-8"))
            payload = apply_incoming(data, json_path=json_path, jsonl_path=jsonl_path)
        except json.JSONDecodeError:
            self.send_error(400, "invalid json")
            return
        except ValueError as exc:
            self.send_error(400, str(exc))
            return
        body = json.dumps({"ok": True, "n": payload["n"], "updated_at": payload["updated_at"]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8778)
    args = ap.parse_args()
    if not DECISIONS_JSON.exists():
        write_payload(
            {
                "dataset": "fontdiffuser-p649-t295-s338-cn2west-v2a-r0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "n": 0,
                "last_stem": "",
                "decisions": {},
            }
        )
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving {ROOT} on http://{args.host}:{args.port}/p649_v2a_review/review.html", flush=True)
    print(f"decisions -> {DECISIONS_JSON}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
