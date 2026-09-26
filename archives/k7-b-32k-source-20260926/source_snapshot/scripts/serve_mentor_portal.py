#!/usr/bin/env python3
"""Serve an allowlisted, read-only mentor review portal.

Mirrors :8777 data/ layout so cn2west review images resolve via ../dataset/...

Routes:
  /                          thin index + weekly
  /briefing/                 2026-09-07 mentor one-pager
  /weekly.html               weekly report HTML
  /assets/                   weekly-report figures
  /cn2west_v2_abc_review/    same as :8777
  /p649_v2a_review/          p649 协议 A ink 审查（可写 decisions.json）
  /fontdiffuser-*/           protocol render datasets (image roots)
  /e1_formal_eval/ /e1/      E1 formal eval
  /f3_ckpt_dashboard/        Official / F0 / F3 visual compare
  /mentor_briefing_20260907/ same briefing via :8777-style path
  /render_qa_hub.html        QA hub
  /bbox/                     bbox_explain alias
  /rendering.html            → /cn2west_v2_abc_review/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))


ROOT = Path(__file__).resolve().parents[1]
PORTAL = ROOT / "public" / "mentor_portal"
DATA = ROOT / "data"
ASSETS = ROOT / "reports" / "weekly_20260905"
BRIEFING = ROOT / "reports" / "mentor_briefing_20260907"
F03 = ROOT / "reports" / "f03_test16_strat"
REVIEW = DATA / "cn2west_v2_abc_review"
BBOX = REVIEW / "bbox_explain"
E1 = (
    ROOT
    / "runs"
    / "E1-FTV2-A-S3407"
    / "eval_formal"
    / "E1-formal-strat-20260904T041520Z"
)
QA_HUB = DATA / "render_qa_hub.html"

# Top-level names under data/ that :8777 review pages may load.
_DATA_PREFIXES = (
    "cn2west_v2_abc_review",
    "e1_formal_eval",
    "e1_ft_v2_dashboard",
    "fontdiffuser-p253-",
    "fontdiffuser_p253",
    "fontdiffuser-p649-",
    "p649_v2a_review",
)


def _is_allowed_data_path(request_path: str) -> bool:
    rel = request_path.lstrip("/")
    if not rel:
        return False
    if rel == "render_qa_hub.html":
        return True
    top = rel.split("/", 1)[0]
    return any(top == p or top.startswith(p) for p in _DATA_PREFIXES)


class MentorHandler(SimpleHTTPRequestHandler):
    server_version = "HRFontMentor/1.0"

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def list_directory(self, path: str):  # type: ignore[override]
        self.send_error(403, "Directory listing disabled")
        return None

    def do_GET(self) -> None:  # noqa: N802
        request_path = unquote(urlsplit(self.path).path)
        if request_path in ("/rendering.html", "/rendering"):
            self.send_response(302)
            self.send_header("Location", "/cn2west_v2_abc_review/")
            self.end_headers()
            return
        super().do_GET()

    def do_OPTIONS(self) -> None:  # noqa: N802
        from serve_p649_review import PUT_STORES

        if unquote(urlsplit(self.path).path) not in PUT_STORES:
            self.send_error(403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_PUT(self) -> None:  # noqa: N802
        """Only p649 ink review may write; the rest of the portal stays read-only."""
        from serve_p649_review import PUT_STORES, apply_incoming

        request_path = unquote(urlsplit(self.path).path)
        store = PUT_STORES.get(request_path)
        if store is None:
            self.send_error(403, "PUT only allowed for p649 review decisions")
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

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)

        if request_path == "/render_qa_hub.html":
            return str(QA_HUB)

        # Dedicated aliases
        routes = (
            ("/e1_formal_eval", E1),
            ("/e1", E1),
            ("/bbox", BBOX),
            ("/assets", ASSETS),
            ("/briefing", BRIEFING),
            ("/mentor_briefing_20260907", BRIEFING),
            ("/f3_ckpt_dashboard", ROOT / "reports" / "f3_ckpt_dashboard"),
            ("/f03_test16_strat", F03),
            ("/f03_eval", F03),
        )
        for prefix, base in routes:
            if request_path == prefix or request_path.startswith(prefix + "/"):
                relative = request_path[len(prefix) :].lstrip("/")
                return str(self._safe_target(base, relative))

        # Same layout as python -m http.server --directory data/
        # so ../fontdiffuser-... from cn2west review resolves.
        if _is_allowed_data_path(request_path):
            return str(self._safe_target(DATA, request_path.lstrip("/")))

        relative = request_path.lstrip("/") or "index.html"
        return str(self._safe_target(PORTAL, relative))

    @staticmethod
    def _safe_target(base: Path, relative: str) -> Path:
        base_resolved = base.resolve()
        target = (base_resolved / relative).resolve()
        if target != base_resolved and base_resolved not in target.parents:
            return base_resolved / "__forbidden__"
        return target

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def verify_inputs() -> None:
    required = (
        PORTAL / "index.html",
        PORTAL / "weekly.html",
        BRIEFING / "index.html",
        F03,
        ASSETS / "fig1_six_protocols.png",
        REVIEW / "index.html",
        BBOX / "index.html",
        E1 / "index.html",
        E1 / "browse_index.json",
        QA_HUB,
        DATA / "p649_v2a_review" / "review.html",
        DATA / "fontdiffuser-p649-t295-s338-cn2west-v2a-r0" / "train" / "TargetImage",
    )
    missing = []
    for path in required:
        if path.is_file() or path.is_dir():
            continue
        missing.append(str(path))
    if missing:
        raise FileNotFoundError("mentor portal inputs missing:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19001)
    args = parser.parse_args()

    verify_inputs()
    server = ThreadingHTTPServer((args.host, args.port), MentorHandler)
    print(
        f"HR-Font mentor portal listening on http://{args.host}:{args.port} "
        f"(pid={os.getpid()})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
