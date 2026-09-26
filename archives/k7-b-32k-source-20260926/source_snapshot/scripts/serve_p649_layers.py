#!/usr/bin/env python3
"""Internal layered usability review on :19003. Does not write L1 decisions.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_p649_review import apply_incoming, DECISIONS_JSON  # noqa: E402

ROOT = Path("/root/projects/hrfont")
DATA = ROOT / "data"
LAYERS = DATA / "p649_v2a_layers"
CHARSET = ROOT / "manifests/charset_cn2west_v2_planned.json"
OUT_NAME = "fontdiffuser-p649-t295-s338-cn2west-v2a-r0"
DATASET_PREFIX = "/fontdiffuser-p649-"
PROBES = list("永和书AaGgWwMm01ilㄅあア")
PUT_STORES = {
    "/p649_v2a_layers/layer2.json": (LAYERS / "layer2.json", LAYERS / "layer2.jsonl"),
    "/p649_v2a_layers/layer3.json": (LAYERS / "layer3.json", LAYERS / "layer3.jsonl"),
}


def _safe_target(base: Path, relative: str) -> Path:
    base_resolved = base.resolve()
    target = (base_resolved / relative).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        return base_resolved / "__forbidden__"
    return target


class Handler(SimpleHTTPRequestHandler):
    server_version = "HRFontLayers/1.0"

    def end_headers(self) -> None:
        path = self.path.split("?")[0]
        if path in set(PUT_STORES) | {
            "/p649_v2a_layers/review.html",
            "/p649_v2a_layers/l1_snapshot.json",
            "/p649_v2a_review/decisions.json",
        }:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        if request_path in {"/", "/p649_v2a_layers", "/p649_v2a_layers/"}:
            self.send_response(302)
            self.send_header("Location", "/p649_v2a_layers/review.html")
            self.end_headers()
            return
        super().do_GET()

    def do_PUT(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        store = PUT_STORES.get(request_path)
        if store is None:
            self.send_error(403, "PUT only for layer2/layer3")
            return
        json_path, jsonl_path = store
        if json_path.resolve() == DECISIONS_JSON.resolve():
            self.send_error(403, "refusing to write layer-1 decisions")
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw.decode("utf-8"))
            payload = apply_incoming(data, json_path=json_path, jsonl_path=jsonl_path)
            layer_name = "layer2" if json_path.name.startswith("layer2") else "layer3"
            payload["layer"] = layer_name
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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

    def list_directory(self, path: str):  # type: ignore[override]
        self.send_error(403, "Directory listing disabled")
        return None

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        if request_path.startswith("/p649_v2a_layers"):
            return str(_safe_target(LAYERS, request_path[len("/p649_v2a_layers") :].lstrip("/")))
        if request_path.startswith("/p649_v2a_review") or request_path.startswith(DATASET_PREFIX):
            return str(_safe_target(DATA, request_path.lstrip("/")))
        return str(_safe_target(LAYERS, request_path.lstrip("/")))

    def log_message(self, fmt: str, *args: object) -> None:
        path = str(args[0]) if args else ""
        if any(k in path for k in ("PUT ", "review.html", "layer2", "layer3")):
            super().log_message(fmt, *args)


def script_probes() -> list[dict]:
    t = json.loads(CHARSET.read_text(encoding="utf-8"))["target"]
    return [
        {"id": "style_han", "label": "Style 汉字", "n": 338, "sample": list("永和书风骨韵天地")},
        {"id": "ascii_digits", "label": "数字", "n": 10, "sample": list(t["ascii_digits"])},
        {"id": "ascii_letters", "label": "拉丁", "n": 52, "sample": list("AaGgWwMmIiLlQq")},
        {"id": "latin_ext_letters", "label": "扩展拉丁", "n": 27, "sample": list(t["latin_ext_letters"])},
        {"id": "hiragana", "label": "平假名", "n": 83, "sample": list("あいうえおかがきぎぱぽん")},
        {"id": "katakana", "label": "片假名", "n": 86, "sample": list("アイウエオカガキギジヴン")},
        {"id": "bopomofo", "label": "注音", "n": 37, "sample": list(t["bopomofo"])},
    ]


def write_pages() -> None:
    LAYERS.mkdir(parents=True, exist_ok=True)
    tpl = Path(__file__).with_name("p649_layer_review_template.html").read_text(encoding="utf-8")
    html = (
        tpl.replace("SCRIPT_PROBES_PLACEHOLDER", json.dumps(script_probes(), ensure_ascii=False))
        .replace("PROBES_PLACEHOLDER", json.dumps(PROBES, ensure_ascii=False))
        .replace("REL_PLACEHOLDER", f"../{OUT_NAME}")
    )
    (LAYERS / "review.html").write_text(html, encoding="utf-8")
    (LAYERS / "index.html").write_text(
        """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/><title>p649 layers</title></head>
<body style="font:14px/1.5 system-ui;margin:32px">
<h1>内部 · 分层可用性（非导师页）</h1>
<p>不要把本端口发给导师。第1层记录只读。</p>
<p><a href="review.html">打开分层审查</a></p>
</body></html>
""",
        encoding="utf-8",
    )


def empty_layer(name: str) -> dict:
    return {
        "dataset": "fontdiffuser-p649-t295-s338-cn2west-v2a-r0",
        "layer": name,
        "updated_at": "",
        "n": 0,
        "reviewer": "",
        "last_stem": "",
        "reviewers": {},
        "decisions": {},
    }


def write_l1_snapshot() -> dict:
    rank = json.loads((DATA / "p649_v2a_review" / "ink_ratio_rank.json").read_text(encoding="utf-8"))
    raw = DECISIONS_JSON.read_bytes()
    dec = json.loads(raw.decode("utf-8"))
    decisions = dec.get("decisions") or {}
    buckets = {
        "overlap260": {"n": 0, "pass": [], "drop": [], "undecided": []},
        "new389": {"n": 0, "pass": [], "drop": [], "undecided": []},
    }
    for font in rank.get("fonts") or []:
        stem = font["stem"]
        key = "overlap260" if font.get("subset") == "overlap260" else "new389"
        buckets[key]["n"] += 1
        decision = (decisions.get(stem) or {}).get("review_decision") or ""
        if decision in {"pass", "drop"}:
            buckets[key][decision].append(stem)
        else:
            buckets[key]["undecided"].append(stem)
    for bucket in buckets.values():
        bucket["pass"].sort()
        bucket["drop"].sort()
        bucket["undecided"].sort()
        bucket["complete"] = bucket["n"] > 0 and not bucket["undecided"]
    overlap = buckets["overlap260"]
    payload = {
        "dataset": dec.get("dataset") or rank.get("dataset"),
        "source": "p649_v2a_review/decisions.json",
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "l1_updated_at": dec.get("updated_at") or "",
        "l1_n": len(decisions),
        "l1_sha256": hashlib.sha256(raw).hexdigest(),
        "overlap260": overlap,
        "new389": buckets["new389"],
        "layer2_candidates_overlap260": len(overlap["drop"]),
        "checks": {
            "overlap260_complete": overlap["complete"] and overlap["n"] == 260,
            "overlap260_pass_plus_drop": len(overlap["pass"]) + len(overlap["drop"]) == overlap["n"],
            "l2_pool_equals_l1_drop": True,
        },
    }
    (LAYERS / "l1_snapshot.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def ensure_files() -> None:
    write_pages()
    write_l1_snapshot()
    for stem in ("layer2", "layer3"):
        path = LAYERS / f"{stem}.json"
        if not path.exists():
            path.write_text(json.dumps(empty_layer(stem), ensure_ascii=False, indent=2), encoding="utf-8")
            (LAYERS / f"{stem}.jsonl").write_text("", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=19003)
    args = ap.parse_args()
    ensure_files()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"p649 layered review http://{args.host}:{args.port}/p649_v2a_layers/review.html "
        f"(pid={os.getpid()})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
