#!/usr/bin/env python3
"""V100 local hub: training board + F2/F2-RL/F2-VEC compare.

This container's :8791 / :19000 are NOT published on the host NIC.
From a laptop, SSH-tunnel or Cursor port-forward:

  ssh -L 19000:127.0.0.1:19000 root@172.18.41.23
  open http://127.0.0.1:19000/

Do not use http://0.0.0.0:8791 or http://172.18.41.23:19000 — those will fail.
"""
from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path("/root/projects/hrfont")
SNAP = ROOT / "reports/v0913_dashboard"
GSNAP = ROOT / "reports/g_dashboard"
GSHOT = ROOT / "reports/g_v0913_shot"
GSHOT_K1248 = ROOT / "reports/g_v0913_shot_k1248"
HBOARD = ROOT / "reports/h_20260915/board"
IBOARD = ROOT / "reports/i_20260915/board"
BOARD = ROOT / "reports/f03_test16_strat"
KBOARD = ROOT / "reports/k_default_k1248_20260917"
KREVIEW = ROOT / "reports/k1_final_review"
KSTAGE = ROOT / "reports/k_stage_reviews_20260917"
KCOMPARE = ROOT / "reports/k_step_compare_20260917"
KFAMILY = ROOT / "reports/k_family_20260917"
K3FINAL = ROOT / "reports/k3_final_review"
KALL = ROOT / "reports/k_all_compare_20260917"
K4AVAL = ROOT / "reports/k4_a_val_compare_20260918"
K4APROTOCOL = ROOT / "reports/k4_a_protocol_20260918"
K4APROTOCOL_DATA = Path("/root/data1/hrfont_k4a_diagnostic_20260918")
K4APROTOCOL_SOURCE = Path("/root/data1/hrfont_dataset_v2_20260917/v2")
K4AHOLLOW = ROOT / "reports/k4_a_hollow_compare_20260918"
K5PROTOCOL = ROOT / "reports/k0_k1_k5_protocol_20260920"
K5STEPS = ROOT / "reports/k5_step_training_20260920"
K5DEFAULT = Path("/root/data1/hrfont_default_inference_20260917")
K5EVAL = Path("/root/data1/hrfont_k5_20260919/evaluation")
KSTAGES = {
    "k1_step2000": ROOT / "reports/k1_original_step2000_review",
    "k1_val192": ROOT / "reports/k1_final_val192_review",
    "k3_step2000": ROOT / "reports/k3_step2000_review",
    "k3_step4000": ROOT / "reports/k3_step4000_review",
    "k3_step6000": ROOT / "reports/k3_step6000_review",
    "k3_step8000": ROOT / "reports/k3_step8000_review",
}
HUB = ROOT / "reports/v100_hub"
RUNS = ROOT / "runs"

INDEX = """<!doctype html>
<meta charset="utf-8">
<title>V100 看板枢纽</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:32px;background:#111;color:#eee;max-width:72ch;line-height:1.5}
a{color:#9cf} .warn{color:#fd6} .ok{color:#7d7} code{color:#c8a24b}
.hero{font-size:18px;margin:18px 0;padding:14px 16px;border:1px solid #345;background:#16202a}
</style>
<h1>V100 看板枢纽</h1>
<p class="hero ok"><b>G0b · I0~I4（k1248 · 区分 train/test/val）：</b>
  <a href="/g_shot_k1248/g0_i0_i1_board.html">打开对照看板</a>
  · <a href="/g_shot_k1248/">完整 k1248</a>
  · <a href="/i/">I fixed-val</a></p>
<p class="note">服务端已就绪。Ports 面板确认 <code>19000</code> 已转发后打开
<code>http://127.0.0.1:19000/g_shot_k1248/g0_i0_i1_board.html</code>。</p>
<h2>页</h2>
<ul>
<li><a href="/g_shot_k1248/g0_i0_i1_board.html"><b>G0b · I0~I4</b> · F 模板 · 测试/训练/验证</a>（当前主对照）</li>
<li><a href="/g_shot_k1248/"><b>G/I k1248 全列</b> · old Ref8 · 含 I0~I4</a></li>
<li><a href="/i/"><b>I 系列</b> · fixed-val 对照（I1~I5）</a></li>
<li><a href="/k/"><b>K0 / K1</b> · Train / Val / Test · 10k 对照</a></li>
<li><a href="/k_review/"><b>K0 / K1 matched review</b> · 同噪声、嵌套参考、CFG 1、DPM++ 20</a></li>
<li><a href="/k_stage/"><b>K1 / K3 阶段评估</b> · checkpoint review 汇总</a></li>
<li><a href="/k_compare/"><b>K1 / K3 跨 step 对比</b> · 同一组样本横向比较</a></li>
<li><a href="/k_family/"><b>K 族实验总览</b> · K0/K1/K2/K3 状态、指标与入口</a></li>
<li><a href="/k_all/"><b>K0 / K1 / K3 视觉对比</b> · 同一 TEST 样本横向看图</a></li>
<li><a href="/k4_a_val/"><b>K4-A VAL 跨 step</b> · 2k / 4k / 6k 人工检查</a></li>
<li><a href="/k4_a_protocol/"><b>K4-A 固定协议</b> · Train / Val / Test · 20k</a></li>
<li><a href="/k4_a_hollow/"><b>K4-A 空心字</b> · 训练集跨 checkpoint · 0k–20k</a></li>
<li><a href="/k5_protocol/"><b>K0 / K1 / K5-A / K5-B</b> · 固定协议逐图对比</a></li>
<li><a href="/k5_steps/"><b>K5-A / K5-B 逐 step</b> · 训练 telemetry、donor guard、checkpoint</a></li>
<li><a href="/h/"><b>H v0915</b> · H3/H0/H4… fixed-val 对照</a></li>
<li><a href="/g_shot/"><b>G v0913</b> one/few-shot · vs dirty F2/F2-RL</a></li>
<li><a href="/f2vec_shot_board.html">F2 / F2-RL / F2-VEC 对照</a></li>
<li><a href="/g/">Group G 训练进度</a></li>
<li><a href="/v0913/">v0913_clean 训练看板</a></li>
<li><a href="/core_shot_board.html">核心模型 Demo-8 对照</a></li>
</ul>
"""


def _safe(base: Path, relative: str, extra_roots: list[Path] | None = None) -> Path:
    root = base.resolve()
    target = (base / relative).resolve()
    allowed = [root, *((p.resolve() for p in extra_roots or []))]
    if any(target == a or a in target.parents for a in allowed):
        return target
    return root / "__forbidden__"


class HubHandler(SimpleHTTPRequestHandler):
    server_version = "HRFontV100Hub/1.0"

    def end_headers(self) -> None:
        path = self.path.split("?")[0]
        if path in {"/", "/v0913/", "/v0913/status.json", "/v0913/index.html",
                    "/g/", "/g/status.json", "/g/index.html",
                    "/g_shot/", "/g_shot/index.html",
                    "/g_shot_k1248/", "/g_shot_k1248/index.html",
                    "/k/", "/k/index.html", "/k_review/", "/k_review/index.html",
                    "/k_stage/", "/k_stage/index.html",
                    "/k_compare/", "/k_compare/index.html",
                    "/k_family/", "/k_family/index.html",
                    "/k3_final/", "/k3_final/index.html",
                    "/k_all/", "/k_all/index.html",
                    "/k4_a_val/", "/k4_a_val/index.html",
                    "/k4_a_protocol/", "/k4_a_protocol/index.html",
                    "/k4_a_hollow/", "/k4_a_hollow/index.html",
                    "/k5_protocol/", "/k5_protocol/index.html",
                    "/k5_steps/", "/k5_steps/index.html",
                    "/h/", "/h/index.html",
                    "/i/", "/i/index.html"}:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        for route in KSTAGES:
            if request_path in {f"/{route}", f"/{route}/"}:
                self.send_response(302)
                self.send_header("Location", f"/{route}/index.html")
                self.end_headers()
                return
        if request_path in {"/v0913", "/v0913/"}:
            self.send_response(302)
            self.send_header("Location", "/v0913/index.html")
            self.end_headers()
            return
        if request_path in {"/g", "/g/"}:
            self.send_response(302)
            self.send_header("Location", "/g/index.html")
            self.end_headers()
            return
        if request_path in {"/i", "/i/"}:
            self.send_response(302)
            self.send_header("Location", "/i/index.html")
            self.end_headers()
            return
        if request_path in {"/h", "/h/"}:
            self.send_response(302)
            self.send_header("Location", "/h/index.html")
            self.end_headers()
            return
        if request_path in {"/g_shot_k1248", "/g_shot_k1248/"}:
            self.send_response(302)
            self.send_header("Location", "/g_shot_k1248/index.html")
            self.end_headers()
            return
        if request_path in {"/g_shot", "/g_shot/"}:
            self.send_response(302)
            self.send_header("Location", "/g_shot/index.html")
            self.end_headers()
            return
        if request_path in {"/k", "/k/"}:
            self.send_response(302)
            self.send_header("Location", "/k/index.html")
            self.end_headers()
            return
        if request_path in {"/k_review", "/k_review/"}:
            self.send_response(302)
            self.send_header("Location", "/k_review/index.html")
            self.end_headers()
            return
        if request_path in {"/k_stage", "/k_stage/"}:
            self.send_response(302)
            self.send_header("Location", "/k_stage/index.html")
            self.end_headers()
            return
        if request_path in {"/k_compare", "/k_compare/"}:
            self.send_response(302)
            self.send_header("Location", "/k_compare/index.html")
            self.end_headers()
            return
        if request_path in {"/k_family", "/k_family/"}:
            self.send_response(302)
            self.send_header("Location", "/k_family/index.html")
            self.end_headers()
            return
        if request_path in {"/k3_final", "/k3_final/"}:
            self.send_response(302)
            self.send_header("Location", "/k3_final/index.html")
            self.end_headers()
            return
        if request_path in {"/k_all", "/k_all/"}:
            self.send_response(302)
            self.send_header("Location", "/k_all/index.html")
            self.end_headers()
            return
        if request_path in {"/k4_a_val", "/k4_a_val/"}:
            self.send_response(302)
            self.send_header("Location", "/k4_a_val/index.html")
            self.end_headers()
            return
        if request_path in {"/k4_a_protocol", "/k4_a_protocol/"}:
            self.send_response(302)
            self.send_header("Location", "/k4_a_protocol/index.html")
            self.end_headers()
            return
        if request_path in {"/k4_a_hollow", "/k4_a_hollow/"}:
            self.send_response(302)
            self.send_header("Location", "/k4_a_hollow/index.html")
            self.end_headers()
            return
        if request_path in {"/k5_protocol", "/k5_protocol/"}:
            self.send_response(302)
            self.send_header("Location", "/k5_protocol/index.html")
            self.end_headers()
            return
        if request_path in {"/k5_steps", "/k5_steps/"}:
            self.send_response(302)
            self.send_header("Location", "/k5_steps/index.html")
            self.end_headers()
            return
        super().do_GET()

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        if request_path in {"/", "/index.html"}:
            return str(HUB / "index.html")
        if request_path == "/v0913" or request_path.startswith("/v0913/"):
            rel = request_path[len("/v0913") :].lstrip("/")
            return str(_safe(SNAP, rel or "index.html"))
        if request_path == "/i" or request_path.startswith("/i/"):
            rel = request_path[len("/i") :].lstrip("/")
            return str(_safe(IBOARD, rel or "index.html", extra_roots=[RUNS]))
        if request_path == "/h" or request_path.startswith("/h/"):
            rel = request_path[len("/h") :].lstrip("/")
            # preds/* are symlinks into runs/<arm>/eval_step_*
            return str(_safe(HBOARD, rel or "index.html", extra_roots=[RUNS]))
        if request_path == "/g_shot_k1248" or request_path.startswith("/g_shot_k1248/"):
            rel = request_path[len("/g_shot_k1248") :].lstrip("/")
            return str(_safe(GSHOT_K1248, rel or "index.html"))
        if request_path == "/g_shot" or request_path.startswith("/g_shot/"):
            rel = request_path[len("/g_shot") :].lstrip("/")
            return str(_safe(GSHOT, rel or "index.html"))
        if request_path == "/k" or request_path.startswith("/k/"):
            rel = request_path[len("/k") :].lstrip("/")
            return str(_safe(KBOARD, rel or "index.html"))
        if request_path == "/k_review" or request_path.startswith("/k_review/"):
            rel = request_path[len("/k_review") :].lstrip("/")
            return str(_safe(KREVIEW, rel or "index.html"))
        if request_path == "/k_stage" or request_path.startswith("/k_stage/"):
            rel = request_path[len("/k_stage") :].lstrip("/")
            return str(_safe(KSTAGE, rel or "index.html"))
        if request_path == "/k_compare" or request_path.startswith("/k_compare/"):
            rel = request_path[len("/k_compare") :].lstrip("/")
            return str(_safe(KCOMPARE, rel or "index.html"))
        if request_path == "/k_family" or request_path.startswith("/k_family/"):
            rel = request_path[len("/k_family") :].lstrip("/")
            return str(_safe(KFAMILY, rel or "index.html"))
        if request_path == "/k3_final" or request_path.startswith("/k3_final/"):
            rel = request_path[len("/k3_final") :].lstrip("/")
            return str(_safe(K3FINAL, rel or "index.html"))
        if request_path == "/k_all" or request_path.startswith("/k_all/"):
            rel = request_path[len("/k_all") :].lstrip("/")
            return str(_safe(KALL, rel or "index.html"))
        if request_path == "/k4_a_val" or request_path.startswith("/k4_a_val/"):
            rel = request_path[len("/k4_a_val") :].lstrip("/")
            if rel.startswith("experiments/"):
                return str(_safe(ROOT / "reports", rel))
            return str(_safe(K4AVAL, rel or "index.html"))
        if request_path == "/k4_a_protocol" or request_path.startswith("/k4_a_protocol/"):
            rel = request_path[len("/k4_a_protocol") :].lstrip("/")
            if rel.startswith("source/"):
                return str(
                    _safe(
                        K4APROTOCOL_SOURCE,
                        rel[len("source/") :],
                        extra_roots=[
                            ROOT / "data",
                            Path("/root/data1/hrfont_dataset_v2_20260917"),
                        ],
                    )
                )
            if rel.startswith("default20k/"):
                return str(_safe(K4APROTOCOL_DATA, rel))
            return str(_safe(K4APROTOCOL, rel or "index.html"))
        if request_path == "/k4_a_hollow" or request_path.startswith("/k4_a_hollow/"):
            rel = request_path[len("/k4_a_hollow") :].lstrip("/")
            if rel.startswith("source/"):
                return str(
                    _safe(
                        K4APROTOCOL_SOURCE,
                        rel[len("source/") :],
                        extra_roots=[ROOT / "data"],
                    )
                )
            if rel.startswith("hollow/"):
                return str(_safe(K4APROTOCOL_DATA, rel))
            return str(_safe(K4AHOLLOW, rel or "index.html"))
        if request_path == "/k5_protocol" or request_path.startswith("/k5_protocol/"):
            rel = request_path[len("/k5_protocol") :].lstrip("/")
            if rel.startswith("source_legacy/"):
                return str(_safe(ROOT / "data", rel[len("source_legacy/") :]))
            if rel.startswith("source_v2/"):
                return str(
                    _safe(
                        K4APROTOCOL_SOURCE,
                        rel[len("source_v2/") :],
                        extra_roots=[
                            ROOT / "data",
                            Path("/root/data1/hrfont_dataset_v2_20260917"),
                        ],
                    )
                )
            if rel.startswith("pred/"):
                parts = rel.split("/", 3)
                if len(parts) == 4:
                    _, arm, split, filename = parts
                    base = K5DEFAULT / arm if arm in {"K0", "K1"} else K5EVAL / arm
                    return str(_safe(base / split, filename))
            return str(_safe(K5PROTOCOL, rel or "index.html"))
        if request_path == "/k5_steps" or request_path.startswith("/k5_steps/"):
            rel = request_path[len("/k5_steps") :].lstrip("/")
            return str(_safe(K5STEPS, rel or "index.html"))
        for route, base in KSTAGES.items():
            if request_path == f"/{route}" or request_path.startswith(f"/{route}/"):
                rel = request_path[len(route) + 2 :].lstrip("/")
                return str(_safe(base, rel or "index.html"))
        if request_path == "/g" or request_path.startswith("/g/"):
            rel = request_path[len("/g") :].lstrip("/")
            return str(_safe(GSNAP, rel or "index.html"))
        return str(_safe(BOARD, request_path.lstrip("/")))

    def log_message(self, fmt: str, *args: object) -> None:
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=19000)
    args = ap.parse_args()
    HUB.mkdir(parents=True, exist_ok=True)
    (HUB / "index.html").write_text(INDEX, encoding="utf-8")
    httpd = ThreadingHTTPServer((args.host, args.port), HubHandler)
    print(
        f"V100 hub http://127.0.0.1:{args.port}/ pid={os.getpid()} "
        f"(tunnel: ssh -L {args.port}:127.0.0.1:{args.port} root@172.18.41.23)",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
