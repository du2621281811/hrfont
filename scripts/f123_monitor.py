#!/usr/bin/env python3
"""F0–F3 live training monitor: health checks, crash resume, HTTP dashboard.

Training itself is local (no network). This process:
  - serves a dashboard that does not load any CDN
  - copies train_log.jsonl out of gitignored runs/ every tick
  - crash-resumes F3 from last_state; chains F2 after F3 DONE, F1 after F2 DONE
    (paused STOP files are removed only at that handoff)

  python scripts/f123_monitor.py --port 8787 --bind 0.0.0.0 --auto-resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
SNAP = ROOT / "reports/f123_dashboard"
TRAIN_LOG_EXPORT = ROOT / "reports/training_logs"
MONITOR_STATE = ROOT / "runs/f123_monitor/state.json"
PID_FILE = ROOT / "logs/f123_monitor.pid"

MAX_STEPS = 80_000
STALL_SEC = 20 * 60          # val@5k can freeze heartbeat 5–15 min
RESUME_COOLDOWN_SEC = 15 * 60
MAX_RESUMES_PER_HOUR = 3

ARMS = (
    {
        "arm": "F0",
        "run_id": "F0-RSIFREE-FT-A-S3407",
        "title": "F0 RSI-free parent",
        "role": "parent (done)",
        "gpu": None,
        "auto_resume": False,
        "max_steps": 100_000,
    },
    {
        "arm": "F1",
        "run_id": "F1-OFFRSI-A-S3407",
        "title": "F1 official RSI",
        "role": "control · last after F2",
        "gpu": 2,
        "auto_resume": False,
        "resume_after": "F2",
        "max_steps": MAX_STEPS,
        "log": "logs/f1/F1-OFFRSI-A-S3407.log",
    },
    {
        "arm": "F2",
        "run_id": "F2-DELTARSI-A-S3407",
        "title": "F2 Δ-RSI",
        "role": "isolates Δ vs F1 · paused until F3 DONE",
        "gpu": 3,
        "auto_resume": False,
        "resume_after": "F3",
        "max_steps": MAX_STEPS,
        "log": "logs/f2/F2-DELTARSI-A-S3407.log",
    },
    {
        "arm": "F3",
        "run_id": "F3-JOINT-DS-A-S3407",
        "title": "F3 Δ + Support",
        "role": "isolates Support vs F2 · exclusive now",
        "gpu": 2,
        "auto_resume": True,
        "max_steps": MAX_STEPS,
        "log": "logs/f3/F3-JOINT-DS-A-S3407.log",
    },
)

_lock = threading.Lock()
_status: dict[str, Any] = {}
_resume_log: list[dict[str, Any]] = []
_started = time.time()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path, last_n: int | None = None) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    if last_n:
        lines = lines[-last_n:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _train_procs() -> list[dict[str, Any]]:
    out = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        cmd = raw.replace(b"\0", b" ").decode("utf-8", "replace")
        if "train.py" not in cmd or "--arm" not in cmd:
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        state = ""
        rss_kb = 0
        for line in status.splitlines():
            if line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
            elif line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
        env = {}
        try:
            for item in (entry / "environ").read_bytes().split(b"\0"):
                if b"=" in item:
                    k, _, v = item.partition(b"=")
                    env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
        except OSError:
            pass
        name = ""
        if "--experience_name " in cmd:
            name = cmd.split("--experience_name ", 1)[1].split(" ", 1)[0]
        out.append({
            "pid": int(entry.name),
            "cmd": cmd[:400],
            "state": state[:40],
            "rss_gb": round(rss_kb / 1024 / 1024, 2),
            "cuda": env.get("CUDA_VISIBLE_DEVICES"),
            "run_id": name,
        })
    return out


def _nvidia() -> list[dict[str, Any]]:
    try:
        text = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.free,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append({
            "index": int(parts[0]),
            "util_pct": int(float(parts[1])),
            "mem_used_mb": int(float(parts[2])),
            "mem_free_mb": int(float(parts[3])),
            "temp_c": int(float(parts[4])),
        })
    return gpus


def _host() -> dict[str, Any]:
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            mem[k] = int(rest.strip().split()[0])
    except OSError:
        pass
    def gb(key: str) -> float:
        return round(mem.get(key, 0) / 1024 / 1024, 2)
    disk = shutil.disk_usage(str(ROOT))
    return {
        "ram_total_gb": gb("MemTotal"),
        "ram_available_gb": gb("MemAvailable"),
        "swap_total_gb": gb("SwapTotal"),
        "swap_used_gb": round((mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / 1024 / 1024, 2),
        "disk_free_tb": round(disk.free / 1024 ** 4, 2),
        "disk_used_pct": round(100 * disk.used / max(1, disk.total), 1),
    }


def _tail_errors(log_path: Path, n_bytes: int = 120_000) -> list[str]:
    if not log_path.is_file():
        return []
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as fh:
            fh.seek(max(0, size - n_bytes))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    keys = ("Traceback", "RuntimeError", "CUDA out of memory", "non-finite",
            "Killed", "NCCL", "ConnectionReset", "HostNotFound")
    hits = []
    for line in text.replace("\r", "\n").splitlines():
        if any(k in line for k in keys) and "FutureWarning" not in line:
            hits.append(line.strip()[:240])
    return hits[-8:]


def _ckpt_info(run: Path) -> dict[str, Any]:
    named = []
    for p in sorted(run.glob("global_step_*")):
        if (p / "unet.pth").is_file():
            try:
                named.append(int(p.name.rsplit("_", 1)[-1]))
            except ValueError:
                pass
    last = run / "last_state" / "trainer_state.pt"
    stopped = run / "stopped_step" / "trainer_state.pt"
    best = run / "best" / "unet.pth"
    return {
        "named_steps": named,
        "has_last_state": last.is_file(),
        "last_state_mtime": datetime.fromtimestamp(last.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        if last.is_file() else None,
        "has_stopped_step": stopped.is_file(),
        "has_best": best.is_file(),
    }


def _curve(rows: list[dict], key: str = "loss") -> list[dict[str, float]]:
    pts = []
    for r in rows:
        if "step" not in r or key not in r:
            continue
        val = r[key]
        if isinstance(val, list):
            val = sum(val) / max(1, len(val))
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        pts.append({"step": int(r["step"]), key: round(fv, 6)})
    return pts


def _rsi_curve(rows: list[dict]) -> list[dict[str, float]]:
    pts = []
    for r in rows:
        g = r.get("rsi_gain")
        if not g or "step" not in r:
            continue
        try:
            pts.append({
                "step": int(r["step"]),
                "up1": round(float(g[0]), 8),
                "up2": round(float(g[1] if len(g) > 1 else g[0]), 8),
            })
        except (TypeError, ValueError, IndexError):
            continue
    return pts


def _eta_hours(step: int, max_steps: int, created_at: str | None, hb_ts: datetime | None) -> float | None:
    if step <= 0 or max_steps <= step:
        return 0.0 if step >= max_steps else None
    t0 = _parse_ts(created_at)
    t1 = hb_ts or _now()
    if t0 is None:
        return None
    elapsed = (t1 - t0).total_seconds()
    if elapsed < 60:
        return None
    rate = step / elapsed
    if rate <= 0:
        return None
    return round((max_steps - step) / rate / 3600, 2)


def collect() -> dict[str, Any]:
    now = _now()
    procs = _train_procs()
    by_run = {p["run_id"]: p for p in procs if p.get("run_id")}
    gpus = _nvidia()
    host = _host()
    alerts: list[dict[str, str]] = []
    arms_out = []

    if host["swap_used_gb"] >= host["swap_total_gb"] - 0.05 and host["swap_total_gb"] > 0:
        alerts.append({
            "level": "warn",
            "msg": f"swap 已满（{host['swap_used_gb']} / {host['swap_total_gb']} GiB）。"
                   "容器 overlay 无法再加 swap；F2/F3 自身 RSS 未进 swap，继续观察。",
        })
    if host["disk_free_tb"] < 0.2:
        alerts.append({"level": "crit", "msg": f"磁盘仅剩 {host['disk_free_tb']} TB"})
    if host["ram_available_gb"] < 4:
        alerts.append({"level": "warn", "msg": f"MemAvailable 仅 {host['ram_available_gb']} GiB"})

    zombie_note = False
    for g in gpus:
        if g["index"] in (0, 1) and g["mem_used_mb"] > 8000 and g["util_pct"] == 0:
            zombie_note = True
    if zombie_note:
        alerts.append({
            "level": "info",
            "msg": "GPU0/1 仍被已死进程占约 21GB，不能在上面续跑 F1。"
                   "训练中禁止 nvidia-smi -r，以免带崩 GPU2/3。",
        })

    for spec in ARMS:
        run = RUNS / spec["run_id"]
        hb = _load_json(run / "heartbeat.json") or {}
        meta = _load_json(run / "launch_meta.json") or {}
        done = _load_json(run / "DONE.json")
        stop = (run / "STOP").is_file()
        log_rows = _read_jsonl(run / "train_log.jsonl")
        if not log_rows:
            alt = TRAIN_LOG_EXPORT / spec["run_id"] / "train_log.jsonl"
            log_rows = _read_jsonl(alt)
        if not log_rows:
            alt2 = TRAIN_LOG_EXPORT / spec["run_id"] / "train_loss.jsonl"
            log_rows = [{"step": r["step"], "loss": r.get("loss", r.get("train_loss"))}
                        for r in _read_jsonl(alt2)]
        val_rows = _read_jsonl(run / "val_log.jsonl")
        if not val_rows:
            val_rows = _read_jsonl(TRAIN_LOG_EXPORT / spec["run_id"] / "val_log.jsonl")
        proc = by_run.get(spec["run_id"])
        hb_ts = _parse_ts(hb.get("ts"))
        age = (now - hb_ts).total_seconds() if hb_ts else None
        last = log_rows[-1] if log_rows else {}
        step = int(hb.get("step") or last.get("step") or (done or {}).get("global_step") or 0)
        loss = last.get("loss", hb.get("loss"))
        losses = [float(r["loss"]) for r in log_rows if isinstance(r.get("loss"), (int, float))
                  and math.isfinite(r["loss"])]
        nan_n = sum(1 for r in log_rows if isinstance(r.get("loss"), (int, float))
                    and not math.isfinite(r["loss"]))
        max_steps = int(spec.get("max_steps") or meta.get("max_steps") or MAX_STEPS)
        errors = _tail_errors(ROOT / spec["log"]) if spec.get("log") else []
        ckpt = _ckpt_info(run) if run.is_dir() else {
            "named_steps": [], "has_last_state": False, "last_state_mtime": None,
            "has_stopped_step": False, "has_best": False,
        }

        if done:
            phase = "done"
        elif stop and not proc:
            phase = "paused"
        elif proc and age is not None and age > STALL_SEC:
            phase = "stalled"
        elif proc and step and step % 5000 == 0 and age is not None and age > 90:
            phase = "ckpt_val"
        elif proc:
            phase = "running"
        elif stop:
            phase = "paused"
        elif done:
            phase = "done"
        else:
            phase = "dead" if run.is_dir() and spec["arm"] != "F0" else ("done" if spec["arm"] == "F0" else "missing")

        if spec["arm"] == "F0" and done:
            phase = "done"

        if nan_n:
            alerts.append({"level": "crit", "msg": f"{spec['arm']} train_log 含 {nan_n} 条非有限 loss"})
            phase = "nan"
        if errors and phase in ("running", "ckpt_val", "stalled", "dead"):
            alerts.append({"level": "warn", "msg": f"{spec['arm']} 日志命中: {errors[-1][:160]}"})
        if phase == "stalled":
            alerts.append({"level": "crit",
                           "msg": f"{spec['arm']} 心跳已 {int(age or 0)}s 未更新，进程仍在（可能卡在 val 或 cache）。"})
        if phase == "dead":
            alerts.append({"level": "crit", "msg": f"{spec['arm']} 进程不在，将尝试从 last_state 续跑。"
                           if spec["auto_resume"] else f"{spec['arm']} 进程不在。"})

        eta = _eta_hours(step, max_steps, meta.get("created_at"), hb_ts)
        arms_out.append({
            "arm": spec["arm"],
            "run_id": spec["run_id"],
            "title": spec["title"],
            "role": spec["role"],
            "phase": phase,
            "step": step,
            "max_steps": max_steps,
            "pct": round(100 * step / max_steps, 2) if max_steps else 0,
            "loss": round(float(loss), 6) if isinstance(loss, (int, float)) else None,
            "min_loss": round(min(losses), 6) if losses else None,
            "lr": last.get("lr"),
            "rsi_gain": last.get("rsi_gain"),
            "heartbeat_age_s": int(age) if age is not None else None,
            "heartbeat_ts": hb.get("ts"),
            "pid": proc["pid"] if proc else None,
            "proc_state": proc["state"] if proc else None,
            "rss_gb": proc["rss_gb"] if proc else None,
            "cuda": proc["cuda"] if proc else (str(spec["gpu"]) if spec.get("gpu") is not None else None),
            "eta_h": eta,
            "ckpt": ckpt,
            "n_log": len(log_rows),
            "n_val": len(val_rows),
            "curve": _curve(log_rows),
            "rsi": _rsi_curve(log_rows),
            "val": [{"step": int(r["step"]), "val_loss": r.get("val_loss")}
                    for r in val_rows if "step" in r],
            "errors": errors,
            "stop": stop,
            "done": bool(done),
        })

    overall = "ok"
    if any(a["level"] == "crit" for a in alerts):
        overall = "crit"
    elif any(a["phase"] in ("stalled", "dead", "nan") for a in arms_out):
        overall = "crit"
    elif any(a["level"] == "warn" for a in alerts):
        overall = "warn"

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "overall": overall,
        "alerts": alerts,
        "arms": arms_out,
        "gpus": gpus,
        "host": host,
        "procs": procs,
        "resume_log": _resume_log[-20:],
        "monitor_uptime_s": int(time.time() - _started),
        "notes": [
            "训练不依赖外网；看板本身也不加载 CDN。",
            "每 100 step 写 heartbeat，每 1000 step 覆盖 last_state，每 5000 step 写 global_step_* 并跑 val。",
            "5k 存盘+val 时心跳可能停 5–20 分钟，属正常。",
            "F1 有 STOP，F2 有 STOP（等 F3 跑完再续）；F3 崩溃会从 last_state/stopped_step 续跑。",
        ],
    }


def _export_logs(status: dict[str, Any]) -> None:
    SNAP.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG_EXPORT.mkdir(parents=True, exist_ok=True)
    for spec in ARMS:
        src = RUNS / spec["run_id"]
        dest = TRAIN_LOG_EXPORT / spec["run_id"]
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("train_log.jsonl", "val_log.jsonl", "heartbeat.json",
                     "launch_meta.json", "parity_gate.json", "parent_load_manifest.json",
                     "STOP_PROVENANCE.json", "DONE.json"):
            p = src / name
            if p.is_file():
                try:
                    shutil.copy2(p, dest / name)
                except OSError:
                    pass
        named = (src and (RUNS / spec["run_id"])).exists()
        if named:
            steps = sorted(p.name for p in src.glob("global_step_*") if (p / "unet.pth").is_file())
            (dest / "milestones.json").write_text(
                json.dumps({"milestones": steps}, indent=2) + "\n", encoding="utf-8")
    (SNAP / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    (SNAP / "index.html").write_text(render_html(api_url="status.json"), encoding="utf-8")
    data_link = ROOT / "data/f123_dashboard"
    try:
        if data_link.is_symlink() or not data_link.exists():
            if data_link.is_symlink():
                data_link.unlink()
            data_link.symlink_to(SNAP)
    except OSError:
        pass


def _cmd_from_log(log_path: Path) -> list[str] | None:
    if not log_path.is_file():
        return None
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for _ in range(5):
                line = fh.readline()
                if line.startswith("CMD "):
                    return line[4:].strip().split()
    except OSError:
        return None
    return None


def _resume_target(run: Path) -> Path | None:
    """Prefer an explicit pause checkpoint over an older last_state overlay."""
    stopped = run / "stopped_step"
    if (stopped / "unet.pth").is_file() and (stopped / "trainer_state.pt").is_file():
        return stopped
    last = run / "last_state"
    if (last / "unet.pth").is_file() and (last / "trainer_state.pt").is_file():
        return last
    named = sorted(
        (p for p in run.glob("global_step_*") if (p / "unet.pth").is_file()),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    return named[-1] if named else None


def _arm_done(arm: str) -> bool:
    spec = next(s for s in ARMS if s["arm"] == arm)
    return (RUNS / spec["run_id"] / "DONE.json").is_file()


def _want_resume(arm: dict[str, Any], spec: dict[str, Any]) -> bool:
    """Crash-resume F3; chain F2 after F3 completes, F1 after F2 completes."""
    after = spec.get("resume_after")
    if after:
        if not _arm_done(after):
            return False
        return arm["phase"] in ("paused", "dead")
    if not spec.get("auto_resume"):
        return False
    return arm["phase"] == "dead"


def _load_monitor_state() -> dict[str, Any]:
    data = _load_json(MONITOR_STATE) or {"resumes": {}}
    return data


def _save_monitor_state(data: dict[str, Any]) -> None:
    MONITOR_STATE.parent.mkdir(parents=True, exist_ok=True)
    MONITOR_STATE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def maybe_resume(status: dict[str, Any], enabled: bool) -> None:
    if not enabled:
        return
    state = _load_monitor_state()
    resumes: dict = state.setdefault("resumes", {})
    now = time.time()
    for arm in status["arms"]:
        spec = next(s for s in ARMS if s["arm"] == arm["arm"])
        if not _want_resume(arm, spec):
            continue
        run = RUNS / spec["run_id"]
        hist = resumes.setdefault(spec["arm"], [])
        hist = [t for t in hist if now - t < 3600]
        resumes[spec["arm"]] = hist
        if hist and now - hist[-1] < RESUME_COOLDOWN_SEC:
            continue
        if len(hist) >= MAX_RESUMES_PER_HOUR:
            _resume_log.append({
                "ts": _now().isoformat(timespec="seconds"),
                "arm": spec["arm"],
                "ok": False,
                "msg": "hourly resume cap reached",
            })
            continue
        ckpt = _resume_target(run)
        cmd = _cmd_from_log(ROOT / spec["log"]) if spec.get("log") else None
        if ckpt is None or cmd is None:
            _resume_log.append({
                "ts": _now().isoformat(timespec="seconds"),
                "arm": spec["arm"],
                "ok": False,
                "msg": f"missing ckpt or CMD line (ckpt={ckpt})",
            })
            continue
        if "--resume_from" not in cmd:
            cmd = cmd + ["--resume_from", str(ckpt)]
        stop_path = run / "STOP"
        if stop_path.exists():
            stop_path.unlink()
        env = os.environ.copy()
        gpu = str(spec.get("gpu") if spec.get("gpu") is not None else 0)
        meta = _load_json(run / "launch_meta.json") or {}
        env["CUDA_VISIBLE_DEVICES"] = str(meta.get("gpu", gpu))
        env["PYTHONUNBUFFERED"] = "1"
        log_path = ROOT / spec["log"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n# auto-resume {_now().isoformat()} from {ckpt}\n")
                fh.flush()
                proc = subprocess.Popen(
                    cmd, cwd=str(VARIANT), env=env,
                    stdout=fh, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            hist.append(now)
            resumes[spec["arm"]] = hist
            _save_monitor_state(state)
            _resume_log.append({
                "ts": _now().isoformat(timespec="seconds"),
                "arm": spec["arm"],
                "ok": True,
                "pid": proc.pid,
                "from": str(ckpt),
            })
        except OSError as exc:
            _resume_log.append({
                "ts": _now().isoformat(timespec="seconds"),
                "arm": spec["arm"],
                "ok": False,
                "msg": str(exc),
            })
    _save_monitor_state(state)


def watchdog_loop(interval: float, auto_resume: bool, stop: threading.Event) -> None:
    while not stop.wait(interval):
        try:
            payload = collect()
            maybe_resume(payload, auto_resume)
            payload["resume_log"] = _resume_log[-20:]
            with _lock:
                _status.clear()
                _status.update(payload)
            _export_logs(payload)
        except Exception as exc:  # noqa: BLE001 — keep the loop alive
            with _lock:
                _status["watchdog_error"] = str(exc)
                _status["generated_at"] = _now().isoformat(timespec="seconds")


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta http-equiv="refresh" content="20"/>
<title>HRFont F0–F3 训练看板</title>
<style>
:root{--bg:#eef1f5;--card:#fff;--line:#d5dbe3;--muted:#5c6570;--ink:#1a1a1a;--accent:#1f4a6f;--ok:#2c6e49;--warn:#9a6b12;--crit:#8b2e2e;--pause:#5c6570}
*{box-sizing:border-box} body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}
h1{margin:0;font-size:1.25rem} .meta{color:var(--muted);font-size:12px;margin-top:4px}
main{max-width:1180px;margin:16px auto;padding:0 14px 48px}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px;margin:14px 0}
.card h2{margin:0 0 10px;font-size:1.05rem}
.row{display:flex;flex-wrap:wrap;gap:10px}
.stat{flex:1 1 160px;border:1px solid var(--line);padding:10px 12px;background:#f7f9fb}
.stat .k{font-size:11px;color:var(--muted)} .stat .v{font-size:20px;font-variant-numeric:tabular-nums;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border-bottom:1px solid var(--line);padding:6px 7px;text-align:left;vertical-align:top}
.bar{height:8px;background:#e4e9ef;position:relative;margin-top:4px}
.bar>i{display:block;height:100%;background:var(--accent)}
.pill{display:inline-block;padding:1px 7px;border:1px solid var(--line);font-size:11px;letter-spacing:.02em}
.pill.ok{color:var(--ok);border-color:#b7d4c3} .pill.warn{color:var(--warn);border-color:#e4d0a0}
.pill.crit,.pill.dead,.pill.stalled,.pill.nan{color:var(--crit);border-color:#e0b4b4}
.pill.paused,.pill.ckpt_val{color:var(--pause)}
.alert{border:1px solid #e4d0a0;background:#fff8e8;padding:8px 10px;margin:6px 0;font-size:13px}
.alert.crit{border-color:#e0b4b4;background:#fbeeee}
canvas{width:100%;height:260px;background:#fff}
.mono{font-family:ui-monospace,monospace;font-size:12px}
</style>
</head>
<body>
<header>
  <h1>HRFont F0–F3 训练看板</h1>
  <div class="meta" id="hdr">加载中…</div>
  <div class="meta">效果出图：<a href="http://127.0.0.1:8777/f3_ckpt_dashboard/">F3 ckpt 看板</a>（:8777，不干扰训练）</div>
</header>
<main>
  <div class="row" id="kpis"></div>
  <div id="alerts"></div>
  <section class="card">
    <h2>各臂状态</h2>
    <table>
      <thead><tr><th>臂</th><th>状态</th><th>step</th><th>loss</th><th>ETA</th><th>进程</th><th>ckpt</th></tr></thead>
      <tbody id="arms"></tbody>
    </table>
  </section>
  <section class="card">
    <h2>Train loss（log 每 100 step）</h2>
    <canvas id="loss" width="1100" height="260"></canvas>
    <p class="meta">横轴 global step · 纵轴 train loss · 源：各 run 的 train_log.jsonl。F0 为 parent 参考。</p>
  </section>
  <section class="card">
    <h2>RSI gain（zero-conv |w| 均值）</h2>
    <canvas id="rsi" width="1100" height="220"></canvas>
    <p class="meta">identity-safe 头从 0 缓慢增大属预期。实线=up block 1，虚线=up block 2。</p>
  </section>
  <section class="card">
    <h2>机器</h2>
    <table><thead><tr><th>GPU</th><th>util %</th><th>used MiB</th><th>free MiB</th><th>temp C</th></tr></thead>
    <tbody id="gpus"></tbody></table>
    <p class="meta" id="host"></p>
  </section>
  <section class="card">
    <h2>自动续跑记录</h2>
    <pre class="mono" id="resume">尚无</pre>
    <p class="meta">F2/F3 进程消失且无 STOP/DONE 时，从 last_state 续跑（恢复 optimizer + RNG）。F1 有 STOP，不自动续。</p>
  </section>
</main>
<script>
const SRC = window.HRFONT_STATUS_URL || "STATUS_URL_PLACEHOLDER";
const COL = {F0:"#7a8490", F1:"#1f4a6f", F2:"#c45c26", F3:"#2c6e49"};
function pill(ph){return '<span class="pill '+ph+'">'+ph+'</span>';}
function num(x,d){return (x==null||x!==x)?'—':Number(x).toFixed(d);}
function drawLines(id, series, ylab){
  const cv=document.getElementById(id); if(!cv) return;
  const ctx=cv.getContext('2d');
  const W=cv.width, H=cv.height, m={l:52,r:12,t:12,b:32};
  ctx.clearRect(0,0,W,H);
  let pts=[]; series.forEach(s=>s.pts.forEach(q=>pts.push(q)));
  if(!pts.length){ctx.fillStyle='#5c6570';ctx.fillText('等待数据',20,20);return;}
  const xs=pts.map(q=>q.x), ys=pts.map(q=>q.y);
  const xmin=0, xmax=Math.max(80000, ...xs);
  let ymin=Math.min(...ys), ymax=Math.max(...ys);
  if(ymax<=ymin){ymax=ymin+1e-6;}
  const ypad=(ymax-ymin)*0.08; ymin=Math.max(0,ymin-ypad); ymax=ymax+ypad;
  const x=v=>m.l+(v-xmin)/(xmax-xmin)*(W-m.l-m.r);
  const y=v=>m.t+(1-(v-ymin)/(ymax-ymin))*(H-m.t-m.b);
  ctx.strokeStyle='#d5dbe3'; ctx.beginPath();
  for(let i=0;i<=4;i++){const yy=m.t+i*(H-m.t-m.b)/4; ctx.moveTo(m.l,yy); ctx.lineTo(W-m.r,yy);}
  ctx.stroke();
  ctx.fillStyle='#5c6570'; ctx.font='11px system-ui';
  ctx.fillText(ylab, 8, 14);
  ctx.fillText(String(xmin), m.l, H-10);
  ctx.fillText(String(xmax), W-80, H-10);
  series.forEach(s=>{
    if(!s.pts.length) return;
    ctx.strokeStyle=s.color; ctx.lineWidth=1.4; ctx.setLineDash(s.dash||[]);
    ctx.beginPath();
    s.pts.forEach((q,i)=>{const X=x(q.x),Y=y(q.y); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
    ctx.stroke(); ctx.setLineDash([]);
  });
  let lx=p.l; series.forEach(s=>{
    ctx.fillStyle=s.color; ctx.fillRect(lx,8,10,10); ctx.fillStyle='#1a1a1a';
    ctx.fillText(s.name, lx+14, 17); lx+=ctx.measureText(s.name).width+32;
  });
}
function render(S){
  document.getElementById('hdr').innerHTML =
    '生成 '+S.generated_at+' · 总评 <b>'+S.overall+'</b> · 监控已运行 '+S.monitor_uptime_s+'s · 每 20s 刷新 · 无 CDN';
  const running=S.arms.filter(a=>a.phase==='running'||a.phase==='ckpt_val').length;
  const kpis=[
    ['总评', S.overall],
    ['在训', running+' / 3'],
    ['RAM 可用', num(S.host.ram_available_gb,1)+' GiB'],
    ['swap', num(S.host.swap_used_gb,1)+' / '+num(S.host.swap_total_gb,1)+' GiB'],
    ['磁盘剩余', num(S.host.disk_free_tb,2)+' TB'],
  ];
  document.getElementById('kpis').innerHTML=kpis.map(([k,v])=>
    '<div class="stat"><div class="k">'+k+'</div><div class="v">'+v+'</div></div>').join('');
  document.getElementById('alerts').innerHTML=(S.alerts||[]).map(a=>
    '<div class="alert '+(a.level||'')+'">'+a.msg+'</div>').join('');
  document.getElementById('arms').innerHTML=S.arms.map(a=>{
    const ck=(a.ckpt.named_steps||[]).join(', ')||'—';
    const last=a.ckpt.has_last_state?' last_state':'';
    const stop=a.ckpt.has_stopped_step?' stopped':'';
    return '<tr><td><b>'+a.arm+'</b><div class="meta">'+a.role+'</div></td>'+
      '<td>'+pill(a.phase)+'</td>'+
      '<td class="mono">'+a.step.toLocaleString()+' / '+a.max_steps.toLocaleString()+
      '<div class="bar"><i style="width:'+a.pct+'%"></i></div></td>'+
      '<td class="mono">'+num(a.loss,5)+'<div class="meta">min '+num(a.min_loss,5)+'</div></td>'+
      '<td>'+(a.eta_h==null?'—':a.eta_h+' h')+'</td>'+
      '<td class="mono">'+(a.pid||'—')+' '+(a.proc_state||'')+
      '<div class="meta">hb '+ (a.heartbeat_age_s==null?'—':a.heartbeat_age_s+'s')+' · rss '+num(a.rss_gb,1)+'G</div></td>'+
      '<td class="mono">'+ck+last+stop+'</td></tr>';
  }).join('');
  const lossSeries=S.arms.map(a=>({name:a.arm,color:COL[a.arm]||'#333',
    pts:(a.curve||[]).map(p=>({x:p.step,y:p.loss}))}));
  drawLines('loss', lossSeries, 'loss');
  const rsiSeries=[];
  S.arms.filter(a=>a.arm!=='F0').forEach(a=>{
    rsiSeries.push({name:a.arm+' u1', color:COL[a.arm], pts:(a.rsi||[]).map(p=>({x:p.step,y:p.up1}))});
    rsiSeries.push({name:a.arm+' u2', color:COL[a.arm], dash:[4,3], pts:(a.rsi||[]).map(p=>({x:p.step,y:p.up2}))});
  });
  drawLines('rsi', rsiSeries, '|w|');
  document.getElementById('gpus').innerHTML=(S.gpus||[]).map(g=>
    '<tr><td>'+g.index+'</td><td>'+g.util_pct+'</td><td>'+g.mem_used_mb+'</td><td>'+g.mem_free_mb+'</td><td>'+g.temp_c+'</td></tr>'
  ).join('');
  document.getElementById('host').textContent =
    'RAM '+S.host.ram_available_gb+' / '+S.host.ram_total_gb+' GiB available · swap '+
    S.host.swap_used_gb+'/'+S.host.swap_total_gb+' · disk '+S.host.disk_used_pct+'% used';
  const rl=S.resume_log||[];
  document.getElementById('resume').textContent = rl.length? JSON.stringify(rl,null,2) : '尚无';
}
async function tick(){
  try{
    const r=await fetch(SRC+'?t='+Date.now(), {cache:'no-store'});
    render(await r.json());
  }catch(e){
    document.getElementById('hdr').textContent='看板读取失败: '+e;
  }
}
tick(); setInterval(tick, 10000);
</script>
</body></html>
"""


def render_html(api_url: str) -> str:
    return HTML.replace("STATUS_URL_PLACEHOLDER", api_url)


class Handler(SimpleHTTPRequestHandler):
    server_version = "HRFontF123/1.0"

    def log_message(self, fmt: str, *args) -> None:
        return

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlsplit(self.path).path)
        if path in ("/", "/index.html"):
            self._send(200, render_html("/api/status").encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            with _lock:
                payload = dict(_status) if _status else collect()
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path == "/status.json":
            p = SNAP / "status.json"
            if p.is_file():
                self._send(200, p.read_bytes(), "application/json; charset=utf-8")
                return
        self.send_error(404, "not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--auto-resume", action="store_true")
    ap.add_argument("--once", action="store_true", help="write snapshot and exit")
    args = ap.parse_args()

    payload = collect()
    with _lock:
        _status.update(payload)
    _export_logs(payload)
    if args.once:
        print(json.dumps({"overall": payload["overall"], "out": str(SNAP)}, ensure_ascii=False))
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    stop = threading.Event()
    t = threading.Thread(target=watchdog_loop, args=(args.interval, args.auto_resume, stop), daemon=True)
    t.start()
    httpd = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"F123 monitor http://{args.bind}:{args.port}/  auto_resume={args.auto_resume}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        stop.set()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
