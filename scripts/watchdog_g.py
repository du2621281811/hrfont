#!/usr/bin/env python3
"""Watch G0 → G0 caches → G2 → G1 + G2-PRL.

Does not overwrite F0-CLEAN / F1-CLEAN / F2-CLEAN. Does not revive F0-c.
Survives SSH/Cursor disconnect (start with setsid).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
PY = "/root/miniforge3/envs/boogu/bin/python"
F123 = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
CLEAN_MAP = ROOT / "manifests/v0913_clean"
G0_RUN = ROOT / "runs/G0-F0-V0913-BS256-A-S3407"
G0_LOG = ROOT / "runs/G0-F0-V0913-BS256-A-S3407.console.log"
CACHE = ROOT / "artifacts/g0"
WD = ROOT / "reports/watchdog_g"
SNAP = ROOT / "reports/g_dashboard"
PIDFILE = WD / "watchdog.pid"
STATUS = SNAP / "status.json"
INCIDENTS = WD / "incidents.jsonl"
POLL_S = 30
MAX_STEPS_G0 = 10_000
MAX_STEPS_KID = 10_000

G2 = {"arm": "F2", "run_id": "G2-F2-V0913-A-S3407", "gpu": 0, "rsi": "delta"}
G1 = {"arm": "F1", "run_id": "G1-F1-V0913-A-S3407", "gpu": 1, "rsi": "official"}
G2PRL = {"arm": "F2PRL", "run_id": "G2-PRL-V0913-A-S3407", "gpu": 2, "rsi": "delta"}

STEP_RE = re.compile(r"Global Step (\d+)")
TQDM_RE = re.compile(r"\|\s+(\d+)/10000")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_incident(kind: str, detail: dict) -> None:
    WD.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utcnow(), "kind": kind, **detail}
    with INCIDENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INCIDENT] {kind} {detail}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [x.decode("utf-8", errors="replace") for x in raw.split(b"\x00") if x]


def find_pid(markers: list[str]) -> int | None:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        args = cmdline(pid)
        if not args or "python" not in Path(args[0]).name:
            continue
        text = " ".join(args)
        if all(m in text for m in markers) and "watchdog_g.py" not in text:
            if alive(pid):
                return pid
    return None


def parse_step_from_text(text: str) -> int:
    best = 0
    for match in STEP_RE.finditer(text):
        best = max(best, int(match.group(1)))
    for match in TQDM_RE.finditer(text):
        best = max(best, int(match.group(1)))
    return best


def run_step(run_dir: Path, console: Path | None = None) -> int:
    best = 0
    done = run_dir / "DONE.json"
    if done.is_file():
        try:
            best = max(best, int(json.loads(done.read_text(encoding="utf-8")).get("global_step", 0)))
        except Exception:
            pass
    hb = run_dir / "heartbeat.json"
    if hb.is_file():
        try:
            best = max(best, int(json.loads(hb.read_text(encoding="utf-8")).get("step", 0)))
        except Exception:
            pass
    for path in run_dir.glob("global_step_*"):
        try:
            best = max(best, int(path.name.split("_")[-1]))
        except ValueError:
            pass
    log = run_dir / "fontdiffuser_training.log"
    if log.is_file():
        try:
            best = max(best, parse_step_from_text(log.read_text(encoding="utf-8", errors="replace")[-20000:]))
        except OSError:
            pass
    if console and console.is_file():
        try:
            best = max(best, parse_step_from_text(console.read_text(encoding="utf-8", errors="replace")[-8000:]))
        except OSError:
            pass
    return best


def done(run_dir: Path, max_steps: int) -> bool:
    p = run_dir / "DONE.json"
    if not p.is_file():
        return False
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "completed" and int(payload.get("global_step", 0)) >= max_steps


def resume_dir(run_dir: Path) -> Path | None:
    last = run_dir / "last_state"
    if (last / "trainer_state.pt").is_file():
        return last
    steps = sorted(run_dir.glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1]), reverse=True)
    for d in steps:
        if (d / "trainer_state.pt").is_file():
            return d
    return None


def gpu_snap() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({"index": int(parts[0]), "util": float(parts[1]), "mem": float(parts[2]), "power": float(parts[3])})
    return rows


def disk_gb() -> float:
    st = os.statvfs(ROOT)
    return st.f_bavail * st.f_frsize / 1024 ** 3


def launch_g0(resume: Path | None) -> int:
    cmd = [PY, str(ROOT / "scripts/launch_g0.py"), "--yes"]
    if resume:
        cmd += ["--resume", "--resume_from", str(resume)]
    G0_LOG.parent.mkdir(parents=True, exist_ok=True)
    G0_RUN.mkdir(parents=True, exist_ok=True)
    with G0_LOG.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== WATCHDOG {'RESUME' if resume else 'START'} {utcnow()} =====\n")
        logf.write("CMD " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return proc.pid


def g0_parent() -> Path:
    for name in ("global_step_10000", "last_state"):
        d = G0_RUN / name
        if (d / "unet.pth").is_file():
            return d
    raise FileNotFoundError("G0 parent weights missing")


def launch_kid(spec: dict, resume: Path | None) -> int:
    run_dir = ROOT / "runs" / spec["run_id"]
    parent = g0_parent()
    log = WD / f"{spec['run_id']}.stdout"
    cmd = [
        PY, str(F123 / "train.py"),
        "--arm", spec["arm"],
        "--rsi_source", spec["rsi"],
        "--no-support",
        "--support_drop", "0.2", "--support_k", "8", "--source_drop", "0.25",
        "--seed", "3407",
        "--experience_name", spec["run_id"],
        "--output_dir", str(run_dir),
        "--data_root", str(DATA),
        "--split_manifest", str(SPLIT),
        "--v0913_clean_map", str(CLEAN_MAP),
        "--es_cache_path", str(CACHE / "es_spatial"),
        "--es_local_cache_path", str(CACHE / "es_local"),
        "--ec_cache_path", str(CACHE / "ec_multiscale"),
        "--phase_1_ckpt_dir", str(parent),
        "--resolution", "96", "--style_image_size", "96", "--content_image_size", "96",
        "--train_batch_size", "8", "--gradient_accumulation_steps", "1",
        "--max_train_steps", str(MAX_STEPS_KID),
        "--learning_rate", "1e-05", "--lr_scheduler", "linear", "--lr_warmup_steps", "500",
        "--mixed_precision", "fp16",
        "--ckpt_interval", "2500", "--state_interval", "500", "--log_interval", "100",
        "--drop_prob", "0.1", "--perceptual_coefficient", "0.01", "--offset_coefficient", "0.5",
        "--parity_check",
    ]
    if resume:
        cmd += ["--resume_from", str(resume)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu"])
    env["PYTHONUNBUFFERED"] = "1"
    run_dir.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== WATCHDOG {'RESUME' if resume else 'START'} {utcnow()} =====\n")
        logf.write("CMD " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(F123), env=env,
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return proc.pid


def ensure_g0() -> dict:
    rec = {"id": "G0", "run_id": G0_RUN.name, "gpu": "0-7"}
    if done(G0_RUN, MAX_STEPS_G0):
        rec.update(state="done", step=run_step(G0_RUN, G0_LOG))
        return rec
    if (G0_RUN / "STOP").is_file():
        rec.update(state="stopped_by_user", step=run_step(G0_RUN, G0_LOG))
        return rec
    launcher = find_pid(["launch_g0.py"])
    pid = find_pid(["G0-F0-V0913-BS256-A-S3407", "train.py"])
    step = run_step(G0_RUN, G0_LOG)
    rec["step"] = step
    if pid:
        write_json(G0_RUN / "heartbeat.json", {"step": step, "status": "running", "pid": pid})
        rec.update(state="running", pid=pid)
        return rec
    if launcher:
        rec.update(state="starting", pid=launcher)
        return rec
    if not G0_RUN.exists():
        rec.update(state="waiting_launch")
        return rec
    rd = resume_dir(G0_RUN)
    new_pid = launch_g0(rd)
    log_incident("g0_resume" if rd else "g0_start", {"pid": new_pid, "resume": str(rd) if rd else None, "step": step})
    rec.update(state="relaunched", pid=new_pid)
    return rec


def cache_ready() -> bool:
    return (CACHE / "READY.json").is_file() and all(
        (CACHE / name / "progress.json").is_file()
        for name in ("es_spatial", "ec_multiscale", "es_local")
    )


def ensure_caches() -> dict:
    if cache_ready():
        return {"id": "g0_caches", "state": "done"}
    pid = find_pid(["rebuild_g0_caches.py"])
    if pid:
        return {"id": "g0_caches", "state": "running", "pid": pid}
    log = WD / "rebuild_caches.stdout"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== G0 CACHE BUILD {utcnow()} =====\n")
        logf.flush()
        proc = subprocess.Popen(
            [PY, str(ROOT / "scripts/rebuild_g0_caches.py"), "--gpu", "0"],
            cwd=str(ROOT), env=env,
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    log_incident("cache_start", {"pid": proc.pid})
    return {"id": "g0_caches", "state": "relaunched", "pid": proc.pid}


def ensure_kid(spec: dict) -> dict:
    run_dir = ROOT / "runs" / spec["run_id"]
    rec = {"id": spec["run_id"], "arm": spec["arm"], "gpu": spec["gpu"]}
    if done(run_dir, MAX_STEPS_KID):
        rec.update(state="done", step=run_step(run_dir))
        return rec
    if (run_dir / "STOP").is_file():
        rec.update(state="stopped_by_user", step=run_step(run_dir))
        return rec
    pid = find_pid([spec["run_id"], "train.py"])
    step = run_step(run_dir, WD / f"{spec['run_id']}.stdout")
    rec["step"] = step
    if pid:
        rec.update(state="running", pid=pid)
        return rec
    rd = resume_dir(run_dir)
    new_pid = launch_kid(spec, rd)
    log_incident("kid_launch", {"arm": spec["arm"], "pid": new_pid, "resume": str(rd) if rd else None, "step": step})
    rec.update(state="relaunched", pid=new_pid)
    return rec


BOARD_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Group G 训练看板</title>
<style>
body{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#111;color:#eee}
h1{font-size:20px} table{border-collapse:collapse;width:100%;margin:16px 0}
th,td{border:1px solid #444;padding:8px 10px;text-align:left} .ok{color:#7d7}.run{color:#fd6}.bad{color:#f66}
.meta{color:#aaa;font-size:13px} a{color:#9cf}
</style>
<h1>Group G</h1>
<p class="meta">G0 8×32=256 → cache → <b>G2</b> → G1 + G2-PRL。远程走
<a href="http://127.0.0.1:19000/g/">http://127.0.0.1:19000/g/</a></p>
<p class="meta"><span id="ts"></span></p>
<p id="phase"></p>
<p class="meta" id="disk"></p>
<table><thead><tr><th>任务</th><th>状态</th><th>步数</th><th>GPU</th><th>PID</th></tr></thead>
<tbody id="rows"></tbody></table>
<h2>GPU</h2><pre id="gpu"></pre>
<h2>最近事件</h2><pre id="inc"></pre>
<script>
async function tick(){
  const s = await (await fetch('status.json',{cache:'no-store'})).json();
  document.getElementById('ts').textContent = s.ts || '';
  document.getElementById('phase').textContent = '阶段：' + (s.phase||'') + '  ' + (s.note||'');
  document.getElementById('disk').textContent = '剩余磁盘 ' + (s.disk_gb||'?') + 'G';
  const rows = (s.jobs||[]).map(j => {
    const cls = j.state==='done'?'ok':(String(j.state).includes('run')?'run':'bad');
    return `<tr><td>${j.id||j.arm||''}</td><td class="${cls}">${j.state||''}</td>
      <td>${j.step??''}</td><td>${j.gpu??''}</td><td>${j.pid??''}</td></tr>`;
  }).join('');
  document.getElementById('rows').innerHTML = rows;
  document.getElementById('gpu').textContent = JSON.stringify(s.gpus||[], null, 2);
  document.getElementById('inc').textContent = (s.incidents||[]).slice(-12).map(x=>JSON.stringify(x)).join('\\n');
}
tick(); setInterval(tick, 5000);
</script>
"""


def write_board(snapshot: dict) -> None:
    SNAP.mkdir(parents=True, exist_ok=True)
    write_json(STATUS, snapshot)
    html = SNAP / "index.html"
    if not html.is_file() or "Group G" not in html.read_text(encoding="utf-8"):
        html.write_text(BOARD_HTML, encoding="utf-8")


def recent_incidents(n: int = 20) -> list:
    if not INCIDENTS.is_file():
        return []
    out = []
    for line in INCIDENTS.read_text(encoding="utf-8").splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def loop() -> None:
    while True:
        try:
            jobs = []
            g0 = ensure_g0()
            jobs.append(g0)
            phase = "g0"
            note = "G0 8×32=256，日程 10k，主看 5k。"
            if g0.get("state") == "done":
                caches = ensure_caches()
                jobs.append(caches)
                phase = "caches"
                note = "用 G0@10k 编 Es/Ec/es_local。盘不够会删 dirty Ec。"
                if caches.get("state") == "done":
                    g2 = ensure_kid(G2)
                    jobs.append(g2)
                    phase = "g2"
                    note = "先 G2（mean-Δ）。G2 跑完再开 G1 与 G2-PRL。"
                    if g2.get("state") == "done":
                        g1 = ensure_kid(G1)
                        prl = ensure_kid(G2PRL)
                        jobs.extend([g1, prl])
                        phase = "g1_g2prl"
                        note = "G1 与 G2-PRL 并行。"
                        if g1.get("state") == "done" and prl.get("state") == "done":
                            phase = "done"
                            note = "Group G：G0 + G2 + G1 + G2-PRL 均已 10k。"
            snap = {
                "ts": utcnow(),
                "phase": phase,
                "note": note,
                "disk_gb": round(disk_gb(), 1),
                "jobs": jobs,
                "gpus": gpu_snap(),
                "incidents": recent_incidents(),
            }
            write_board(snap)
            if phase == "done":
                log_incident("pipeline_complete", {"jobs": [j.get("id") for j in jobs]})
                break
        except Exception as exc:
            log_incident("watchdog_exception", {"error": str(exc), "tb": traceback.format_exc()})
        time.sleep(POLL_S)


def main() -> int:
    WD.mkdir(parents=True, exist_ok=True)
    SNAP.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()) + "\n")
    (SNAP / "index.html").write_text(BOARD_HTML, encoding="utf-8")
    log_incident("watchdog_start", {"pid": os.getpid()})
    loop()
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    raise SystemExit(main())
