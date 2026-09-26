#!/usr/bin/env python3
"""Watch G0b → caches → G2 → G2-RL → G-RL-pilot → G1 → G0c.

Queue override 2026-09-14 (see reports/G_QUEUE_DECISIONS_20260914.md).
Does not overwrite F0-CLEAN / failed G0 / G0b. Does not kill running kids on restart.
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
G0_RUN = ROOT / "runs/G0b-F0-V0913-BS256-A-S3407"
G0_LOG = ROOT / "runs/G0b-F0-V0913-BS256-A-S3407.console.log"
G0C_RUN_ID = "G0c-F0-V0913-BS256-A-S3407"
G0C_RUN = ROOT / "runs" / G0C_RUN_ID
G0C_LOG = ROOT / "runs" / f"{G0C_RUN_ID}.console.log"
CACHE = ROOT / "artifacts/g0"
QUEUE = ROOT / "artifacts/g_queue"
PILOT_READY = QUEUE / "G_RL_PILOT_READY.json"
PILOT_DONE_MARK = QUEUE / "G_RL_PILOT_DONE.json"
WD = ROOT / "reports/watchdog_g"
SNAP = ROOT / "reports/g_dashboard"
PIDFILE = WD / "watchdog.pid"
STATUS = SNAP / "status.json"
INCIDENTS = WD / "incidents.jsonl"
DECISIONS = WD / "decisions.jsonl"
POLL_S = 30
MAX_STEPS_G0 = 10_000
MAX_STEPS_KID = 10_000
MAX_STEPS_G0C = 20_000

G2 = {"arm": "F2", "run_id": "G2-F2-V0913-A-S3407", "rsi": "delta"}
G1 = {"arm": "F1", "run_id": "G1-F1-V0913-A-S3407", "rsi": "official"}
G2RL = {"arm": "F2RL", "run_id": "G2-RL-V0913-A-S3407", "rsi": "delta"}
KID_GPUS = "0,1,2,3,4,5,6,7"
KID_NPROC = 8
KID_PORT = "29522"
ACCEL = "/root/miniforge3/envs/boogu/bin/accelerate"

STEP_RE = re.compile(r"Global Step (\d+)")
TQDM_RE = re.compile(r"\|\s+(\d+)/10000")
TQDM_20K_RE = re.compile(r"\|\s+(\d+)/20000")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_incident(kind: str, detail: dict) -> None:
    WD.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utcnow(), "kind": kind, **detail}
    with INCIDENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INCIDENT] {kind} {detail}", flush=True)


def log_decision(decision_id: str, summary: str, detail: dict | None = None) -> None:
    WD.mkdir(parents=True, exist_ok=True)
    rec = {"ts": utcnow(), "id": decision_id, "summary": summary, **(detail or {})}
    with DECISIONS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log_incident("decision", rec)


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
        if all(m in text for m in markers) and "watchdog_g.py" not in text and "watchdog_g_git" not in text:
            if alive(pid):
                return pid
    return None


def parse_step_from_text(text: str) -> int:
    best = 0
    for match in STEP_RE.finditer(text):
        best = max(best, int(match.group(1)))
    for match in TQDM_RE.finditer(text):
        best = max(best, int(match.group(1)))
    for match in TQDM_20K_RE.finditer(text):
        best = max(best, int(match.group(1)))
    return best


def run_step(run_dir: Path, console: Path | None = None) -> int:
    best = 0
    done_p = run_dir / "DONE.json"
    if done_p.is_file():
        try:
            best = max(best, int(json.loads(done_p.read_text(encoding="utf-8")).get("global_step", 0)))
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
            best = max(best, parse_step_from_text(console.read_text(encoding="utf-8", errors="replace")[-200000:]))
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


def g0_parent() -> Path:
    for name in ("global_step_10000", "last_state"):
        d = G0_RUN / name
        if (d / "unet.pth").is_file() or (d / "trainer_state.pt").is_file():
            return d
    raise FileNotFoundError("G0b parent weights missing")


def launch_kid(spec: dict, resume: Path | None) -> int:
    run_dir = ROOT / "runs" / spec["run_id"]
    parent = g0_parent()
    log = WD / f"{spec['run_id']}.stdout"
    cmd = [
        ACCEL, "launch",
        "--num_processes", str(KID_NPROC),
        "--num_machines", "1",
        "--mixed_precision", "fp16",
        "--multi_gpu",
        "--main_process_port", KID_PORT,
        str(F123 / "train.py"),
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
        "--learning_rate", "1e-05", "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "500",
        "--mixed_precision", "fp16",
        "--ckpt_interval", "2500", "--state_interval", "500", "--log_interval", "100",
        "--drop_prob", "0.1", "--perceptual_coefficient", "0.01", "--offset_coefficient", "0.5",
        "--parity_check",
    ]
    if resume:
        cmd += ["--resume_from", str(resume)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = KID_GPUS
    env["PYTHONUNBUFFERED"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
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


def ensure_g0b() -> dict:
    rec = {"id": "G0b", "run_id": G0_RUN.name, "gpu": "0-7"}
    if done(G0_RUN, MAX_STEPS_G0):
        rec.update(state="done", step=run_step(G0_RUN, G0_LOG))
        return rec
    if (G0_RUN / "STOP").is_file():
        rec.update(state="stopped_by_user", step=run_step(G0_RUN, G0_LOG))
        return rec
    pid = find_pid([G0_RUN.name, "train.py"])
    step = run_step(G0_RUN, G0_LOG)
    rec["step"] = step
    if pid:
        rec.update(state="running", pid=pid)
        return rec
    rec.update(state="waiting_or_incomplete", step=step)
    return rec


def cache_ready() -> bool:
    if not (CACHE / "READY.json").is_file():
        return False
    for name in ("es_spatial", "ec_multiscale", "es_local"):
        d = CACHE / name
        man = d / "manifest.json"
        prog = d / "progress.json"
        if not man.is_file() or not prog.is_file():
            return False
        p = json.loads(prog.read_text(encoding="utf-8"))
        if p.get("done") != p.get("total") or not p.get("total"):
            return False
    return True


def ensure_caches() -> dict:
    if cache_ready():
        return {"id": "g0_caches", "state": "done"}
    hold = CACHE / "HOLD"
    if hold.is_file():
        return {"id": "g0_caches", "state": "held", "note": hold.read_text(encoding="utf-8").strip() or "HOLD"}
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
            [PY, str(ROOT / "scripts/rebuild_g0_caches.py"), "--workers", "8"],
            cwd=str(ROOT), env=env,
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    log_incident("cache_start", {"pid": proc.pid})
    return {"id": "g0_caches", "state": "relaunched", "pid": proc.pid}


def ensure_kid(spec: dict) -> dict:
    run_dir = ROOT / "runs" / spec["run_id"]
    rec = {"id": spec["run_id"], "arm": spec["arm"], "gpu": "0-7"}
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


def pilot_ready_payload() -> dict | None:
    if not PILOT_READY.is_file():
        return None
    try:
        return json.loads(PILOT_READY.read_text(encoding="utf-8"))
    except Exception:
        return None


def pilot_finished() -> bool:
    if PILOT_DONE_MARK.is_file():
        return True
    payload = pilot_ready_payload() or {}
    run_id = payload.get("run_id")
    max_steps = int(payload.get("max_steps") or 5000)
    if run_id and done(ROOT / "runs" / run_id, max_steps):
        return True
    # also accept DONE in payload path
    out = payload.get("output_dir")
    if out and done(Path(out), max_steps):
        return True
    return False


def ensure_pilot() -> dict:
    """Launch G-RL-pilot via READY recipe; never invent hyperparams here."""
    rec = {"id": "G-RL-pilot", "gpu": "0-7"}
    if pilot_finished():
        rec.update(state="done")
        return rec
    payload = pilot_ready_payload()
    if not payload:
        rec.update(state="waiting_code")
        return rec
    run_id = str(payload.get("run_id") or "G-RL-pilot-V0913-A-S3407")
    rec["run_id"] = run_id
    run_dir = Path(payload.get("output_dir") or (ROOT / "runs" / run_id))
    max_steps = int(payload.get("max_steps") or 5000)
    if done(run_dir, max_steps):
        write_json(PILOT_DONE_MARK, {"ts": utcnow(), "run_id": run_id, "step": run_step(run_dir)})
        rec.update(state="done", step=run_step(run_dir))
        return rec
    if (run_dir / "STOP").is_file():
        rec.update(state="stopped_by_user", step=run_step(run_dir))
        return rec
    pid = find_pid([run_id, "train.py"]) or find_pid(["launch_g_rl_pilot.py"])
    step = run_step(run_dir)
    rec["step"] = step
    if pid:
        rec.update(state="running", pid=pid)
        return rec
    cmd = payload.get("cmd")
    if not cmd or not isinstance(cmd, list):
        script = payload.get("script") or str(ROOT / "scripts/launch_g_rl_pilot.py")
        if not Path(script).is_file():
            rec.update(state="waiting_code", note="READY missing cmd/script")
            return rec
        cmd = [PY, script, "--yes"]
        for key in ("extra_args", "argv_tail"):
            extra = payload.get(key)
            if isinstance(extra, list):
                cmd += [str(x) for x in extra]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = KID_GPUS
    env["PYTHONUNBUFFERED"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    log = WD / f"{run_id}.stdout"
    run_dir.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== WATCHDOG PILOT START {utcnow()} =====\n")
        logf.write("CMD " + " ".join(str(x) for x in cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            [str(x) for x in cmd], cwd=str(ROOT), env=env,
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    log_incident("pilot_launch", {"pid": proc.pid, "run_id": run_id, "cmd": cmd})
    rec.update(state="relaunched", pid=proc.pid)
    return rec


def launch_g0c(resume: Path) -> int:
    cmd = [
        PY, str(ROOT / "scripts/launch_g0.py"),
        "--yes",
        "--run_id", G0C_RUN_ID,
        "--resume_from", str(resume),
        "--max_steps", str(MAX_STEPS_G0C),
        "--warmup", "500",
    ]
    G0C_LOG.parent.mkdir(parents=True, exist_ok=True)
    with G0C_LOG.open("a", encoding="utf-8") as logf:
        logf.write(f"\n===== WATCHDOG G0c START {utcnow()} =====\n")
        logf.write("CMD " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return proc.pid


def ensure_g0c() -> dict:
    rec = {"id": "G0c", "run_id": G0C_RUN_ID, "gpu": "0-7"}
    if done(G0C_RUN, MAX_STEPS_G0C):
        rec.update(state="done", step=run_step(G0C_RUN, G0C_LOG))
        return rec
    if (G0C_RUN / "STOP").is_file():
        rec.update(state="stopped_by_user", step=run_step(G0C_RUN, G0C_LOG))
        return rec
    launcher = find_pid(["launch_g0.py", G0C_RUN_ID])
    pid = find_pid([G0C_RUN_ID, "train.py"])
    step = run_step(G0C_RUN, G0C_LOG)
    rec["step"] = step
    if pid:
        rec.update(state="running", pid=pid)
        return rec
    if launcher:
        rec.update(state="starting", pid=launcher)
        return rec
    parent = g0_parent()
    rd = resume_dir(G0C_RUN) or parent
    new_pid = launch_g0c(rd)
    log_decision(
        "D3_g0c_launch",
        "start G0c continue from G0b@10k toward 20k",
        {"pid": new_pid, "resume": str(rd), "max_steps": MAX_STEPS_G0C},
    )
    rec.update(state="relaunched", pid=new_pid)
    return rec


def render_board(snapshot: dict) -> str:
    jobs = snapshot.get("jobs") or []
    rows = []
    for j in jobs:
        state = str(j.get("state") or "")
        cls = "ok" if state == "done" else ("run" if "run" in state or state in ("starting", "waiting_code", "deferred") else "bad")
        rows.append(
            f"<tr><td>{j.get('id') or j.get('arm') or ''}</td>"
            f"<td class='{cls}'>{state}</td><td>{j.get('step', '')}</td>"
            f"<td>{j.get('gpu', '')}</td><td>{j.get('pid', '')}</td></tr>"
        )
    gpu = json.dumps(snapshot.get("gpus") or [], ensure_ascii=False, indent=2)
    inc = "\n".join(json.dumps(x, ensure_ascii=False) for x in (snapshot.get("incidents") or [])[-12:])
    payload = json.dumps(snapshot, ensure_ascii=False)
    return f"""<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>Group G 训练看板</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
h1{{font-size:20px}} table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{border:1px solid #444;padding:8px 10px;text-align:left}} .ok{{color:#7d7}}.run{{color:#fd6}}.bad{{color:#f66}}
.meta{{color:#aaa;font-size:13px}} a{{color:#9cf}} pre{{white-space:pre-wrap}}
.warn{{color:#fd6}}
</style>
<h1>Group G</h1>
<p class="warn">队列：G2 → G2-RL → G-RL-pilot → G1 → G0c。决策见 reports/G_QUEUE_DECISIONS_20260914.md</p>
<p class="meta">阶段：{snapshot.get('phase','')}　{snapshot.get('note','')}　磁盘 {snapshot.get('disk_gb','?')}G　{snapshot.get('ts','')}</p>
<table><thead><tr><th>任务</th><th>状态</th><th>步数</th><th>GPU</th><th>PID</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>GPU</h2><pre>{gpu}</pre>
<h2>最近事件</h2><pre>{inc}</pre>
<script id="snap" type="application/json">{payload}</script>
"""


def write_board(snapshot: dict) -> None:
    SNAP.mkdir(parents=True, exist_ok=True)
    write_json(STATUS, snapshot)
    html = render_board(snapshot)
    (SNAP / "index.html").write_text(html, encoding="utf-8")
    (ROOT / "G_BOARD.html").write_text(html, encoding="utf-8")


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
    deferred_pilot = False
    while True:
        try:
            jobs = []
            g0 = ensure_g0b()
            jobs.append(g0)
            phase = "g0"
            note = "等待 G0b@10k。"
            if g0.get("state") == "done":
                caches = ensure_caches()
                jobs.append(caches)
                phase = "caches"
                note = "Es/Ec/es_local。"
                if caches.get("state") == "done":
                    g2 = ensure_kid(G2)
                    jobs.append(g2)
                    phase = "g2"
                    note = "G2（8×8=64）。完成后 → G2-RL → pilot → G1 → G0c。"
                    if g2.get("state") == "done":
                        rl = ensure_kid(G2RL)
                        jobs.append(rl)
                        phase = "g2rl"
                        note = "G2-RL。完成后优先 G-RL-pilot。"
                        if rl.get("state") == "done":
                            # Prefer pilot before G1 when READY; else defer to keep GPUs busy.
                            if pilot_finished():
                                jobs.append({"id": "G-RL-pilot", "state": "done"})
                                g1 = ensure_kid(G1)
                                jobs.append(g1)
                                phase = "g1"
                                note = "pilot 已完成；跑 G1。"
                                if g1.get("state") == "done":
                                    g0c = ensure_g0c()
                                    jobs.append(g0c)
                                    phase = "g0c"
                                    note = "G0c 续训 → 20k。"
                                    if g0c.get("state") == "done":
                                        phase = "done"
                                        note = "G2 + G2-RL + pilot + G1 + G0c 完成。"
                            else:
                                pilot = ensure_pilot()
                                jobs.append(pilot)
                                if pilot.get("state") in ("running", "relaunched", "starting"):
                                    phase = "g_rl_pilot"
                                    note = "G-RL-pilot 训练中。"
                                elif pilot.get("state") == "done":
                                    g1 = ensure_kid(G1)
                                    jobs.append(g1)
                                    phase = "g1"
                                    note = "pilot 完成；G1。"
                                    if g1.get("state") == "done":
                                        g0c = ensure_g0c()
                                        jobs.append(g0c)
                                        phase = "g0c" if g0c.get("state") != "done" else "done"
                                        note = "G0c 续训。" if phase == "g0c" else "全流水线完成。"
                                else:
                                    # waiting_code → defer: run G1 to avoid idle
                                    if not deferred_pilot:
                                        deferred_pilot = True
                                        log_decision(
                                            "D2_pilot_deferred",
                                            "G-RL-pilot code not READY after G2-RL; start G1 to keep GPUs busy",
                                            {"pilot_state": pilot.get("state")},
                                        )
                                    g1 = ensure_kid(G1)
                                    jobs.append(g1)
                                    phase = "g1_deferred_pilot"
                                    note = "pilot 代码未到，先跑 G1（卡不空）。"
                                    if g1.get("state") == "done":
                                        # try pilot again before G0c
                                        pilot2 = ensure_pilot()
                                        jobs.append(pilot2)
                                        if pilot2.get("state") in ("running", "relaunched", "starting"):
                                            phase = "g_rl_pilot"
                                            note = "G1 完；补跑 deferred G-RL-pilot。"
                                        elif pilot2.get("state") == "done":
                                            g0c = ensure_g0c()
                                            jobs.append(g0c)
                                            phase = "g0c" if g0c.get("state") != "done" else "done"
                                            note = "G0c。" if phase == "g0c" else "完成。"
                                        else:
                                            g0c = ensure_g0c()
                                            jobs.append(g0c)
                                            phase = "g0c"
                                            note = "pilot 仍未 READY；先 G0c，pilot 队尾等待。"
                                            if g0c.get("state") == "done":
                                                # keep trying pilot after G0c
                                                pilot3 = ensure_pilot()
                                                jobs.append(pilot3)
                                                if pilot3.get("state") == "done":
                                                    phase = "done"
                                                    note = "G0c + 迟到 pilot 均完成。"
                                                elif pilot3.get("state") in ("running", "relaunched", "starting"):
                                                    phase = "g_rl_pilot"
                                                    note = "G0c 完；补跑 pilot。"
                                                else:
                                                    phase = "waiting_pilot_code"
                                                    note = "主队列完成；仍等 G-RL-pilot 代码。"
            snap = {
                "ts": utcnow(),
                "phase": phase,
                "note": note,
                "disk_gb": round(disk_gb(), 1),
                "jobs": jobs,
                "gpus": gpu_snap(),
                "incidents": recent_incidents(),
                "queue": ["G2", "G2-RL", "G-RL-pilot", "G1", "G0c"],
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
    QUEUE.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()) + "\n")
    log_incident("watchdog_start", {"pid": os.getpid(), "queue": "G2→G2-RL→pilot→G1→G0c"})
    log_decision("D1_queue_live", "watchdog_g live with new serial queue; will not interrupt running G2")
    loop()
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    raise SystemExit(main())
