#!/usr/bin/env python3
"""G-series 1/2/4/8-shot eval with OLD ref8=永和书风骨韵天地.

Reuses existing test 1-shot/8-shot preds from reports/g_v0913_shot when identical.
Writes reports/g_v0913_shot_k1248/. Sampling: DPM++20 CFG7.5 seed3407.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))
import eval_f03_test16_strat as E  # noqa: E402

OUT = ROOT / "reports/g_v0913_shot_k1248"
OLD = ROOT / "reports/g_v0913_shot"
F123 = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
F0V = ROOT / "code/variants/cn2west_f0_rsifree/FontDiffuser"
G0_ES = ROOT / "artifacts/g0/es_spatial"
G0_EC = ROOT / "artifacts/g0/ec_multiscale"
G0_LOCAL = ROOT / "artifacts/g0/es_local"
TC_CACHE = ROOT / "artifacts/tc_v2_cache"
PY = "/root/miniforge3/envs/boogu/bin/python"

REF8 = "永和书风骨韵天地"
TRAIN5 = ["FZPTYJW", "FZDuHJW_Cu", "FZShuLTJW-H", "FZYiMSJW-T", "FZFeiSJW-EL"]
VAL5 = ["FZYouHK_511M", "FZLTHProGBK_H", "FZJunYTJW-H", "FZBangSKKXJW", "FZKANGJW"]
SHOTS = (1, 2, 4, 8)

REUSE_TEST = {
    "G0b_s1",
    "G0c_s1",
    "G1_s1",
    "G1_s8",
    "G2_s1",
    "G2_s8",
    "G2RL_s1",
    "G2RL_s8",
    "pilot_s1",
    "pilot_s8",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_methods() -> dict:
    g0c = ROOT / "runs/G0c-F0-V0913-BS256-A-S3407/global_step_20000"
    base_g = {
        "variant": F123,
        "es_cache": str(G0_ES),
        "ec_cache": str(G0_EC),
        "es_local_cache": str(G0_LOCAL),
    }
    methods: dict = {
        "G0b_s1": {
            "label": "G0b@10k",
            "kind": "image",
            "variant": F0V,
            "ckpt": ROOT / "runs/G0b-F0-V0913-BS256-A-S3407/global_step_10000",
            "style_k": 1,
            "style_oneshot": True,
        },
        "G0c_s1": {
            "label": "G0c@20k",
            "kind": "image",
            "variant": F0V,
            "ckpt": g0c,
            "style_k": 1,
            "style_oneshot": True,
        },
    }
    multi = [
        ("G1", ROOT / "runs/G1-F1-V0913-A-S3407/global_step_10000", "f1", False, False, False),
        ("G2", ROOT / "runs/G2-F2-V0913-A-S3407/global_step_10000", "f2", False, False, False),
        ("G2RL", ROOT / "runs/G2-RL-V0913-A-S3407/global_step_10000", "f2", False, True, False),
        ("pilot", ROOT / "runs/G-RL-pilot-V0913-A-S3407/global_step_1000", "f2", False, True, False),
        ("pilot8", ROOT / "runs/G-RL-pilot-8gpu-V0913-A-S3407/best", "f2", False, True, False),
        ("TCG2", ROOT / "runs/G-TC-G2-8gpu-V0913-A-S3407/best", "f2", False, False, True),
        ("TCG2RL", ROOT / "runs/G-TC-G2RL-8gpu-V0913-A-S3407/best", "f2", False, True, True),
    ]
    for name, ckpt, kind, pattn, rl, tc in multi:
        for k in SHOTS:
            mid = f"{name}_s{k}"
            spec = {
                **base_g,
                "label": f"{name} {k}-shot",
                "kind": kind,
                "ckpt": ckpt,
                "style_k": k,
                "style_oneshot": k == 1,
                "delta_oneshot": k == 1,
                "style_pattn": pattn,
                "style_rl128": rl,
                "tc_enabled": tc,
            }
            if tc:
                spec["tc_cache"] = str(TC_CACHE)
            methods[mid] = spec
    return methods


def link_tree(src: Path, dst: Path) -> int:
    n = 0
    if not src.is_dir():
        return 0
    for png in src.rglob("*.png"):
        rel = png.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        for s, d in ((png, out), (png.with_suffix(".json"), out.with_suffix(".json"))):
            if not s.is_file():
                continue
            if d.exists() or d.is_symlink():
                continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
            n += 1
    return n


def reuse_test_preds(methods: dict) -> dict:
    stats = {}
    for mid in REUSE_TEST:
        if mid not in methods:
            continue
        n = link_tree(OLD / "preds" / mid, OUT / "preds" / mid)
        stats[mid] = n
        print(json.dumps({"reuse": mid, "linked_files": n}), flush=True)
    return stats


def job_list(methods: dict) -> list[dict]:
    test16 = json.loads((ROOT / "manifests/split_v3_228_16_16.json").read_text())["stems"]["test"]
    split_fonts = {"test": test16, "train": TRAIN5, "val": VAL5}
    jobs = []
    for mid, spec in methods.items():
        splits = []
        for split, stems in split_fonts.items():
            if split == "test" and mid in REUSE_TEST:
                continue
            splits.append({"split": split, "stems": stems})
        if splits:
            jobs.append({"mid": mid, "kind": spec["kind"], "splits": splits})
    return jobs


def expected_pngs(mid: str) -> int:
    """test16+train5+val5 * 47 chars; reuse-test arms skip test."""
    n_chars = len(E.STRATIFIED)
    n_fonts = 10 if mid in REUSE_TEST else 26
    return n_fonts * n_chars


def mid_done(mid: str) -> bool:
    p = OUT / "preds" / mid
    if not p.is_dir():
        return False
    return sum(1 for _ in p.rglob("*.png")) >= expected_pngs(mid)


def run_one(mid: str, splits: list[dict], device: str, overwrite: bool, shard: str = "0/1") -> None:
    E.METHODS.update(build_methods())
    spec = E.METHODS[mid]
    kind = spec["kind"]
    for item in splits:
        split = item["split"]
        stems = E.shard_list(list(item["stems"]), shard)
        if not stems:
            continue
        E.configure_split(split, out=OUT)
        E.OUT = OUT
        print(
            json.dumps(
                {
                    "start": mid,
                    "split": split,
                    "n_fonts": len(stems),
                    "device": device,
                    "shard": shard,
                }
            ),
            flush=True,
        )
        if kind == "image":
            E.generate_image_method(mid, device, stems, overwrite)
        elif kind == "f1":
            E.generate_f1(device, stems, overwrite, mid=mid)
        elif kind == "f2":
            E.generate_f2(device, stems, overwrite, mid=mid)
        else:
            raise SystemExit(f"unknown kind {kind}")
        print(json.dumps({"done": mid, "split": split, "shard": shard}), flush=True)


def write_protocol(methods: dict, reuse: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proto = {
        "created_at": utc_now(),
        "ref8": REF8,
        "shot_sets": {str(k): REF8[:k] for k in SHOTS},
        "sampling": {"sampler": "dpmsolver++", "steps": 20, "cfg": 7.5, "seed": 3407},
        "fonts": {
            "test": json.loads((ROOT / "manifests/split_v3_228_16_16.json").read_text())["stems"]["test"],
            "train5": TRAIN5,
            "val5": VAL5,
        },
        "chars": E.STRATIFIED,
        "reuse_from": str(OLD / "preds"),
        "reuse": reuse,
        "methods": {
            mid: {
                "label": v["label"],
                "kind": v["kind"],
                "ckpt": str(v["ckpt"]),
                "style_k": v.get("style_k"),
                "tc_enabled": bool(v.get("tc_enabled")),
            }
            for mid, v in methods.items()
        },
    }
    (OUT / "PROTOCOL.json").write_text(json.dumps(proto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--mid")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--procs-per-gpu", type=int, default=4, help="parallel bs=1 workers per card")
    ap.add_argument("--shard", default="0/1", help="font shard i/n inside one mid")
    ap.add_argument("--link-only", action="store_true")
    ap.add_argument("--resume-pending", action="store_true", help="skip link+protocol; only unfinished mids")
    ap.add_argument("--job-json", type=str, default="")
    args = ap.parse_args()

    methods = build_methods()
    OUT.mkdir(parents=True, exist_ok=True)
    E.OUT = OUT
    E.METHODS.update(methods)

    if args.worker:
        assert args.mid and args.job_json
        payload = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
        run_one(args.mid, payload["splits"], args.device, args.overwrite, shard=args.shard)
        return 0

    if not args.resume_pending:
        reuse = reuse_test_preds(methods)
        write_protocol(methods, reuse)
        if args.link_only:
            print(json.dumps({"phase": "link_only_done", "reuse": reuse}), flush=True)
            return 0
    else:
        reuse = {}

    jobs = [j for j in job_list(methods) if args.overwrite or not mid_done(j["mid"])]
    jobs_dir = OUT / "job_specs"
    jobs_dir.mkdir(exist_ok=True)
    (OUT / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    ppg = max(1, int(args.procs_per_gpu))
    log_dir = OUT / "logs"
    log_dir.mkdir(exist_ok=True)
    print(
        json.dumps(
            {
                "phase": "dispatch",
                "mode": "one_mid_per_gpu_multi_proc",
                "n_jobs": len(jobs),
                "gpus": gpus,
                "procs_per_gpu": ppg,
            }
        ),
        flush=True,
    )

    # One model (mid) per GPU; on that GPU spawn ppg font-sharded bs=1 workers.
    pending = list(jobs)
    # gpu -> list[(Popen, log_fh, mid, shard)]
    slots: dict[str, list[tuple[subprocess.Popen, object, str, str]]] = {g: [] for g in gpus}
    mid_on_gpu: dict[str, str | None] = {g: None for g in gpus}
    rc = 0
    open_logs: list = []

    def reap(gpu: str) -> None:
        nonlocal rc
        alive = []
        for p, lf, mid, shard in slots[gpu]:
            r = p.poll()
            if r is None:
                alive.append((p, lf, mid, shard))
                continue
            if r != 0:
                rc = r
                print(json.dumps({"worker_fail": mid, "shard": shard, "gpu": gpu, "rc": r}), flush=True)
            try:
                lf.close()
            except Exception:
                pass
        slots[gpu] = alive
        if not alive:
            mid_on_gpu[gpu] = None

    def launch_mid(gpu: str, job: dict) -> None:
        mid = job["mid"]
        mid_on_gpu[gpu] = mid
        spec_p = jobs_dir / f"{mid}.json"
        spec_p.write_text(json.dumps(job, ensure_ascii=False) + "\n", encoding="utf-8")
        for si in range(ppg):
            shard = f"{si}/{ppg}"
            log_p = log_dir / f"{mid}.s{si}of{ppg}.log"
            cmd = [
                PY,
                str(Path(__file__).resolve()),
                "--worker",
                "--mid",
                mid,
                "--device",
                "cuda:0",
                "--job-json",
                str(spec_p),
                "--shard",
                shard,
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            lf = open(log_p, "w", encoding="utf-8")
            open_logs.append(lf)
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            slots[gpu].append((p, lf, mid, shard))
        print(json.dumps({"launch_mid": mid, "gpu": gpu, "procs": ppg}), flush=True)

    while pending or any(slots[g] for g in gpus):
        for gpu in gpus:
            reap(gpu)
            if slots[gpu]:
                continue
            if not pending:
                continue
            launch_mid(gpu, pending.pop(0))
        time.sleep(3)

    for lf in open_logs:
        try:
            lf.close()
        except Exception:
            pass

    print(json.dumps({"phase": "all_workers_done", "rc": rc}), flush=True)
    (OUT / "DONE.json").write_text(
        json.dumps({"finished_at": utc_now(), "rc": rc}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
