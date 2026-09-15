#!/usr/bin/env python3
"""I0/I1 inference on G k1248 protocol (test16+train5+val5 × 47 × k∈{1,2,4,8}).

Matches reports/g_v0913_shot_k1248: Ref8=永和书风骨韵天地, DPM++20, CFG7.5, seed 3407.
Writes preds into the same tree as G columns: preds/I0_s{k}/… and preds/I1_s{k}/….
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

CODE = Path(__file__).resolve().parents[1]
ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(CODE))

from i34_runtime import T, model_for, DataContext, batch_to, atomic_json  # noqa: E402
from i_eval import sample  # noqa: E402

OUT = ROOT / "reports/g_v0913_shot_k1248"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
PROTO = json.loads((OUT / "PROTOCOL.json").read_text(encoding="utf-8"))
REF8 = list(PROTO["ref8"])
REF8_CPS = [f"u{ord(c):04X}" for c in REF8]
CHARS = list(PROTO["chars"])
SEED = int(PROTO["sampling"]["seed"])
SHOTS = (1, 2, 4, 8)

FONT_SPLITS: list[tuple[str, str]] = (
    [("test", f) for f in PROTO["fonts"]["test"]]
    + [("train", f) for f in PROTO["fonts"]["train5"]]
    + [("val", f) for f in PROTO["fonts"]["val5"]]
)


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def content_path(cp: str) -> Path:
    for sp in ("test", "val", "train"):
        p = DATA / sp / "ContentImage" / f"{cp}.png"
        if p.is_file():
            return p
    raise FileNotFoundError(cp)


def target_path(split: str, font: str, cp: str) -> Path:
    p = DATA / split / "TargetImage" / font / f"{font}+{cp}.png"
    if p.is_file():
        return p
    for sp in ("test", "train", "val"):
        q = DATA / sp / "TargetImage" / font / f"{font}+{cp}.png"
        if q.is_file():
            return q
    raise FileNotFoundError(f"{font}+{cp}")


def style_path(split: str, font: str, cp: str) -> Path:
    p = DATA / split / "StyleImage" / font / f"{font}+{cp}.png"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def pred_path(mid: str, split: str, font: str, cp: str) -> Path:
    return OUT / "preds" / mid / split / font / f"{split}__{font}__{cp}__s{SEED}.png"


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        assert im.size == (96, 96), (path, im.size)
        arr = np.array(im.convert("RGB"), copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1


def jobs_for(arm: str, shots: tuple[int, ...]) -> list[dict]:
    jobs = []
    for split, font in FONT_SPLITS:
        for ch in CHARS:
            cp = cp_of(ch)
            tgt = target_path(split, font, cp)
            for k in shots:
                mid = f"{arm}_s{k}"
                refs = REF8_CPS[:k]
                jobs.append(
                    dict(
                        mid=mid,
                        arm=arm,
                        k=k,
                        split=split,
                        font=font,
                        cp=cp,
                        ch=ch,
                        refs=refs,
                        target=str(tgt),
                        content=str(content_path(cp)),
                        styles=[str(style_path(split, font, r)) for r in refs],
                        out=str(pred_path(mid, split, font, cp)),
                    )
                )
    return jobs


def make_sample(job: dict) -> dict:
    tgt = load_rgb(Path(job["target"]))
    return dict(
        content_image=load_rgb(Path(job["content"])),
        target_image=tgt,
        nonorm_target_image=(tgt + 1) / 2,
        target_image_path=job["target"],
        font_stem=job["font"],
        char_cp=job["cp"],
        split=job["split"],
        ref_chars=list(job["refs"]),
        ref_image_paths=list(job["styles"]),
    )


def load_model(arm: str, device: torch.device, scratch: Path):
    if arm == "I0":
        model, args = model_for("I0", device, scratch / "load")
        with torch.no_grad():
            model.local_gain.zero_()
        for block in model.base.unet.up_blocks:
            if hasattr(block, "rsi_enabled"):
                block.rsi_enabled = False
        step = 10000
    elif arm == "I1":
        ckpt = ROOT / "runs/I1-V0915-S3407/global_step_10000"
        meta = json.loads((ckpt / "checkpoint.json").read_text())
        assert meta["complete"] and meta["arm"] == "I1"
        model, args = model_for("I1", device, scratch / "load")
        model.load_train_state(
            torch.load(ckpt / "ema.pth", map_location=device, weights_only=True)
        )
        step = int(meta.get("step") or 10000)
    else:
        raise ValueError(arm)
    return model, args, step


def run_arm(arm: str, shots: tuple[int, ...], overwrite: bool) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    out_scratch = OUT / "preds" / f"_i_k1248_scratch_{arm}"
    out_scratch.mkdir(parents=True, exist_ok=True)

    all_jobs = jobs_for(arm, shots)
    pending = [
        j
        for j in all_jobs
        if overwrite or not Path(j["out"]).is_file()
    ]
    if rank == 0:
        print(
            f"ARM {arm} total={len(all_jobs)} pending={len(pending)} shots={shots}",
            flush=True,
        )
    dist.barrier()
    if not pending:
        if rank == 0:
            print(f"ARM {arm} already complete", flush=True)
        return

    model, args, step = load_model(arm, device, out_scratch / f"rank{rank}")
    data = DataContext(args, device)
    scheduler = T.build_ddpm_scheduler(args)
    model.eval()
    gate = min(1.0, step / 1000)

    mine = pending[rank :: dist.get_world_size()]
    begin = time.time()
    done = 0
    for job in mine:
        path = Path(job["out"])
        path.parent.mkdir(parents=True, exist_ok=True)
        sample_dict = make_sample(job)
        b = batch_to([sample_dict], device)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = sample(model, data, b, scheduler, SEED, gate)
        Image.fromarray(pred).save(path)
        done += 1
        if done % 20 == 0 or done == len(mine):
            rate = done / max(time.time() - begin, 1e-6)
            print(
                f"INFER {arm} rank{rank} {done}/{len(mine)} {rate:.3f}/s last={path.name}",
                flush=True,
            )
    dist.barrier()
    if rank == 0:
        missing = [j["out"] for j in pending if not Path(j["out"]).is_file()]
        assert not missing, f"missing {len(missing)} e.g. {missing[:3]}"
        for k in shots:
            mid = f"{arm}_s{k}"
            n = sum(1 for _ in (OUT / "preds" / mid).rglob("*.png"))
            atomic_json(
                OUT / "preds" / mid / "DONE.json",
                dict(
                    status="completed",
                    mid=mid,
                    arm=arm,
                    k=k,
                    images=n,
                    expected=len(FONT_SPLITS) * len(CHARS),
                    seed=SEED,
                    cfg=7.5,
                    dpm_steps=20,
                    ref8="".join(REF8),
                    seconds=time.time() - begin,
                ),
            )
            print(f"DONE {mid} n={n}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("I0", "I1"), required=True)
    ap.add_argument("--shots", default="1,2,4,8")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    shots = tuple(int(x) for x in args.shots.split(",") if x.strip())
    assert all(k in SHOTS for k in shots)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
    assert dist.get_world_size() == 8
    try:
        run_arm(args.arm, shots, args.overwrite)
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
