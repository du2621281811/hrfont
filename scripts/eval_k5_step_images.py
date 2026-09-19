#!/usr/bin/env python3
"""Render a fixed visual panel from each K5 checkpoint.

The K5 telemetry board is intentionally separate from this job: telemetry is
available for every update, while images are rendered only at selected
checkpoints on the same fixed VAL samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


EXTERNAL = Path("/root/projects/hrfont_k5_20260919_r2")
sys.path.insert(0, str(EXTERNAL))
sys.path.insert(0, str(EXTERNAL / "scripts"))

import k4_runtime as K4  # noqa: E402
import k5_runtime as K5  # noqa: E402
from scripts.k_components import detail_distances  # noqa: E402
from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP  # noqa: E402


ROOT = Path("/root/projects/hrfont")
OUT = ROOT / "reports/k5_step_training_20260920/image_data"
CHECKPOINTS = {
    "K5-A": Path("/root/data1/hrfont_k5_20260919/checkpoints/K5-A-V2-K1RECIPE-S3407"),
    "K5-B": Path("/root/data2/hrfont_k5_20260919/checkpoints/K5-B-V2-K1RECIPE-S3407"),
}


def monitor_jobs(max_per_script: int) -> list[dict]:
    manifest = json.loads(
        (EXTERNAL / "experiments/K4/v2_val192_monitor.json").read_text(encoding="utf-8")
    )["jobs"]
    selected = []
    counts: dict[str, int] = {}
    for job in sorted(manifest, key=lambda x: (x["script"], x["font"], x["cp"])):
        if job["shot"] != 4:
            continue
        if counts.get(job["script"], 0) >= max_per_script:
            continue
        counts[job["script"]] = counts.get(job["script"], 0) + 1
        selected.append(dict(job, k=job["shot"], refs=job["refs"]))
    return selected


@torch.no_grad()
def sample_batch(model, data, samples, scheduler, seeds: list[int]) -> np.ndarray:
    batch_size = len(seeds)
    device = data.device
    active = torch.zeros(batch_size, device=device, dtype=torch.bool)
    style, refs, query, keep, content, structure = data.conditions(
        samples, active, active, need_refs=model.use_tc
    )
    context = model.conditions(style, refs, query, keep, active)
    schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)

    def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = ((t - 1.0 / schedule.total_N) * 1000.0).expand(batch_size)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction, _ = model.denoise(
                x, t, style, content, structure, context, 1.0
            )
        return prediction.float()

    noise = torch.cat(
        [
            torch.randn(
                (1, 3, 96, 96),
                device=device,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            for seed in seeds
        ]
    )
    solver = DPM_Solver(
        model_fn=model_fn,
        noise_schedule=schedule,
        algorithm_type="dpmsolver++",
    )
    prediction = solver.sample(
        x=noise, steps=20, order=2, skip_type="time_uniform", method="multistep"
    )
    assert torch.isfinite(prediction).all()
    return (
        (prediction.float() / 2 + 0.5)
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
        * 255
    ).round().astype("uint8")


def sample_from_dataset(dataset, job: dict) -> dict:
    sample = dataset.sample(dataset.lookup[(job["font"], job["cp"])], job["refs"])
    assert sample["font_stem"] == job["font"]
    assert sample["char_cp"] == job["cp"]
    return sample


def evaluate(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda", rank)

    jobs = monitor_jobs(args.per_script)
    shard = jobs[rank::world]
    out = OUT / args.arm / f"step{args.step:05d}"
    out.mkdir(parents=True, exist_ok=True)

    model, model_args = K5.model_for(
        args.arm, device, out / f"provenance_rank{rank}", evaluation=True
    )
    checkpoint = CHECKPOINTS[args.arm] / f"global_step_{args.step}" / "ema.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model.load_train_state(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    data = K5.DataContext(model_args, device, model)
    dataset = K5.dataset(model_args, "val")
    scheduler = K5.T.build_ddpm_scheduler(model_args)

    rows = []
    start = time.time()
    for offset in range(0, len(shard), 4):
        chunk = shard[offset : offset + 4]
        samples = [sample_from_dataset(dataset, job) for job in chunk]
        predictions = sample_batch(model, data, K4.batch_to(samples, device), scheduler,
                                   [job["seed"] for job in chunk])
        batch = K4.batch_to(samples, device)
        prediction_tensor = (
            torch.from_numpy(predictions.copy()).permute(0, 3, 1, 2).float().to(device) / 255
        )
        details = detail_distances(
            prediction_tensor, batch["content_image"] / 2 + 0.5, batch["nonorm_target_image"]
        )
        for prediction, job, sample, detail_index in zip(
            predictions, chunk, samples, range(len(chunk))
        ):
            name = f'{job["font"]}__{job["cp"]}__k4.png'
            Image.fromarray(prediction).save(out / name)
            target = np.asarray(Image.open(sample["target_image_path"]).convert("L"), dtype="float32") / 255
            gray = np.asarray(Image.fromarray(prediction).convert("L"), dtype="float32") / 255
            rows.append(
                dict(
                    **job,
                    png=name,
                    prediction=str(out / name),
                    target=sample["target_image_path"],
                    content=str(dataset.root / "val" / "ContentImage" / f'{job["cp"]}.png'),
                    refs_paths=sample["ref_image_paths"],
                    l1=float(np.abs(gray - target).mean()),
                    ssim=float(structural_similarity(gray, target, data_range=1.0, win_size=7)),
                    **{
                        key: float(value[detail_index])
                        for key, value in details.items()
                    },
                )
            )

    rank_path = out / f"rank{rank}.json"
    rank_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        all_rows = []
        for other in range(world):
            all_rows.extend(
                json.loads((out / f"rank{other}.json").read_text(encoding="utf-8"))["rows"]
            )
        all_rows.sort(key=lambda row: (row["script"], row["font"], row["cp"]))
        (out / "metrics.json").write_text(
            json.dumps({"arm": args.arm, "step": args.step, "rows": all_rows}, ensure_ascii=False),
            encoding="utf-8",
        )
        (out / "DONE.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "arm": args.arm,
                    "step": args.step,
                    "rows": len(all_rows),
                    "seconds": time.time() - start,
                }
            ),
            encoding="utf-8",
        )
    if world > 1:
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--per-script", type=int, default=4)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.distributed.init_process_group("nccl")
    evaluate(args)


if __name__ == "__main__":
    main()
