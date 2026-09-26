"""K6 normal-protocol inference and metrics.

This is deliberately a K6-runtime evaluator rather than a training utility:
it loads the completed EMA checkpoint, freezes the existing K_TEST/K_VAL192
jobs, and emits the same row-level metrics as the K0/K1 evaluator.
"""
import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from skimage.metrics import structural_similarity

from k6_runtime import CODE, ROOT, T, DataContext, atomic_json, batch_to, dataset, model_for, sha256_file
from scripts.k_components import detail_distances
from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP


PROTOCOL = "K-original-CFG1-DPM20-v1"
STEP = 10000


def _edge(x):
    return np.diff(x, axis=0, prepend=x[:1]) ** 2 + np.diff(x, axis=1, prepend=x[:, :1]) ** 2


@torch.no_grad()
def sample(model, data, samples, scheduler, seed):
    device = data.device
    active = torch.zeros(1, device=device, dtype=torch.bool)
    style, refs, query, keep, content, structure = data.conditions(
        samples,
        active,
        active,
        no_delta=False,
        official=False,
        need_refs=model.use_tc,
    )
    ctx = model.conditions(style, refs, query, keep, active)
    schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)

    def model_fn(x, continuous_t):
        t = (continuous_t - 1.0 / schedule.total_N) * 1000.0
        with torch.autocast("cuda", dtype=torch.float16):
            pred, _ = model.denoise(x, t, style, content, structure, ctx, 1.0)
        return pred.float()

    solver = DPM_Solver(model_fn=model_fn, noise_schedule=schedule, algorithm_type="dpmsolver++")
    noise = torch.randn(
        (1, 3, 96, 96),
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    pred = solver.sample(x=noise, steps=20, order=2, skip_type="time_uniform", method="multistep")
    assert torch.isfinite(pred).all()
    return ((pred[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")


def evaluate(model, data, args, scheduler, manifest_path, out, checkpoint, smoke, eval_step):
    rank, world = dist.get_rank(), dist.get_world_size()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text())
    jobs = manifest["jobs"]
    if smoke:
        jobs = [next(j for j in jobs if j["script"] == g) for g in ("western", "kana", "bopomofo")]

    ds = dataset(args, manifest["split"])
    # The frozen manifest's `index` is provenance from the evaluator that
    # created it, while K6's v2 FontDataset enumerates physical files after
    # its own clean-pair filter.  Resolve by the semantic (font, cp) key so
    # the manifest, not an incidental list position, defines the sample.
    by_key = {}
    for index, path in enumerate(ds.target_images):
        path = Path(path)
        font = path.parent.name
        cp = path.stem[len(font) + 1 :]
        by_key[(font, cp)] = index
    model.eval()
    begin = time.time()
    rows = []
    if rank == 0:
        atomic_json(
            out / "protocol.json",
            dict(
                protocol=manifest["protocol"],
                split=manifest["split"],
                cfg=manifest["cfg"],
                steps=manifest["steps"],
                order=manifest["order"],
                seed=manifest["seed"],
                jobs=jobs,
                arm=model.arm,
                step=eval_step,
                weights="EMA",
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256_file(checkpoint / "ema.pth"),
                manifest_sha256=sha256_file(manifest_path),
                code_commit=json.loads((CODE / "K6_CODE_IDENTITY.json").read_text()).get("commit"),
                smoke=smoke,
            ),
        )

    for job in jobs[rank::world]:
        dataset_index = by_key[(job["font"], job["cp"])]
        sample_row = ds[dataset_index]
        assert (sample_row["font_stem"], sample_row["char_cp"]) == (job["font"], job["cp"])
        sample_row["ref_chars"] = job["refs"]
        sample_row["ref_image_paths"] = [str(ds.style_by_font_char[job["font"]][c]) for c in job["refs"]]
        batch = batch_to([sample_row], data.device)
        pred = sample(model, data, batch, scheduler, job["seed"])
        name = f"{job['font']}__{job['cp']}__k{job['k']}.png"
        Image.fromarray(pred).save(out / name)

        gt = np.asarray(Image.open(sample_row["target_image_path"]).convert("L"), dtype="float32") / 255
        gray = np.asarray(Image.fromarray(pred).convert("L"), dtype="float32") / 255
        tensor = torch.from_numpy(pred.copy()).permute(2, 0, 1).float()[None].to(data.device) / 255
        distances = detail_distances(tensor, batch["content_image"] / 2 + 0.5, batch["nonorm_target_image"])
        change = detail_distances(
            batch["content_image"] / 2 + 0.5,
            batch["content_image"] / 2 + 0.5,
            batch["nonorm_target_image"],
        )
        rows.append(
            dict(
                **job,
                png=name,
                target=sample_row["target_image_path"],
                refs_paths=sample_row["ref_image_paths"],
                content=str(ds.root / ds.phase / "ContentImage" / f"{job['cp']}.png"),
                l1=float(np.abs(gray - gt).mean()),
                ssim=float(structural_similarity(gray, gt, data_range=1.0, win_size=7)),
                edge=float(np.abs(_edge(gray) - _edge(gt)).mean()),
                ink_fraction=float((gray < 0.95).mean()),
                change_magnitude=float(change["D_change"][0]),
                **{k: float(v[0]) for k, v in distances.items()},
            )
        )
        if len(rows) % 20 == 0:
            print("K6_INFER", model.arm, rank, len(rows), len(jobs[rank::world]), flush=True)

    atomic_json(out / f"rank{rank}.json", dict(rows=rows, seconds=time.time() - begin))
    dist.barrier()
    if rank == 0:
        all_rows = sum(
            [json.loads((out / f"rank{i}.json").read_text())["rows"] for i in range(world)], []
        )
        assert len(all_rows) == len(jobs)
        assert len({r["png"] for r in all_rows}) == len(jobs)
        keys = ["l1", "ssim", "edge", "D_region", "D_change", "D_add", "D_remove", "D_high"]
        summary = {}
        for group in ["all", "western", "kana", "bopomofo"]:
            selected = [r for r in all_rows if group == "all" or r["script"] == group]
            summary[group] = {k: float(np.mean([r[k] for r in selected])) for k in keys}
        stratified = {}
        for field in ["font", "k"]:
            stratified[field] = {
                str(value): {k: float(np.mean([r[k] for r in all_rows if r[field] == value])) for k in keys}
                for value in sorted({r[field] for r in all_rows})
            }
        ordered = sorted(all_rows, key=lambda r: r["change_magnitude"])
        stratified["change_quartile"] = {
            str(i + 1): {k: float(np.mean([r[k] for r in chunk])) for k in keys}
            for i, chunk in enumerate(np.array_split(ordered, 4))
            if len(chunk)
        }
        atomic_json(out / "metrics.json", dict(summary=summary, stratified=stratified, rows=all_rows))
        atomic_json(
            out / "DONE.json",
            dict(status="completed", images=len(all_rows), step=eval_step, seconds=time.time() - begin, smoke=smoke),
        )
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["K6-A", "K6-B", "K6-B-RSI-NOOFFSETLOSS"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    device = torch.device("cuda", rank)
    model, runtime_args = model_for(args.arm, device, args.out / f"provenance_rank{rank}", evaluation=True)
    model.load_train_state(torch.load(args.checkpoint / "ema.pth", map_location=device, weights_only=True))
    data = DataContext(runtime_args, device, model)
    evaluate(model, data, runtime_args, T.build_ddpm_scheduler(runtime_args), args.manifest, args.out, args.checkpoint, args.smoke, args.step)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
