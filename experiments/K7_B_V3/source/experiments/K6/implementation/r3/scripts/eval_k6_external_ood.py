#!/usr/bin/env python3
"""K6 external-font OOD inference.

The external font is used only for raw reference/target pixels and frozen
style-encoder features.  Content features and Delta donor candidates remain
the validated V2/K6 ones.  Results are written to a separate report tree and
must not be merged into K_TEST/K_VAL192 metrics.
"""
from __future__ import annotations

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
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

import k6_runtime as K6
import k5_runtime as K5
from scripts.k_components import detail_distances
from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP


PROTOCOL = "K-original-CFG1-DPM20-v1-external-font-OOD"
DATA_ROOT = Path("/root/data1/hrfont_dataset_v2_20260917/v2")
EXTERNAL_ROOT = Path("/root/data2/hrfont_k6_20260920/external_font_challenge_20260921/rendered")
MANIFEST = Path("/root/data2/hrfont_k6_20260920/external_font_challenge_20260921/external_font_manifest.json")
K6_TEST = Path("/root/projects/hrfont/experiments/K/K_TEST.json")
CONTENT_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_unit(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (96, 96):
            raise ValueError(f"expected 96x96 RGB: {path}")
        return torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float() / 255.0


def _edge(x):
    return np.diff(x, axis=0, prepend=x[:1]) ** 2 + np.diff(x, axis=1, prepend=x[:, :1]) ** 2


class ExternalEs:
    """Current frozen Es cache plus on-demand Es for the external font."""
    def __init__(self, current, encoder, device, root: Path, stems: set[str]):
        self.current = current
        self.encoder = encoder
        self.device = device
        self.root = root
        self.stems = stems
        self.cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def _encode(self, font: str, cp: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = (font, cp)
        if key not in self.cache:
            path = self.root / "StyleImage" / font / f"{font}+{cp}.png"
            x = ((read_unit(path) - 0.5) / 0.5).unsqueeze(0).to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=False):
                spatial, pooled, _ = self.encoder(x.float())
                pooled = F.normalize(pooled.float(), dim=1)
            self.cache[key] = (spatial[0].cpu(), pooled[0].cpu())
        return self.cache[key]

    def spatial_tensor(self, split: str, font: str, cp: str) -> torch.Tensor:
        if font in self.stems:
            return self._encode(font, cp)[0]
        return self.current.spatial_tensor(split, font, cp)

    def pooled_tensor(self, split: str, font: str, cp: str) -> torch.Tensor:
        if font in self.stems:
            return self._encode(font, cp)[1]
        return self.current.pooled_tensor(split, font, cp)


class ExternalFamilyPolicy:
    """No same-family exclusion exists for an unseen external family."""
    def __init__(self, current):
        self.current = current

    def exclude(self, font, valid):
        if font in self.current.groups:
            return self.current.exclude(font, valid)
        return valid.clone()

    def verify(self, font, selected):
        if font in self.current.groups:
            return self.current.verify(font, selected)


class ExternalEc:
    """Current donor Ec cache plus neutral content Ec for new Chinese cps."""
    def __init__(self, current, encoder, device, root: Path, external_cps: set[str]):
        self.current = current
        self.encoder = encoder
        self.device = device
        self.root = root
        self.external_cps = external_cps
        self.cache: dict[str, list[torch.Tensor]] = {}

    def _content_path(self, cp: str) -> Path:
        path = self.root / "ContentImage" / f"{cp}.png"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            ch = chr(int(cp[1:], 16))
            im = Image.new("RGB", (96, 96), (255, 255, 255))
            dr = ImageDraw.Draw(im)
            lo, hi, best = 8, 800, None
            while lo <= hi:
                fs = (lo + hi) // 2
                font = ImageFont.truetype(str(CONTENT_FONT), fs, index=0)
                box = dr.textbbox((0, 0), ch, font=font)
                w, h = box[2] - box[0], box[3] - box[1]
                if w <= 88 and h <= 88:
                    best = fs
                    lo = fs + 1
                else:
                    hi = fs - 1
            fs = best or 48
            font = ImageFont.truetype(str(CONTENT_FONT), fs, index=0)
            box = dr.textbbox((0, 0), ch, font=font)
            w, h = box[2] - box[0], box[3] - box[1]
            dr.text(((96 - w) // 2 - box[0], (96 - h) // 2 - box[1]), ch, font=font, fill=(0, 0, 0))
            im.save(path, optimize=True)
        return path

    @torch.no_grad()
    def _encode_content(self, cp: str) -> list[torch.Tensor]:
        if cp not in self.cache:
            x = ((read_unit(self._content_path(cp)) - 0.5) / 0.5).unsqueeze(0).to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=False):
                final, residuals = self.encoder(x.float())
            scales = list(residuals) + [final]
            self.cache[cp] = [x.detach().cpu() for x in scales]
        return self.cache[cp]

    def features_many(self, role, items):
        if role != "content":
            return self.current.features_many(role, items)
        unique = list(dict.fromkeys(items))
        result = {}
        normal = [item for item in unique if item[1] not in self.external_cps]
        if normal:
            result.update(self.current.features_many("content", normal))
        for _, cp in unique:
            if cp in self.external_cps:
                result[("", cp)] = self._encode_content(cp)
        return result


class ExternalDataContext(K5.DataContext):
    def __init__(self, args, device, model, root: Path, stems: set[str], external_cps: set[str]):
        super().__init__(args, device, model)
        self.es = ExternalEs(self.es, model.base.style_encoder, device, root, stems)
        self.ec = ExternalEc(self.ec, model.base.content_encoder, device, root, external_cps)
        self.library.family_policy = ExternalFamilyPolicy(self.library.family_policy)


class ExternalDataset:
    def __init__(self, manifest: dict, root: Path):
        self.root = root
        self.fonts = [x["stem"] for x in manifest["fonts"]]
        self.refs = list(manifest["refs"])
        self.targets = list(manifest["targets"])
        self.content_root = DATA_ROOT / "test" / "ContentImage"

    def sample(self, font: str, ch: str, refs: list[str]) -> dict:
        cp = f"u{ord(ch):04X}"
        target_path = self.root / "TargetImage" / font / f"{font}+{cp}.png"
        ref_paths = [self.root / "StyleImage" / font / f"{font}+{x}.png" for x in refs]
        target = read_unit(target_path)
        content_path = self.content_root / f"{cp}.png"
        content = read_unit(content_path)
        return dict(
            content_image=(content - 0.5) / 0.5,
            target_image=(target - 0.5) / 0.5,
            target_image_path=str(target_path),
            nonorm_target_image=target,
            font_stem=font,
            char_cp=cp,
            split="test",
            ref_chars=refs,
            ref_image_paths=[str(x) for x in ref_paths],
        )


@torch.no_grad()
def sample(model, data, samples, scheduler, seed):
    device = data.device
    active = torch.zeros(1, device=device, dtype=torch.bool)
    style, refs, query, keep, content, structure = data.conditions(
        samples, active, active, no_delta=False, official=False, need_refs=model.use_tc
    )
    ctx = model.conditions(style, refs, query, keep, active)
    schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)

    def model_fn(x, continuous_t):
        t = (continuous_t - 1.0 / schedule.total_N) * 1000.0
        with torch.autocast("cuda", dtype=torch.float16):
            pred, _ = model.denoise(x, t, style, content, structure, ctx, 1.0)
        return pred.float()

    solver = DPM_Solver(model_fn=model_fn, noise_schedule=schedule,
                        algorithm_type="dpmsolver++")
    noise = torch.randn((1, 3, 96, 96), device=device,
                        generator=torch.Generator(device=device).manual_seed(seed))
    pred = solver.sample(x=noise, steps=20, order=2, skip_type="time_uniform", method="multistep")
    if not torch.isfinite(pred).all():
        raise FloatingPointError("non-finite external OOD prediction")
    return ((pred[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")


def make_jobs(manifest: dict, smoke: bool) -> list[dict]:
    jobs = []
    n = 0
    for font in [x["stem"] for x in manifest["fonts"]]:
        for ch in manifest["targets"]:
            for k in (1, 2, 4, 8):
                cp = f"u{ord(ch):04X}"
                refs = [f"u{ord(x):04X}" for x in manifest["refs"][:k]]
                seed = 3407 + n * 7919
                jobs.append(dict(font=font, cp=cp, char=ch, script="external_cn",
                                 split="external", k=k, refs=refs, seed=seed))
                n += 1
    if smoke:
        # One 8-shot character per external family, still exercises all arms.
        jobs = [j for j in jobs if j["k"] == 8 and j["cp"] == f"u{ord(manifest['targets'][0]):04X}"]
    return jobs


def evaluate(model, data, ds, manifest, args, scheduler, out, checkpoint, jobs):
    rank, world = dist.get_rank(), dist.get_world_size()
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        K6.atomic_json(out / "protocol.json", dict(
            protocol=PROTOCOL, manifest=str(MANIFEST), manifest_sha256=sha256(MANIFEST),
            font_sources=manifest["fonts"], refs=manifest["refs"], targets=manifest["targets"],
            shot_schedule=[1, 2, 4, 8], cfg=1.0, steps=20, order=2, seed=3407,
            weights="EMA", checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint / "ema.pth"),
            arm=model.arm, smoke=args.smoke,
        ))
    rows = []
    begin = time.time()
    for job in jobs[rank::world]:
        sample_row = ds.sample(job["font"], chr(int(job["cp"][1:], 16)), job["refs"])
        batch = K6.batch_to([sample_row], data.device)
        pred = sample(model, data, batch, scheduler, job["seed"])
        name = f"{job['font']}__{job['cp']}__k{job['k']}.png"
        Image.fromarray(pred).save(out / name)
        gt = np.asarray(Image.open(sample_row["target_image_path"]).convert("L"), dtype="float32") / 255
        gray = np.asarray(Image.fromarray(pred).convert("L"), dtype="float32") / 255
        tensor = torch.from_numpy(pred.copy()).permute(2, 0, 1).float()[None].to(data.device) / 255
        distances = detail_distances(tensor, batch["content_image"] / 2 + 0.5,
                                     batch["nonorm_target_image"])
        change = detail_distances(batch["content_image"] / 2 + 0.5,
                                  batch["content_image"] / 2 + 0.5,
                                  batch["nonorm_target_image"])
        rows.append(dict(**job, png=name, target=sample_row["target_image_path"],
                         refs_paths=sample_row["ref_image_paths"],
                         content=str(ds.content_root / f"{job['cp']}.png"),
                         l1=float(np.abs(gray - gt).mean()),
                         ssim=float(structural_similarity(gray, gt, data_range=1.0, win_size=7)),
                         edge=float(np.abs(_edge(gray) - _edge(gt)).mean()),
                         ink_fraction=float((gray < 0.95).mean()),
                         change_magnitude=float(change["D_change"][0]),
                         **{k: float(v[0]) for k, v in distances.items()}))
    K6.atomic_json(out / f"rank{rank}.json", dict(rows=rows, seconds=time.time() - begin))
    dist.barrier()
    if rank == 0:
        all_rows = sum([json.loads((out / f"rank{i}.json").read_text())["rows"] for i in range(world)], [])
        assert len(all_rows) == len(jobs)
        keys = ["l1", "ssim", "edge", "D_region", "D_change", "D_add", "D_remove", "D_high"]
        summary = {k: float(np.mean([r[k] for r in all_rows])) for k in keys}
        by_font = {f: {k: float(np.mean([r[k] for r in all_rows if r["font"] == f])) for k in keys}
                   for f in sorted({r["font"] for r in all_rows})}
        by_k = {str(k): {m: float(np.mean([r[m] for r in all_rows if r["k"] == k])) for m in keys}
                for k in sorted({r["k"] for r in all_rows})}
        K6.atomic_json(out / "metrics.json", dict(summary=summary, stratified={"font": by_font, "k": by_k}, rows=all_rows))
        K6.atomic_json(out / "DONE.json", dict(status="completed", images=len(all_rows),
                                                seconds=time.time() - begin, smoke=args.smoke,
                                                external_only=True))
    dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["K6-A", "K6-B-RSI-NOOFFSETLOSS", "K6-C-RSI-AUX-NOOFFSETLOSS"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
    device = torch.device("cuda", rank)
    base_arm = "K6-B-RSI-NOOFFSETLOSS" if args.arm.startswith("K6-C") else args.arm
    model, runtime_args = K6.model_for(base_arm, device, args.out / f"provenance_rank{rank}", evaluation=True)
    model.load_train_state(torch.load(args.checkpoint / "ema.pth", map_location=device, weights_only=True))
    model.arm = args.arm
    manifest = json.loads(MANIFEST.read_text())
    ds = ExternalDataset(manifest, EXTERNAL_ROOT)
    data = ExternalDataContext(runtime_args, device, model, EXTERNAL_ROOT,
                               {x["stem"] for x in manifest["fonts"]}, set())
    jobs = make_jobs(manifest, args.smoke)
    evaluate(model, data, ds, manifest, args, K6.T.build_ddpm_scheduler(runtime_args),
             args.out, args.checkpoint, jobs)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
