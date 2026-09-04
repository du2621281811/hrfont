#!/usr/bin/env python3
"""Train HR-Font E2 Stage A with online feature-mix delta."""
from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.hrfont_delta_v2 import DeltaConfig, compute_alpha, feature_delta

from configs.fontdiffuser import get_parser
from dataset.collate_fn import CollateFN
from dataset.font_dataset import FontDataset
from src import (ContentPerceptualLoss, FontDiffuserModel, build_content_encoder,
                 build_ddpm_scheduler, build_style_encoder, build_unet)
from utils import normalize_mean_std, reNormalize_img, save_args_to_yaml, x0_from_epsilon


def get_args():
    args = get_parser().parse_args()
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)
    return args


def _load_image(path: str, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (resolution, resolution):
        raise ValueError(f"expected native {resolution}x{resolution}: {path} ({image.size})")
    return transforms.functional.normalize(transforms.functional.to_tensor(image), [0.5], [0.5])


def _load_unique(paths: list[str], resolution: int, device) -> tuple[list[str], torch.Tensor]:
    unique = list(dict.fromkeys(paths))
    return unique, torch.stack([_load_image(p, resolution) for p in unique]).to(device)


@torch.no_grad()
def _style_conditions(model, ref_paths: list[list[str]], resolution: int, device):
    paths, images = _load_unique([p for refs in ref_paths for p in refs], resolution, device)
    _, pooled, _ = model.style_encoder(images)
    pooled = F.normalize(pooled.float(), dim=1)
    by_path = {path: pooled[i] for i, path in enumerate(paths)}
    per_ref = [torch.stack([by_path[p] for p in refs]) for refs in ref_paths]
    return torch.stack([rows.mean(0) for rows in per_ref]), per_ref


class _FeatureLookup(nn.Module):
    """Expose one already-batched Ec pass through feature_delta's encoder contract."""

    def __init__(self, final: torch.Tensor, residuals: list[torch.Tensor]):
        super().__init__()
        self.final = final
        self.residuals = residuals

    def forward(self, index):
        i = int(index.reshape(-1)[0])
        return self.final[i:i + 1], [x[i:i + 1] for x in self.residuals]


def _cache_feature(cache: dict, font: str, cp: str) -> torch.Tensor:
    key = (font, cp)
    if key not in cache:
        raise KeyError(f"Es cache missing {key}")
    return cache[key].float()


@torch.no_grad()
def _structure_features(model, dataset, cache, samples, ref_rows, cfg, delta_draw, device):
    fonts = samples["font_stem"]
    chars = samples["char_cp"]
    ref_chars = samples["ref_chars"]
    ref_paths = samples["ref_image_paths"]
    library = sorted(dataset.target_by_font_char)

    plans = []
    all_paths = []
    if cfg.rsi_source == "official":
        for refs in ref_paths:
            plans.append((refs[0],))
            all_paths.append(refs[0])
    elif cfg.delta_enabled:
        alpha_cfg = DeltaConfig(tau=cfg.delta_tau, eps_alpha=cfg.delta_eps_alpha,
                                k_max=cfg.delta_k_max, mode=cfg.delta_mode,
                                rng_seed=cfg.seed)
        for font, cp, rchars, query in zip(fonts, chars, ref_chars, ref_rows):
            prototypes = torch.stack([
                torch.stack([_cache_feature(cache, candidate, rcp) for rcp in rchars])
                for candidate in library
            ]).to(device)
            exclude = library.index(font) if font in library else None
            indices, weights, _ = compute_alpha(query, prototypes, exclude, alpha_cfg)
            neighbor_paths = [dataset.target_path(library[i], cp) for i in indices]
            # B0 = Noto ContentImage (same render as the Identity/MCA input).
            neutral_path = dataset.content_path(cp)
            plans.append((neighbor_paths, weights, neutral_path))
            all_paths.extend(neighbor_paths)
            all_paths.append(neutral_path)
    else:
        return None

    paths, images = _load_unique(all_paths, cfg.resolution, device)
    final, residuals = model.content_encoder(images)
    residuals = list(residuals)
    lookup = _FeatureLookup(final, residuals)
    path_index = {path: i for i, path in enumerate(paths)}
    keys = [torch.tensor([[i]], device=device) for i in range(len(paths))]

    if cfg.rsi_source == "official":
        return [torch.cat([x[path_index[p]:path_index[p] + 1] for (p,) in plans])
                for x in residuals + [final]]

    example = residuals + [final]
    sample_features = []
    for dropped, (neighbors, weights, neutral) in zip(delta_draw, plans):
        if bool(dropped) or not neighbors:
            sample_features.append([torch.zeros_like(x[:1]) for x in example])
            continue
        sample_features.append(feature_delta(
            lookup,
            [keys[path_index[p]] for p in neighbors],
            weights,
            keys[path_index[neutral]],
        ))
    return [torch.cat([row[scale] for row in sample_features]) for scale in range(len(example))]


def _save_checkpoint(model, directory: Path, optimizer, scheduler, step: int):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.unet.state_dict(), directory / "unet.pth")
    torch.save(model.style_encoder.state_dict(), directory / "style_encoder.pth")
    torch.save(model.content_encoder.state_dict(), directory / "content_encoder.pth")
    torch.save({}, directory / "delta_layers.pth")  # Direct Ec features need no adapter.
    torch.save({"step": step, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}, directory / "trainer_state.pt")


def _load_checkpoint(model, directory: Path, optimizer=None, scheduler=None) -> int:
    model.unet.load_state_dict(torch.load(directory / "unet.pth", map_location="cpu"))
    model.style_encoder.load_state_dict(torch.load(directory / "style_encoder.pth", map_location="cpu"))
    model.content_encoder.load_state_dict(torch.load(directory / "content_encoder.pth", map_location="cpu"))
    state_path = directory / "trainer_state.pt"
    if not state_path.is_file():
        return 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("step", state.get("global_step", 0)))


def _record_config(args):
    out = Path(args.output_dir)
    save_args_to_yaml(args, out / f"{args.experience_name}_config.yaml")
    if args.config_path:
        copied = out / "input_config.yaml"
        shutil.copy2(args.config_path, copied)
        digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    else:
        digest = "none"
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout.strip()
    (out / "run_note.txt").write_text(f"git_sha={sha} config_sha256={digest}\n", encoding="utf-8")


def main():
    args = get_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.report_to,
                              project_dir=str(Path(args.output_dir) / args.logging_dir))
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = FontDiffuserModel(unet=build_unet(args),
                              style_encoder=build_style_encoder(args),
                              content_encoder=build_content_encoder(args))
    if args.phase_1_ckpt_dir:
        _load_checkpoint(model, Path(args.phase_1_ckpt_dir))
    if args.freeze_encoders:
        model.style_encoder.requires_grad_(False).eval()
        model.content_encoder.requires_grad_(False).eval()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon)
    scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                              num_warmup_steps=args.lr_warmup_steps,
                              num_training_steps=args.max_train_steps)
    noise_scheduler = build_ddpm_scheduler(args)
    perceptual_loss = ContentPerceptualLoss()

    def native(image):
        if image.size != (args.resolution, args.resolution):
            raise ValueError(f"expected native {args.resolution}x{args.resolution}, got {image.size}")
        return image

    normalized = transforms.Compose([native, transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])
    dataset = FontDataset(args, "train", [normalized, normalized, normalized], scr=False)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                         batch_size=args.train_batch_size,
                                         collate_fn=CollateFN())
    cache = torch.load(args.es_cache_path, map_location="cpu", weights_only=False)
    if isinstance(cache, dict) and "features" in cache:
        cache = cache["features"]

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    raw = accelerator.unwrap_model(model)
    global_step = _load_checkpoint(raw, Path(args.resume_from), optimizer, scheduler) \
        if args.resume_from else 0
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        _record_config(args)

    progress = tqdm(range(args.max_train_steps), initial=global_step,
                    disable=not accelerator.is_local_main_process)
    epochs = math.ceil(args.max_train_steps / max(1, len(loader)))
    for _ in range(epochs):
        for samples in loader:
            if global_step >= args.max_train_steps:
                break
            model.train()
            raw.style_encoder.eval()
            raw.content_encoder.eval()
            target = samples["target_image"]
            content = samples["content_image"].clone()
            nonorm_target = samples["nonorm_target_image"]
            bsz = target.shape[0]

            with accelerator.accumulate(model):
                with torch.no_grad():
                    style, ref_rows = _style_conditions(raw, samples["ref_image_paths"],
                                                        args.resolution, target.device)
                    delta_draw = torch.rand(bsz, device=target.device) < args.delta_drop
                    structure = _structure_features(raw, dataset, cache, samples, ref_rows,
                                                    args, delta_draw, target.device)
                    cfg_mask = torch.rand(bsz, device=target.device) < args.drop_prob
                    content[cfg_mask] = 0
                    style[cfg_mask] = 0

                noise = torch.randn_like(target)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps,
                                          (bsz,), device=target.device).long()
                noisy = noise_scheduler.add_noise(target, noise, timesteps)
                noise_pred, offset_sum = model(
                    x_t=noisy, timesteps=timesteps, content_images=content,
                    style_features=style, structure_features=structure,
                    content_encoder_downsample_size=args.content_encoder_downsample_size)
                diffusion = F.mse_loss(noise_pred.float(), noise.float())
                x0_norm = x0_from_epsilon(noise_scheduler, noise_pred, noisy, timesteps)
                x0 = reNormalize_img(x0_norm)
                perceptual = perceptual_loss.calculate_loss(
                    normalize_mean_std(x0), normalize_mean_std(nonorm_target), target.device)
                loss = diffusion + args.perceptual_coefficient * perceptual + \
                    args.offset_coefficient * (offset_sum / 2)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)
                if accelerator.is_main_process and global_step % args.ckpt_interval == 0:
                    _save_checkpoint(raw, Path(args.output_dir) / f"global_step_{global_step}",
                                     optimizer, scheduler, global_step)
                    _save_checkpoint(raw, Path(args.output_dir) / "last_state",
                                     optimizer, scheduler, global_step)
            if global_step % args.log_interval == 0:
                progress.set_postfix(loss=float(loss.detach()), lr=scheduler.get_last_lr()[0])
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        _save_checkpoint(raw, Path(args.output_dir) / "last_state", optimizer, scheduler, global_step)
        (Path(args.output_dir) / "DONE.json").write_text(
            f'{{"global_step": {global_step}, "status": "completed"}}\n', encoding="utf-8")
    accelerator.end_training()


if __name__ == "__main__":
    main()
