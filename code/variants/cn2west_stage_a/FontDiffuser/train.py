#!/usr/bin/env python3
"""Train HR-Font Stage A with cache-only Es/Ec and 9-token style."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from scripts.hrfont_delta_v2 import DeltaConfig, compute_alpha
from scripts.hrfont_feature_cache import EcCache, EsCache, mix_cached_delta

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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ban_encoder_forward(model):
    def hook(_module, _inputs):
        raise RuntimeError("cache-only: Es/Ec forward is forbidden")
    model.style_encoder.register_forward_pre_hook(hook)
    model.content_encoder.register_forward_pre_hook(hook)


def _verify_caches(args):
    es_dir = Path(args.es_cache_path)
    ec_dir = Path(args.ec_cache_path)
    es_man = json.loads((es_dir / "manifest.json").read_text(encoding="utf-8"))
    ec_man = json.loads((ec_dir / "manifest.json").read_text(encoding="utf-8"))
    init = Path(args.phase_1_ckpt_dir)
    es_sha = _sha256_file(init / "style_encoder.pth")
    ec_sha = _sha256_file(init / "content_encoder.pth")
    if es_man.get("es_checkpoint_sha256") != es_sha:
        raise RuntimeError("D-A1: Es cache SHA does not match init style_encoder")
    if ec_man.get("ec_checkpoint_sha256") != ec_sha:
        raise RuntimeError("D-A1: Ec cache SHA does not match init content_encoder")
    es_prog = json.loads((es_dir / "progress.json").read_text(encoding="utf-8"))
    ec_prog = json.loads((ec_dir / "progress.json").read_text(encoding="utf-8"))
    if es_prog.get("done") != es_prog.get("total") or ec_prog.get("done") != ec_prog.get("total"):
        raise RuntimeError("encoder cache is incomplete")
    return es_sha, ec_sha


class _LibraryEs:
    """Keep train228 pooled Es in RAM so alpha does not mmap 228 fonts per sample."""

    def __init__(self, es_cache: EsCache, train_fonts: list[str], style_chars: list[str]):
        self.fonts = train_fonts
        self.font_index = {font: i for i, font in enumerate(train_fonts)}
        self.char_index = {cp: i for i, cp in enumerate(style_chars)}
        rows = []
        for font in train_fonts:
            idx = [es_cache._row("train", font, cp) for cp in style_chars]
            rows.append(np.array(es_cache.pooled[idx], copy=True))
        self.table = torch.from_numpy(np.stack(rows)).float()  # [228, 338, D]

    def prototypes(self, rchars: list[str], device) -> torch.Tensor:
        idx = [self.char_index[cp] for cp in rchars]
        return self.table[:, idx, :].to(device)


def _style_chars_from_cache(es_cache: EsCache) -> list[str]:
    first_font = None
    ordered = []
    for key in es_cache.table.keys:
        _, split, font, cp = key.split("|")
        if split != "train":
            continue
        if first_font is None:
            first_font = font
        if font != first_font:
            break
        ordered.append(cp)
    if len(ordered) != 338:
        raise RuntimeError(f"expected 338 style chars in Es cache, got {len(ordered)}")
    return ordered


def _style_conditions(es_cache: EsCache, samples, device):
    maps, queries = [], []
    for split, font, refs in zip(samples["split"], samples["font_stem"], samples["ref_chars"]):
        spatial = torch.stack([es_cache.spatial_tensor(split, font, cp) for cp in refs], 0)
        maps.append(spatial.mean(0))
        queries.append(torch.stack([
            F.normalize(es_cache.pooled_tensor(split, font, cp), dim=0) for cp in refs
        ]))
    return torch.stack(maps).to(device), queries


def _write_heartbeat(output_dir: Path, **payload):
    payload = {"ts": __import__("datetime").datetime.now().isoformat(timespec="seconds"), **payload}
    (output_dir / "heartbeat.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _content_features(ec_cache: EcCache, samples, cfg_mask, device):
    rows = []
    for dropped, cp in zip(cfg_mask, samples["char_cp"]):
        feats = [tensor.to(device) for tensor in ec_cache.features("content", "", cp)]
        if bool(dropped):
            feats = [torch.zeros_like(item) for item in feats]
        rows.append(feats)
    return [torch.cat([row[scale] for row in rows]) for scale in range(len(rows[0]))]


def _structure_features(es_cache, ec_cache, library: _LibraryEs, samples, queries, cfg, delta_draw, device):
    fonts = samples["font_stem"]
    chars = samples["char_cp"]
    ref_chars = samples["ref_chars"]
    if cfg.rsi_source == "official":
        rows = []
        for font, refs in zip(fonts, ref_chars):
            rows.append([tensor.to(device) for tensor in ec_cache.features("style", font, refs[0])])
        return [torch.cat([row[scale] for row in rows]) for scale in range(len(rows[0]))]
    if not cfg.delta_enabled:
        return None
    alpha_cfg = DeltaConfig(tau=cfg.delta_tau, eps_alpha=cfg.delta_eps_alpha,
                            k_max=cfg.delta_k_max, k_top=cfg.delta_k_top,
                            mode=cfg.delta_mode, rng_seed=cfg.seed)
    sample_features = []
    example = None
    train_fonts = library.fonts
    for dropped, font, cp, rchars, query in zip(delta_draw, fonts, chars, ref_chars, queries):
        prototypes = library.prototypes(rchars, device)
        exclude = library.font_index.get(font)
        indices, weights, _ = compute_alpha(query.to(device), prototypes, exclude, alpha_cfg)
        if len(indices) == 0:
            raise RuntimeError(f"empty top-K neighborhood font={font} char={cp}")
        neighbors = [ec_cache.features("target", train_fonts[i], cp) for i in indices]
        neighbors = [[tensor.to(device) for tensor in feats] for feats in neighbors]
        neutral = [tensor.to(device) for tensor in ec_cache.features("content", "", cp)]
        example = example or neutral
        if bool(dropped):
            sample_features.append([torch.zeros_like(item) for item in example])
            continue
        sample_features.append(mix_cached_delta(neighbors, weights, neutral))
    return [torch.cat([row[scale] for row in sample_features]) for scale in range(len(example))]


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng(state: dict):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(model, directory: Path, optimizer, scheduler, step: int, scaler=None):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.unet.state_dict(), directory / "unet.pth")
    torch.save(model.style_encoder.state_dict(), directory / "style_encoder.pth")
    torch.save(model.content_encoder.state_dict(), directory / "content_encoder.pth")
    torch.save({}, directory / "delta_layers.pth")
    payload = {"step": step, "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "rng": _rng_state()}
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, directory / "trainer_state.pt")


def _load_checkpoint(model, directory: Path, optimizer=None, scheduler=None, scaler=None,
                     restore_rng: bool = False) -> int:
    model.unet.load_state_dict(torch.load(directory / "unet.pth", map_location="cpu"))
    model.style_encoder.load_state_dict(torch.load(directory / "style_encoder.pth", map_location="cpu"))
    model.content_encoder.load_state_dict(torch.load(directory / "content_encoder.pth", map_location="cpu"))
    state_path = directory / "trainer_state.pt"
    if not state_path.is_file():
        return 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        _set_rng(state.get("rng") or {})
    return int(state.get("step", state.get("global_step", 0)))


def _record_config(args, es_sha: str, ec_sha: str):
    out = Path(args.output_dir)
    save_args_to_yaml(args, out / f"{args.experience_name}_config.yaml")
    digest = "none"
    if args.config_path:
        src = Path(args.config_path)
        if not src.is_file():
            src = REPO / args.config_path
        copied = out / "input_config.yaml"
        shutil.copy2(src, copied)
        digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout.strip()
    (out / "run_note.txt").write_text(
        f"git_sha={sha} config_sha256={digest} es_sha={es_sha} ec_sha={ec_sha}\n",
        encoding="utf-8")


def _forward_batch(model, noise_scheduler, perceptual_loss, args, batch, style, structure,
                   content_feats, train: bool):
    target = batch["target_image"]
    nonorm_target = batch["nonorm_target_image"]
    bsz = target.shape[0]
    noise = torch.randn_like(target)
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target.device).long()
    noisy = noise_scheduler.add_noise(target, noise, timesteps)
    noise_pred, offset_sum = model(
        x_t=noisy, timesteps=timesteps, content_images=batch["content_image"],
        style_features=style, structure_features=structure, content_features=content_feats,
        content_encoder_downsample_size=args.content_encoder_downsample_size)
    diffusion = F.mse_loss(noise_pred.float(), noise.float())
    if not train:
        return diffusion, offset_sum.detach()
    x0_norm = x0_from_epsilon(noise_scheduler, noise_pred, noisy, timesteps)
    x0 = reNormalize_img(x0_norm)
    perceptual = perceptual_loss.calculate_loss(
        normalize_mean_std(x0), normalize_mean_std(nonorm_target), target.device)
    loss = diffusion + args.perceptual_coefficient * perceptual + args.offset_coefficient * (offset_sum / 2)
    return loss, diffusion.detach()


@torch.no_grad()
def _run_val(raw, es_cache, ec_cache, library, val_loader, noise_scheduler, args, device):
    raw.eval()
    losses = []
    for samples in val_loader:
        target = samples["target_image"].to(device)
        samples = {**samples, "target_image": target}
        style, queries = _style_conditions(es_cache, samples, device)
        delta_draw = torch.zeros(target.shape[0], dtype=torch.bool, device=device)
        cfg_mask = torch.zeros_like(delta_draw)
        structure = _structure_features(es_cache, ec_cache, library, samples, queries, args,
                                        delta_draw, device)
        content_feats = _content_features(ec_cache, samples, cfg_mask, device)
        dummy = samples
        dummy["target_image"] = target
        dummy["content_image"] = samples["content_image"].to(device)
        dummy["nonorm_target_image"] = samples["nonorm_target_image"].to(device)
        loss, _ = _forward_batch(raw, noise_scheduler, None, args, dummy, style, structure,
                                 content_feats, train=False)
        losses.append(float(loss))
    raw.train()
    raw.style_encoder.eval()
    raw.content_encoder.eval()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = get_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.report_to,
                              project_dir=str(Path(args.output_dir) / args.logging_dir))
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    es_sha, ec_sha = _verify_caches(args)
    if accelerator.is_main_process:
        _write_heartbeat(Path(args.output_dir), status="loading_caches", step=0)
    es_cache = EsCache(Path(args.es_cache_path))
    ec_cache = EcCache(Path(args.ec_cache_path))
    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    train_fonts = sorted(split.get("stems", split)["train"])
    if accelerator.is_main_process:
        _write_heartbeat(Path(args.output_dir), status="loading_library", step=0)
    library = _LibraryEs(es_cache, train_fonts, _style_chars_from_cache(es_cache))
    if accelerator.is_main_process:
        print(f"library Es table {tuple(library.table.shape)} fonts={len(train_fonts)}", flush=True)

    model = FontDiffuserModel(unet=build_unet(args),
                              style_encoder=build_style_encoder(args),
                              content_encoder=build_content_encoder(args))
    if args.phase_1_ckpt_dir:
        _load_checkpoint(model, Path(args.phase_1_ckpt_dir), restore_rng=False)
    if args.freeze_encoders:
        model.style_encoder.requires_grad_(False).eval()
        model.content_encoder.requires_grad_(False).eval()
    if args.encoder_runtime == "cache_only":
        _ban_encoder_forward(model)

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
                                         collate_fn=CollateFN(),
                                         generator=torch.Generator().manual_seed(args.seed or 3407))
    val_set = FontDataset(args, "val", [normalized, normalized, normalized], scr=False)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False,
                                             batch_size=args.train_batch_size,
                                             collate_fn=CollateFN())

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    raw = accelerator.unwrap_model(model)
    global_step = _load_checkpoint(raw, Path(args.resume_from), optimizer, scheduler,
                                   restore_rng=True) if args.resume_from else 0
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        _record_config(args, es_sha, ec_sha)

    best = {"step": None, "val": None}
    progress = tqdm(range(args.max_train_steps), initial=global_step,
                    disable=not accelerator.is_local_main_process)
    epochs = math.ceil(args.max_train_steps / max(1, len(loader)))
    stop_file = Path(args.output_dir) / "STOP"
    for _ in range(epochs):
        for samples in loader:
            if global_step >= args.max_train_steps:
                break
            model.train()
            raw.style_encoder.eval()
            raw.content_encoder.eval()
            bsz = samples["target_image"].shape[0]
            with accelerator.accumulate(model):
                with torch.no_grad():
                    style, queries = _style_conditions(es_cache, samples, samples["target_image"].device)
                    delta_draw = torch.rand(bsz, device=style.device) < args.delta_drop
                    structure = _structure_features(es_cache, ec_cache, library, samples, queries,
                                                    args, delta_draw, style.device)
                    cfg_mask = torch.rand(bsz, device=style.device) < args.drop_prob
                    content_feats = _content_features(ec_cache, samples, cfg_mask, style.device)
                    style = style.clone()
                    style[cfg_mask] = 0
                loss, _ = _forward_batch(model, noise_scheduler, perceptual_loss, args, samples,
                                         style, structure, content_feats, train=True)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                loss_value = float(loss.detach().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"non-finite train loss at step {global_step}: {loss_value}")
                accelerator.log({"train_loss": loss_value}, step=global_step)
                if accelerator.is_main_process:
                    if global_step <= 64:
                        with (Path(args.output_dir) / "draw_log.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "step": global_step,
                                "fonts": list(samples["font_stem"]),
                                "chars": list(samples["char_cp"]),
                                "refs": list(samples["ref_chars"]),
                                "cfg": [bool(x) for x in cfg_mask.detach().cpu()],
                                "delta_draw": [bool(x) for x in delta_draw.detach().cpu()],
                                "loss": loss_value,
                            }, ensure_ascii=False) + "\n")
                    if global_step % args.log_interval == 0 or global_step <= 20:
                        with (Path(args.output_dir) / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({
                                "step": global_step, "loss": loss_value,
                                "lr": scheduler.get_last_lr()[0],
                            }) + "\n")
                        _write_heartbeat(Path(args.output_dir), status="running",
                                         step=global_step, loss=loss_value)
                    if global_step % args.state_interval == 0:
                        _save_checkpoint(raw, Path(args.output_dir) / "last_state",
                                         optimizer, scheduler, global_step)
                if accelerator.is_main_process and global_step % args.ckpt_interval == 0:
                    _save_checkpoint(raw, Path(args.output_dir) / f"global_step_{global_step}",
                                     optimizer, scheduler, global_step)
                    _save_checkpoint(raw, Path(args.output_dir) / "last_state",
                                     optimizer, scheduler, global_step)
                    val_loss = _run_val(raw, es_cache, ec_cache, library, val_loader,
                                        noise_scheduler, args, style.device)
                    eligible = global_step >= 10000
                    if eligible and (best["val"] is None or val_loss < best["val"]):
                        best = {"step": global_step, "val": val_loss}
                        _save_checkpoint(raw, Path(args.output_dir) / "best",
                                         optimizer, scheduler, global_step)
                    (Path(args.output_dir) / "val_log.jsonl").open("a", encoding="utf-8").write(
                        json.dumps({"step": global_step, "val_loss": val_loss, "best": best}) + "\n")
                    accelerator.log({"val_loss": val_loss}, step=global_step)
                if stop_file.exists():
                    if accelerator.is_main_process:
                        _save_checkpoint(raw, Path(args.output_dir) / "stopped_step",
                                         optimizer, scheduler, global_step)
                    return
            if global_step % args.log_interval == 0:
                progress.set_postfix(loss=float(loss.detach()), lr=scheduler.get_last_lr()[0])
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        _save_checkpoint(raw, Path(args.output_dir) / "last_state", optimizer, scheduler, global_step)
        (Path(args.output_dir) / "DONE.json").write_text(
            json.dumps({"global_step": global_step, "status": "completed", "best": best}) + "\n",
            encoding="utf-8")
    accelerator.end_training()


if __name__ == "__main__":
    main()
