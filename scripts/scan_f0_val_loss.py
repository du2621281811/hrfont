#!/usr/bin/env python3
"""Offline val16 diffusion loss for every F0 5k milestone.

Pre-registered selection (reports/DECISION_F123_20260906.md §2):
  candidates = global_step_{5000,...,100000}
  metric     = mean train-time loss on val split (no CFG dropout, no SCR)
  pick       = min loss; ties -> smaller step
  report     = selected step AND the 100k endpoint

Mirrors scripts/build_e1_train_dashboard.py::eval_val_loss_one, but:
  variant = cn2west_f0_rsifree
  offset_coefficient = 0.0  (F0 trained with offset=0)

Read-only w.r.t. checkpoints. Writes under --out.

Example:
  python scripts/scan_f0_val_loss.py --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/root/projects/hrfont")
VARIANT = ROOT / "code/variants/cn2west_f0_rsifree/FontDiffuser"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
RUN = ROOT / "runs/F0-RSIFREE-FT-A-S3407"


def list_milestones(run: Path) -> list[tuple[int, Path]]:
    rows = []
    for p in run.glob("global_step_*"):
        if not (p / "unet.pth").is_file():
            continue
        try:
            step = int(p.name.split("_")[-1])
        except ValueError:
            continue
        if step % 5000 == 0 and 5000 <= step <= 100000:
            rows.append((step, p))
    return sorted(rows)


def load_model(ckpt_dir: Path, device: str, offset_coefficient: float):
    sys.path.insert(0, str(VARIANT))
    import torch
    from src import (
        FontDiffuserModel,
        build_ddpm_scheduler,
        build_unet,
        build_content_encoder,
        build_style_encoder,
        ContentPerceptualLoss,
    )

    args = SimpleNamespace(
        resolution=96,
        unet_channels=(64, 128, 256, 512),
        style_image_size=(96, 96),
        content_image_size=(96, 96),
        content_encoder_downsample_size=3,
        channel_attn=True,
        content_start_channel=64,
        style_start_channel=64,
        beta_scheduler="scaled_linear",
        model_type="noise",
        data_root=str(DATA),
        perceptual_coefficient=0.01,
        offset_coefficient=offset_coefficient,
        drop_prob=0.0,
        phase_2=False,
        train_batch_size=8,
    )
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    unet.load_state_dict(torch.load(ckpt_dir / "unet.pth", map_location="cpu", weights_only=True))
    style_encoder.load_state_dict(
        torch.load(ckpt_dir / "style_encoder.pth", map_location="cpu", weights_only=True))
    content_encoder.load_state_dict(
        torch.load(ckpt_dir / "content_encoder.pth", map_location="cpu", weights_only=True))
    model = FontDiffuserModel(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    ).to(device)
    model.eval()
    return model, build_ddpm_scheduler(args), ContentPerceptualLoss(), args


def eval_one(ckpt_dir: Path, device: str, seed: int, offset_coefficient: float) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms

    sys.path.insert(0, str(VARIANT))
    from dataset.font_dataset import FontDataset
    from dataset.collate_fn import CollateFN
    from utils import x0_from_epsilon, reNormalize_img, normalize_mean_std

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, noise_scheduler, percep, args = load_model(ckpt_dir, device, offset_coefficient)

    def native(img):
        if img.size != (96, 96):
            raise ValueError(f"expected native 96x96, got {img.size}")
        return img

    tfm = transforms.Compose([native, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    ds = FontDataset(args=args, phase="val", transforms=[tfm, tfm, tfm], scr=False)
    loader = torch.utils.data.DataLoader(
        ds, shuffle=False, batch_size=args.train_batch_size, collate_fn=CollateFN(), num_workers=0
    )

    total = 0.0
    n = 0
    with torch.no_grad():
        for samples in loader:
            content_images = samples["content_image"].to(device)
            style_images = samples["style_image"].to(device)
            target_images = samples["target_image"].to(device)
            nonorm_target_images = samples["nonorm_target_image"].to(device)
            noise = torch.randn_like(target_images)
            bsz = target_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bsz,), device=device
            ).long()
            noisy = noise_scheduler.add_noise(target_images, noise, timesteps)
            noise_pred, offset_out_sum = model(
                x_t=noisy,
                timesteps=timesteps,
                style_images=style_images,
                content_images=content_images,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
            )
            diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            pred_x0 = reNormalize_img(
                x0_from_epsilon(scheduler=noise_scheduler, noise_pred=noise_pred,
                                x_t=noisy, timesteps=timesteps)
            )
            percep_loss = percep.calculate_loss(
                generated_images=normalize_mean_std(pred_x0),
                target_images=normalize_mean_std(nonorm_target_images),
                device=device,
            )
            loss = (
                diff_loss
                + args.perceptual_coefficient * percep_loss
                + args.offset_coefficient * (offset_out_sum / 2)
            )
            total += float(loss.item()) * bsz
            n += bsz

    del model
    import torch as _t
    _t.cuda.empty_cache()
    return {"loss": total / max(n, 1), "n": n}


def select(rows: list[dict]) -> dict:
    eligible = [r for r in rows if r.get("loss") is not None]
    if not eligible:
        raise RuntimeError("no val losses computed")
    best = min(eligible, key=lambda r: (float(r["loss"]), int(r["step"])))
    end = next(r for r in eligible if int(r["step"]) == 100000)
    return {
        "rule": "min val16 diffusion loss among 5k milestones; ties -> smaller step",
        "selected_step": int(best["step"]),
        "selected_loss": float(best["loss"]),
        "endpoint_step": 100000,
        "endpoint_loss": float(end["loss"]),
        "n": int(best["n"]),
        "seed": 3407,
        "offset_coefficient": 0.0,
        "variant": "cn2west_f0_rsifree",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--out", default=str(ROOT / "reports/training_logs/F0-RSIFREE-FT-A-S3407"))
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hist_path = out / "val_loss_history.json"
    hist = []
    if hist_path.is_file():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
    have = {int(r["step"]) for r in hist if "loss" in r}

    for step, ckpt in list_milestones(run):
        if step in have:
            print(f"skip {step}: already have loss", flush=True)
            continue
        print(f"eval step={step} ckpt={ckpt}", flush=True)
        row = eval_one(ckpt, args.device, args.seed, offset_coefficient=0.0)
        row.update({
            "step": step,
            "tag": f"global_step_{step}",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        hist = [r for r in hist if int(r["step"]) != step] + [row]
        hist.sort(key=lambda r: int(r["step"]))
        hist_path.write_text(json.dumps(hist, indent=2) + "\n", encoding="utf-8")
        print(f"  loss={row['loss']:.6f} n={row['n']}", flush=True)

    decision = select(hist)
    (out / "F0_MILESTONE.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    selected = run / f"global_step_{decision['selected_step']}"
    best = run / "best"
    if best.is_symlink() or best.exists():
        if best.is_symlink() or best.is_file():
            best.unlink()
        elif best.is_dir() and not any(best.iterdir()):
            best.rmdir()
    if not best.exists():
        best.symlink_to(selected.name)
        print(f"symlink {best} -> {selected.name}", flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
