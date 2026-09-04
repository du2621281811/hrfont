#!/usr/bin/env python3
"""Build / refresh E1 training dashboard: loss curve + val/test sample grids.

Uses a non-training GPU (default 2) so it does not disturb E1 on GPU3.
Outputs under runs/E1-FTV2-A-S3407/viz/ — open via data http server symlink or direct path.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
RUN = ROOT / "runs/E1-FTV2-A-S3407"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
VARIANT = ROOT / "code/variants/cn2west_ft_v2/FontDiffuser"
P1 = ROOT / "code/official/FontDiffuser/ckpt"
VIZ = RUN / "viz"
VAL_STEMS = ROOT / "manifests/pipeline_v3_val_stems.txt"
TEST_STEMS = ROOT / "manifests/pipeline_v3_test_stems.txt"

# Fixed probe set for dashboard (latin targets + one CJK style ref)
PROBE_CHARS = ["A", "a", "G", "0", "永"]
STYLE_PREF = ["永", "和", "书", "一"]


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def parse_losses() -> list[dict]:
    rows: dict[int, float] = {}
    log = RUN / "fontdiffuser_training.log"
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"Global Step\s+(\d+)\s+=>\s+train_loss\s*=\s*([0-9.eE+-]+)", line)
            if m:
                rows[int(m.group(1))] = float(m.group(2))
    # also scrape tqdm step_loss occasionally from watchdog log (denser early signal)
    wlog = RUN / "watchdog_train.log"
    if wlog.is_file():
        text = wlog.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"\|\s*(\d+)/100000\s+.*?step_loss=([0-9.eE+-]+)", text):
            step = int(m.group(1))
            if step % 50 == 0 or step < 200:
                rows.setdefault(step, float(m.group(2)))
    out = [{"step": s, "loss": rows[s]} for s in sorted(rows)]
    return out


def plot_loss(train_losses: list[dict], out_png: Path, val_losses: list[dict] | None = None) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not train_losses and not val_losses:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.text(0.5, 0.5, "waiting for loss…", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    if train_losses:
        steps = [r["step"] for r in train_losses]
        vals = [r["loss"] for r in train_losses]
        ax.plot(steps, vals, color="#1f4a6f", lw=1.15, alpha=0.85, label="train")
        if len(vals) >= 10:
            w = max(3, min(25, len(vals) // 10))
            kernel = np.ones(w) / w
            ma = np.convolve(vals, kernel, mode="valid")
            ax.plot(steps[w - 1 :], ma, color="#7aa0c4", lw=1.5, alpha=0.95, label=f"train MA-{w}")
    if val_losses:
        vsteps = [r["step"] for r in val_losses]
        vvals = [r["loss"] for r in val_losses]
        ax.plot(vsteps, vvals, color="#c45c26", lw=2.0, marker="o", ms=3.5, label="val (ckpt)")
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.set_title("E1-FTV2-A-S3407 train / val loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pick_style_path(split: str, stem: str) -> Path | None:
    d = DATA / split / "StyleImage" / stem
    if not d.is_dir():
        return None
    for ch in STYLE_PREF:
        p = d / f"{stem}+{cp_of(ch)}.png"
        if p.is_file():
            return p
    pngs = sorted(d.glob("*.png"))
    return pngs[0] if pngs else None


def content_path(split: str, ch: str) -> Path:
    # Content lives per-split in A disk
    for sp in (split, "train"):
        p = DATA / sp / "ContentImage" / f"{cp_of(ch)}.png"
        if p.is_file():
            return p
    raise FileNotFoundError(ch)


def gt_path(split: str, stem: str, ch: str) -> Path | None:
    # 永 may be style-only; targets are latin/digits etc.
    p = DATA / split / "TargetImage" / stem / f"{stem}+{cp_of(ch)}.png"
    if p.is_file():
        return p
    p2 = DATA / split / "StyleImage" / stem / f"{stem}+{cp_of(ch)}.png"
    return p2 if p2.is_file() else None


def list_ckpt_tags() -> list[tuple[str, Path, int]]:
    """(tag, ckpt_dir, step) for regeneratable weights only.

    Formal milestones: step0_P1 + every global_step_* (5k).
    Plus current last_state only when it is not exactly on a 5k milestone
    (mid-interval progress without spawning every-1k orphan dirs).
    """
    tags: list[tuple[str, Path, int]] = [("step0_P1", P1, 0)]
    formal_steps: set[int] = {0}
    for p in sorted(RUN.glob("global_step_*")):
        if not (p / "unet.pth").is_file():
            continue
        try:
            step = int(p.name.split("_")[-1])
        except ValueError:
            continue
        tags.append((f"step{step}", p, step))
        formal_steps.add(step)

    last = RUN / "last_state"
    if (last / "unet.pth").is_file() and (last / "trainer_state.pt").is_file():
        try:
            import torch
            st = torch.load(str(last / "trainer_state.pt"), map_location="cpu", weights_only=False)
            step = int(st.get("global_step", 0))
        except Exception:
            step = -1
        if step > 0 and step not in formal_steps:
            tags.append((f"step{step}_last", last, step))

    by_step: dict[int, tuple[str, Path, int]] = {}
    for tag, path, step in tags:
        prev = by_step.get(step)
        if prev is None or path.name.startswith("global_step"):
            by_step[step] = (tag, path, step)
    return [by_step[s] for s in sorted(by_step)]


def load_split_fonts(max_fonts_per_split: int | None) -> tuple[list[str], list[str]]:
    val = [l.strip() for l in VAL_STEMS.read_text().splitlines() if l.strip()]
    test = [l.strip() for l in TEST_STEMS.read_text().splitlines() if l.strip()]
    if max_fonts_per_split is not None and max_fonts_per_split > 0:
        val = val[:max_fonts_per_split]
        test = test[:max_fonts_per_split]
    return val, test


def load_pipe(ckpt_dir: Path, device: str):
    sys.path.insert(0, str(VARIANT))
    import torch
    from types import SimpleNamespace
    from src import (
        FontDiffuserDPMPipeline,
        FontDiffuserModelDPM,
        build_ddpm_scheduler,
        build_unet,
        build_content_encoder,
        build_style_encoder,
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
    )
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    unet.load_state_dict(torch.load(ckpt_dir / "unet.pth", map_location="cpu", weights_only=True))
    style_encoder.load_state_dict(torch.load(ckpt_dir / "style_encoder.pth", map_location="cpu", weights_only=True))
    content_encoder.load_state_dict(torch.load(ckpt_dir / "content_encoder.pth", map_location="cpu", weights_only=True))
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    ).to(device)
    model.eval()
    noise_scheduler = build_ddpm_scheduler(args)
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=noise_scheduler,
        version="V3",
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5,
    )
    return pipe, args


def to_tensor96(img: Image.Image, device: str):
    import torch
    import torchvision.transforms as T

    if img.size != (96, 96):
        raise ValueError(f"expected 96x96, got {img.size}")
    t = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return t(img.convert("RGB"))[None].to(device)


def sample_one(pipe, args, content_img: Image.Image, style_img: Image.Image, device: str, seed: int = 3407):
    import torch
    from accelerate.utils import set_seed

    set_seed(seed)
    content = to_tensor96(content_img, device)
    style = to_tensor96(style_img, device)
    with torch.no_grad():
        images = pipe.generate(
            content_images=content,
            style_images=style,
            batch_size=1,
            order=2,
            num_inference_step=20,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=None,
            t_end=None,
            dm_size=args.content_image_size,
            algorithm_type="dpmsolver++",
            skip_type="time_uniform",
            method="multistep",
            correcting_x0_fn=None,
        )
    im = images[0]
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if hasattr(im, "cpu"):
        arr = im.detach().cpu()
        if arr.ndim == 4:
            arr = arr[0]
        arr = ((arr.clamp(-1, 1) + 1) * 0.5 * 255).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(arr)
    raise TypeError(type(im))


def make_panel(content: Image.Image, style: Image.Image, gt: Image.Image | None, pred: Image.Image, label: str) -> Image.Image:
    cell = 96
    pad = 6
    cols = 4
    w = cols * cell + (cols + 1) * pad
    h = cell + pad * 2 + 18
    canvas = Image.new("RGB", (w, h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    imgs = [content, style, gt if gt is not None else Image.new("RGB", (96, 96), (220, 220, 220)), pred]
    titles = ["Content", "Style", "GT", "Pred"]
    for i, im in enumerate(imgs):
        x = pad + i * (cell + pad)
        canvas.paste(im.resize((96, 96)), (x, pad))
        draw.text((x, pad + cell + 2), titles[i], fill=(60, 60, 60))
    draw.text((pad, 0), label, fill=(30, 30, 30))
    return canvas


def ensure_samples(device: str, max_fonts_per_split: int | None = None) -> dict:
    """Generate sample panels for each regeneratable ckpt × full val/test fonts.

    Default: all fonts in each split (16+16). Pass a positive max_fonts_per_split
    only for smoke/debug; 0 or None means all.
    """
    import shutil

    out_root = VIZ / "samples"
    out_root.mkdir(parents=True, exist_ok=True)
    meta: dict = {"generated": [], "skipped": [], "errors": [], "removed_orphans": []}

    val_fonts, test_fonts = load_split_fonts(max_fonts_per_split)
    probes = [("val", s) for s in val_fonts] + [("test", s) for s in test_fonts]
    chars = ["A", "a", "G", "0"]
    expected_n = len(probes) * len(chars)
    expected_fonts = {"val": len(val_fonts), "test": len(test_fonts)}

    regenerable = list_ckpt_tags()
    regenerable_tags = {t[0] for t in regenerable}

    # Drop incomplete historical sample dirs that are no longer regenerable
    # (old every-1k last_state snapshots with only 2 fonts each).
    if out_root.is_dir():
        for tag_dir in list(out_root.iterdir()):
            if not tag_dir.is_dir():
                continue
            if tag_dir.name in regenerable_tags:
                continue
            shutil.rmtree(tag_dir, ignore_errors=True)
            meta["removed_orphans"].append(tag_dir.name)

    for tag, ckpt_dir, step in regenerable:
        tag_dir = out_root / tag
        marker = tag_dir / "_done.json"
        if marker.is_file():
            try:
                prev = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
            if (
                int(prev.get("expected_n", -1)) == expected_n
                and int(prev.get("n", 0)) >= expected_n
                and prev.get("fonts") == expected_fonts
            ):
                meta["skipped"].append(tag)
                continue
            for p in tag_dir.glob("*.png"):
                p.unlink(missing_ok=True)

        tag_dir.mkdir(parents=True, exist_ok=True)
        try:
            pipe, args = load_pipe(ckpt_dir, device)
        except Exception as e:
            meta["errors"].append({"tag": tag, "error": f"load:{e}"})
            continue
        made = []
        for split, stem in probes:
            style_p = pick_style_path(split, stem)
            if style_p is None:
                meta["errors"].append({"tag": tag, "stem": stem, "error": "no_style"})
                continue
            style_img = Image.open(style_p).convert("RGB")
            for ch in chars:
                try:
                    c_img = Image.open(content_path(split, ch)).convert("RGB")
                except FileNotFoundError:
                    meta["errors"].append({"tag": tag, "stem": stem, "ch": ch, "error": "no_content"})
                    continue
                gtp = gt_path(split, stem, ch)
                gt_img = Image.open(gtp).convert("RGB") if gtp else None
                try:
                    pred = sample_one(pipe, args, c_img, style_img, device=device, seed=3407)
                except Exception as e:
                    meta["errors"].append({"tag": tag, "stem": stem, "ch": ch, "error": str(e)})
                    continue
                panel = make_panel(c_img, style_img, gt_img, pred, f"{tag} · {split}/{stem} · '{ch}'")
                panel.save(tag_dir / f"{split}_{stem}_{cp_of(ch)}.png")
                made.append(f"{tag}/{split}_{stem}_{cp_of(ch)}.png")
        marker.write_text(
            json.dumps(
                {
                    "step": step,
                    "n": len(made),
                    "expected_n": expected_n,
                    "fonts": expected_fonts,
                    "chars": chars,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        meta["generated"].append({"tag": tag, "step": step, "n": len(made), "expected_n": expected_n})
        del pipe
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    meta["expected_fonts"] = expected_fonts
    meta["expected_n"] = expected_n
    return meta


def current_step_guess() -> int:
    # Prefer last_state
    last = RUN / "last_state" / "trainer_state.pt"
    if last.is_file():
        try:
            import torch
            st = torch.load(str(last), map_location="cpu", weights_only=False)
            return int(st.get("global_step", 0))
        except Exception:
            pass
    losses = parse_losses()
    if losses:
        return losses[-1]["step"]
    # tqdm scrape
    wlog = RUN / "watchdog_train.log"
    if wlog.is_file():
        ms = re.findall(r"\|\s*(\d+)/100000", wlog.read_text(encoding="utf-8", errors="replace")[-5000:])
        if ms:
            return int(ms[-1])
    return 0


# Panel layout from make_panel: pad=6, cell=96 → Content/Style/GT/Pred boxes
_PANEL_BOXES = {
    "content": (6, 6, 102, 102),
    "style": (108, 6, 204, 102),
    "gt": (210, 6, 306, 102),
    "pred": (312, 6, 408, 102),
}


def parse_sample_name(name: str) -> dict | None:
    # val_FontStem_u0041.png
    m = re.match(r"^(val|test)_(.+)_((?:u[0-9A-Fa-f]{4,6}))\.png$", name)
    if not m:
        return None
    split, stem, cp = m.group(1), m.group(2), m.group(3)
    try:
        ch = chr(int(cp[1:], 16))
    except Exception:
        ch = cp
    return {"split": split, "stem": stem, "cp": cp, "ch": ch, "key": f"{split}|{stem}|{cp}"}


def ensure_pred_crops() -> dict:
    """Crop Pred from panels; build Content/Style/GT refs from source data (clean 96×96)."""
    samples_root = VIZ / "samples"
    pred_root = VIZ / "preds"
    ref_root = VIZ / "refs"
    pred_root.mkdir(parents=True, exist_ok=True)
    for role in ("content", "style", "gt"):
        (ref_root / role).mkdir(parents=True, exist_ok=True)
    meta = {"tags": 0, "preds": 0, "refs": 0, "ref_errors": []}
    if not samples_root.is_dir():
        return meta

    # Prefer source data for refs (no panel title bleed). One pass over keys.
    seen_keys: set[str] = set()
    for tag_dir in samples_root.iterdir():
        if not tag_dir.is_dir():
            continue
        for png in tag_dir.glob("*.png"):
            info = parse_sample_name(png.name)
            if info is None or info["key"] in seen_keys:
                continue
            seen_keys.add(info["key"])
            split, stem, ch = info["split"], info["stem"], info["ch"]
            targets = {
                "content": None,
                "style": pick_style_path(split, stem),
                "gt": gt_path(split, stem, ch),
            }
            try:
                targets["content"] = content_path(split, ch)
            except FileNotFoundError:
                targets["content"] = None
            for role, src in targets.items():
                out = ref_root / role / png.name
                if out.is_file():
                    continue
                if src is None or not Path(src).is_file():
                    meta["ref_errors"].append({"file": png.name, "role": role, "error": "missing_src"})
                    continue
                Image.open(src).convert("RGB").resize((96, 96)).save(out)
                meta["refs"] += 1

    for tag_dir in sorted(samples_root.iterdir()):
        if not tag_dir.is_dir():
            continue
        out_tag = pred_root / tag_dir.name
        out_tag.mkdir(parents=True, exist_ok=True)
        n = 0
        for png in tag_dir.glob("*.png"):
            info = parse_sample_name(png.name)
            if info is None:
                continue
            pred_path = out_tag / png.name
            if pred_path.is_file():
                continue
            im = Image.open(png).convert("RGB")
            # Pred cell only; slight top inset avoids panel title bleed
            box = _PANEL_BOXES["pred"]
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            im.crop((box[0], box[1] + 2, box[2], box[3])).resize((96, 96)).save(pred_path)
            n += 1
            # Fallback refs from panel if source missing
            for role in ("content", "style", "gt"):
                ref_path = ref_root / role / png.name
                if ref_path.is_file():
                    continue
                im.crop(_PANEL_BOXES[role]).resize((96, 96)).save(ref_path)
                meta["refs"] += 1
        if n:
            meta["tags"] += 1
            meta["preds"] += n
    return meta


def load_val_loss_history() -> list[dict]:
    p = VIZ / "val_loss_history.json"
    if not p.is_file():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
        return [r for r in rows if isinstance(r, dict) and "step" in r and "loss" in r]
    except Exception:
        return []


def save_val_loss_history(rows: list[dict]) -> None:
    by_step = {int(r["step"]): r for r in rows}
    out = [by_step[s] for s in sorted(by_step)]
    (VIZ / "val_loss_history.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def load_train_model(ckpt_dir: Path, device: str):
    """Train-time FontDiffuserModel (for val loss), not DPM sampler."""
    sys.path.insert(0, str(VARIANT))
    import torch
    from types import SimpleNamespace
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
        offset_coefficient=0.5,
        drop_prob=0.0,  # eval: no CFG dropout
        phase_2=False,
        train_batch_size=16,
    )
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    unet.load_state_dict(torch.load(ckpt_dir / "unet.pth", map_location="cpu", weights_only=True))
    style_encoder.load_state_dict(torch.load(ckpt_dir / "style_encoder.pth", map_location="cpu", weights_only=True))
    content_encoder.load_state_dict(torch.load(ckpt_dir / "content_encoder.pth", map_location="cpu", weights_only=True))
    model = FontDiffuserModel(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    ).to(device)
    model.eval()
    noise_scheduler = build_ddpm_scheduler(args)
    percep = ContentPerceptualLoss()
    return model, noise_scheduler, percep, args


def eval_val_loss_one(ckpt_dir: Path, device: str, max_batches: int | None = None, seed: int = 3407) -> dict:
    """Mirror train.py loss on val split (no dropout, no SCR)."""
    import random
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

    model, noise_scheduler, percep, args = load_train_model(ckpt_dir, device)

    def _assert_native_size(img):
        if img.size != (96, 96):
            raise ValueError(f"expected native 96x96, got {img.size}")
        return img

    tfm = transforms.Compose(
        [_assert_native_size, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    ds = FontDataset(args=args, phase="val", transforms=[tfm, tfm, tfm], scr=False)
    loader = torch.utils.data.DataLoader(
        ds, shuffle=False, batch_size=args.train_batch_size, collate_fn=CollateFN(), num_workers=0
    )

    total = 0.0
    n = 0
    with torch.no_grad():
        for bi, samples in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
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
            offset_loss = offset_out_sum / 2
            pred_x0_norm = x0_from_epsilon(
                scheduler=noise_scheduler,
                noise_pred=noise_pred,
                x_t=noisy,
                timesteps=timesteps,
            )
            pred_x0 = reNormalize_img(pred_x0_norm)
            percep_loss = percep.calculate_loss(
                generated_images=normalize_mean_std(pred_x0),
                target_images=normalize_mean_std(nonorm_target_images),
                device=device,
            )
            loss = (
                diff_loss
                + args.perceptual_coefficient * percep_loss
                + args.offset_coefficient * offset_loss
            )
            total += float(loss.item()) * bsz
            n += bsz

    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    mean = total / max(n, 1)
    return {"loss": mean, "n": n, "max_batches": max_batches}


def ensure_val_losses(device: str, max_batches: int | None = None) -> dict:
    """Compute missing val losses for regeneratable ckpts (offline; does not touch train)."""
    VIZ.mkdir(parents=True, exist_ok=True)
    hist = load_val_loss_history()
    have = {int(r["step"]) for r in hist}
    meta = {"computed": [], "skipped": [], "errors": []}
    for tag, ckpt_dir, step in list_ckpt_tags():
        prev = next((r for r in hist if int(r["step"]) == step), None)
        if prev and int(prev.get("n", 0)) > 0:
            meta["skipped"].append(tag)
            continue
        try:
            row = eval_val_loss_one(ckpt_dir, device=device, max_batches=max_batches, seed=3407)
            row.update({"step": step, "tag": tag, "ts": datetime.now(timezone.utc).isoformat()})
            hist = [r for r in hist if int(r["step"]) != step] + [row]
            save_val_loss_history(hist)
            meta["computed"].append({"tag": tag, "step": step, "loss": row["loss"], "n": row["n"]})
            have.add(step)
        except Exception as e:
            meta["errors"].append({"tag": tag, "error": str(e)})
    meta["history"] = load_val_loss_history()
    return meta


def collect_sample_index() -> tuple[list[dict], list[dict]]:
    """Return (steps, keys) for comparison UI."""
    samples_root = VIZ / "samples"
    pred_root = VIZ / "preds"
    steps: list[dict] = []
    keys_map: dict[str, dict] = {}
    if not samples_root.is_dir():
        return steps, []
    for tag_dir in samples_root.iterdir():
        if not tag_dir.is_dir():
            continue
        imgs = [p.name for p in tag_dir.glob("*.png")]
        if not imgs:
            continue
        step = 0
        m = re.search(r"step(\d+)", tag_dir.name)
        if m:
            step = int(m.group(1))
        # prefer pred crops when present
        has_preds = (pred_root / tag_dir.name).is_dir() and any((pred_root / tag_dir.name).glob("*.png"))
        steps.append({"tag": tag_dir.name, "step": step, "n": len(imgs), "preds": has_preds})
        for name in imgs:
            info = parse_sample_name(name)
            if info is None:
                continue
            keys_map.setdefault(
                info["key"],
                {
                    "key": info["key"],
                    "split": info["split"],
                    "stem": info["stem"],
                    "cp": info["cp"],
                    "ch": info["ch"],
                    "file": name,
                },
            )
    steps.sort(key=lambda g: (g["step"], g["tag"]))
    keys = sorted(keys_map.values(), key=lambda k: (k["split"], k["stem"], k["cp"]))
    return steps, keys


def write_html(train_losses: list[dict], sample_meta: dict, val_losses: list[dict] | None = None) -> None:
    VIZ.mkdir(parents=True, exist_ok=True)
    ensure_pred_crops()
    steps, keys = collect_sample_index()
    val_losses = val_losses if val_losses is not None else load_val_loss_history()
    step_now = current_step_guess()
    payload = {
        "steps": steps,
        "keys": keys,
        "val_losses": val_losses,
        "train_tail": train_losses[-5:] if train_losses else [],
    }
    (VIZ / "compare_index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    val_best = None
    if val_losses:
        val_best = min(val_losses, key=lambda r: float(r["loss"]))

    best_txt = (
        f"当前最优 val loss = <b>{val_best['loss']:.4f}</b> @ step {val_best['step']}（{val_best.get('tag','')}）"
        if val_best
        else "验证集 loss 尚未计算（离线评测中…）"
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="120"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>E1 FT-v2 训练看板</title>
<style>
:root{{--bg:#eef1f5;--card:#fff;--line:#d5dbe3;--muted:#5c6570;--ink:#1a1a1a;--accent:#1f4a6f}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}}
header{{background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}}
h1{{margin:0;font-size:1.25rem}} .meta{{color:var(--muted);font-size:12px;margin-top:4px}}
main{{max-width:1200px;margin:16px auto;padding:0 14px 40px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px;margin:14px 0}}
.card h2{{margin:0 0 8px;font-size:1.05rem}}
img.loss{{width:100%;height:auto;background:#fff}}
.controls{{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:8px 0 12px}}
.controls label{{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--muted)}}
.controls select{{min-width:160px;padding:6px 8px;border:1px solid var(--line);border-radius:4px;background:#fff}}
.refs{{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;margin-bottom:12px;padding:8px 10px;border:1px solid var(--line);background:#f7f9fb}}
.refs .ref-meta{{flex:1 1 180px;min-width:160px;font-size:12px;color:var(--muted);align-self:center}}
.refs figure{{margin:0;text-align:center;flex:0 0 auto}}
.refs img{{width:96px;height:96px;image-rendering:pixelated;border:1px solid var(--line);background:#fff;display:block}}
.refs figcaption{{font-size:11px;color:var(--muted);margin-top:4px}}
.timeline-wrap{{overflow-x:auto;border:1px solid var(--line);background:#f7f9fb;padding:10px}}
.timeline{{display:flex;gap:8px;align-items:flex-end;min-height:130px}}
.timeline figure{{margin:0;flex:0 0 auto;width:104px;text-align:center}}
.timeline img{{width:96px;height:96px;image-rendering:pixelated;border:1px solid var(--line);background:#fff;display:block}}
.timeline figcaption{{font-size:10px;color:var(--muted);margin-top:3px;font-family:ui-monospace,monospace}}
.timeline figure.best img{{outline:2px solid #c45c26;outline-offset:1px}}
.timeline .col-label{{font-size:10px;color:var(--muted);margin-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:12px}} td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}}
.tag{{font-family:ui-monospace,monospace;font-size:12px;color:var(--accent)}}
details.meta-box{{font-size:12px;color:var(--muted)}}
</style>
</head>
<body>
<header>
  <h1>E1-FTV2-A-S3407 训练看板</h1>
  <div class="meta">每 120s 自动刷新 · 当前约 step <b>{step_now}</b> / 100000 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
  采样/验证评测在空闲 GPU，不干扰训练</div>
</header>
<main>
  <section class="card">
    <h2>Train / Val loss</h2>
    <img class="loss" src="loss.png?t={int(datetime.now().timestamp())}" alt="loss curve"/>
    <p class="meta">蓝=训练 loss（log 每 100 step）。橙=验证集 loss（各 ckpt 离线评测，与 train.py 同一公式：diff+percep+offset，drop_prob=0）。{best_txt}</p>
  </section>

  <section class="card">
    <h2>多样本时间步对比（Pred）</h2>
    <p class="meta">Content / Style / GT 只显示一次；下方横向滚动对比各 step 的 Pred。字集：val16+test16 × A/a/G/0 · seed=3407 · DPM++20 / CFG7.5</p>
    <div class="controls">
      <label>Split<select id="sel-split"></select></label>
      <label>字体<select id="sel-font"></select></label>
      <label>字符<select id="sel-char"></select></label>
      <label>跳到最优 val step
        <button type="button" id="btn-best" style="padding:6px 10px">Best val</button>
      </label>
    </div>
    <div class="refs" id="refs"></div>
    <div class="timeline-wrap"><div class="timeline" id="timeline"></div></div>
  </section>

  <section class="card">
    <h2>验证集 loss 表</h2>
    <table><thead><tr><th>step</th><th>tag</th><th>val loss</th><th>n</th></tr></thead>
    <tbody id="val-tbody"></tbody></table>
  </section>

  <section class="card">
    <details class="meta-box"><summary>生成元数据 / 调试</summary>
      <pre style="white-space:pre-wrap;font-size:11px">{json.dumps(sample_meta, ensure_ascii=False, indent=2)}</pre>
    </details>
  </section>
</main>
<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};
const bestStep = {json.dumps(val_best["step"] if val_best else None)};
const selSplit = document.getElementById('sel-split');
const selFont = document.getElementById('sel-font');
const selChar = document.getElementById('sel-char');
const refs = document.getElementById('refs');
const timeline = document.getElementById('timeline');
const tbody = document.getElementById('val-tbody');

(DATA.val_losses || []).slice().sort((a,b)=>a.step-b.step).forEach(r => {{
  const tr = document.createElement('tr');
  const isBest = bestStep !== null && r.step === bestStep;
  tr.innerHTML = `<td>${{r.step}}${{isBest?' ★':''}}</td><td class="tag">${{r.tag||''}}</td><td>${{Number(r.loss).toFixed(5)}}</td><td>${{r.n||''}}</td>`;
  tbody.appendChild(tr);
}});

function uniq(arr){{ return [...new Set(arr)]; }}
function keysFiltered(){{
  return DATA.keys.filter(k => k.split === selSplit.value);
}}
function fillSelect(sel, values, preferred){{
  const cur = preferred || sel.value;
  sel.innerHTML = '';
  values.forEach(v => {{
    const o = document.createElement('option');
    o.value = v; o.textContent = v; sel.appendChild(o);
  }});
  if (values.includes(cur)) sel.value = cur;
  else if (values.length) sel.value = values[0];
}}
function rebuildFonts(){{
  const fonts = uniq(keysFiltered().map(k => k.stem)).sort();
  fillSelect(selFont, fonts);
  rebuildChars();
}}
function rebuildChars(){{
  const chars = uniq(keysFiltered().filter(k => k.stem === selFont.value).map(k => k.ch));
  // keep A a G 0 order
  const order = ['A','a','G','0'];
  chars.sort((a,b) => (order.indexOf(a)+99) - (order.indexOf(b)+99) || a.localeCompare(b));
  fillSelect(selChar, chars);
  render();
}}
function currentKey(){{
  return DATA.keys.find(k => k.split===selSplit.value && k.stem===selFont.value && k.ch===selChar.value);
}}
function render(){{
  const k = currentKey();
  refs.innerHTML = '';
  timeline.innerHTML = '';
  if (!k) {{ refs.textContent = '无样本'; return; }}
  const bust = '{int(datetime.now().timestamp())}';
  const meta = document.createElement('div');
  meta.className = 'ref-meta';
  meta.innerHTML = `<b>${{k.split}}</b> / ${{k.stem}} / '<b>${{k.ch}}</b>'<br/>固定参考（不随 step 变化）`;
  refs.appendChild(meta);
  [['content','Content'],['style','Style'],['gt','GT']].forEach(([role, label]) => {{
    const f = document.createElement('figure');
    const img = document.createElement('img');
    img.src = `refs/${{role}}/${{k.file}}?t=${{bust}}`;
    img.alt = label;
    img.width = 96; img.height = 96;
    const c = document.createElement('figcaption');
    c.textContent = label;
    f.appendChild(img); f.appendChild(c);
    refs.appendChild(f);
  }});

  DATA.steps.forEach(s => {{
    const f = document.createElement('figure');
    if (bestStep !== null && s.step === bestStep) f.classList.add('best');
    const img = document.createElement('img');
    img.src = s.preds ? `preds/${{s.tag}}/${{k.file}}?t=${{bust}}` : `samples/${{s.tag}}/${{k.file}}?t=${{bust}}`;
    img.alt = s.tag;
    img.width = 96; img.height = 96;
    img.loading = 'lazy';
    const c = document.createElement('figcaption');
    c.textContent = s.step === 0 ? 'P1' : String(s.step);
    f.appendChild(img); f.appendChild(c);
    timeline.appendChild(f);
  }});
}}
selSplit.addEventListener('change', rebuildFonts);
selFont.addEventListener('change', rebuildChars);
selChar.addEventListener('change', render);
document.getElementById('btn-best').addEventListener('click', () => {{
  if (bestStep === null) return;
  const el = [...timeline.querySelectorAll('figure')].find(f => f.classList.contains('best'));
  if (el) el.scrollIntoView({{inline:'center', block:'nearest', behavior:'smooth'}});
}});
fillSelect(selSplit, uniq(DATA.keys.map(k => k.split)).sort().reverse()); // val first
rebuildFonts();
</script>
</body>
</html>
"""
    (VIZ / "index.html").write_text(html, encoding="utf-8")


def link_into_data_server() -> None:
    """Expose under data/ for port 8777 hub."""
    dst = ROOT / "data/e1_ft_v2_dashboard"
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink():
            return
    try:
        if dst.exists():
            return
        dst.symlink_to(VIZ, target_is_directory=True)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0", help="Visible device after CUDA_VISIBLE_DEVICES")
    ap.add_argument("--gpu", type=int, default=2, help="Physical GPU for sampling / val eval")
    ap.add_argument("--skip-samples", action="store_true")
    ap.add_argument(
        "--eval-val-loss",
        action="store_true",
        help="Compute missing validation losses for regeneratable ckpts",
    )
    ap.add_argument(
        "--val-max-batches",
        type=int,
        default=0,
        help="Limit val batches (0 = full val split)",
    )
    ap.add_argument(
        "--max-fonts",
        type=int,
        default=0,
        help="Max fonts per split (val/test). 0 = all (16+16).",
    )
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    VIZ.mkdir(parents=True, exist_ok=True)

    train_losses = parse_losses()
    (VIZ / "loss_history.json").write_text(json.dumps(train_losses, indent=2) + "\n", encoding="utf-8")

    sample_meta: dict = {"skipped_samples": True}
    val_meta: dict = {"skipped_val": True}
    if not args.skip_samples:
        mf = None if args.max_fonts <= 0 else args.max_fonts
        sample_meta = ensure_samples(device=args.device, max_fonts_per_split=mf)
    if args.eval_val_loss:
        mb = None if args.val_max_batches <= 0 else args.val_max_batches
        val_meta = ensure_val_losses(device=args.device, max_batches=mb)

    val_losses = load_val_loss_history()
    plot_loss(train_losses, VIZ / "loss.png", val_losses=val_losses)
    crop_meta = ensure_pred_crops()
    write_html(train_losses, {"samples": sample_meta, "val": val_meta, "crops": crop_meta}, val_losses=val_losses)
    link_into_data_server()

    # hub card bump
    hub = ROOT / "data/render_qa_hub.html"
    if hub.is_file() and "e1_ft_v2_dashboard" not in hub.read_text(encoding="utf-8"):
        text = hub.read_text(encoding="utf-8")
        card = """
<a class="card" href="e1_ft_v2_dashboard/">
  <b>E1 FT-v2 训练看板</b>
  <div class="s">train/val loss · 多时间步 Pred 对比 · 随 checkpoint 更新</div>
</a>
"""
        text = text.replace("</body>", card + "\n</body>")
        hub.write_text(text, encoding="utf-8")

    print(
        json.dumps(
            {
                "train_losses": len(train_losses),
                "val_losses": len(val_losses),
                "step": current_step_guess(),
                "samples": sample_meta,
                "val": val_meta,
                "crops": crop_meta,
                "url": "http://127.0.0.1:8777/e1_ft_v2_dashboard/",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
