#!/usr/bin/env python3
"""Cache-backed TC-v2 sampler for the real F2RL/G-RL condition path.

This intentionally does not touch the legacy DPM sampler.  It reconstructs the
same cached G9 + local-L128 + Mean-Delta + TC inputs used by ``train.py`` and
uses the existing DDPM scheduler for a small, reproducible inference path.
Only the requested content glyph and reference cache rows are read; target GT
images are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(VARIANT))

import train as tr  # noqa: E402
from configs.fontdiffuser import get_parser  # noqa: E402
from hrfont_feature_cache import EcCache, EsCache, EsLocalCache  # noqa: E402
from src import (FontDiffuserModel, TCV2Cache, TCV2Global9Adapter, TCV2Head,
                 build_content_encoder, build_ddpm_scheduler, build_style_encoder,
                 build_unet)  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = get_parser()
    parser.set_defaults(arm="F2RL", tc_enabled=True, rsi_source="delta",
                        support=False, encoder_runtime="cache_only")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--font", required=True)
    parser.add_argument("--content-cp", required=True)
    parser.add_argument("--refs", nargs="+", required=True,
                        help="One to eight reference codepoint tokens, e.g. u6C38 u548C")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    # The shared training parser marks --arm required.  This sampler has a
    # deliberate F2RL+TC default, while still allowing an explicit F2/F2RL
    # baseline with --no-tc_enabled, so inject the default only when absent.
    argv = sys.argv[1:]
    if not any(item == "--arm" or item.startswith("--arm=") for item in argv):
        argv = ["--arm", "F2RL", *argv]
    args = parser.parse_args(argv)
    args.style_image_size = (args.style_image_size, args.style_image_size)
    args.content_image_size = (args.content_image_size, args.content_image_size)
    if args.arm not in {"F2", "F2RL"}:
        raise ValueError("the dedicated DDPM sampler supports only F2 or F2RL")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not 1 <= len(args.refs) <= 8:
        raise ValueError("--refs must contain 1..8 references")
    return args


def load_model(args):
    model = FontDiffuserModel(unet=build_unet(args),
                              style_encoder=build_style_encoder(args),
                              content_encoder=build_content_encoder(args))
    if args.arm == "F2RL":
        tr._attach_local_proj(model)
    if args.tc_enabled:
        model.tc_head = TCV2Head()
        model.tc_global_adapter = TCV2Global9Adapter()
    ckpt = args.checkpoint
    model.unet.load_state_dict(torch.load(ckpt / "unet.pth", map_location="cpu", weights_only=True))
    style_path = ckpt / "style_encoder.pth"
    content_path = ckpt / "content_encoder.pth"
    model.style_encoder.load_state_dict(torch.load(style_path, map_location="cpu", weights_only=True))
    model.content_encoder.load_state_dict(torch.load(content_path, map_location="cpu", weights_only=True))
    modules = []
    if args.arm == "F2RL":
        modules.append((model.local_style_proj, "local_style_proj.pth"))
    if args.tc_enabled:
        modules.extend(((model.tc_head, "tc_head.pth"),
                        (model.tc_global_adapter, "tc_global_adapter.pth")))
    for module, name in modules:
        path = ckpt / name
        if not path.is_file():
            raise RuntimeError(f"required {name} missing from checkpoint: {path}")
        module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.style_encoder.requires_grad_(False).eval()
    model.content_encoder.requires_grad_(False).eval()
    return model.to(args.device).eval(), sha256_file(style_path), sha256_file(content_path)


def load_content(args):
    path = args.data_root / args.split / "ContentImage" / f"{args.content_cp}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    if image.size != (args.resolution, args.resolution):
        raise ValueError(f"content must be native {args.resolution}px: {path}")
    image = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])(image)
    return image.unsqueeze(0)


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    output_png = args.output.with_suffix(".png")
    output_meta = args.output.with_suffix(".json")
    if args.output.exists() or output_png.exists() or output_meta.exists():
        raise RuntimeError(f"refusing to overwrite sampler outputs near {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_manifest = Path(args.split_manifest)
    if not split_manifest.is_file():
        split_manifest = ROOT / args.split_manifest
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    stems = split.get("stems", split)
    if args.font not in stems[args.split]:
        raise RuntimeError(f"font {args.font} is not in split {args.split}")
    es = EsCache(Path(args.es_cache_path))
    ec = EcCache(Path(args.ec_cache_path))
    local = (EsLocalCache(Path(args.es_local_cache_path))
             if args.arm == "F2RL" else None)
    tc = TCV2Cache(Path(args.tc_cache_path)) if args.tc_enabled else None
    # Reuse the training preflight so sampler and training enforce identical
    # encoder/local/TC cache completion and checkpoint SHA contracts.
    tr._verify_caches(args, args.checkpoint)
    if es.manifest.get("es_checkpoint_sha256") != sha256_file(args.checkpoint / "style_encoder.pth"):
        raise RuntimeError("Es cache/checkpoint SHA mismatch")
    if ec.manifest.get("ec_checkpoint_sha256") != sha256_file(args.checkpoint / "content_encoder.pth"):
        raise RuntimeError("Ec cache/checkpoint SHA mismatch")
    if tc is not None and tc.manifest.get("split_manifest_sha256") != sha256_file(split_manifest):
        raise RuntimeError("TC cache/split manifest SHA mismatch")
    if tc is not None:
        binding_path = args.checkpoint / "tc_binding.json"
        if not binding_path.is_file():
            raise RuntimeError("TC checkpoint is missing tc_binding.json")
        expected_binding = {
            "tc_cache_manifest_sha256": tr.cache_fingerprint(tc.manifest),
            "ec_checkpoint_sha256": ec.manifest.get("ec_checkpoint_sha256"),
            "split_manifest_sha256": tc.manifest.get("split_manifest_sha256"),
        }
        actual_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if actual_binding != expected_binding:
            raise RuntimeError("TC checkpoint/cache/Ec binding mismatch")
    model, _, _ = load_model(args)
    train_fonts = sorted(stems["train"])
    if getattr(args, "v0913_clean_map", ""):
        from v0913_clean_lib import extra_exclude_indices, load_cp_group, load_donors
        clean_map = Path(args.v0913_clean_map)
        if not clean_map.is_dir():
            clean_map = ROOT / args.v0913_clean_map
        donors = load_donors(str(clean_map))
        train_fonts = donors["all"]
        cp_group = load_cp_group(str(clean_map))
        args._v0913_donor_exclude = lambda cp: extra_exclude_indices(
            train_fonts, cp, donors, cp_group)
    library = tr._LibraryEs(es, train_fonts, tr._style_chars_from_cache(es))
    samples = {"split": [args.split], "font_stem": [args.font],
               "char_cp": [args.content_cp], "ref_chars": [args.refs]}
    device = torch.device(args.device)
    with torch.no_grad():
        style, queries, style_seq, style_mask, local_raw, local_mask = tr._style_bundle(
            es, samples, device, args, local)
        source_on = torch.zeros(1, dtype=torch.bool, device=device)
        cfg_on = torch.zeros(1, dtype=torch.bool, device=device)
        structure = tr._structure_features(es, ec, library, samples, queries, args,
                                            source_on, device)
        content_feats = tr._content_features(ec, samples, cfg_on, device)
        seq, mask = tr._pack_up_style(model, style, style_seq, style_mask,
                                      local_raw, local_mask, cfg_on)
        if args.tc_enabled:
            tc_residual, _, _ = tr._tc_conditions(
                tc, ec, samples, model.tc_head, model.tc_global_adapter, cfg_on,
                device, include_target=False)
        else:
            tc_residual = None
        # Exact unconditional branch: source/content/global/local/TC conditions
        # are all masked, matching the training CFG convention.
        cfg_all = torch.ones(1, dtype=torch.bool, device=device)
        zero_style = torch.zeros_like(style)
        zero_content = tr._content_features(ec, samples, cfg_all, device)
        zero_structure = [torch.zeros_like(x) for x in structure]
        zero_seq, zero_mask = tr._pack_up_style(model, zero_style, style_seq, style_mask,
                                                local_raw, local_mask, cfg_all)
        zero_tc = torch.zeros_like(tc_residual) if tc_residual is not None else None

    content = load_content(args).to(device)
    scheduler = build_ddpm_scheduler(args)
    scheduler.set_timesteps(args.steps, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sample = torch.randn(1, 3, args.resolution, args.resolution, device=device,
                          generator=generator)
    for timestep in scheduler.timesteps:
        t = timestep.expand(1)
        cond, _ = model(sample, t, content_images=content, style_features=style,
                        structure_features=structure, content_features=content_feats,
                        style_seq_tokens=seq, style_seq_mask=mask,
                        tc_global_residual=tc_residual,
                        content_encoder_downsample_size=args.content_encoder_downsample_size)
        uncond, _ = model(sample, t, content_images=content, style_features=zero_style,
                          structure_features=zero_structure, content_features=zero_content,
                          style_seq_tokens=zero_seq, style_seq_mask=zero_mask,
                          tc_global_residual=zero_tc,
                          content_encoder_downsample_size=args.content_encoder_downsample_size)
        noise = uncond + args.guidance_scale * (cond - uncond)
        sample = scheduler.step(noise, timestep, sample, generator=generator).prev_sample
    image = ((sample[0].clamp(-1, 1) / 2 + 0.5) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image).save(output_png)
    output_meta.write_text(json.dumps({
        "kind": "cache_ddpm_sampler", "font": args.font, "content_cp": args.content_cp,
        "refs": args.refs, "split": args.split, "steps": args.steps,
        "guidance_scale": args.guidance_scale, "checkpoint": str(args.checkpoint),
        "es_checkpoint_sha256": sha256_file(args.checkpoint / "style_encoder.pth"),
        "ec_checkpoint_sha256": sha256_file(args.checkpoint / "content_encoder.pth"),
        "tc_enabled": bool(args.tc_enabled),
        "tc_cache_manifest_sha256": (tr.cache_fingerprint(tc.manifest)
                                      if tc is not None else None),
        "clean_map": args.v0913_clean_map,
        "target_gt_read": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
