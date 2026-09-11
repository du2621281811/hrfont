#!/usr/bin/env python3
"""Train/val style-domain probe: F2@40k vs F2-P@25k (Mode B ref8).

Locks fonts/chars before generate. Writes reports/style_domain_probe/.
Does not touch test16. Prefer GPU2 while F2-P trains on GPU1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
OUT = ROOT / "reports/style_domain_probe"
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
REF8 = list("永和书风骨韵天地")
REF8_CPS = [f"u{ord(c):04X}" for c in REF8]
SEED = 3407

# Locked before any generate (do not edit after first run without bumping probe_id).
PROBE_ID = "v1_20260911"
CHARS = list("Il1oAaあの")  # 8: latin + kana identity/style probes
# 12 train: weight extremes + unmarked (all must exist in split train).
TRAIN_STEMS = [
    "FZBaiZBYTJW",      # unmarked
    "FZBaiZJHTJW",      # unmarked
    "FZXinZYHJW_EB",    # ExtraBold (paired lineage)
    "FZXinZYHJW_UL",    # UltraLight
    "FZSiNTJW-H",       # Heavy
    "FZSiNTJW-UL",      # UltraLight
    "FZSiNTJW-UB",      # UltraBold
    "FZDongGSJW-L",     # Light
    "FZJingQSTJW_Cu",   # 粗
    "FZRunKJW-T",       # Thin
    "FZYouYSJW-R",      # Regular
    "FZFeiSJW-EL",      # ExtraLight
]
VAL_STEMS = []  # filled from split val[:8] after audit


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def audit_stems(preferred: list[str], pool: list[str], n: int, split_name: str) -> list[str]:
    """Keep preferred if present in pool; fill from pool in order."""
    have = set(pool)
    out = [s for s in preferred if s in have]
    for s in pool:
        if len(out) >= n:
            break
        if s not in out:
            out.append(s)
    if len(out) < n:
        raise RuntimeError(f"{split_name}: need {n} stems, got {len(out)}")
    return out[:n]


def content_path(ch: str) -> Path:
    for sp in ("test", "val", "train"):
        p = DATA / sp / "ContentImage" / f"{cp_of(ch)}.png"
        if p.is_file():
            return p
    raise FileNotFoundError(ch)


def gt_path(split_name: str, stem: str, ch: str) -> Path | None:
    p = DATA / split_name / "TargetImage" / stem / f"{stem}+{cp_of(ch)}.png"
    return p if p.is_file() else None


def style_ref_paths(split_name: str, stem: str) -> list[Path]:
    d = DATA / split_name / "StyleImage" / stem
    paths = []
    for c in REF8:
        p = d / f"{stem}+{cp_of(c)}.png"
        if not p.is_file():
            raise FileNotFoundError(p)
        paths.append(p)
    return paths


def pred_path(method: str, split_name: str, stem: str, ch: str) -> Path:
    return OUT / "preds" / method / split_name / stem / f"{stem}+{cp_of(ch)}.png"


def make_dpm_adapter(fd, torch, style_pattn: bool):
    class DPMAdapter(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.fd = inner
            self.style_pattn = style_pattn

        def forward(self, x_t, timesteps, cond, content_encoder_downsample_size, version):
            kw = dict(
                content_images=cond[0],
                content_encoder_downsample_size=content_encoder_downsample_size,
                style_features=cond[2],
                structure_features=cond[3],
                content_features=cond[4],
                support_tokens=cond[5],
            )
            if self.style_pattn:
                kw["style_seq_tokens"] = cond[6]
                kw["style_seq_mask"] = cond[7]
            noise, _ = self.fd(x_t, timesteps, **kw)
            return noise

    return DPMAdapter(fd)


def cat_cond(uncond, cond, torch):
    out = []
    for u, c in zip(uncond, cond):
        if u is None and c is None:
            out.append(None)
        elif isinstance(u, list):
            out.append([torch.cat([a, b], 0) for a, b in zip(u, c)])
        else:
            out.append(torch.cat([u, c], 0))
    return out


def generate_method(
    method: str,
    ckpt: Path,
    style_pattn: bool,
    device: str,
    jobs: list[tuple[str, str]],  # (split, stem)
    chars: list[str],
    overwrite: bool,
) -> dict:
    import torch
    from accelerate.utils import set_seed

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(VARIANT))
    os.chdir(VARIANT)
    import train as T
    from scripts.hrfont_feature_cache import EcCache, EsCache
    from src.build import (
        build_content_encoder,
        build_ddpm_scheduler,
        build_style_encoder,
        build_unet,
    )
    from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
    from src.model import FontDiffuserModel

    print(f"[{method}] load caches + {ckpt}", flush=True)
    es = EsCache(ROOT / "artifacts/f0/es_spatial_f0")
    ec = EcCache(ROOT / "artifacts/f0/ec_multiscale_f0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    library = T._LibraryEs(es, sorted(split["stems"]["train"]), T._style_chars_from_cache(es))
    cfg = SimpleNamespace(
        rsi_source="delta",
        delta_enabled=True,
        delta_tau=0.07,
        delta_eps_alpha=0.01,
        delta_k_max=10,
        delta_k_top=10,
        delta_mode="topk",
        seed=SEED,
        support=False,
        support_k=0,
        style_start_channel=64,
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
    )
    fd = FontDiffuserModel(
        unet=build_unet(args),
        style_encoder=build_style_encoder(args),
        content_encoder=build_content_encoder(args),
    )
    fd.unet.load_state_dict(torch.load(ckpt / "unet.pth", map_location="cpu", weights_only=True))
    fd.style_encoder.load_state_dict(
        torch.load(ckpt / "style_encoder.pth", map_location="cpu", weights_only=True)
    )
    fd.content_encoder.load_state_dict(
        torch.load(ckpt / "content_encoder.pth", map_location="cpu", weights_only=True)
    )
    T._ban_encoder_forward(fd)
    device_t = torch.device(device)
    fd.to(device_t).eval()
    model = make_dpm_adapter(fd, torch, style_pattn=style_pattn).to(device_t).eval()
    scheduler = build_ddpm_scheduler(args)
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)
    keep = torch.zeros(1, dtype=torch.bool, device=device_t)

    def pack_one(split_name: str, font: str, ch: str):
        base = {"split": [split_name], "font_stem": [font], "char_cp": [cp_of(ch)]}
        style, queries, style_seq, style_mask = T._style_conditions(
            es, {**base, "ref_chars": [REF8_CPS]}, device_t, style_pattn=style_pattn
        )
        structure = T._structure_features(
            es, ec, library, {**base, "ref_chars": [REF8_CPS]}, queries, cfg, keep, device_t
        )
        content = T._content_features(ec, {**base, "ref_chars": [REF8_CPS]}, keep, device_t)
        return style, structure, content, style_seq, style_mask

    def sample_one(packed):
        set_seed(SEED)
        img = torch.zeros(1, 3, 96, 96, device=device_t)
        style, structure, content, style_seq, style_mask = packed
        if style_pattn:
            cond = [img, img, style, structure, content, None, style_seq, style_mask]
            uncond = [
                torch.ones_like(img),
                torch.ones_like(img),
                torch.zeros_like(style),
                [torch.zeros_like(x) for x in structure],
                [torch.zeros_like(x) for x in content],
                None,
                torch.zeros_like(style_seq),
                style_mask.clone(),
            ]
        else:
            cond = [img, img, style, structure, content, None]
            uncond = [
                torch.ones_like(img),
                torch.ones_like(img),
                torch.zeros_like(style),
                [torch.zeros_like(x) for x in structure],
                [torch.zeros_like(x) for x in content],
                None,
            ]

        def get_t_input(t_continuous):
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0

        def model_fn(x, t_continuous):
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t_continuous, t_continuous], dim=0)
            noise_uncond, noise = model(
                x_in,
                get_t_input(t_in),
                cat_cond(uncond, cond, torch),
                content_encoder_downsample_size=3,
                version="V3",
            ).chunk(2)
            return noise_uncond + 7.5 * (noise - noise_uncond)

        solver = DPM_Solver(model_fn=model_fn, noise_schedule=noise_schedule, algorithm_type="dpmsolver++")
        x = torch.randn(1, 3, 96, 96, device=device_t)
        with torch.no_grad():
            x = solver.sample(x=x, steps=20, order=2, skip_type="time_uniform", method="multistep")
        x = (x / 2 + 0.5).clamp(0, 1)[0].detach().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray((x * 255).round().astype("uint8"))

    n_total = len(jobs) * len(chars)
    done = skipped = 0
    t0 = time.time()
    for split_name, stem in jobs:
        for ch in chars:
            out_p = pred_path(method, split_name, stem, ch)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.is_file() and not overwrite:
                skipped += 1
                continue
            pred = sample_one(pack_one(split_name, stem, ch))
            pred.save(out_p)
            meta = {
                "method": method,
                "split": split_name,
                "font": stem,
                "char": ch,
                "cp": cp_of(ch),
                "seed": SEED,
                "style_k": 8,
                "style_chars": "".join(REF8),
                "ckpt": str(ckpt.relative_to(ROOT)),
                "style_pattn": style_pattn,
                "gt": str(gt_path(split_name, stem, ch).relative_to(ROOT))
                if gt_path(split_name, stem, ch)
                else None,
            }
            out_p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
            done += 1
            if (done + skipped) % 10 == 0 or done + skipped == n_total:
                rate = done / max(1e-6, time.time() - t0)
                print(
                    f"[{method}] {done+skipped}/{n_total} done={done} skip={skipped} "
                    f"{rate:.2f}/s last={split_name}/{stem}/{ch}",
                    flush=True,
                )
    return {"method": method, "done": done, "skipped": skipped, "elapsed_s": round(time.time() - t0, 1)}


def tile_row(images: list[Image.Image], labels: list[str], cell=96, label_h=18) -> Image.Image:
    n = len(images)
    canvas = Image.new("RGB", (cell * n, cell + label_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for i, (im, lab) in enumerate(zip(images, labels)):
        canvas.paste(im.resize((cell, cell)), (i * cell, label_h))
        draw.text((i * cell + 4, 2), lab, fill=(30, 30, 30))
    return canvas


def build_html(manifest: dict) -> Path:
    methods = manifest["methods"]
    train = manifest["train_stems"]
    val = manifest["val_stems"]
    chars = manifest["chars"]

    def img_tag(method: str, split_name: str, stem: str, ch: str) -> str:
        rel = f"preds/{method}/{split_name}/{stem}/{stem}+{cp_of(ch)}.png"
        return f'<img src="{rel}" width="96" height="96" loading="lazy" alt="{method} {stem} {ch}"/>'

    def gt_tag(split_name: str, stem: str, ch: str) -> str:
        g = gt_path(split_name, stem, ch)
        if not g:
            return "<span class='miss'>∅</span>"
        # copy symlink into out for portable html
        dst = OUT / "gt" / split_name / stem / f"{stem}+{cp_of(ch)}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.symlink(g, dst)
            except OSError:
                Image.open(g).convert("RGB").resize((96, 96)).save(dst)
        rel = f"gt/{split_name}/{stem}/{stem}+{cp_of(ch)}.png"
        return f'<img src="{rel}" width="96" height="96" loading="lazy" alt="gt"/>'

    def ref_strip(split_name: str, stem: str) -> str:
        dst_dir = OUT / "refs" / split_name / stem
        dst_dir.mkdir(parents=True, exist_ok=True)
        tags = []
        for p in style_ref_paths(split_name, stem):
            dst = dst_dir / p.name
            if not dst.exists():
                try:
                    os.symlink(p, dst)
                except OSError:
                    Image.open(p).convert("RGB").resize((64, 64)).save(dst)
            tags.append(f'<img src="refs/{split_name}/{stem}/{p.name}" width="48" height="48"/>')
        return "".join(tags)

    sections = []
    for domain, stems, title in (
        ("train", train, "T0 · train fonts（应能学到的域）"),
        ("val", val, "V0 · val fonts（未见字体，近似 test 泛化）"),
    ):
        rows = []
        for stem in stems:
            cells = [
                f"<td class='font'><code>{stem}</code><div class='refs'>{ref_strip(domain, stem)}</div></td>"
            ]
            for ch in chars:
                cells.append(
                    "<td class='cell'>"
                    f"<div class='lab'>{ch}</div>"
                    f"<div class='pair'>"
                    f"<figure><figcaption>F2@40k</figcaption>{img_tag(methods[0]['id'], domain, stem, ch)}</figure>"
                    f"<figure><figcaption>F2-P@25k</figcaption>{img_tag(methods[1]['id'], domain, stem, ch)}</figure>"
                    f"<figure><figcaption>GT</figcaption>{gt_tag(domain, stem, ch)}</figure>"
                    f"</div></td>"
                )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        sections.append(
            f"<h2>{title}</h2><div class='scroll'><table><thead><tr>"
            f"<th>font / refs</th>" + "".join(f"<th>{c}</th>" for c in chars) + "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>Style domain probe · F2 vs F2-P</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#f7f5f1;color:#1a1a1a}}
h1{{font-size:1.4rem;margin:0 0 .4rem}}
.meta{{color:#555;font-size:.9rem;max-width:70ch;line-height:1.45}}
.scroll{{overflow-x:auto;border:1px solid #ddd;background:#fff;margin:12px 0 28px}}
table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #e5e5e5;padding:6px;vertical-align:top}}
th{{background:#eee;position:sticky;top:0}}
.font code{{font-size:11px}}
.refs img{{margin:2px;border:1px solid #ddd}}
.pair{{display:flex;gap:4px}}
figure{{margin:0;text-align:center}}
figcaption{{font-size:10px;color:#666}}
.lab{{font-weight:600;margin-bottom:2px}}
.miss{{color:#aaa}}
.verdict{{background:#fff;border:1px solid #ccc;padding:12px 16px;max-width:70ch}}
</style></head><body>
<h1>Style domain probe · F2@40k vs F2-P@25k</h1>
<p class="meta">probe_id=<code>{manifest['probe_id']}</code> · seed={SEED} · Mode B ref8=<code>{''.join(REF8)}</code> ·
DPM++20 CFG7.5 · chars=<code>{''.join(chars)}</code><br/>
{manifest['methods'][0]['id']}: {manifest['methods'][0]['ckpt']}<br/>
{manifest['methods'][1]['id']}: {manifest['methods'][1]['ckpt']}<br/>
读法：T0 差 → 训练域没学好；T0 好而 V0 差 → 泛化问题。先看 refs 脾气，再看 GT。</p>
<div class="verdict"><strong>眼检记录（填）</strong><br/>
T0 上 F2-P 是否更像 refs？ ____ &nbsp; V0 上是否仍更好？ ____<br/>
粗细是否跟 refs？ ____ &nbsp; 方圆/收笔是否跟 refs？ ____</div>
{''.join(sections)}
</body></html>"""
    path = OUT / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_stems = audit_stems(TRAIN_STEMS, sorted(split["stems"]["train"]), 12, "train")
    val_stems = audit_stems(VAL_STEMS, sorted(split["stems"]["val"]), 8, "val")

    methods = [
        {
            "id": "F2_40k",
            "label": "F2@40k",
            "ckpt": "runs/F2-DELTARSI-A-S3407/global_step_40000",
            "style_pattn": False,
        },
        {
            "id": "F2P_25k",
            "label": "F2-P@25k",
            "ckpt": "runs/f2_pattn_s3407/global_step_25000",
            "style_pattn": True,
        },
    ]
    for m in methods:
        if not (ROOT / m["ckpt"]).is_dir():
            raise FileNotFoundError(m["ckpt"])

    manifest = {
        "probe_id": PROBE_ID,
        "created": utc(),
        "seed": SEED,
        "chars": CHARS,
        "ref8": "".join(REF8),
        "train_stems": train_stems,
        "val_stems": val_stems,
        "methods": methods,
        "protocol": "Mode B ref8 · DPM++20 · CFG7.5 · matched seed 3407",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("LOCKED train:", train_stems)
    print("LOCKED val:", val_stems)
    print("LOCKED chars:", "".join(CHARS))

    if not args.html_only:
        jobs = [("train", s) for s in train_stems] + [("val", s) for s in val_stems]
        results = []
        for m in methods:
            results.append(
                generate_method(
                    m["id"],
                    ROOT / m["ckpt"],
                    m["style_pattn"],
                    args.device,
                    jobs,
                    CHARS,
                    args.overwrite,
                )
            )
        (OUT / "generate_summary.json").write_text(
            json.dumps({"ts": utc(), "results": results}, indent=2) + "\n", encoding="utf-8"
        )

    html = build_html(manifest)
    print("HTML", html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
