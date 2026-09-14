#!/usr/bin/env python3
"""Paper-6 OOD fonts: F0/F2 dirty vs CLEAN board (no collaborator G*).

Fonts are NOT in the 260 split. Renders Style/GT, builds Es overlays for
ref8, samples F0 image + F2 delta (layered Es: overlay query + base train
library), writes reports/paper6_0914_f0f2/index.html.
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

import numpy as np
import torch
import torch.nn.functional as F
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts"))
from build_cn2west_v2_proto_abc import CANVAS_96, INNER_AB, find_size_A, render_glyph_AB  # noqa: E402
from hrfont_feature_cache import ES_POOLED, ES_SPATIAL, EcCache, EsCache, MemmapTable, key_es, sha256_file  # noqa: E402

OUT = ROOT / "reports/paper6_0914_f0f2"
DATA_PNG = OUT / "png"
FONT_DIR = Path("/root/font_files_0914/founder_fonts_download")
REC = ROOT / "reports/paper_fonts_0914_screen/recommended_paper8.json"

REF8 = list("永和书风骨韵天地")
BOARD_CHARS = list("08AGQaegàěあさアンㄅㄚij")
SEED = 3407
SPLIT = "test"

F0_DIRTY = ROOT / "runs/F0-RSIFREE-FT-A-S3407/global_step_100000"
F0_CLEAN_95 = ROOT / "runs/F0-CLEAN-V0913-A-S3407/global_step_95000"
F0_CLEAN_100 = ROOT / "runs/F0-CLEAN-V0913-A-S3407/global_step_100000"
F2_DIRTY = ROOT / "runs/F2-DELTARSI-A-S3407"
F2_CLEAN = ROOT / "runs/F2-CLEAN-V0913-A-S3407"
ES_DIRTY = ROOT / "artifacts/f0/es_spatial_f0"
EC_DIRTY = ROOT / "artifacts/f0/ec_multiscale_f0"
ES_CLEAN = ROOT / "artifacts/f0_clean_v0913/es_spatial"
EC_CLEAN = ROOT / "artifacts/f0_clean_v0913/ec_multiscale"
F0_VARIANT = ROOT / "code/variants/cn2west_f0_rsifree/FontDiffuser"
F2_VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
CONTENT = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2/test/ContentImage"

STEPS = [5000, 40000, 80000]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def load_fonts() -> list[dict]:
    payload = json.loads(REC.read_text(encoding="utf-8"))
    return [r for r in payload["recommended"] if r.get("eye") == "pass"][:6]


def cmap_ok(path: Path, chars: list[str]) -> list[str]:
    f = TTFont(str(path), lazy=True, fontNumber=0)
    c = set((f.getBestCmap() or {}).keys())
    f.close()
    return [ch for ch in chars if ord(ch) not in c]


def render_font_pngs(rec: dict) -> dict:
    stem = rec["clean"]
    ttf = Path(rec["path"])
    miss = cmap_ok(ttf, REF8 + BOARD_CHARS)
    if miss:
        raise RuntimeError(f"{stem} missing cmap: {miss}")
    style_dir = DATA_PNG / "StyleImage" / stem
    gt_dir = DATA_PNG / "TargetImage" / stem
    style_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    all_chars = list(dict.fromkeys(REF8 + BOARD_CHARS))
    fs = find_size_A(str(ttf), all_chars)
    for ch in REF8:
        p = style_dir / f"{stem}+{cp_of(ch)}.png"
        if not p.is_file():
            render_glyph_AB(str(ttf), ch, fs).convert("RGB").save(p)
    for ch in BOARD_CHARS:
        p = gt_dir / f"{stem}+{cp_of(ch)}.png"
        if not p.is_file():
            render_glyph_AB(str(ttf), ch, fs).convert("RGB").save(p)
    return {"stem": stem, "fs": fs, "ttf": str(ttf), "disp": rec["disp"], "cat": rec["cat"]}


class LayeredEsCache:
    """Query fonts from overlay; train library from base."""

    def __init__(self, base: EsCache, overlay: EsCache):
        self.base = base
        self.overlay = overlay

    def spatial_tensor(self, split: str, font: str, cp: str) -> torch.Tensor:
        try:
            return self.overlay.spatial_tensor(split, font, cp)
        except KeyError:
            return self.base.spatial_tensor(split, font, cp)

    def pooled_tensor(self, split: str, font: str, cp: str) -> torch.Tensor:
        try:
            return self.overlay.pooled_tensor(split, font, cp)
        except KeyError:
            return self.base.pooled_tensor(split, font, cp)


def load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (96, 96):
        raise ValueError(path)
    array = np.array(image, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1


def purge_fontdiffuser_modules() -> None:
    doomed = [k for k in list(sys.modules) if k == "src" or k.startswith("src.") or k == "train"]
    for k in doomed:
        del sys.modules[k]


def build_es_overlay(name: str, ckpt: Path, stems: list[str], device: str) -> Path:
    out = OUT / f"es_overlay_{name}"
    jobs = []
    for stem in stems:
        for ch in REF8:
            cp = cp_of(ch)
            path = DATA_PNG / "StyleImage" / stem / f"{stem}+{cp}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            jobs.append((key_es(SPLIT, stem, cp), path))
    table = MemmapTable(out, create=True)
    keys = [k for k, _ in jobs]
    if (out / "keys.txt").is_file() and (out / "spatial.dat").is_file() and (out / "progress.json").is_file():
        prog = json.loads((out / "progress.json").read_text())
        if prog.get("done") == prog.get("total") == len(keys):
            existing = [ln for ln in (out / "keys.txt").read_text().splitlines() if ln]
            if existing == keys:
                print(f"overlay {name} ready n={len(keys)}", flush=True)
                return out
    table.set_keys(keys)
    n = len(keys)
    spatial = table.open_array("spatial", (n, *ES_SPATIAL), "w+")
    pooled = table.open_array("pooled", (n, *ES_POOLED), "w+")

    purge_fontdiffuser_modules()
    enc_root = str(ROOT / "code/variants/cn2west_stage_a/FontDiffuser")
    sys.path.insert(0, enc_root)
    from src.modules.style_encoder import StyleEncoder

    model = StyleEncoder(G_ch=64, resolution=96)
    model.load_state_dict(torch.load(ckpt / "style_encoder.pth", map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval().requires_grad_(False)

    bs = 16
    for i in range(0, n, bs):
        batch = jobs[i : i + bs]
        imgs = torch.stack([load_rgb(p) for _, p in batch]).to(device)
        with torch.no_grad():
            sp, po, _ = model(imgs)
            po = F.normalize(po.float(), dim=1)
        spatial[i : i + len(batch)] = sp.detach().half().cpu().numpy()
        pooled[i : i + len(batch)] = po.detach().half().cpu().numpy()
        print(f"  es {name} {min(i+bs,n)}/{n}", flush=True)
    table.flush()
    table.save_progress(n, n)
    table.save_manifest(
        {
            "kind": "es_overlay",
            "entries": n,
            "ckpt_dir": str(ckpt),
            "encoder_sha256": sha256_file(ckpt / "style_encoder.pth"),
            "stems": stems,
            "chars": REF8,
            "split": SPLIT,
            "created_at": utc_now(),
        }
    )
    # drop stage_a from path so F0/F2 variants load cleanly
    if enc_root in sys.path:
        sys.path.remove(enc_root)
    purge_fontdiffuser_modules()
    del model
    torch.cuda.empty_cache()
    return out


def to_tensor96(img: Image.Image, device: str):
    import torchvision.transforms as T

    t = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return t(img.convert("RGB"))[None].to(device)


def load_image_pipe(ckpt_dir: Path, device: str):
    purge_fontdiffuser_modules()
    sys.path.insert(0, str(F0_VARIANT))
    from src import (
        FontDiffuserDPMPipeline,
        FontDiffuserModelDPM,
        build_content_encoder,
        build_ddpm_scheduler,
        build_style_encoder,
        build_unet,
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
    model = FontDiffuserModelDPM(unet=unet, style_encoder=style_encoder, content_encoder=content_encoder).to(device)
    model.eval()
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=build_ddpm_scheduler(args),
        version="V3",
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5,
    )
    return pipe, args


def sample_f0(pipe, args, content: Image.Image, style: Image.Image, device: str) -> Image.Image:
    from accelerate.utils import set_seed

    set_seed(SEED)
    with torch.no_grad():
        images = pipe.generate(
            content_images=to_tensor96(content, device),
            style_images=to_tensor96(style, device),
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
    return images[0]


def run_f0_arm(mid: str, ckpt: Path, stems: list[str], device: str, overwrite: bool) -> None:
    print(f"F0 {mid} <- {ckpt}", flush=True)
    pipe, args = load_image_pipe(ckpt, device)
    for stem in stems:
        style = Image.open(DATA_PNG / "StyleImage" / stem / f"{stem}+{cp_of('永')}.png")
        for ch in BOARD_CHARS:
            out = OUT / "preds" / mid / stem / f"{stem}__{cp_of(ch)}__s{SEED}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.is_file() and not overwrite:
                continue
            content = Image.open(CONTENT / f"{cp_of(ch)}.png")
            sample_f0(pipe, args, content, style, device).save(out)
        print(f"  done {mid}/{stem}", flush=True)
    del pipe
    torch.cuda.empty_cache()


def cat_cond(a, b):
    out = []
    for x, y in zip(a, b):
        if x is None and y is None:
            out.append(None)
        elif isinstance(x, list):
            out.append([torch.cat([xi, yi], 0) for xi, yi in zip(x, y)])
        else:
            out.append(torch.cat([x, y], 0))
    return out


def run_f2_arm(
    prefix: str,
    run: Path,
    es_base: Path,
    es_overlay: Path,
    ec_dir: Path,
    stems: list[str],
    steps: list[int],
    device: str,
    overwrite: bool,
) -> None:
    purge_fontdiffuser_modules()
    # Ensure F0 variant is not preferred on sys.path
    f0s = str(F0_VARIANT)
    while f0s in sys.path:
        sys.path.remove(f0s)
    f2s = str(F2_VARIANT)
    while f2s in sys.path:
        sys.path.remove(f2s)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, f2s)
    os.chdir(F2_VARIANT)
    import train as T
    from accelerate.utils import set_seed
    from src.build import build_content_encoder, build_ddpm_scheduler, build_style_encoder, build_unet
    from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
    from src.model import FontDiffuserModel

    base = EsCache(es_base)
    overlay = EsCache(es_overlay)
    es = LayeredEsCache(base, overlay)
    ec = EcCache(ec_dir)
    split = json.loads((ROOT / "manifests/split_v3_228_16_16.json").read_text(encoding="utf-8"))
    train_fonts = sorted(split["stems"]["train"])
    library = T._LibraryEs(base, train_fonts, T._style_chars_from_cache(base))
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
    device_t = torch.device(device)
    fd = FontDiffuserModel(
        unet=build_unet(args),
        style_encoder=build_style_encoder(args),
        content_encoder=build_content_encoder(args),
    )
    T._ban_encoder_forward(fd)
    fd.to(device_t).eval()

    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_f03_test16_strat as E

    model = E.make_dpm_adapter(fd, torch).to(device_t).eval()
    scheduler = build_ddpm_scheduler(args)
    noise_schedule = NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)
    keep = torch.zeros(1, dtype=torch.bool, device=device_t)

    def pack_one(font: str, ch: str):
        samples = {
            "split": [SPLIT],
            "font_stem": [font],
            "char_cp": [cp_of(ch)],
            "ref_chars": [[cp_of(c) for c in REF8]],
        }
        style, queries, *_ = T._style_conditions(es, samples, device_t)
        structure = T._structure_features(es, ec, library, samples, queries, cfg, keep, device_t)
        content = T._content_features(ec, samples, keep, device_t)
        return {"style": style, "structure": structure, "content": content}

    packed = {(stem, ch): pack_one(stem, ch) for stem in stems for ch in BOARD_CHARS}
    print(f"packed F2 conditions {len(packed)}", flush=True)

    for step in steps:
        mid = f"{prefix}{step}"
        ckpt = run / f"global_step_{step}"
        print(f"F2 {mid} <- {ckpt}", flush=True)
        fd.unet.load_state_dict(torch.load(ckpt / "unet.pth", map_location="cpu", weights_only=True))
        fd.style_encoder.load_state_dict(torch.load(ckpt / "style_encoder.pth", map_location="cpu", weights_only=True))
        fd.content_encoder.load_state_dict(torch.load(ckpt / "content_encoder.pth", map_location="cpu", weights_only=True))
        fd.to(device_t).eval()

        for stem in stems:
            for ch in BOARD_CHARS:
                out = OUT / "preds" / mid / stem / f"{stem}__{cp_of(ch)}__s{SEED}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.is_file() and not overwrite:
                    continue
                p = packed[(stem, ch)]
                set_seed(SEED)
                img = torch.zeros(1, 3, 96, 96, device=device_t)
                cond = [img, img, p["style"], p["structure"], p["content"], None]
                uncond = [
                    torch.ones_like(img),
                    torch.ones_like(img),
                    torch.zeros_like(p["style"]),
                    [torch.zeros_like(x) for x in p["structure"]],
                    [torch.zeros_like(x) for x in p["content"]],
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
                        cat_cond(uncond, cond),
                        content_encoder_downsample_size=3,
                        version="V3",
                    ).chunk(2)
                    return noise_uncond + 7.5 * (noise - noise_uncond)

                solver = DPM_Solver(model_fn=model_fn, noise_schedule=noise_schedule, algorithm_type="dpmsolver++")
                x = torch.randn(1, 3, 96, 96, device=device_t)
                with torch.no_grad():
                    x = solver.sample(x=x, steps=20, order=2, skip_type="time_uniform", method="multistep")
                x = (x / 2 + 0.5).clamp(0, 1)[0].detach().cpu().permute(1, 2, 0).numpy()
                Image.fromarray((x * 255).round().astype("uint8")).save(out)
            print(f"  done {mid}/{stem}", flush=True)
    torch.cuda.empty_cache()


def resolve_split(stem: str, by_split: dict[str, set[str]]) -> dict:
    """Label train/val/test. New fonts (not in 260) → test (added OOD)."""
    for sp in ("train", "val", "test"):
        if stem in by_split[sp]:
            return {
                "split": sp,
                "in_260": True,
                "origin": "split_v3_228_16_16",
                "split_label": {"train": "训练集", "val": "验证集", "test": "测试集"}[sp],
            }
    return {
        "split": "test",
        "in_260": False,
        "origin": "added_ood_paper6_0914",
        "split_label": "测试集（新增）",
    }


def write_html(meta: list[dict]) -> None:
    cols = [
        ("ref", "Ref8", None),
        ("gt", "GT", None),
        ("F0_100000", "F0脏@100k", "dirty"),
        ("F0C_95000", "F0净@95k", "clean"),
        ("F0C_100000", "F0净@100k", "clean"),
    ]
    for s in STEPS:
        cols.append((f"F2_{s}", f"F2脏@{s//1000}k", "dirty"))
        cols.append((f"F2C_{s}", f"F2净@{s//1000}k", "clean"))

    sections = []
    for rec in meta:
        stem = rec["stem"]
        split_lab = rec.get("split_label") or "测试集（新增）"
        split_tag = rec.get("split") or "test"
        badge = f"<span class='badge {split_tag}'>{split_lab}</span>"
        rows = []
        for ch in BOARD_CHARS:
            cells = [f"<td class='ch'>{ch}</td>"]
            refs = "".join(
                f"<img src='png/StyleImage/{stem}/{stem}+{cp_of(c)}.png' title='{c}'>" for c in REF8[:4]
            )
            cells.append(f"<td class='ref'>{refs}</td>")
            gt = f"png/TargetImage/{stem}/{stem}+{cp_of(ch)}.png"
            cells.append(f"<td><img src='{gt}'></td>")
            for mid, _lab, _kind in cols[2:]:
                p = f"preds/{mid}/{stem}/{stem}__{cp_of(ch)}__s{SEED}.png"
                if (OUT / p).is_file():
                    cells.append(f"<td class='{_kind}'><img src='{p}'></td>")
                else:
                    cells.append(f"<td class='miss {_kind}'>—</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        head = "".join(
            f"<th class='{kind or ''}'>{lab}</th>" for _mid, lab, kind in [("ch", "char", None)] + cols
        )
        sections.append(
            f"<h2 id='{stem}'>{badge} {rec['disp']} <code>{stem}</code> · {rec['cat']}</h2>"
            f"<div class='wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )

    split_rows = "".join(
        f"<tr><td>{m.get('split_label','')}</td><td><code>{m['stem']}</code></td>"
        f"<td>{m['disp']}</td><td>{'260内' if m.get('in_260') else '新增→测试集'}</td></tr>"
        for m in meta
    )
    nav = " · ".join(
        f"<a href='#{m['stem']}'>{m.get('split_label','测试集（新增）')} · {m['disp']}</a>" for m in meta
    )
    html = f"""<!doctype html>
<meta charset=utf-8>
<title>Paper6 · F0/F2 dirty vs CLEAN</title>
<style>
:root{{--dirty:#8b3a2a;--clean:#1f5a3a}}
body{{font:13px/1.4 system-ui,sans-serif;margin:16px;background:#eef1f5;color:#1a1a1a}}
h1{{font-size:1.2rem;margin:0 0 8px}} h2{{font-size:1rem;margin:28px 0 8px}}
.note{{color:#5c6570;max-width:1100px}} a{{color:#1f4a6f}}
.wrap{{overflow:auto;border:1px solid #d5dbe3;background:#fff;margin-bottom:8px}}
table{{border-collapse:separate;border-spacing:0}}
th,td{{border:1px solid #d5dbe3;padding:3px;text-align:center;vertical-align:bottom;background:#fff}}
th{{position:sticky;top:0;background:#f7f9fb;z-index:2;font-size:11px}}
th.dirty,td.dirty{{background:#fffaf8}} th.clean,td.clean{{background:#f7fcf8}}
th.dirty{{color:var(--dirty)}} th.clean{{color:var(--clean)}}
td.ch{{font:700 14px ui-serif,serif;min-width:1.6em}}
img{{width:72px;height:72px;image-rendering:pixelated;display:block;margin:0 auto;background:#fff}}
td.ref img{{width:36px;height:36px;display:inline-block}}
.miss{{color:#999}}
.badge{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;margin-right:6px;font-weight:600;vertical-align:middle}}
.badge.train{{background:#e8eef7;color:#1f4a6f}}
.badge.val{{background:#f3eef8;color:#5a3d7a}}
.badge.test{{background:#e7f5ec;color:#1f5a3a}}
.sum{{border-collapse:collapse;margin:10px 0;background:#fff}}
.sum th,.sum td{{border:1px solid #d5dbe3;padding:6px 8px;text-align:left;font-size:12px}}
.sum th{{background:#f7f9fb}}
</style>
<h1>论文 6 套 · F0/F2 脏臂 vs CLEAN（非 G 系）</h1>
<p class="note">生成 {utc_now()} · 协议对齐 timeline_f2_clean：DPM++20 CFG7.5 seed3407 · 8-shot Es+Δ · F0 单 style「永」。
<strong>不用</strong>合作者 G2/G2-RL。</p>
<p class="note"><b>划分约定：</b>原 260 = 训练 228 / 验证 16 / 测试 16（<code>split_v3_228_16_16</code>）。
本页 6 套均<strong>不在原 260</strong>，按约定<strong>归入测试集（新增）</strong>，与原 test16 并列作 OOD 展示，不进训练/验证。</p>
<table class="sum">
<thead><tr><th>划分</th><th>stem</th><th>字体</th><th>来源</th></tr></thead>
<tbody>{split_rows}</tbody>
</table>
<p class="note">导航：{nav}</p>
{''.join(sections)}
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print("wrote", OUT / "index.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--f2-only", action="store_true", help="skip F0 (reuse existing preds)")
    ap.add_argument("--steps", default="5000,40000,80000")
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preds").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    fonts = load_fonts()
    data = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
    by_split = {
        sp: {p.name for p in (data / sp / "TargetImage").iterdir() if p.is_dir()}
        for sp in ("train", "val", "test")
    }
    membership = []
    for r in fonts:
        info = resolve_split(r["clean"], by_split)
        membership.append(
            {
                "clean": r["clean"],
                "disp": r["disp"],
                "cat": r.get("cat"),
                **info,
            }
        )
    (OUT / "membership.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "base_split": "split_v3_228_16_16",
                    "train_n": len(by_split["train"]),
                    "val_n": len(by_split["val"]),
                    "test_n_original": len(by_split["test"]),
                    "rule": "原260按 train/val/test 标明；不在260的新增字体一律归为测试集（新增）",
                },
                "fonts": membership,
                "test_added": [m["clean"] for m in membership if not m["in_260"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    meta = []
    for r, mem in zip(fonts, membership):
        print("render", r["clean"], flush=True)
        row = render_font_pngs(r)
        row.update(
            {
                "split": mem["split"],
                "split_label": mem["split_label"],
                "in_260": mem["in_260"],
                "origin": mem["origin"],
            }
        )
        meta.append(row)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.html_only:
        write_html(meta)
        return

    stems = [m["stem"] for m in meta]
    device = args.device
    ov_dirty = build_es_overlay("dirty", F0_DIRTY, stems, device)
    ov_clean = build_es_overlay("clean", F0_CLEAN_95, stems, device)

    if not args.f2_only:
        run_f0_arm("F0_100000", F0_DIRTY, stems, device, args.overwrite)
        run_f0_arm("F0C_95000", F0_CLEAN_95, stems, device, args.overwrite)
        run_f0_arm("F0C_100000", F0_CLEAN_100, stems, device, args.overwrite)

    run_f2_arm("F2_", F2_DIRTY, ES_DIRTY, ov_dirty, EC_DIRTY, stems, steps, device, args.overwrite)
    run_f2_arm("F2C_", F2_CLEAN, ES_CLEAN, ov_clean, EC_CLEAN, stems, steps, device, args.overwrite)

    write_html(meta)
    print("OPEN:")
    print(f"  http://127.0.0.1:8780/paper6_0914_f0f2/")
    print(f"  http://172.19.45.13:19000/reports/paper6_0914_f0f2/")
    print(f"  (8791 is f03 board only; this page is under reports/)")


if __name__ == "__main__":
    main()
