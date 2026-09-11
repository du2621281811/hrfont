#!/usr/bin/env python3
"""F3 checkpoint visual dashboard.

Samples named ckpts with the same frozen Es/Ec Δ + Support conditions used in
training (no source/support/CFG drop). Uses a custom CFG path because the
stock V3 DPM wrapper only concatenates the two image slots and would silently
drop Δ / Support.

Runs on a non-training GPU (default CUDA_VISIBLE_DEVICES=3). Do not point this
at GPU2 while F3 is exclusive.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path("/root/projects/hrfont")
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
RUN = ROOT / "runs/F3-JOINT-DS-A-S3407"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
SPLIT = ROOT / "manifests/split_v3_228_16_16.json"
TEST_STEMS = ROOT / "manifests/pipeline_v3_test_stems.txt"
OUT = ROOT / "reports/f3_ckpt_dashboard"
F0_PREVIEW = ROOT / "reports/preview_f0_100k"
REF8 = [f"u{ord(c):04X}" for c in "永和书风骨韵天地"]
PROBE = list("AaG0e")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def content_png(ch: str) -> Path:
    for sp in ("test", "val", "train"):
        p = DATA / sp / "ContentImage" / f"{cp_of(ch)}.png"
        if p.is_file():
            return p
    raise FileNotFoundError(ch)


def gt_png(split: str, stem: str, ch: str) -> Path | None:
    p = DATA / split / "TargetImage" / stem / f"{stem}+{cp_of(ch)}.png"
    return p if p.is_file() else None


def style_png(split: str, stem: str) -> Path:
    d = DATA / split / "StyleImage" / stem
    for ch in "永和书风骨韵天地":
        p = d / f"{stem}+{cp_of(ch)}.png"
        if p.is_file():
            return p
    pngs = sorted(d.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(d)
    return pngs[0]


class DPMAdapter(torch.nn.Module):
    """DPM cond-list → FontDiffuserModel cache-feature forward (returns noise only)."""

    def __init__(self, fd, device: torch.device):
        super().__init__()
        self.fd = fd
        self.device = device

    def forward(self, x_t, timesteps, cond, content_encoder_downsample_size, version):
        noise, _ = self.fd(
            x_t,
            timesteps,
            content_images=cond[0],
            content_encoder_downsample_size=content_encoder_downsample_size,
            style_features=cond[2],
            structure_features=cond[3],
            content_features=cond[4],
            support_tokens=cond[5],
        )
        return noise


def cat_cond(uncond, cond):
    out = []
    for u, c in zip(uncond, cond):
        if u is None and c is None:
            out.append(None)
        elif isinstance(u, list):
            out.append([torch.cat([a, b], dim=0) for a, b in zip(u, c)])
        else:
            out.append(torch.cat([u, c], dim=0))
    return out


def import_train():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(VARIANT))
    os.chdir(VARIANT)
    import train as T
    from scripts.hrfont_feature_cache import EcCache, EsCache
    from scripts.hrfont_support_adapter import SupportAdapter
    from src.build import (
        build_content_encoder,
        build_ddpm_scheduler,
        build_style_encoder,
        build_unet,
    )
    from src.dpm_solver.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP
    from src.model import FontDiffuserModel
    return SimpleNamespace(
        T=T, EcCache=EcCache, EsCache=EsCache, SupportAdapter=SupportAdapter,
        build_content_encoder=build_content_encoder,
        build_ddpm_scheduler=build_ddpm_scheduler,
        build_style_encoder=build_style_encoder,
        build_unet=build_unet,
        DPM_Solver=DPM_Solver, NoiseScheduleVP=NoiseScheduleVP,
        FontDiffuserModel=FontDiffuserModel,
    )


def load_caches(M):
    es = M.EsCache(ROOT / "artifacts/f0/es_spatial_f0")
    ec = M.EcCache(ROOT / "artifacts/f0/ec_multiscale_f0")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_fonts = sorted(split["stems"]["train"])
    print(f"building LibraryEs fonts={len(train_fonts)} …", flush=True)
    library = M.T._LibraryEs(es, train_fonts, M.T._style_chars_from_cache(es))
    bank = M.T._load_support_bank(
        str(ROOT / "artifacts/f0/support_bank.json"),
        SimpleNamespace(support=True),
    )
    cfg = SimpleNamespace(
        rsi_source="delta", delta_enabled=True, delta_tau=0.07,
        delta_eps_alpha=0.01, delta_k_max=10, delta_k_top=10,
        delta_mode="topk", seed=3407, support=True, support_k=8,
        style_start_channel=64,
    )
    return es, ec, library, bank, cfg


def pool_support(M, ec, bank, cfg, device, font, cp):
    vecs = []
    for scp in list(bank.get(cp, []))[: cfg.support_k]:
        try:
            feats = [x.to(device) for x in ec.features("style", font, scp)]
        except KeyError:
            continue
        vecs.append(M.T._pool_ec(feats))
    if not vecs:
        return None
    return torch.stack(vecs, dim=0).unsqueeze(0)


def precompute_conditions(M, es, ec, library, bank, cfg, device, split, fonts, chars):
    keep = torch.zeros(1, dtype=torch.bool, device=device)
    packed = {}
    for font in fonts:
        for ch in chars:
            samples = {
                "split": [split],
                "font_stem": [font],
                "char_cp": [cp_of(ch)],
                "ref_chars": [REF8],
            }
            style, queries, *_ = M.T._style_conditions(es, samples, device)
            structure = M.T._structure_features(
                es, ec, library, samples, queries, cfg, keep, device)
            content = M.T._content_features(ec, samples, keep, device)
            pooled = pool_support(M, ec, bank, cfg, device, font, cp_of(ch))
            packed[(font, ch)] = {
                "style": style,
                "structure": structure,
                "content": content,
                "support_pooled": pooled,
            }
            print(f"  cond {font} {ch} support_k="
                  f"{0 if pooled is None else pooled.shape[1]}", flush=True)
    return packed


def load_model(M, ckpt: Path, device: torch.device, cfg):
    args = SimpleNamespace(
        resolution=96, unet_channels=(64, 128, 256, 512),
        style_image_size=(96, 96), content_image_size=(96, 96),
        content_encoder_downsample_size=3, channel_attn=True,
        content_start_channel=64, style_start_channel=64,
        beta_scheduler="scaled_linear",
    )
    fd = M.FontDiffuserModel(
        unet=M.build_unet(args),
        style_encoder=M.build_style_encoder(args),
        content_encoder=M.build_content_encoder(args),
    )
    fd.unet.load_state_dict(torch.load(ckpt / "unet.pth", map_location="cpu", weights_only=True))
    fd.style_encoder.load_state_dict(
        torch.load(ckpt / "style_encoder.pth", map_location="cpu", weights_only=True))
    fd.content_encoder.load_state_dict(
        torch.load(ckpt / "content_encoder.pth", map_location="cpu", weights_only=True))
    adapter = M.SupportAdapter(sum(M.T.EC_SCALE_CHANNELS), cfg.style_start_channel * 16)
    payload = torch.load(ckpt / "support_adapter.pth", map_location="cpu", weights_only=True)
    adapter.load_state_dict(payload)
    fd.support_adapter = adapter
    M.T._ban_encoder_forward(fd)
    fd.to(device).eval()
    wrapped = DPMAdapter(fd, device).to(device).eval()
    scheduler = M.build_ddpm_scheduler(args)
    noise_schedule = M.NoiseScheduleVP(schedule="discrete", betas=scheduler.betas)
    return wrapped, fd, noise_schedule, args


def sample_one(M, model, noise_schedule, packed, adapter, device, seed: int, cfg_scale=7.5, steps=20):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    img = torch.zeros(1, 3, 96, 96, device=device)
    style = packed["style"]
    structure = packed["structure"]
    content = packed["content"]
    pooled = packed["support_pooled"]
    support = adapter(pooled) if pooled is not None else None
    cond = [img, img, style, structure, content, support]
    uncond = [
        torch.ones_like(img),
        torch.ones_like(img),
        torch.zeros_like(style),
        [torch.zeros_like(x) for x in structure],
        [torch.zeros_like(x) for x in content],
        torch.zeros_like(support) if support is not None else None,
    ]

    def get_t_input(t_continuous):
        return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0

    def model_fn(x, t_continuous):
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t_continuous, t_continuous], dim=0)
        t_input = get_t_input(t_in)
        c_in = cat_cond(uncond, cond)
        noise_uncond, noise = model(
            x_in, t_input, c_in,
            content_encoder_downsample_size=3, version="V3",
        ).chunk(2)
        return noise_uncond + cfg_scale * (noise - noise_uncond)

    solver = M.DPM_Solver(model_fn=model_fn, noise_schedule=noise_schedule, algorithm_type="dpmsolver++")
    x = torch.randn(1, 3, 96, 96, device=device)
    with torch.no_grad():
        x = solver.sample(x=x, steps=steps, order=2, skip_type="time_uniform", method="multistep")
    x = (x / 2 + 0.5).clamp(0, 1)[0].detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((x * 255).round().astype("uint8"))


def last_state_step() -> int:
    """last_state is written every 1000 steps; use the latest logged multiple of 1000."""
    rows = [json.loads(l) for l in (RUN / "train_log.jsonl").read_text().splitlines() if l.strip()]
    if not rows:
        return 0
    return (int(rows[-1]["step"]) // 1000) * 1000


def plot_loss(out: Path) -> None:
    rows = [json.loads(l) for l in (RUN / "train_log.jsonl").read_text().splitlines() if l.strip()]
    valp = RUN / "val_log.jsonl"
    val = [json.loads(l) for l in valp.read_text().splitlines() if l.strip()] if valp.is_file() else []
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    ax.plot(steps, losses, color="#1f4a6f", lw=1.05, alpha=0.75, label="train")
    if len(losses) >= 8:
        w = 9
        ma = np.convolve(losses, np.ones(w) / w, mode="valid")
        ax.plot(steps[w - 1 :], ma, color="#7aa0c4", lw=1.6, label=f"train MA-{w}")
    if val:
        ax.plot([r["step"] for r in val], [r["val_loss"] for r in val],
                color="#c45c26", lw=1.8, marker="o", ms=5, label="val (no perceptual)")
    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.set_title("F3-JOINT-DS-A-S3407")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out / "loss.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_html(out: Path, tags: list[dict], fonts: list[str], chars: list[str], meta: dict) -> None:
    val_rows = ""
    for r in meta.get("val", []):
        best = r.get("best") or {}
        mark = " ← best" if best.get("step") == r["step"] else ""
        val_rows += (
            f"<tr><td>{r['step']}</td><td>{r['val_loss']:.6f}</td>"
            f"<td>{best.get('step')}</td><td>{mark}</td></tr>"
        )
    hb = meta.get("heartbeat") or {}
    status = (
        f"训练 {hb.get('status','?')} · heartbeat step {hb.get('step')} · "
        f"loss {hb.get('loss')} · {hb.get('ts')}"
    )
    font_opts = "".join(f'<option value="{f}">{f}</option>' for f in fonts)
    char_opts = "".join(f'<option value="{c}">{c}</option>' for c in chars)
    tag_js = json.dumps([{"step": t["step"], "tag": t["tag"]} for t in tags])
    sheets = "".join(
        f'<p class="meta">{t["tag"]}</p><img class="loss" src="{t["tag"]}/contact_sheet.png" alt="{t["tag"]}"/>'
        for t in tags
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>F3 checkpoint 效果看板</title>
<style>
:root{{--bg:#eef1f5;--card:#fff;--line:#d5dbe3;--muted:#5c6570;--ink:#1a1a1a;--accent:#1f4a6f}}
*{{box-sizing:border-box}} body{{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink)}}
header{{background:var(--card);border-bottom:1px solid var(--line);padding:14px 18px}}
h1{{margin:0;font-size:1.25rem}} .meta{{color:var(--muted);font-size:12px;margin-top:4px}}
.note{{background:#fff8e8;border:1px solid #e6d7a8;padding:8px 10px;margin:10px 0;border-radius:4px;font-size:12px}}
main{{max-width:1180px;margin:16px auto;padding:0 14px 48px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px;margin:14px 0}}
.card h2{{margin:0 0 8px;font-size:1.05rem}}
img.loss{{width:100%;height:auto}}
.refs,.timeline{{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end}}
.timeline-wrap{{overflow-x:auto;border:1px solid var(--line);background:#f7f9fb;padding:10px}}
figure{{margin:0;text-align:center}} figcaption{{font-size:11px;color:var(--muted);margin-top:3px;font-family:ui-monospace,monospace}}
img.g{{width:96px;height:96px;image-rendering:pixelated;border:1px solid var(--line);background:#fff;display:block}}
select{{padding:6px 8px;border:1px solid var(--line);background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left}}
.ok{{color:#1b6b3a}}
</style></head><body>
<header>
  <h1>F3-JOINT-DS-A-S3407 checkpoint 效果</h1>
  <div class="meta">{status} · 看板生成 {meta['generated_at']}</div>
</header>
<main>
<div class="note">采样条件与训练 val 一致：F0@100k 的 Es/Ec cache、Δ top-k、同字体 Support×8、无 drop。
DPM++ 20 / CFG 7.5 / seed 3407。官方 V3 DPM wrapper 会丢掉 Δ/Support，本页用自定义 CFG。
trainer val <b>不含</b> perceptual，不能直接和 F0 扫描的 0.031 比。F0@100k 列仅 overlapping 字有图。</div>
<section class="card">
  <h2>Train / Val loss</h2>
  <img class="loss" src="loss.png" alt="loss"/>
  <table><thead><tr><th>step</th><th>val_loss</th><th>best step</th><th></th></tr></thead>
  <tbody>{val_rows}</tbody></table>
</section>
<section class="card">
  <h2>多样本时间步（Pred）</h2>
  <p class="meta">test split · 字 {" ".join(chars)} · 先看字形是否立住，再看风格是否贴 GT</p>
  <p><label>字体 <select id="font">{font_opts}</select></label>
     <label>字 <select id="ch">{char_opts}</select></label></p>
  <div class="refs" id="refs"></div>
  <div class="timeline-wrap"><div class="timeline" id="timeline"></div></div>
</section>
<section class="card">
  <h2>各 ckpt contact sheet</h2>
  {sheets}
</section>
</main>
<script>
const TAGS = {tag_js};
function cp(ch){{return 'u'+ch.codePointAt(0).toString(16).toUpperCase().padStart(4,'0');}}
function fig(src, cap){{
  return `<figure><img class="g" src="${{src}}" onerror="this.style.opacity=0.15"/><figcaption>${{cap}}</figcaption></figure>`;
}}
function render(){{
  const font=document.getElementById('font').value;
  const ch=document.getElementById('ch').value;
  const c=cp(ch);
  document.getElementById('refs').innerHTML =
    fig(`refs/content/${{c}}.png`,'Content '+ch)+
    fig(`refs/style/${{font}}.png`,'Style 永…')+
    fig(`refs/gt/${{font}}+${{c}}.png`,'GT')+
    fig(`refs/f0/${{font}}+${{c}}.png`,'F0@100k');
  document.getElementById('timeline').innerHTML = TAGS.map(t=>
    fig(`preds/${{t.tag}}/${{font}}+${{c}}.png`, t.tag)).join('');
}}
document.getElementById('font').onchange=render;
document.getElementById('ch').onchange=render;
render();
</script></body></html>
"""
    (out / "index.html").write_text(html, encoding="utf-8")


def contact_sheet(fonts, chars, rows_by_font, title: str) -> Image.Image:
    cell, pad, head, left = 96, 6, 26, 170
    n_rows = len(fonts) * 2
    w = left + len(chars) * (cell + pad) + pad
    h = head + n_rows * (cell + pad) + pad
    sheet = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), title, fill="black")
    for j, ch in enumerate(chars):
        d.text((left + j * (cell + pad) + 40, 10), ch, fill="black")
    i = 0
    for stem in fonts:
        gts, preds = rows_by_font[stem]
        for name, imgs in ((f"{stem} GT", gts), (f"{stem} Pred", preds)):
            y = head + i * (cell + pad)
            d.text((8, y + 40), name[:24], fill="black")
            for j, im in enumerate(imgs):
                sheet.paste(im.resize((cell, cell)), (left + j * (cell + pad), y))
            i += 1
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fonts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"device={device} cuda={torch.cuda.is_available()}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("preds", "refs/content", "refs/style", "refs/gt", "refs/f0"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    fonts = [l.strip() for l in TEST_STEMS.read_text().splitlines() if l.strip()][: args.fonts]
    chars = PROBE
    split = "test"

    plot_loss(OUT)
    val = []
    if (RUN / "val_log.jsonl").is_file():
        val = [json.loads(l) for l in (RUN / "val_log.jsonl").read_text().splitlines() if l.strip()]
    heartbeat = {}
    if (RUN / "heartbeat.json").is_file():
        heartbeat = json.loads((RUN / "heartbeat.json").read_text(encoding="utf-8"))

    for ch in chars:
        Image.open(content_png(ch)).convert("RGB").resize((96, 96)).save(
            OUT / "refs/content" / f"{cp_of(ch)}.png")
    for stem in fonts:
        Image.open(style_png(split, stem)).convert("RGB").resize((96, 96)).save(
            OUT / "refs/style" / f"{stem}.png")
        for ch in chars:
            g = gt_png(split, stem, ch)
            dst = OUT / "refs/gt" / f"{stem}+{cp_of(ch)}.png"
            if g:
                Image.open(g).convert("RGB").resize((96, 96)).save(dst)
            else:
                Image.new("RGB", (96, 96), (245, 245, 245)).save(dst)
            f0 = F0_PREVIEW / f"{stem}+{cp_of(ch)}.png"
            if f0.is_file():
                Image.open(f0).convert("RGB").resize((96, 96)).save(
                    OUT / "refs/f0" / f"{stem}+{cp_of(ch)}.png")

    M = import_train()
    es, ec, library, bank, cfg = load_caches(M)
    print("precomputing Δ+Support conditions …", flush=True)
    packed = precompute_conditions(M, es, ec, library, bank, cfg, device, split, fonts, chars)

    last_step = last_state_step()
    ckpts = [
        (5000, RUN / "global_step_5000", "step5000"),
        (10000, RUN / "global_step_10000", "step10000"),
        (15000, RUN / "global_step_15000", "step15000"),
        (20000, RUN / "global_step_20000", "step20000"),
        (last_step, RUN / "last_state", f"step{last_step}_last"),
    ]
    ckpts = [(s, p, t) for s, p, t in ckpts if (p / "unet.pth").is_file()]

    tags = []
    for step, ckpt, tag in ckpts:
        print(f"=== {tag} {ckpt} ===", flush=True)
        pred_dir = OUT / "preds" / tag
        pred_dir.mkdir(parents=True, exist_ok=True)
        model, fd, noise_schedule, _ = load_model(M, ckpt, device, cfg)
        rows = {}
        for stem in fonts:
            gts, preds = [], []
            for ch in chars:
                try:
                    pred = sample_one(
                        M, model, noise_schedule, packed[(stem, ch)],
                        fd.support_adapter, device, args.seed)
                except Exception as exc:
                    print(f"  FAIL {stem} {ch}: {exc}", flush=True)
                    pred = Image.new("RGB", (96, 96), (180, 40, 40))
                pred.save(pred_dir / f"{stem}+{cp_of(ch)}.png")
                preds.append(pred)
                g = gt_png(split, stem, ch)
                gts.append(
                    Image.open(g).convert("RGB").resize((96, 96))
                    if g else Image.new("RGB", (96, 96), "white"))
                print(f"  {stem} {ch}", flush=True)
            rows[stem] = (gts, preds)
        tag_dir = OUT / tag
        tag_dir.mkdir(exist_ok=True)
        contact_sheet(fonts, chars, rows, f"F3 {tag}").save(tag_dir / "contact_sheet.png")
        tags.append({"step": step, "tag": tag, "ckpt": str(ckpt)})
        del model, fd
        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": str(RUN),
        "fonts": fonts,
        "chars": chars,
        "seed": args.seed,
        "sampler": "dpmsolver++ 20 CFG7.5, cache Δ+Support, custom CFG (not stock V3 wrapper)",
        "val": val,
        "tags": tags,
        "heartbeat": heartbeat,
        "note": "trainer val has no perceptual term; not comparable to F0 scan 0.031",
    }
    (OUT / "SNAPSHOT.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_html(OUT, tags, fonts, chars, meta)
    (OUT / "README.md").write_text(
        "# F3 checkpoint 效果看板\n\n"
        "- 训练机：http://127.0.0.1:8777/f3_ckpt_dashboard/\n"
        "- 或 reports 服务：http://127.0.0.1:8765/f3_ckpt_dashboard/\n"
        "- 训练进度实时页仍是 http://127.0.0.1:8787/\n",
        encoding="utf-8",
    )
    hub = ROOT / "data/f3_ckpt_dashboard"
    if hub.is_symlink() or hub.exists():
        if hub.is_symlink() or hub.is_file():
            hub.unlink()
        elif hub.is_dir() and not any(hub.iterdir()):
            hub.rmdir()
    if not hub.exists():
        hub.symlink_to(OUT)
    print("wrote", OUT / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
