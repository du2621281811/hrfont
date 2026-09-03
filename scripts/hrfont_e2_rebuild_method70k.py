#!/usr/bin/env python3
"""Sample StageA@70k + ft for 4x8 method grids. GPU3."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HRFONT_PREVIEW_GPU", "3"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path("/root/projects/hrfont")
BANK = ROOT / "data/hrfont/e0_bank"
FT = ROOT / "runs/ft_cnstyle/global_step_25000"
STAGEA = ROOT / "runs/e2_stageA96"
OUT = ROOT / "reports/hrfont_overnight/mentor_preview"
REF8 = list("永和书风骨韵天地")
CHARS = list("AaOoRg8S")
SIZE = 96
CELL = 64
TAU = 0.07
TOP_M = 3
STEPS = 25
SEED = 42
BEST = 70000

def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)

def png(font, ch):
    return BANK / "r96" / font / f"u{ord(ch):04X}.png"

def load_im(path):
    tfm = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    return tfm(Image.open(path).convert("RGB"))

def to_pil(t):
    x = t.detach().float().cpu()
    if x.dim() == 4: x = x[0]
    x = (x.clamp(-1, 1) + 1) * 0.5
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

def load_pt(path):
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")

def main():
    device = torch.device("cuda:0")
    meta = json.loads((BANK / "meta.json").read_text())
    demo = list(meta["demo8"])
    train = [f for f in meta["train_fonts"] if f not in set(demo)]
    bank_proto = load_pt(BANK / "cache/ec_es_r96.pt")["style_proto"]
    fonts = [f for f in demo if sum(png(f, c).exists() for c in CHARS) >= 6][:4]
    log(f"fonts={fonts} gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    sys.path.insert(0, str(ROOT / "code/ours/FontDiffuser"))
    from configs.fontdiffuser import get_parser
    from src import build_unet, build_style_encoder, build_content_encoder, build_ddpm_scheduler
    args = get_parser().parse_args([])
    args.resolution = SIZE
    args.content_image_size = (SIZE, SIZE)
    args.style_image_size = (SIZE, SIZE)
    args.unet_channels = (64, 128, 256, 512)
    args.channel_attn = True
    args.content_encoder_downsample_size = 3
    args.content_start_channel = 64
    args.style_start_channel = 64
    args.beta_scheduler = "scaled_linear"
    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    se.load_state_dict(load_pt(FT / "style_encoder.pth")); se.to(device).eval()
    ce.load_state_dict(load_pt(FT / "content_encoder.pth")); ce.to(device).eval()

    def style_proto(font):
        feats = []
        for r in REF8:
            p = png(font, r)
            if not p.exists(): continue
            feat, _, _ = se(load_im(p).unsqueeze(0).to(device))
            feats.append(feat.flatten(1).mean(0).cpu())
        return torch.stack(feats).mean(0)

    def alpha_mix(proto_q, candidates, ch):
        scores = []
        for g in candidates:
            if g not in bank_proto or not png(g, ch).exists(): continue
            v = bank_proto[g].float().flatten()
            scores.append((g, float(F.cosine_similarity(proto_q.flatten()[None], v[None]).item())))
        scores.sort(key=lambda x: -x[1])
        top = scores[:TOP_M]
        w = torch.softmax(torch.tensor([s / TAU for _, s in top]), dim=0)
        mix = None
        for i, (g, _) in enumerate(top):
            im = load_im(png(g, ch))
            mix = im * float(w[i]) if mix is None else mix + im * float(w[i])
        return mix

    @torch.no_grad()
    def sample(unet, content, style, struct_img, use_delta, generator):
        style_feat, _, _ = se(style)
        b, c, h, w = style_feat.shape
        style_hidden = style_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        content_feat, content_res = ce(content)
        struct_feat, struct_res = ce(struct_img)
        content_res = list(content_res) + [content_feat]
        struct_res = list(struct_res) + [struct_feat]
        if use_delta:
            struct_res = [s - c_ for s, c_ in zip(struct_res, content_res)]
        hidden = [style_feat, content_res, style_hidden, struct_res]
        sched = build_ddpm_scheduler(args)
        sched.set_timesteps(STEPS, device=device)
        x = torch.randn(content.shape, device=device, generator=generator)
        for t in sched.timesteps:
            out = unet(x, t.expand(x.shape[0]), encoder_hidden_states=hidden, content_encoder_downsample_size=args.content_encoder_downsample_size)
            x = sched.step(out[0], t, x).prev_sample
        return x

    jobs = []
    for font in fonts:
        proto = style_proto(font)
        style_ch = next(r for r in REF8 if png(font, r).exists())
        style = load_im(png(font, style_ch))
        others = [g for g in train if g != font]
        for ch in CHARS:
            if not png(font, ch).exists() or not png("_B0_", ch).exists(): continue
            jobs.append({"font": font, "ch": ch, "content": load_im(png("_B0_", ch)), "style": style, "gt": load_im(png(font, ch)), "mix": alpha_mix(proto, others, ch)})
    log(f"jobs={len(jobs)}")

    unet_ft = build_unet(args=args); unet_ft.load_state_dict(load_pt(FT / "unet.pth")); unet_ft.to(device).eval()
    ft_preds = {}
    for j, job in enumerate(jobs):
        g = torch.Generator(device=device); g.manual_seed(SEED + j * 17)
        pred = sample(unet_ft, job["content"].unsqueeze(0).to(device), job["style"].unsqueeze(0).to(device), job["style"].unsqueeze(0).to(device), False, g)
        ft_preds[(job["font"], job["ch"])] = pred.cpu()
        job["l1_ft"] = float((pred.cpu() - job["gt"]).abs().mean())
    del unet_ft; torch.cuda.empty_cache()
    log(f"ft L1={float(np.mean([j['l1_ft'] for j in jobs])):.4f}")

    blob = load_pt(STAGEA / f"step_{BEST}.pt")
    unet_a = build_unet(args=args); unet_a.load_state_dict(blob["unet"]); unet_a.to(device).eval()
    a_preds = {}
    for j, job in enumerate(jobs):
        g = torch.Generator(device=device); g.manual_seed(SEED + j * 17)
        pred = sample(unet_a, job["content"].unsqueeze(0).to(device), job["style"].unsqueeze(0).to(device), job["mix"].unsqueeze(0).to(device), True, g)
        a_preds[(job["font"], job["ch"])] = pred.cpu()
        job["l1_a"] = float((pred.cpu() - job["gt"]).abs().mean())
    del unet_a; torch.cuda.empty_cache()
    log(f"A@70k L1={float(np.mean([j['l1_a'] for j in jobs])):.4f} wins={sum(1 for j in jobs if j['l1_a'] < j['l1_ft'])}/{len(jobs)}")

    try: font_ui = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception: font_ui = ImageFont.load_default()

    def panel(font):
        rows = [j for j in jobs if j["font"] == font]
        pad, lw, cw = 8, 28, CELL + 10
        hh, rh = 36, CELL + 18
        im = Image.new("RGB", (pad*2 + lw + 3*cw, hh + len(rows)*rh + pad), (255,255,255))
        dr = ImageDraw.Draw(im)
        dr.text((pad, 6), font.replace("FZ","")[:16] + " @70k", fill=(18,21,26), font=font_ui)
        for i, lab in enumerate(["GT", "ft", "A"]):
            dr.text((pad + lw + i*cw + 8, 20), lab, fill=(92,101,112), font=font_ui)
        for ri, j in enumerate(rows):
            y = hh + ri * rh
            key = (j["font"], j["ch"])
            better = j["l1_a"] < j["l1_ft"]
            border = (15,122,74) if better else (180,83,9)
            dr.text((pad+4, y + CELL//2 - 6), j["ch"], fill=(18,21,26), font=font_ui)
            for xi, (pil, br) in enumerate([(to_pil(j["gt"]), None), (to_pil(ft_preds[key]), None), (to_pil(a_preds[key]), border)]):
                g = pil.resize((CELL, CELL))
                if br:
                    fr = Image.new("RGB", (CELL+4, CELL+4), br); fr.paste(g, (2,2)); im.paste(fr, (pad+lw+xi*cw-2, y-2))
                else:
                    im.paste(g, (pad+lw+xi*cw, y))
            dr.text((pad+lw+cw, y+CELL+2), f"{j['l1_ft']:.2f}", fill=(180,83,9), font=font_ui)
            dr.text((pad+lw+2*cw, y+CELL+2), f"{j['l1_a']:.2f}", fill=border, font=font_ui)
        return im

    panels = [panel(f) for f in fonts]
    gap = 12
    canvas = Image.new("RGB", (sum(p.width for p in panels) + gap*(len(panels)-1), max(p.height for p in panels)), (246,247,249))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0)); x += p.width + gap
    # also 2x2
    gw, gh = panels[0].width, panels[0].height
    grid2 = Image.new("RGB", (2*gw + 3*gap, 2*gh + 3*gap), (246,247,249))
    for i, p in enumerate(panels):
        r, c = divmod(i, 2)
        grid2.paste(p, (gap + c*(gw+gap), gap + r*(gh+gap)))
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = OUT / "viz_method_70k.png"; canvas.save(p1)
    p2 = OUT / "viz_method_70k_2x2.png"; grid2.save(p2)
    log(f"wrote {p1} {canvas.size} and {p2} {grid2.size}")
    meta = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "fonts": fonts, "chars": CHARS, "best_step": BEST,
            "mean_l1_ft": float(np.mean([j["l1_ft"] for j in jobs])),
            "mean_l1_a": float(np.mean([j["l1_a"] for j in jobs])),
            "wins": sum(1 for j in jobs if j["l1_a"] < j["l1_ft"]), "n": len(jobs)}
    (OUT / "viz_method_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(meta)

if __name__ == "__main__":
    main()
