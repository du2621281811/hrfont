#!/usr/bin/env python3
"""Prototype style metrics SRR + FIC on Demo-8 subset (8 chars × 8 fonts).

Uses frozen style encoder; compares generated vs same-font GT neighborhood.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HRFONT_EVAL_GPU", "2"))

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/root/projects/hrfont")
REP = ROOT / "reports/hrfont_overnight/formal_eval"
BANK = ROOT / "data/hrfont/e0_bank"
CHARS = list("AaOoRg8S")


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from hrfont_e2_official_eval import (
        FT,
        META_V2,
        SIZE,
        FontDiffuserDeltaDPM,
        alpha_mix,
        build_pipe,
        load_bank_im,
        make_args,
        png,
        render,
        resolve_font,
        to_tensor,
        _load,
    )

    device = "cuda:0"
    args = make_args()
    sys.path.insert(0, str(ROOT / "code/FontDiffuser"))
    from src import build_content_encoder, build_style_encoder, build_unet
    from accelerate.utils import set_seed

    set_seed(42)
    meta = json.loads((BANK / "meta.json").read_text())
    demo8 = list(META_V2["test_fonts"])
    train = [f for f in meta["train_fonts"] if f not in set(demo8)]
    bank_proto = _load(BANK / "cache/ec_es_r96.pt")["style_proto"]

    se = build_style_encoder(args=args)
    ce = build_content_encoder(args=args)
    se.load_state_dict(_load(FT / "style_encoder.pth"))
    ce.load_state_dict(_load(FT / "content_encoder.pth"))
    se.eval().to(device)
    ce.eval().to(device)

    ck_a = ROOT / "runs/e2_stageA96_formal/step_80000.pt"
    blob = _load(ck_a)
    unet = build_unet(args=args)
    unet.load_state_dict(blob["unet"])
    placeholder = torch.zeros(1, 3, SIZE, SIZE, device=device)
    model = FontDiffuserDeltaDPM(unet, se, ce, placeholder)
    pipe = build_pipe(model, device, args)

    from hrfont_e2_official_eval import DEJAVU, STYLE_CHAR

    @torch.no_grad()
    def es_vec(im_t):
        feat, _, _ = se(im_t)
        # Match official eval: pool spatial dims → (1, D) style vector
        v = feat.flatten(1).mean(0).unsqueeze(0)
        return F.normalize(v, dim=1)

    # GT style vectors per font (from bank latin chars excluding eval char)
    font_gt_es: dict[str, dict[str, torch.Tensor]] = {}
    for font in demo8:
        font_gt_es[font] = {}
        for ch in CHARS:
            p = png(font, ch)
            if p.exists():
                font_gt_es[font][ch] = es_vec(load_bank_im(p).to(device)).cpu()

    rows = []
    for font in demo8:
        fp = resolve_font(font)
        style_t = to_tensor(render(fp, STYLE_CHAR, SIZE), SIZE).to(device)
        proto = bank_proto.get(font)
        for ch in CHARS:
            if not png(font, ch).exists():
                continue
            content_t = to_tensor(render(DEJAVU, ch, SIZE), SIZE).to(device)
            delta_img = alpha_mix(proto.to(device), bank_proto, train, ch, device) if proto is not None else None
            if delta_img is None:
                delta_img = content_t
            model.set_delta(delta_img)
            pred = pipe.generate(
                content_images=content_t,
                style_images=style_t,
                batch_size=1,
                order=2,
                num_inference_step=20,
                content_encoder_downsample_size=3,
                dm_size=(SIZE, SIZE),
                algorithm_type="dpmsolver++",
                method="multistep",
            )[0]
            pred_t = to_tensor(pred, SIZE).to(device)
            v_gen = es_vec(pred_t).cpu()
            others = [font_gt_es[font][c] for c in CHARS if c != ch and c in font_gt_es[font]]
            if not others:
                continue
            stack = torch.cat(others, dim=0)
            d_min = float(torch.cdist(v_gen, stack).min().item())
            # FIC: gen vs GT same char
            v_gt = font_gt_es[font][ch]
            d_gt = float(torch.cdist(v_gen, v_gt).item())
            rows.append({"font": font, "char": ch, "srr_dist": d_min, "fic_dist_gt": d_gt})

    srr_thresh = float(np.percentile([r["srr_dist"] for r in rows], 75)) if rows else 1.0
    for r in rows:
        r["srr_pass"] = r["srr_dist"] < srr_thresh

    payload = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "Demo-8 × AaOoRg8S subset · Stage A @80k",
        "n": len(rows),
        "mean_srr_dist": float(np.mean([r["srr_dist"] for r in rows])) if rows else None,
        "srr_pass_rate": float(np.mean([r["srr_pass"] for r in rows])) if rows else None,
        "mean_fic_dist_gt": float(np.mean([r["fic_dist_gt"] for r in rows])) if rows else None,
        "note": "Prototype: SRR=Es(gen) to same-font other chars; lower is better",
        "rows": rows,
    }
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "style_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"style_metrics n={len(rows)} srr_pass={payload['srr_pass_rate']}")


if __name__ == "__main__":
    main()
