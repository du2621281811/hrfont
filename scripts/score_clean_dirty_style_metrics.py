#!/usr/bin/env python3
"""Clean/dirty metrics board with FROZEN style columns + vs-GT diagnostics.

Frozen (do not redefine):
  - Ours φ : cos(ref8_proto, query) → S01  (E12 teacher style score)
  - E12 mem: membership P(same|ref8)

vs GT only:
  - CLIP / DINOv2 / LPIPS-Alex feature cosine → S01
  - LPIPS Alex distance

    export HF_ENDPOINT=https://hf-mirror.com
    /root/miniforge3/envs/boogu/bin/python scripts/score_clean_dirty_style_metrics.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path("/root/projects/hrfont")
sys.path.insert(0, str(ROOT / "scripts" / "eval_framework"))
from models import load_phi_checkpoint  # noqa: E402

OUT = ROOT / "reports/f03_test16_strat"
DATA = ROOT / "data/fontdiffuser-p253-t295-s338-cn2west-v2"
COMPARE = OUT / "clean_dirty_compare.json"
COS_ITEMS = ROOT / "reports/e12_paper/scores_test16_cosine_items.json"
E12_ITEMS = ROOT / "reports/e12_paper/scores_test16_items.json"
PHI = ROOT / "runs/e12_phi_s2_b_s3407/best.pt"
REF8 = list("永和书风骨韵天地")
SEED = 3407

METHODS = [
    ("P1", "官方 P1", "baseline"),
    ("F0_100k", "F0 脏@100k", "dirty"),
    ("F0C_95000", "F0 净@95k(best)", "clean"),
    ("F0C_100000", "F0 净@100k", "clean"),
    ("F2_40000", "F2 脏@40k", "dirty"),
    ("F2C_40000", "F2 净@40k", "clean"),
    ("F2_80000", "F2 脏@80k", "dirty"),
    ("F2C_80000", "F2 净@80k", "clean"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cp_of(ch: str) -> str:
    return f"u{ord(ch):04X}"


def to_tensor(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im.convert("RGB").resize((96, 96)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def to_pil_rgb(tensor_chw: torch.Tensor) -> Image.Image:
    x = tensor_chw.detach().cpu()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def style_path(font: str, ch: str) -> Path | None:
    for split in ("test", "val", "train"):
        p = DATA / split / "StyleImage" / font / f"{font}+{cp_of(ch)}.png"
        if p.is_file():
            return p
    return None


def gt_path(font: str, cp: str) -> Path | None:
    for split in ("test", "val", "train"):
        p = DATA / split / "TargetImage" / font / f"{font}+{cp}.png"
        if p.is_file():
            return p
    return None


def pred_path(mid: str, font: str, cp: str) -> Path | None:
    p = OUT / "preds" / mid / "test" / font / f"test__{font}__{cp}__s{SEED}.png"
    return p if p.is_file() else None


def mean_std(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=0))}


def s01(cos: float) -> float:
    return float((cos + 1.0) / 2.0)


def fmt(x, d=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{d}f}"


def load_matched_keys() -> list[tuple[str, str, str]]:
    items_p = OUT / "clean_dirty_compare_items.json"
    items = json.loads(items_p.read_text())
    return sorted({(r["font"], r["cp"], r["char"]) for r in items if r.get("method") == "F0_100k"})


class OursPhi:
    name = "Ours"

    def __init__(self, device: torch.device):
        self.model = load_phi_checkpoint(PHI, 3, 512, str(device)).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(tensors).to(self.device)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return F.normalize(self.model(x), dim=1)


class CLIPBackbone:
    name = "CLIP"

    def __init__(self, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

    @torch.no_grad()
    def embed_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        images = [to_pil_rgb(t) for t in tensors]
        inputs = self.processor(images=images, return_tensors="pt")
        vision = self.model.vision_model(pixel_values=inputs["pixel_values"].to(self.device))
        z = self.model.visual_projection(vision.pooler_output)
        return F.normalize(z, dim=1)


class DINOv2Backbone:
    name = "DINOv2"

    def __init__(self, device: torch.device):
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.device = device

    @torch.no_grad()
    def embed_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        images = [to_pil_rgb(t) for t in tensors]
        inputs = self.processor(images=images, return_tensors="pt")
        out = self.model(pixel_values=inputs["pixel_values"].to(self.device))
        return F.normalize(out.last_hidden_state[:, 0], dim=1)


class LPIPSAlexFeature:
    name = "LPIPS-Alex"

    def __init__(self, device: torch.device):
        import lpips

        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def embed_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        xs = []
        for t in tensors:
            x = t if t.shape[0] == 3 else t.repeat(3, 1, 1)
            xs.append(self.tf(x))
        x = torch.stack(xs).to(self.device)
        feats = self.lpips.net.forward(x)
        flat = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        return F.normalize(torch.cat(flat, dim=1), dim=1)

    @torch.no_grad()
    def distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        def prep(t):
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            t = F.interpolate(t[None], size=(64, 64), mode="bilinear", align_corners=False)
            return t * 2 - 1

        d = self.lpips(prep(a).to(self.device), prep(b).to(self.device))
        return float(d.view(-1)[0])


def embed_cached(backbone, tensors: list[torch.Tensor], batch: int = 32) -> torch.Tensor:
    zs = []
    for i in range(0, len(tensors), batch):
        zs.append(backbone.embed_batch(tensors[i : i + batch]).cpu())
    return torch.cat(zs, dim=0)


def load_frozen_style(matched_fc: set[tuple[str, str]]) -> tuple[dict, dict]:
    """Load Ours φ / E12 mem from archived protocol files — never recompute differently."""
    by_ours: dict[str, list] = defaultdict(list)
    by_e12: dict[str, list] = defaultdict(list)
    if COS_ITEMS.is_file():
        for it in json.loads(COS_ITEMS.read_text()):
            if it["method"] == "GT_wrong_family":
                continue
            if (it["font"], it["char"]) not in matched_fc:
                continue
            by_ours[it["method"]].append(float(it["style_score_01"]))
    if E12_ITEMS.is_file():
        for it in json.loads(E12_ITEMS.read_text()):
            if it["method"] == "GT_wrong_family":
                continue
            if (it["font"], it["char"]) not in matched_fc:
                continue
            by_e12[it["method"]].append(float(it["e12_mem_prob"]))
    return by_ours, by_e12


def render_html(report: dict, path: Path) -> None:
    pixel = json.loads(COMPARE.read_text()).get("methods", {}) if COMPARE.is_file() else {}
    cols = [
        ("E12_mem", "E12mem↑", "sty"),
        ("Ours_phi_S01", "Oursφ↑", "sty"),
        ("CLIP_gt_S01", "CLIP↑", "gtm"),
        ("DINOv2_gt_S01", "DINO↑", "gtm"),
        ("LPIPSAlex_gt_S01", "Alex↑", "gtm"),
        ("LPIPS_gt", "LPIPS↓", "gtm"),
        ("L1", "L1↓", "gtm"),
        ("SSIM", "SSIM↑", "gtm"),
    ]
    rows = []
    for mid, label, fam in METHODS:
        m = report["methods"][mid]
        pix = pixel.get(mid, {}).get("overall", {})
        vals = {
            "E12_mem": (m.get("E12_mem") or {}).get("mean"),
            "Ours_phi_S01": (m.get("Ours_phi_S01") or {}).get("mean"),
            "CLIP_gt_S01": (m.get("CLIP_gt_S01") or {}).get("mean"),
            "DINOv2_gt_S01": (m.get("DINOv2_gt_S01") or {}).get("mean"),
            "LPIPSAlex_gt_S01": (m.get("LPIPSAlex_gt_S01") or {}).get("mean"),
            "LPIPS_gt": (m.get("LPIPS_gt") or {}).get("mean"),
            "L1": pix.get("L1_mean"),
            "SSIM": pix.get("SSIM_mean"),
        }
        n = pix.get("n") or (m.get("CLIP_gt_S01") or {}).get("n") or 0
        tds = "".join(f'<td class="{cls}">{fmt(vals[k])}</td>' for k, _, cls in cols)
        rows.append(f"<tr class='{fam}'><td>{label}</td><td>{fam}</td><td>{n}</td>{tds}</tr>")

    pairs = [
        ("F0C_100000", "F0_100k", "F0 净100k − 脏"),
        ("F2C_40000", "F2_40000", "F2 净40k − 脏"),
        ("F2C_80000", "F2_80000", "F2 净80k − 脏"),
        ("F2C_40000", "P1", "F2 净40k − P1"),
    ]
    dkeys = ["E12_mem", "Ours_phi_S01", "CLIP_gt_S01", "DINOv2_gt_S01", "LPIPSAlex_gt_S01", "LPIPS_gt"]
    drows = []
    for a, b, name in pairs:
        cells = []
        for k in dkeys:
            va = (report["methods"][a].get(k) or {}).get("mean")
            vb = (report["methods"][b].get(k) or {}).get("mean")
            cells.append(fmt(None if va is None or vb is None else va - vb))
        drows.append(f"<tr><td>{name}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>干净 vs 脏 · 风格(固定) + vs GT</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#f5f3ef;color:#1a1a1a;line-height:1.45}}
h1{{font-size:1.35rem;margin:0 0 .35rem}} h2{{font-size:1.1rem;margin:1.4rem 0 .5rem}}
.meta,.note{{color:#555;font-size:.92rem;max-width:92ch}}
table{{border-collapse:collapse;background:#fff;margin:12px 0;font-size:13px}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:right}}
th:first-child,td:first-child,td:nth-child(2){{text-align:left}}
tr.clean{{background:#f3faf3}} tr.dirty{{background:#fffaf3}} tr.baseline{{background:#f3f5fa}}
.links a{{margin-right:12px}}
th.sty,td.sty{{background:#eef6ff}}
th.gtm,td.gtm{{background:#f7f7f7}}
</style></head><body>
<h1>干净 vs 脏 · 指标总表</h1>
<p class="meta">生成 {report['computed_at']} · 匹配 n={report.get('n_matched')}</p>
<p class="note">
<strong>蓝列固定不动：</strong>E12 mem / Ours φ = 中文 ref8→西文 query（风格协议，不是 vs GT）。<br>
<strong>灰列 vs GT：</strong>CLIP / DINO / Alex / LPIPS（及 L1/SSIM）= 相对 TargetImage。
</p>
<p class="links">
  <a href="clean_dirty_compare.html">像素板</a>
  <a href="timeline_f2_clean.html">时间线</a>
</p>
<h2>总表</h2>
<table>
<thead><tr><th>方法</th><th>臂</th><th>n</th>
<th class="sty">E12mem↑</th><th class="sty">Oursφ↑</th>
<th class="gtm">CLIP↑</th><th class="gtm">DINO↑</th><th class="gtm">Alex↑</th><th class="gtm">LPIPS↓</th>
<th class="gtm">L1↓</th><th class="gtm">SSIM↑</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
<h2>成对差值（A−B）</h2>
<table>
<thead><tr><th>对比</th>
<th class="sty">ΔE12</th><th class="sty">ΔOursφ</th>
<th class="gtm">ΔCLIP</th><th class="gtm">ΔDINO</th><th class="gtm">ΔAlex</th><th class="gtm">ΔLPIPS</th>
</tr></thead>
<tbody>{''.join(drows)}</tbody></table>
</body></html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--reuse-gt-items",
        action="store_true",
        help="reuse existing clean_dirty_style_items.json for CLIP/DINO/Alex/LPIPS vs GT",
    )
    args = ap.parse_args()

    keys = load_matched_keys()
    matched_fc = {(f, c) for f, _, c in keys}
    fonts = sorted({f for f, _, _ in keys})
    print(f"[board] matched={len(keys)}", flush=True)

    by_ours, by_e12 = load_frozen_style(matched_fc)

    # vs-GT backbones only (NOT Ours)
    items_path = OUT / "clean_dirty_style_items.json"
    if args.reuse_gt_items and items_path.is_file():
        gt_items = json.loads(items_path.read_text())
        print("[board] reusing existing vs-GT items", flush=True)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        query_tensors: dict[tuple[str, str, str], torch.Tensor] = {}
        gt_tensors: dict[tuple[str, str], torch.Tensor] = {}
        for font, cp, ch in keys:
            gtp = gt_path(font, cp)
            if gtp is None:
                continue
            gt_tensors[(font, cp)] = to_tensor(Image.open(gtp))
            for mid, _, _ in METHODS:
                pp = pred_path(mid, font, cp)
                if pp is None:
                    continue
                query_tensors[(mid, font, cp)] = to_tensor(Image.open(pp))

        backbones = [CLIPBackbone(device), DINOv2Backbone(device), LPIPSAlexFeature(device)]
        lpips_bb = backbones[-1]
        gt_keys = list(gt_tensors.keys())
        q_keys = list(query_tensors.keys())
        emb_gt, emb_q = {}, {}
        for bb in backbones:
            print(f"[gt] embedding {bb.name} …", flush=True)
            emb_gt[bb.name] = {
                gt_keys[i]: z
                for i, z in enumerate(embed_cached(bb, [gt_tensors[k] for k in gt_keys], args.batch))
            }
            emb_q[bb.name] = {
                q_keys[i]: z
                for i, z in enumerate(embed_cached(bb, [query_tensors[k] for k in q_keys], args.batch))
            }

        gt_items = []
        print("[gt] scoring …", flush=True)
        for mid, _, _ in METHODS:
            for font, cp, ch in keys:
                if (mid, font, cp) not in query_tensors or (font, cp) not in gt_tensors:
                    continue
                rec = {"method": mid, "font": font, "char": ch, "cp": cp}
                for bb_name, key in [
                    ("CLIP", "CLIP_gt_S01"),
                    ("DINOv2", "DINOv2_gt_S01"),
                    ("LPIPS-Alex", "LPIPSAlex_gt_S01"),
                ]:
                    q = emb_q[bb_name][(mid, font, cp)]
                    g = emb_gt[bb_name][(font, cp)]
                    cos = float(F.cosine_similarity(q[None], g[None]).item())
                    rec[key] = s01(cos)
                rec["LPIPS_gt"] = lpips_bb.distance(
                    query_tensors[(mid, font, cp)], gt_tensors[(font, cp)]
                )
                gt_items.append(rec)
        items_path.write_text(json.dumps(gt_items, ensure_ascii=False) + "\n")

    by_gt: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for it in gt_items:
        mid = it["method"]
        for k in ("CLIP_gt_S01", "DINOv2_gt_S01", "LPIPSAlex_gt_S01", "LPIPS_gt"):
            if k in it and it[k] is not None:
                by_gt[mid][k].append(float(it[k]))

    methods_out = {}
    for mid, _, _ in METHODS:
        methods_out[mid] = {
            "Ours_phi_S01": mean_std(by_ours.get(mid, [])),
            "E12_mem": mean_std(by_e12.get(mid, [])),
            "CLIP_gt_S01": mean_std(by_gt[mid]["CLIP_gt_S01"]),
            "DINOv2_gt_S01": mean_std(by_gt[mid]["DINOv2_gt_S01"]),
            "LPIPSAlex_gt_S01": mean_std(by_gt[mid]["LPIPSAlex_gt_S01"]),
            "LPIPS_gt": mean_std(by_gt[mid]["LPIPS_gt"]),
        }

    report = {
        "computed_at": utc_now(),
        "protocol": {
            "Ours_phi": "FROZEN: cos(ref8_proto, query) S01 from scores_test16_cosine_items",
            "E12_mem": "FROZEN: membership from scores_test16_items",
            "CLIP_DINO_Alex_LPIPS": "vs TargetImage GT only",
        },
        "n_matched": len(keys),
        "methods": methods_out,
        "caveat": "Ours φ / E12 mem must not be redefined as vs-GT.",
    }
    out_json = OUT / "clean_dirty_style_metrics.json"
    out_html = OUT / "clean_dirty_style_metrics.html"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    render_html(report, out_html)

    print(f"\n{'method':<12} {'E12':>7} {'Oursφ':>7} {'CLIP':>7} {'DINO':>7} {'Alex':>7} {'LPIPS':>7}")
    for mid, _, _ in METHODS:
        m = methods_out[mid]

        def g(k):
            v = (m[k] or {}).get("mean")
            return f"{v:.4f}" if v is not None else "  —  "

        print(
            f"{mid:<12} {g('E12_mem')} {g('Ours_phi_S01')} {g('CLIP_gt_S01')} "
            f"{g('DINOv2_gt_S01')} {g('LPIPSAlex_gt_S01')} {g('LPIPS_gt')}"
        )
    print(f"wrote {out_html}")


if __name__ == "__main__":
    main()
