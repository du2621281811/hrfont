#!/usr/bin/env python3
"""FontDiffuser content-font sweep on Demo-8 (FD0).

Which content fonts to test (and why):
  ★ NotoSerifSC     — 思源宋，原产品页 / 旧 launch
  ★ NotoSansSC      — 思源黑，早期 designer 页
  ★ uming / gbsn    — 开源明/宋，贴 FD 论文 Song 域
  ★ FZ悦宋          — 方正宋
  ★ FZKTJW          — 方正楷体，当前 GAR 冻结
  ★ ukai / gkai     — 开源楷
  ★ FZ正楷          — 方正正楷
  ○ KaiXinSongA     — FD 代码默认（仓未附带；有则自动纳入）

Protocol: Demo-8 styles/GT from stages_fzkt; only content changes; FD0 1-shot「永」.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/projects/hrfont")
FD_ROOT = ROOT / "code/ours/FontDiffuser"
FD_CKPT = FD_ROOT / "ckpt"
FONTS_KAI = ROOT / "repos/_fonts_kai"
WIN_DROP = FONTS_KAI / "windows"
STAGES = ROOT / "runs_retrain_v2/designer_cn2cn_stages_fzkt"
OUT = ROOT / "runs_retrain_v2/fd_content_font_sweep"
REP = ROOT / "reports/retrain_v2/fd_content_font_sweep"

REF8 = list("永和书风骨韵天地")
GEN_CHARS = list("的一是不了在人有我他这中文设计测试")

SPECS = [
    # (id, path, ttc_index, label, why)
    ("NotoSerifSC", Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"), 2,
     "思源宋(旧)", "原产品页"),
    ("NotoSansSC", Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), 2,
     "思源黑", "早期对照"),
    ("uming_CN", Path("/usr/share/fonts/truetype/arphic/uming.ttc"), 0,
     "uming明", "开源明≈宋·贴FD域"),
    ("gbsn00lp", Path("/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf"), 0,
     "gbsn宋", "开源宋·贴FD域"),
    ("FZQingKeBenYueSong", Path("/root/data/font_50/FZQKBYSJW.TTF"), 0,
     "FZ悦宋", "方正宋"),
    ("FZKTJW", ROOT / "data/FZKTJW.TTF", 0,
     "方正楷体", "GAR冻结"),
    ("ukai", Path("/usr/share/fonts/truetype/arphic/ukai.ttc"), 0,
     "ukai楷", "开源楷"),
    ("gkai00mp", FONTS_KAI / "gkai_ext/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf", 0,
     "gkai楷", "开源楷"),
    ("FZHanWenZhengKai", Path("/root/data/font_50/FZHanWZKJW.TTF"), 0,
     "FZ正楷", "方正楷"),
    ("FZFengYaKaiSong", Path("/root/data/font_50/FZFengYKSJ.TTF"), 0,
     "FZ雅楷", "方正楷宋"),
]

KAIXIN_PATHS = [
    WIN_DROP / "KaiXinSongA.ttf",
    FD_ROOT / "ttf" / "KaiXinSongA.ttf",
    ROOT / "data" / "KaiXinSongA.ttf",
]


def fonts() -> list[str]:
    meta = json.loads((ROOT / "data/retrain_v2/meta.json").read_text())
    return [Path(f).stem for f in meta["test_fonts"]]


def build_candidates():
    cands = []
    for p in KAIXIN_PATHS:
        if p.is_file():
            cands.append({"id": "KaiXinSongA", "path": p, "index": 0,
                          "label": "开心宋★", "note": "FD代码默认"})
            break
    else:
        print(f"[missing] KaiXinSongA.ttf — put under {WIN_DROP}/ or FontDiffuser/ttf/", flush=True)
    for cid, path, idx, lab, note in SPECS:
        if not Path(path).is_file():
            print(f"[skip] {cid}: {path}", flush=True)
            continue
        cands.append({"id": cid, "path": Path(path), "index": idx, "label": lab, "note": note})
    return cands


def to_arr(im: Image.Image, size=96) -> np.ndarray:
    return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def ssim(a, b):
    C1, C2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    return float(
        ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2))
        / ((mu_a**2 + mu_b**2 + C1) * (a.var() + b.var() + C2))
    )


def render_char(ttf: Path, ch: str, index: int = 0, size=96, canvas=96) -> Image.Image:
    font = ImageFont.truetype(str(ttf), size=size, index=index)
    im = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    d = ImageDraw.Draw(im)
    bb = d.textbbox((0, 0), ch, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    if w <= 0 or h <= 0:
        raise RuntimeError(f"empty glyph {ch} in {ttf}")
    d.text(((canvas - w) // 2 - bb[0], (canvas - h) // 2 - bb[1]), ch, fill=(0, 0, 0), font=font)
    return im


def prep_contents(cands):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("style_ref", "gt"):
        src = STAGES / name
        if not src.exists():
            raise SystemExit(f"need {src}")
        dst = OUT / name
        if dst.is_symlink() or dst.exists():
            dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
        dst.symlink_to(src)
    for c in cands:
        named = OUT / "content_named" / c["id"]
        if named.exists() and (named / f"{GEN_CHARS[0]}.png").exists():
            print(f"[prep skip] {c['id']}", flush=True)
            continue
        if named.exists():
            shutil.rmtree(named)
        named.mkdir(parents=True)
        for ch in REF8 + GEN_CHARS:
            render_char(c["path"], ch, index=c["index"]).save(named / f"{ch}.png")
        print(f"[prep] {c['id']}", flush=True)


def load_fd(gpu: str):
    if str(FD_ROOT) not in sys.path:
        sys.path.insert(0, str(FD_ROOT))
    from src import (  # noqa: E402
        FontDiffuserModelDPM, FontDiffuserDPMPipeline,
        build_ddpm_scheduler, build_unet, build_content_encoder, build_style_encoder,
    )
    from configs.fontdiffuser import get_parser
    from accelerate.utils import set_seed

    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    parser = get_parser()
    args, _ = parser.parse_known_args([])
    args.ckpt_dir = str(FD_CKPT)
    args.device = device
    args.style_image_size = (96, 96)
    args.content_image_size = (96, 96)
    args.resolution = 96
    args.algorithm_type = "dpmsolver++"
    args.guidance_type = "classifier-free"
    args.guidance_scale = 7.5
    args.num_inference_steps = 20
    args.method = "multistep"
    args.order = 2
    args.seed = 123
    args.model_type = "noise"
    args.t_start = None
    args.t_end = None
    args.skip_type = "time_uniform"
    args.correcting_x0_fn = None

    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location="cpu"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location="cpu"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location="cpu"))
    model = FontDiffuserModelDPM(unet=unet, style_encoder=style_encoder, content_encoder=content_encoder)
    model.to(device).eval()
    set_seed(123)
    tfm = T.Compose([
        T.Resize((96, 96), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=build_ddpm_scheduler(args),
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    return args, pipe, tfm, device


def run_fd0(cands, flist, gpu: str, force: bool = False):
    args, pipe, tfm, device = load_fd(gpu)
    for c in cands:
        cid = c["id"]
        for stem in flist:
            out = OUT / "pred" / cid / stem
            n_done = sum(1 for ch in GEN_CHARS if (out / f"{ch}.png").exists())
            if n_done == len(GEN_CHARS) and not force:
                print(f"[skip] {cid}/{stem}", flush=True)
                continue
            if out.exists() and (force or n_done < len(GEN_CHARS)):
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            print(f"[FD0] {cid} · {stem}", flush=True)
            style = tfm(Image.open(OUT / "style_ref" / stem / "0000.png").convert("RGB")).unsqueeze(0).to(device)
            for ch in GEN_CHARS:
                content = tfm(
                    Image.open(OUT / "content_named" / cid / f"{ch}.png").convert("RGB")
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    images = pipe.generate(
                        content_images=content,
                        style_images=style,
                        batch_size=1,
                        order=args.order,
                        num_inference_step=args.num_inference_steps,
                        content_encoder_downsample_size=args.content_encoder_downsample_size,
                        t_start=args.t_start,
                        t_end=args.t_end,
                        dm_size=args.content_image_size,
                        algorithm_type=args.algorithm_type,
                        skip_type=args.skip_type,
                        method=args.method,
                        correcting_x0_fn=args.correcting_x0_fn,
                    )
                pred = images[0]
                if not isinstance(pred, Image.Image):
                    t = pred.detach().cpu()
                    if t.dim() == 4:
                        t = t[0]
                    t = (t.clamp(-1, 1) + 1) * 127.5
                    arr = t.numpy()
                    if arr.shape[0] in (1, 3):
                        arr = np.transpose(arr, (1, 2, 0))
                    pred = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
                else:
                    pred = pred.convert("RGB")
                out.mkdir(parents=True, exist_ok=True)
                # ASCII tmp — Chinese in ".{ch}.tmp.png" broke Path.replace here
                tmp = out / f"_tmp_{GEN_CHARS.index(ch):02d}.png"
                pred.save(tmp)
                dest = out / f"{ch}.png"
                if dest.exists():
                    dest.unlink()
                tmp.rename(dest)
            print(f"[FD0 done] {cid}/{stem} n={len(list(out.glob('*.png')))}", flush=True)


def eval_all(cands, flist):
    rows = {}
    for c in cands:
        cid = c["id"]
        ls, ss = [], []
        for stem in flist:
            for ch in GEN_CHARS:
                p = OUT / "pred" / cid / stem / f"{ch}.png"
                g = OUT / "gt" / stem / f"{ch}.png"
                if not p.exists() or not g.exists():
                    continue
                pa, ga = to_arr(Image.open(p)), to_arr(Image.open(g))
                ls.append(float(np.abs(pa - ga).mean()))
                ss.append(ssim(pa, ga))
        if not ls:
            continue
        rows[cid] = {
            "n": len(ls), "L1": float(np.mean(ls)), "SSIM": float(np.mean(ss)),
            "label": c["label"], "note": c["note"], "path": str(c["path"]),
        }
    ranked = sorted(rows.items(), key=lambda kv: kv[1]["L1"])
    return rows, ranked


def write_page(cands, flist, rows, ranked):
    REP.mkdir(parents=True, exist_ok=True)
    assets = REP / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    vis_fonts = flist[:3]
    show_chars = GEN_CHARS[:8]
    for c in cands:
        if c["id"] not in rows:
            continue
        d = assets / "content" / c["id"]
        d.mkdir(parents=True)
        for ch in show_chars:
            src = OUT / "content_named" / c["id"] / f"{ch}.png"
            if src.exists():
                shutil.copy2(src, d / f"{ch}.png")
    for stem in vis_fonts:
        gd = assets / "gt" / stem
        gd.mkdir(parents=True)
        for ch in show_chars:
            g = OUT / "gt" / stem / f"{ch}.png"
            if g.exists():
                shutil.copy2(g, gd / f"{ch}.png")
        for c in cands:
            cid = c["id"]
            if cid not in rows:
                continue
            pd = assets / "pred" / cid / stem
            pd.mkdir(parents=True)
            for ch in show_chars:
                p = OUT / "pred" / cid / stem / f"{ch}.png"
                if p.exists():
                    shutil.copy2(p, pd / f"{ch}.png")

    table = "".join(
        f"<tr class=\"{'best' if i<3 else ''}{' old' if n=='NotoSerifSC' else ''}{' kai' if n=='FZKTJW' else ''}\">"
        f"<td>{v['label']}</td><td><code>{n}</code></td><td>{v['note']}</td>"
        f"<td>{v['L1']:.4f}</td><td>{v['SSIM']:.4f}</td><td>{v['n']}</td></tr>"
        for i, (n, v) in enumerate(ranked)
    )
    cols_js = json.dumps(
        [{"id": n, "label": rows[n]["label"],
          "cls": "base" if n == "NotoSerifSC" else ("good" if i < 3 else "")}
         for i, (n, _) in enumerate(ranked)],
        ensure_ascii=False,
    )
    missing_kx = not any(c["id"] == "KaiXinSongA" for c in cands)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"/>
<title>FD Content 字体扫描 · Demo-8</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
:root{{--bg:#0e1217;--panel:#171d25;--line:rgba(255,255,255,.1);--muted:#93a0ad;--accent:#c9a46a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:#e9eef4;font:14px/1.5 system-ui,sans-serif}}
.wrap{{max-width:1400px;margin:0 auto;padding:22px 18px 60px}}
h1{{font:700 1.35rem Georgia,serif}} a{{color:var(--accent)}} .muted{{color:var(--muted)}}
.panel{{background:var(--panel);border:1px solid var(--line);padding:14px;margin:14px 0;border-radius:4px}}
.call{{border-left:3px solid var(--accent);padding:8px 12px;background:rgba(201,164,106,.08);margin:12px 0}}
.warn{{border-left:3px solid #c97a6a;padding:8px 12px;background:rgba(201,122,106,.08);margin:12px 0}}
table.metrics{{width:100%;border-collapse:collapse;font-size:13px}}
table.metrics th,td{{padding:6px 8px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left;white-space:nowrap}}
tr.best td{{color:#b8e0c2}} tr.old td{{color:var(--accent)}} tr.kai td{{color:#9ec5e8}}
.scroll{{overflow-x:auto}}
.grid{{display:grid;gap:6px;align-items:end;min-width:max-content}}
.grid.head{{color:var(--muted);font-size:11px;padding-bottom:4px;border-bottom:1px solid var(--line)}}
.grid.head>div,.cell{{text-align:center}}
.grid.row{{padding:6px 0;border-bottom:1px solid rgba(255,255,255,.06)}}
.ch{{font-size:16px;font-weight:600;padding-top:14px}}
.cell img{{width:56px;height:56px;object-fit:contain;background:#fff;border-radius:2px;display:block;margin:0 auto}}
.sticky{{position:sticky;left:0;background:var(--panel);z-index:1;padding-right:6px}}
.tag{{font-size:10px;padding:1px 5px;border:1px solid var(--line);border-radius:2px;margin-left:3px;color:var(--muted)}}
.tag.base{{border-color:var(--accent);color:var(--accent)}} .tag.good{{border-color:#5a9a6a;color:#b8e0c2}}
select{{background:#0f1419;color:#e9eef4;border:1px solid rgba(255,255,255,.25);padding:6px 8px}}
</style></head><body><div class="wrap">
<h1>FontDiffuser · Content 字体影响（Demo-8 · FD0）</h1>
<p class="muted">只换 content；风格/GT 同
<a href="../designer_cn2cn_launch_fzkt/">launch_fzkt</a>。
对照 <a href="../gar_content_font_sweep/">GAR content 扫</a> ·
<a href="../DESIGNER_LAUNCH_PLAN.md">计划</a></p>
<div class="call">FD 论文=<b>宋</b>；代码默认 <code>KaiXinSongA</code>（缺）。
必含原 <b>思源宋</b>；并测宋/明/楷/黑。</div>
{"<div class='warn'>缺 KaiXinSongA.ttf → 放到 <code>repos/_fonts_kai/windows/</code> 后重跑可补官方默认。</div>" if missing_kx else ""}
<div class="panel">
<table class="metrics">
<tr><th>简称</th><th>id</th><th>为何测</th><th>生成 L1↓</th><th>SSIM↑</th><th>n</th></tr>
{table}
</table>
<p class="muted">橙=原思源宋 · 蓝=方正楷体 · 绿=L1 前三</p>
</div>
<div class="panel">
<label>字体 <select id="f">{"".join(f'<option value="{s}">{s}</option>' for s in vis_fonts)}</select></label>
<div class="scroll" id="grid" style="margin-top:10px"></div>
</div>
<script>
const CHARS={json.dumps(show_chars, ensure_ascii=False)};
const CANDS={cols_js};
const cols=`36px 64px `+CANDS.map(()=>"64px").join(" ");
function render(){{
  const f=document.getElementById('f').value;
  document.getElementById('grid').innerHTML=
    `<div class="grid head" style="grid-template-columns:${{cols}}"><div class="sticky"></div><div>GT</div>${{
      CANDS.map(c=>`<div>${{c.label}}${{c.cls?`<span class="tag ${{c.cls}}">${{c.cls==='base'?'旧':'优'}}</span>`:''}}</div>`).join('')
    }}</div>`+
    CHARS.map(ch=>`<div class="grid row" style="grid-template-columns:${{cols}}">
      <div class="ch sticky">${{ch}}</div>
      <div class="cell"><img src="assets/gt/${{f}}/${{ch}}.png"/></div>
      ${{CANDS.map(c=>`<div class="cell"><img src="assets/pred/${{c.id}}/${{f}}/${{ch}}.png"/></div>`).join('')}}
    </div>`).join('');
}}
document.getElementById('f').onchange=render; render();
</script>
</div></body></html>"""
    (REP / "index.html").write_text(html, encoding="utf-8")
    (REP / "summary.json").write_text(json.dumps({
        "rows": rows, "ranked": [n for n, _ in ranked],
        "protocol": {"model": "FD0", "fonts": flist, "gen": "".join(GEN_CHARS),
                     "missing_KaiXinSongA": missing_kx},
        "candidates": [{k: str(c[k]) if k == "path" else c[k] for k in ("id", "label", "note", "path")} for c in cands],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({n: {"L1": v["L1"], "SSIM": v["SSIM"]} for n, v in ranked},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"[report] {REP/'index.html'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--fonts", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    cands = build_candidates()
    flist = fonts() if not args.fonts else [x.strip() for x in args.fonts.split(",") if x.strip()]
    print(f"[cands] {[c['id'] for c in cands]}", flush=True)
    print(f"[fonts] {flist}", flush=True)
    if not args.report_only:
        prep_contents(cands)
        run_fd0(cands, flist, args.gpu, force=args.force)
    rows, ranked = eval_all(cands, flist)
    write_page(cands, flist, rows, ranked)


if __name__ == "__main__":
    main()
