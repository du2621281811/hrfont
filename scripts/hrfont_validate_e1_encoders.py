#!/usr/bin/env python3
"""Validation battery for frozen E1 style/content encoders.

Expected A layout is ``DATA_ROOT/{train,val,test}/{StyleImage,TargetImage}/FONT/``
with ``FONT+uXXXX.png`` files.  B0 content prototypes may either use the normal
FZKTJW TargetImage directory or a dedicated directory passed via ``--b0-dir``
containing ``FZKTJW+uXXXX.png`` (or ``uXXXX.png``).  Images must be native RGB
96x96.  The train228 set is the only alpha/delta library; val16+test16 are unseen.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrfont_delta_v2 import (DeltaConfig, DummyEc, DummyEs, _ec_features,
                             _font_paths, _style_vector, compute_alpha,
                             feature_delta, load_rgb_tensor, render_glyph,
                             sha256_file)


def auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Mann-Whitney AUC with ties worth one half."""
    if not scores_pos or not scores_neg:
        return float("nan")
    wins = sum((p > n) + .5 * (p == n) for p in scores_pos for n in scores_neg)
    return wins / (len(scores_pos) * len(scores_neg))


def finite_mean(values: list[float]) -> float:
    values = [x for x in values if math.isfinite(x)]
    return float(sum(values) / len(values)) if values else float("nan")


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(torch.tensor(values).median())


class RenderSet:
    def __init__(self, root: Path, split: dict):
        self.root, self.split = root, split
        self.where = {stem: part for part, stems in split.items() for stem in stems}

    @staticmethod
    def cp(ch: str) -> str:
        return f"u{ord(ch):04X}"

    def path(self, stem: str, ch: str, role: str) -> Path:
        cp = self.cp(ch)
        part = self.where.get(stem, "train")
        options = [self.root / part / role / stem / f"{stem}+{cp}.png"]
        for path in options:
            if path.exists():
                return path
        # Transitional datasets sometimes have an extra directory level.
        hits = sorted(self.root.glob(f"**/{role}/{stem}/{stem}+{cp}.png"))
        if hits:
            return hits[0]
        raise FileNotFoundError(f"missing {role} render: {stem} {ch} under {self.root}")

    def image(self, stem: str, ch: str, role: str = "StyleImage") -> torch.Tensor:
        return load_rgb_tensor(self.path(stem, ch, role))

    def content_tensor(self, ch: str) -> torch.Tensor:
        cp = self.cp(ch)
        path = self.root / "train" / "ContentImage" / f"{cp}.png"
        if not path.exists():
            raise FileNotFoundError(f"ContentImage missing: {path}")
        return load_rgb_tensor(path)


class Validator:
    def __init__(self, es, ec, renders: RenderSet, ref8: str, target_chars: str,
                 device: torch.device, dtype: torch.dtype, cfg: DeltaConfig):
        self.es, self.ec = es.to(device=device, dtype=dtype).eval(), ec.to(device=device, dtype=dtype).eval()
        self.r, self.ref8, self.chars = renders, ref8, target_chars
        self.device, self.dtype, self.cfg = device, dtype, cfg
        self._style: dict[tuple[str, str, str], torch.Tensor] = {}
        self._content: dict[tuple[str, str, str], list[torch.Tensor]] = {}

    @torch.no_grad()
    def style(self, font: str, ch: str, role: str = "StyleImage") -> torch.Tensor:
        key = font, ch, role
        if key not in self._style:
            x = self.r.image(font, ch, role).to(self.device, self.dtype)
            self._style[key] = F.normalize(_style_vector(self.es(x)).float(), dim=1).squeeze(0).cpu()
        return self._style[key]

    def proto(self, font: str) -> torch.Tensor:
        # per-char [n, D], L2-normalized per char (matches compute_alpha contract)
        return torch.stack([self.style(font, ch) for ch in self.ref8])

    @torch.no_grad()
    def content_neutral(self, ch: str) -> list[torch.Tensor]:
        # Ec features of the Noto ContentImage (B0 = Content per collaborator decision)
        x = self.r.content_tensor(ch).to(self.device, self.dtype)
        return [F.normalize(v.float().flatten(1), dim=1).squeeze(0).cpu()
                for v in _ec_features(self.ec(x))]

    @torch.no_grad()
    def content(self, font: str, ch: str, role: str = "TargetImage") -> list[torch.Tensor]:
        key = font, ch, role
        if key not in self._content:
            x = self.r.image(font, ch, role).to(self.device, self.dtype)
            self._content[key] = [F.normalize(v.float().flatten(1), dim=1).squeeze(0).cpu()
                                  for v in _ec_features(self.ec(x))]
        return self._content[key]

    def v1(self, library: list[str], eval_fonts: list[str], val_fonts: list[str]) -> dict:
        # Gallery = train library + eval identities; per-char cosine mean over full ref8.
        gallery = sorted(set(library + eval_fonts))
        gp = torch.stack([self.proto(f) for f in gallery])  # [G, 8, D]
        ranks, pos, neg = [], [], []
        for font in eval_fonts:
            qchars = [self.style(font, ch) for ch in self.ref8]
            q = torch.stack(qchars)  # [8, D]
            scores = torch.einsum("gd,fgd->fg", q, gp).mean(dim=1)
            order = sorted(range(len(gallery)), key=lambda i: (-float(scores[i]), gallery[i]))
            ranks.append(order.index(gallery.index(font)) + 1)
            if font in val_fonts:
                for a in qchars[::2]:
                    for b in qchars[1::2]:
                        pos.append(float(a @ b))
                    for other in val_fonts:
                        if other != font:
                            neg.append(float(a @ self.style(other, self.ref8[1])))
        return {"median_rank": median(ranks), "r_at_1": sum(r <= 1 for r in ranks) / len(ranks),
                "r_at_3": sum(r <= 3 for r in ranks) / len(ranks),
                "r_at_5": sum(r <= 5 for r in ranks) / len(ranks),
                "pairwise_auc": (auc(pos, neg) if pos and neg else None), "ranks": ranks,
                "gallery_note": "train library plus eval self identities; queries use disjoint ref8 halves"}

    def v2(self, eval_fonts: list[str]) -> dict:
        latin = next((c for c in self.chars if c.isascii() and c.isalpha()), None)
        han = self.ref8[0]
        if latin is None:
            raise ValueError("target charset needs a Latin letter for V2")
        same, cross = [], []
        for i, font in enumerate(eval_fonts):
            same.append(float(self.style(font, han) @ self.style(font, latin, "TargetImage")))
            cross.append(float(self.style(font, han) @ self.style(
                eval_fonts[(i + 1) % len(eval_fonts)], latin, "TargetImage")))
        return {"auc": auc(same, cross), "median_gap": median(same) - median(cross),
                "same_mean": finite_mean(same), "cross_mean": finite_mean(cross)}

    def v3(self, eval_fonts: list[str]) -> dict:
        chars = sorted(set(self.chars))
        base = torch.stack([self.content_neutral(c)[-1] for c in chars])
        correct, confusions = 0, Counter()
        for font in eval_fonts:
            for ch in chars:
                score = base @ self.content(font, ch)[-1]
                guess = chars[int(score.argmax())]
                correct += guess == ch
                if guess != ch:
                    confusions[(ch, guess)] += 1
        n = len(eval_fonts) * len(chars)
        return {"top1_acc": correct / n, "n": n,
                "top_confusions": [{"true": a, "pred": b, "count": n}
                                   for (a, b), n in confusions.most_common(10)]}

    def v4(self, eval_fonts: list[str]) -> dict:
        chars = sorted(set(self.chars))
        within, cross = [], []
        for ch in chars:
            for a, b in zip(eval_fonts, eval_fonts[1:]):
                within.append(float(self.content(a, ch)[-1] @ self.content(b, ch)[-1]))
        for font in eval_fonts:
            for a, b in zip(chars, chars[1:]):
                cross.append(float(self.content(font, a)[-1] @ self.content(font, b)[-1]))
        return {"within_char_across_font": finite_mean(within),
                "cross_char_within_font": finite_mean(cross),
                "gap": finite_mean(within) - finite_mean(cross), "auc": auc(within, cross)}

    def v5(self, library: list[str], eval_fonts: list[str]) -> dict:
        libp = torch.stack([self.proto(f) for f in library])
        char = sorted(set(self.chars))[0]
        magnitudes, distances, active, entropy = [], [], [], []
        for font in eval_fonts:
            refs = torch.stack([self.style(font, c) for c in self.ref8])
            idx, weights, meta = compute_alpha(refs, libp, None, self.cfg)
            neighbors = [self.r.image(library[i], char) for i in idx]
            neutral = self.r.content_tensor(char)
            delta = feature_delta(self.ec, neighbors, weights, neutral)
            magnitudes.append(0.0 if delta is None else finite_mean([
                float(x.float().flatten(1).norm(dim=1).mean()) for x in delta]))
            distances.append(1.0 - meta["max_cosine"])
            active.append(meta["n_active"]); entropy.append(meta["entropy"])
        neutral = self.r.content_tensor(char)
        zero = feature_delta(self.ec, [neutral], torch.ones(1), neutral)
        zero_max = max(float(x.abs().max()) for x in zero)
        pair = torch.tensor([magnitudes, distances])
        corr = (float(torch.corrcoef(pair)[0, 1]) if len(magnitudes) > 1
                and pair[0].std() > 0 and pair[1].std() > 0 else 0.0)
        return {"delta_magnitude_mean": finite_mean(magnitudes), "magnitude_style_distance_corr": corr,
                "neutral_self_max_abs": zero_max, "n_active_mean": finite_mean(active),
                "alpha_entropy_mean": finite_mean(entropy), "n_active": active, "alpha_entropy": entropy}

    def v6(self, library: list[str]) -> dict:
        libp = torch.stack([self.proto(f) for f in library])
        top_cos, top_mass = [], []
        top_fonts = []
        for self_idx, font in enumerate(library):
            refs = torch.stack([self.style(font, c) for c in self.ref8])
            idx, weights, meta = compute_alpha(refs, libp, self_idx, self.cfg)
            top_cos.append(meta["max_cosine"])
            top_mass.append(float(weights.max()) if len(weights) else 0.0)
            top_fonts.append(library[idx[int(weights.argmax())]] if len(idx) else None)
        return {"top1_neighbor_cosine_mean": finite_mean(top_cos), "top1_mass_mean": finite_mean(top_mass),
                "top1_neighbor_cosines": top_cos, "top1_neighbors": top_fonts,
                "note": "train-library leave-one-out; no semantic neighbor labels, so cosine/mass diagnose confidence"}


def load_split(path: Path) -> dict[str, list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    value = value.get("stems", value)
    return {key: sorted(value[key]) for key in ("train", "val", "test")}


def load_real_encoders(variant_path: Path, ckpt_dir: Path):
    ec_path, es_path = ckpt_dir / "content_encoder.pth", ckpt_dir / "style_encoder.pth"
    for path in (ec_path, es_path):
        if not path.exists():
            raise FileNotFoundError(f"E1 checkpoint missing: {path}")
    sys.path.insert(0, str(variant_path))
    from src.modules import ContentEncoder, StyleEncoder
    ec, es = ContentEncoder(G_ch=64, resolution=96), StyleEncoder(G_ch=64, resolution=96)
    for model, path in ((ec, ec_path), (es, es_path)):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("state_dict", "model", "module"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]; break
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    return es, ec, {"content_encoder": sha256_file(ec_path), "style_encoder": sha256_file(es_path)}


def run_battery(es, ec, root: Path, split: dict, ref8: str, chars: str,
                device: str, precision: str, cfg: DeltaConfig) -> dict:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    if device == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 validation requires CUDA")
    validator = Validator(es, ec, RenderSet(root, split), ref8, chars,
                          torch.device(device), dtype, cfg)
    library = split["train"]

    def battery(eval_fonts):
        return {
            "V1_style_retrieval": validator.v1(library, eval_fonts, split["val"]),
            "V2_cross_script": validator.v2(eval_fonts),
            "V3_content_identity": validator.v3(eval_fonts),
            "V4_content_invariance": validator.v4(eval_fonts),
            "V5_delta_sanity": validator.v5(library, eval_fonts),
        }

    # val/test 隔离：gate 只由 val16（+train 的 V6）决定；test16 仅只读报告。
    gates = battery(split["val"])
    gates["V6_alpha_quality"] = validator.v6(library)
    test_report = battery(split["test"])
    return {"config": {"data_root": str(root), "split": split, "ref8": ref8,
                       "target_chars": chars, "precision": precision,
                       "delta": cfg.to_dict()},
            "gates": gates, "test_report": test_report,
            "isolation_note": "gates computed on val16/train228 only; test16 rows are report-only"}


def print_summary(results: dict, smoke: bool = False) -> bool:
    t = results["gates"]
    rows = [
        ("V1", t["V1_style_retrieval"]["pairwise_auc"], .50 if smoke else .80, ">="),
        ("V2", t["V2_cross_script"]["auc"], .40 if smoke else .75, ">="),
        ("V3", t["V3_content_identity"]["top1_acc"], 0.0 if smoke else .80, ">="),
        ("V4", t["V4_content_invariance"]["gap"], -1.0 if smoke else 0.05, ">="),
        ("V5", t["V5_delta_sanity"]["neutral_self_max_abs"], 1e-6, "<="),
        ("V6", t["V6_alpha_quality"]["top1_neighbor_cosine_mean"], -1.0 if smoke else .50, ">="),
    ]
    print("test metric                         gate       result")
    passed = True
    for name, value, gate, op in rows:
        ok = math.isfinite(value) and (value >= gate if op == ">=" else value <= gate)
        passed &= ok
        print(f"{name:<4} {value:>10.4f}              {op}{gate:<7g} {'PASS' if ok else 'FAIL'}")
    return passed


def write_png(path: Path, tensor: torch.Tensor) -> None:
    array = ((tensor.squeeze(0).permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().numpy()
    Image.fromarray(array).save(path)


def run_smoke(output: Path) -> bool:
    fonts = _font_paths()
    base_paths = (fonts[:4] if len(fonts) >= 4 else fonts + [None] * (4 - len(fonts)))
    stems = [f"font{i}" for i in range(6)]
    paths = [base_paths[i % 4] for i in range(6)]
    split = {"train": stems[:2], "val": stems[2:4], "test": stems[4:]}
    ref8, chars = "永和书风骨韵天地", "永A和B书C"
    with tempfile.TemporaryDirectory(prefix="hrfont-e1-smoke-") as tmp:
        root = Path(tmp)
        for part, part_fonts in split.items():
            for role in ("StyleImage", "TargetImage"):
                for stem in part_fonts:
                    d = root / part / role / stem; d.mkdir(parents=True)
                    font_path = paths[stems.index(stem)]
                    for ch in sorted(set(ref8 + chars)):
                        write_png(d / f"{stem}+u{ord(ch):04X}.png", render_glyph(ch, font_path, stems.index(stem) * 12))
        cdir = root / "train" / "ContentImage"; cdir.mkdir(parents=True)
        for ch in sorted(set(ref8 + chars)):
            write_png(cdir / f"u{ord(ch):04X}.png", render_glyph(ch, paths[0], 0))
        results = run_battery(DummyEs(), DummyEc(), root, split, ref8, chars,
                              "cpu", "fp32", DeltaConfig(eps_alpha=0.0, k_max=2))
        results["checkpoint_sha256"] = {"content_encoder": "dummy", "style_encoder": "dummy"}
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        ok = print_summary(results, smoke=True)
    print(f"hrfont_validate_e1_encoders smoke: {'PASS' if ok else 'FAIL'} ({output})")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--ckpt_dir", type=Path)
    p.add_argument("--variant_path", type=Path, default=Path("code/variants/cn2west_ft_v2/FontDiffuser"))
    p.add_argument("--data_root", type=Path)
    p.add_argument("--split", type=Path)
    p.add_argument("--charset", type=Path, default=Path("manifests/charset_cn2west_v2_planned.json"))
    p.add_argument("--ref8", default="永和书风骨韵天地")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    p.add_argument("--output", type=Path, default=Path("/tmp/hrfont_e1_encoder_validation.json"))
    p.add_argument("--tau", type=float, default=.07)
    p.add_argument("--eps-alpha", type=float, default=.01)
    p.add_argument("--k-max", type=int, default=10)
    args = p.parse_args()
    if args.smoke:
        raise SystemExit(0 if run_smoke(args.output) else 1)
    if not args.data_root or not args.split:
        p.error("--data_root and --split are required outside smoke")
    split = load_split(args.split)
    charset = json.loads(args.charset.read_text(encoding="utf-8"))
    chars = charset["target_string"]
    if args.dummy:
        es, ec, shas = DummyEs(), DummyEc(), {"content_encoder": "dummy", "style_encoder": "dummy"}
    else:
        if not args.ckpt_dir:
            p.error("--ckpt_dir is required unless --dummy")
        es, ec, shas = load_real_encoders(args.variant_path.resolve(), args.ckpt_dir)
    cfg = DeltaConfig(tau=args.tau, eps_alpha=args.eps_alpha, k_max=args.k_max)
    results = run_battery(es, ec, args.data_root, split, args.ref8, chars,
                          args.device, args.precision, cfg)
    results["checkpoint_sha256"] = shas
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print_summary(results)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
