"""Support retrieval for HR-Font Stage B (§4.5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path("/root/projects/hrfont")
BANK = ROOT / "data/hrfont/e0_bank"
META = json.loads((BANK / "meta.json").read_text(encoding="utf-8"))
META_U = json.loads((ROOT / "data/unified_v1/meta.json").read_text(encoding="utf-8"))
P1_CHARS = list(META_U["L_p1"])
DONORS = list(META["donors"])
REF8 = list(META["ref8"])

THETA = 0.25
GAMMA = 0.35
MMR_LAMBDA = 0.7
TOP_K = 3
TOP_M = 3
TAU = 0.07

_GAP: dict[str, float] | None = None
_GAP_BEST: dict[str, str] | None = None
_CONTENT_VEC: dict[str, torch.Tensor] | None = None
_STYLE_PROTO: dict[str, torch.Tensor] | None = None


def _load_cache() -> None:
    global _GAP, _GAP_BEST, _CONTENT_VEC, _STYLE_PROTO
    if _CONTENT_VEC is not None:
        return
    gap = json.loads((BANK / "gap_b0_iou.json").read_text(encoding="utf-8"))
    _GAP = {r["ch"]: float(r["gap"]) for r in gap["rows"]}
    _GAP_BEST = {r["ch"]: r["best_ref"] for r in gap["rows"]}
    cache = torch.load(BANK / "cache/ec_es_r96.pt", map_location="cpu", weights_only=False)
    _CONTENT_VEC = cache["content_vec"]
    _STYLE_PROTO = cache["style_proto"]


def gap_for_char(ch: str) -> float:
    _load_cache()
    assert _GAP is not None
    return float(_GAP.get(ch, 1.0))


def candidate_q(target_ch: str) -> list[str]:
    latin = [c for c in P1_CHARS if c.isalnum()]
    qs = [q for q in latin + DONORS if q != target_ch and q in (_CONTENT_VEC or {})]
    _load_cache()
    assert _CONTENT_VEC is not None
    return [q for q in qs if q in _CONTENT_VEC]


def _norm_vec(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v.float().flatten(), dim=0, eps=1e-8)


def residual_vec(target_ch: str) -> torch.Tensor:
    _load_cache()
    assert _CONTENT_VEC is not None and _GAP_BEST is not None
    if target_ch not in _CONTENT_VEC:
        # Punctuation / missing in content cache — no support scoring
        dim = next(iter(_CONTENT_VEC.values())).numel()
        return torch.zeros(dim)
    v_c = _CONTENT_VEC[target_ch]
    ref = _GAP_BEST.get(target_ch, REF8[0])
    v_r = _CONTENT_VEC.get(ref, _CONTENT_VEC[target_ch])
    return _norm_vec(v_c - v_r)


def cover(target_ch: str, q: str) -> float:
    if q == target_ch:
        return 0.0
    _load_cache()
    assert _CONTENT_VEC is not None
    if target_ch not in _CONTENT_VEC or q not in _CONTENT_VEC:
        return 0.0
    res = residual_vec(target_ch)
    v_q = _norm_vec(_CONTENT_VEC[q])
    return float(torch.dot(res, v_q).clamp(-1, 1).item())


def mmr_select(target_ch: str, k: int = TOP_K, theta: float = THETA) -> list[tuple[str, float]]:
    cands = candidate_q(target_ch)
    scores = [(q, cover(target_ch, q)) for q in cands]
    scores = [(q, s) for q, s in scores if s >= theta]
    if not scores:
        return []
    selected: list[tuple[str, float]] = []
    _load_cache()
    assert _CONTENT_VEC is not None
    while len(selected) < k and scores:
        best_i, best_score = 0, -1e9
        for i, (q, cov) in enumerate(scores):
            if not selected:
                mmr = cov
            else:
                v_q = _norm_vec(_CONTENT_VEC[q])
                div = max(float(torch.dot(v_q, _norm_vec(_CONTENT_VEC[sq])).item()) for sq, _ in selected)
                mmr = MMR_LAMBDA * cov - (1 - MMR_LAMBDA) * div
            if mmr > best_score:
                best_score, best_i = mmr, i
        selected.append(scores.pop(best_i))
    return selected


def alpha_top_m(
    font: str,
    train_fonts: list[str],
    font_proto: torch.Tensor | None = None,
) -> list[tuple[str, float]]:
    _load_cache()
    assert _STYLE_PROTO is not None
    if font_proto is not None:
        q = font_proto.float().flatten()
    elif font in _STYLE_PROTO:
        q = _STYLE_PROTO[font].float().flatten()
    else:
        return []
    scores = []
    for g in train_fonts:
        if g == font or g not in _STYLE_PROTO:
            continue
        v = _STYLE_PROTO[g].float().flatten()
        scores.append((g, float(F.cosine_similarity(q[None], v[None]).item())))
    scores.sort(key=lambda x: -x[1])
    top = scores[:TOP_M]
    if not top:
        return []
    logits = torch.tensor([s / TAU for _, s in top])
    w = torch.softmax(logits, dim=0)
    return [(top[i][0], float(w[i])) for i in range(len(top))]


def pick_support(
    font: str,
    target_ch: str,
    train_fonts: list[str],
    png_fn,
    font_proto: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return support spec: empty if gap low, else imgs/weights/meta."""
    g = gap_for_char(target_ch)
    if g < GAMMA:
        return {"gap": g, "support_imgs": [], "weights": [], "qs": []}
    qs = mmr_select(target_ch)
    if not qs:
        return {"gap": g, "support_imgs": [], "weights": [], "qs": []}
    alpha = alpha_top_m(font, train_fonts, font_proto=font_proto)
    if not alpha:
        return {"gap": g, "support_imgs": [], "weights": [], "qs": []}
    imgs, weights, picked_q = [], [], []
    for q, cov in qs:
        for fnt, aw in alpha:
            p = png_fn(fnt, q)
            if p is None or (hasattr(p, "exists") and not p.exists()):
                continue
            imgs.append(p)
            weights.append(aw * cov)
            picked_q.append(q)
    return {"gap": g, "support_imgs": imgs, "weights": weights, "qs": picked_q, "alpha": alpha}
