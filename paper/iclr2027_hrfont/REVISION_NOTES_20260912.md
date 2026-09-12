# HR-Font paper revision notes — 2026-09-12

Revision base: `fba024d1`.

This revision applies the useful parts of `REVIEW_OPERATIONAL_20260912.md` while preserving the paper's central product story: the font bank supplies a structured space of compatible glyph variation, and the observed references personalize appearance.

| Review item | Decision | Applied change |
|---|---|---|
| A1 | Modify | D2−D1 is labeled as a canonicalized-geometry package comparison, not a single-factor appearance-removal result. D1n is conditional on D2 winning and on a need to separate preprocessing from carrier. |
| A2 | Correct and expand | Added verified FTransGAN, FCAGAN, and DRA-font citations. Corrected `CFFont` to FCAGAN and described DRA-font as diffusion-regularized adversarial learning. |
| A3 | Accept | Removed the self-questioning sentence from Method; D3−D2 remains a preregistered experiment question. |
| A4 | Expand | Added baseline protocol cards and separated direct cross-script competitors from architecture and mechanism controls. |
| A5 | Accept | Added the explicit distinction between anchor-relative residual candidates and fused absolute glyph features. |
| A6 | Modify | Kept alpha as a retrieval/logit prior and added zero-training entropy/effective-donor diagnostics before deciding whether it supports a claim. |
| D1–D8 | Selective accept | Replaced defensive explanations with concise method or protocol facts, shortened captions, and rewrote the legacy observation without claiming that local appearance itself is unimportant. |

## Citation provenance

- FTransGAN: DOI `10.1109/WACV48630.2021.00048`; verified against the WACV proceedings page and DOI metadata.
- FCAGAN: DOI `10.1016/j.patcog.2024.110709`; verified against Pattern Recognition metadata and the Cardiff author manuscript.
- DRA-font: DOI `10.1587/transinf.2026EDP7009`; verified against IEICE and DOI metadata.

## Experiment impact

The revision adds no mandatory training arm to the current screening queue. It adds retrieval diagnostics to Wave 0 and direct cross-script baseline audits. D1n remains a conditional attribution control; DRA-font remains related work until a faithful executable protocol is confirmed.

Changes are grouped into one coherent revision commit rather than one commit per sentence so collaborators can review the paper, experiment contract, and bibliography as a consistent unit.
