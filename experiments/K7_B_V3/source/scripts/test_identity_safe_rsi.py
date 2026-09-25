#!/usr/bin/env python3
"""Gate for the F1/F2/F3 identity-safe RSI claim. No caches or checkpoints needed.

Checks two things:

  1. `StyleRSIUpBlockIdentitySafe` at init is bit-identical to the raw-skip path,
     i.e. adding RSI on top of F0 changes nothing at step 0.
  2. The rejected alternative -- zero-offset DeformConv -- is NOT identity, which is
     why the design uses a zero-init 1x1 conv on the residual instead.

Run: python scripts/test_identity_safe_rsi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/root/projects/hrfont")
VARIANT = ROOT / "code/variants/cn2west_f123_rsi/FontDiffuser"
sys.path.insert(0, str(VARIANT))

from src.modules.unet_blocks import StyleRSIUpBlockIdentitySafe  # noqa: E402

B, HW = 2, 24
SKIP_C, OUT_C, PREV_C, TEMB_C, STRUCT_C, CTX = 256, 256, 512, 256, 128, 1024


def build_inputs(device, dtype):
    hidden = torch.randn(B, PREV_C, HW, HW, device=device, dtype=dtype)
    skips = tuple(torch.randn(B, SKIP_C, HW, HW, device=device, dtype=dtype) for _ in range(2))
    # StyleRSIUpBlock*(upblock_index=1) reads structure_features[-3].
    structure = [None, None, torch.randn(B, STRUCT_C, HW, HW, device=device, dtype=dtype), None, None]
    temb = torch.randn(B, TEMB_C, device=device, dtype=dtype)
    context = torch.randn(B, 9, CTX, device=device, dtype=dtype)
    return hidden, skips, structure, temb, context


def check_block_identity(device, dtype) -> float:
    torch.manual_seed(3407)
    block = StyleRSIUpBlockIdentitySafe(
        in_channels=SKIP_C, out_channels=OUT_C, prev_output_channel=PREV_C,
        temb_channels=TEMB_C, num_layers=2, cross_attention_dim=CTX,
        structure_feature_begin=64, upblock_index=1, add_upsample=True,
    ).to(device=device, dtype=dtype).eval()

    hidden, skips, structure, temb, context = build_inputs(device, dtype)
    with torch.no_grad():
        block.rsi_enabled = True
        with_rsi, offset_on = block(hidden, skips, structure, temb, context)
        block.rsi_enabled = False
        without_rsi, offset_off = block(hidden, skips, structure, temb, context)

    max_abs = float((with_rsi - without_rsi).abs().max())
    print(f"  [{dtype}] max|with_RSI - raw_skip| = {max_abs:.3e}   "
          f"rsi_gain={block.rsi_gain():.3e}  offset_on={float(offset_on):.4f} "
          f"offset_off={float(offset_off):.4f}")
    return max_abs


def check_zero_offset_dcn_is_not_identity(device) -> float:
    """The alternative the design rejected: zero offsets still apply a learned 3x3 kernel."""
    from torchvision.ops import DeformConv2d

    torch.manual_seed(3407)
    dcn = DeformConv2d(SKIP_C, SKIP_C, kernel_size=(3, 3), stride=1, padding=1).to(device).eval()
    skip = torch.randn(B, SKIP_C, HW, HW, device=device)
    with torch.no_grad():
        warped = dcn(skip, torch.zeros(B, 18, HW, HW, device=device))
    max_abs = float((warped - skip).abs().max())
    print(f"  zero-offset DeformConv: max|DCN(skip,0) - skip| = {max_abs:.3e}")
    return max_abs


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    print("1. StyleRSIUpBlockIdentitySafe is identity at init")
    failures = []
    for dtype in (torch.float32, torch.float16) if device == "cuda" else (torch.float32,):
        if check_block_identity(device, dtype) != 0.0:
            failures.append(f"identity-safe block not exact in {dtype}")

    print("2. rejected alternative (zero-offset DeformConv) is not identity")
    if check_zero_offset_dcn_is_not_identity(device) == 0.0:
        failures.append("zero-offset DCN unexpectedly exact; design rationale needs revisiting")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
