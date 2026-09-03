#!/usr/bin/env python3
"""Monkeypatch FontDiffuser StyleRSIUpBlock2D to pick structure feats by channel+HW.

Official code uses style_structure_features[-upblock_index-2], which breaks for
Ec arch[256] (wrong channel). Call apply_rsi_pick_by_hw() ONLY in arch256 trainers
so hybrid / data-ablation paths stay unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _want_channels(sc_inter) -> int:
    # GroupNorm on style branch is sized to style_feat_in_channels
    return int(sc_inter.gnorm_s.num_channels)


def pick_structure_feat(feats, want_c: int, hw: tuple[int, int], log_list=None):
    """Pick feat with matching channels; interpolate spatial to hw if needed."""
    cands = []
    for f in feats:
        if torch.is_tensor(f) and f.dim() == 4 and int(f.shape[1]) == want_c:
            cands.append(f)
    if not cands:
        # closest channel count
        scored = []
        for f in feats:
            if torch.is_tensor(f) and f.dim() == 4:
                scored.append((abs(int(f.shape[1]) - want_c), f))
        if not scored:
            raise RuntimeError(f"no structure feats to pick want_c={want_c}")
        scored.sort(key=lambda x: x[0])
        feat = scored[0][1]
        # 1x1 proj not available — fail loudly if channel mismatch
        if int(feat.shape[1]) != want_c:
            raise RuntimeError(
                f"no feat with channels={want_c}; closest={tuple(feat.shape)}"
            )
    else:
        # prefer exact spatial match, else largest spatial among channel matches
        h, w = hw
        exact = [f for f in cands if int(f.shape[2]) == h and int(f.shape[3]) == w]
        feat = exact[-1] if exact else max(cands, key=lambda f: int(f.shape[2]) * int(f.shape[3]))

    if int(feat.shape[2]) != hw[0] or int(feat.shape[3]) != hw[1]:
        feat = F.interpolate(feat, size=hw, mode="bilinear", align_corners=False)
    if log_list is not None:
        log_list.append((want_c, hw, tuple(feat.shape)))
    return feat


def apply_rsi_pick_by_hw(verbose: bool = False):
    from src.modules.unet_blocks import StyleRSIUpBlock2D

    if getattr(StyleRSIUpBlock2D, "_rsi_pick_by_hw", False):
        return

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        style_structure_features,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
    ):
        total_offset = 0
        picks = [] if verbose else None

        for i, (sc_inter_offset, dcn_deform, resnet, attn) in enumerate(
            zip(self.sc_interpreter_offsets, self.dcn_deforms, self.resnets, self.attentions)
        ):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            want_c = _want_channels(sc_inter_offset)
            hw = (int(res_hidden_states.shape[2]), int(res_hidden_states.shape[3]))
            style_content_feat = pick_structure_feat(
                style_structure_features, want_c, hw, log_list=picks
            )

            offset = sc_inter_offset(res_hidden_states, style_content_feat)
            offset = offset.contiguous()
            total_offset += torch.mean(torch.abs(offset))

            res_hidden_states = res_hidden_states.contiguous()
            res_hidden_states = dcn_deform(res_hidden_states, offset)
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn), hidden_states, encoder_hidden_states
                )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, context=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        if verbose and picks:
            print(f"[rsi_pick] upblock={self.upblock_index} picks={picks}", flush=True)

        return hidden_states, total_offset / self.num_layers

    StyleRSIUpBlock2D.forward = forward
    StyleRSIUpBlock2D._rsi_pick_by_hw = True
    print("[patch] StyleRSIUpBlock2D.forward -> pick_by_hw", flush=True)
