import math
import torch
import torch.nn as nn

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin,
                                           register_to_config)

class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(
        self,
        x_t,
        timesteps,
        content_images,
        content_encoder_downsample_size,
        style_images=None,
        style_features=None,
        structure_features=None,
        content_features=None,
        support_tokens=None,
        style_seq_tokens=None,
        style_seq_mask=None,
    ):
        if style_features is None:
            if style_images is None:
                raise ValueError("style_images or precomputed style_features is required")
            style_img_feature, _, _ = self.style_encoder(style_images)
        else:
            style_img_feature = style_features
            if style_img_feature.ndim == 2:
                style_img_feature = style_img_feature[:, :, None, None]
            if style_img_feature.ndim != 4:
                raise ValueError("style_features must be [B,C] or [B,C,H,W]")

        batch_size, channel, height, width = style_img_feature.shape
        # Legacy path: flatten mean map → 9 tokens. F2-P/F3b-P: per-ref pooled h tokens.
        if style_seq_tokens is not None:
            style_hidden_states = style_seq_tokens
            style_mask = style_seq_mask
        else:
            style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
            style_mask = None

        if content_features is None:
            content_img_feature, content_residual_features = self.content_encoder(content_images)
            content_residual_features = list(content_residual_features) + [content_img_feature]
        else:
            content_residual_features = content_features
        if structure_features is None:
            structure_features = [torch.zeros_like(x) for x in content_residual_features]

        # F3: SupportAdapter tokens extend the up-path cross-attention context only.
        # The MCA down-path still sees the unmodified style map, so F2 vs F3 differ
        # in exactly one place.
        if support_tokens is not None and support_tokens.numel() > 0:
            style_hidden_states = torch.cat([style_hidden_states, support_tokens.to(style_hidden_states.dtype)], dim=1)
            if style_mask is not None:
                # support positions are always valid when provided
                bsz, k_s, _ = support_tokens.shape
                support_mask = torch.ones(bsz, k_s, dtype=torch.bool, device=style_mask.device)
                style_mask = torch.cat([style_mask, support_mask], dim=1)

        style_context = (style_hidden_states, style_mask) if style_mask is not None else style_hidden_states
        input_hidden_states = [style_img_feature, content_residual_features, style_context]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            structure_features=structure_features,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """
    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder

    def forward(
        self,
        x_t,
        timesteps,
        cond,
        content_encoder_downsample_size,
        version,
    ):
        content_images = cond[0]
        style_images = cond[1]
        style_img_feature = cond[2] if len(cond) > 2 else None
        structure_features = cond[3] if len(cond) > 3 else None
        if style_img_feature is None:
            style_img_feature, _, _ = self.style_encoder(style_images)
        elif style_img_feature.ndim == 2:
            style_img_feature = style_img_feature[:, :, None, None]

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)

        # Get content feature
        content_img_feture, content_residual_features = self.content_encoder(content_images)
        content_residual_features = list(content_residual_features) + [content_img_feture]
        if structure_features is None:
            structure_features = [torch.zeros_like(x) for x in content_residual_features]

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            structure_features=structure_features,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred
