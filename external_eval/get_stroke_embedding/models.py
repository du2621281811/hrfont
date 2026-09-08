"""Standalone stroke feature modules extracted from the training model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class StrokeFeatureExtractor(nn.Module):
    def __init__(self, feature_layers=(3, 8, 17, 26)):
        super().__init__()
        try:
            vgg = models.vgg19(weights=None).features
        except TypeError:
            vgg = models.vgg19(pretrained=False).features

        self.slices = nn.ModuleList()
        previous = 0
        for layer_index in feature_layers:
            self.slices.append(nn.Sequential(*vgg[previous:layer_index + 1]))
            previous = layer_index + 1

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image + 1.0) / 2.0
        image = (image - self.mean) / self.std
        features = []
        for slice_module in self.slices:
            image = slice_module(image)
            features.append(image)
        return features


class StrokeStyleTokenAdapter(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 512), token_dim=768, num_tokens=4, hidden_dim=1024):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.proj = nn.Sequential(
            nn.Linear(sum(in_channels), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_tokens * self.token_dim),
        )

    def forward(self, features):
        pooled = [feature.mean(dim=(2, 3)) for feature in features]
        merged = torch.cat(pooled, dim=1)
        return self.proj(merged).view(merged.shape[0], self.num_tokens, self.token_dim)


class StrokeEmbedder(nn.Module):
    def __init__(self, weights_path, device="cuda", mode="style_token"):
        super().__init__()
        self.device = torch.device(device)
        self.mode = mode
        self.feature_extractor = StrokeFeatureExtractor()
        self.style_adapter = StrokeStyleTokenAdapter()

        try:
            payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(weights_path, map_location="cpu")
        feature_state = payload["stroke_feature_extractor"]
        adapter_state = payload["stroke_style_adapter"]
        self.feature_extractor.load_state_dict(feature_state, strict=True)
        self.style_adapter.load_state_dict(adapter_state, strict=True)

        self.requires_grad_(False)
        self.to(self.device)
        self.eval()

    @torch.inference_mode()
    def extract(self, image_rgb_255):
        image = torch.from_numpy(image_rgb_255.copy()).to(dtype=torch.float32)
        image = image.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        image = image.to(self.device) / 255.0 * 2.0 - 1.0
        features = self.feature_extractor(image)

        if self.mode == "style_token":
            embedding = self.style_adapter(features).mean(dim=1)
        elif self.mode == "pooled_feature":
            embedding = torch.cat([feature.mean(dim=(2, 3)) for feature in features], dim=1)
        else:
            raise ValueError(f"unsupported stroke embedding mode: {self.mode}")

        embedding = F.normalize(embedding, dim=1, eps=1e-6)
        return embedding[0].cpu().numpy()
