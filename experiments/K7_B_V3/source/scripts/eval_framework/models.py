#!/usr/bin/env python3
"""E12 模型；默认不下载预训练权重，适合离线 GPU 节点。"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def _backbone(in_channels=3):
    net = resnet18(weights=None)
    if in_channels != 3: net.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
    dim = net.fc.in_features; net.fc = nn.Identity()
    return net, dim


class PhiS2(nn.Module):
    def __init__(self, in_channels=3, feature_dim=512):
        super().__init__(); self.backbone, dim = _backbone(in_channels); self.projection = nn.Identity() if dim == feature_dim else nn.Linear(dim, feature_dim)
    def forward(self, x): return F.normalize(self.projection(self.backbone(x)), dim=1)


class IDCLS(nn.Module):
    def __init__(self, num_classes, in_channels=3, feature_dim=512):
        super().__init__(); self.backbone, dim = _backbone(in_channels); self.embedding = nn.Identity() if dim == feature_dim else nn.Linear(dim, feature_dim); self.head = nn.Linear(feature_dim, num_classes)
    def forward(self, x): return self.head(self.embedding(self.backbone(x)))


class MembershipVerifier(nn.Module):
    def __init__(self, encoder: PhiS2, feature_dim=512, hidden_dim=256, freeze_encoder=True):
        super().__init__(); self.encoder=encoder
        if freeze_encoder:
            for p in self.encoder.parameters(): p.requires_grad_(False)
        self.ref_aggregator=nn.Sequential(nn.Linear(feature_dim*2, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,feature_dim))
        self.fusion=nn.Sequential(nn.Linear(feature_dim*4,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1))
    def forward(self, query, refs):
        batch,n,c,h,w=refs.shape; q=self.encoder(query); r=self.encoder(refs.reshape(batch*n,c,h,w)).reshape(batch,n,-1)
        aggregate=self.ref_aggregator(torch.cat((r.mean(1),r.max(1).values),1))
        aggregate=F.normalize(aggregate,dim=1)
        return self.fusion(torch.cat((q,aggregate,torch.abs(q-aggregate),q*aggregate),1)).squeeze(1)


def load_phi_checkpoint(path, in_channels=3, feature_dim=512, device="cpu"):
    model=PhiS2(in_channels,feature_dim); payload=torch.load(path,map_location=device,weights_only=False)
    state=payload.get("model",payload); model.load_state_dict(state); return model
