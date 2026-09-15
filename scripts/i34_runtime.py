"""Independent I3/I4 model factory; never alters the running I1/I2 snapshot."""
import types
import torch
from torch import nn
from i_runtime import *
from scripts.hrfont_i import IModel, LocalMemory
from scripts.i34_components import InkReadout


def detail_forward(self, refs, query, keep):
    tokens, _ = LocalMemory.forward(self, refs, query, keep)
    # Teacher-free, actual generator-token supervision; omit readout compute at inference.
    image = self.ink(tokens) if self.training else tokens.new_zeros(tokens.shape[0], 1, 96, 96)
    return tokens, image


def model_for(arm, device, output):
    if arm not in ('I3', 'I4'):
        raise ValueError(arm)
    args = args_for(PARENT0)
    T._verify_caches(args, PARENT0)
    torch.manual_seed(3407)
    base = T.FontDiffuserModel(unet=T.build_unet(args), style_encoder=T.build_style_encoder(args),
                              content_encoder=T.build_content_encoder(args))
    T._load_parent(base, PARENT0, output)
    base.style_encoder.requires_grad_(False).eval()
    base.content_encoder.requires_grad_(False).eval()
    model = IModel(base, arm)
    model.reader.readout = nn.Identity()
    model.reader.ink = InkReadout()
    model.reader.forward = types.MethodType(detail_forward, model.reader)
    return model.to(device), args
