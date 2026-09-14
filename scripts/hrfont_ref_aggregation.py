"""Target-independent, reference-balanced global9 residual reader (R)."""
import torch
from torch import nn
import torch.nn.functional as F


class RefGlobal9Reader(nn.Module):
    def __init__(self, channels=1024, hidden=128, heads=4):
        super().__init__()
        self.heads = heads
        self.hidden = hidden
        self.queries = nn.Parameter(torch.randn(9, hidden) * 0.02)
        self.key = nn.Linear(channels, hidden)
        self.value = nn.Linear(channels, hidden)
        self.out = nn.Linear(hidden, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, maps, keep):
        # [B,K,C,3,3]; keep is ref validity, NOT the CFG mask.
        b, k, c, h, w = maps.shape
        if (h, w) != (3, 3) or keep.shape != (b, k):
            raise ValueError("R expects B,K,C,3,3 and a B,K validity mask")
        if not bool(keep.any(dim=1).all()):
            raise ValueError("R requires at least one valid reference per sample")
        weight = keep.to(maps.dtype)[:, :, None, None, None]
        count = weight.sum(1)
        mean = (maps * weight).sum(1) / count
        x = maps.permute(0, 1, 3, 4, 2).reshape(b * k, 9, c)
        d = self.hidden // self.heads
        key = self.key(x).reshape(b * k, 9, self.heads, d).transpose(1, 2)
        value = self.value(x).reshape(b * k, 9, self.heads, d).transpose(1, 2)
        query = self.queries.reshape(9, self.heads, d).permute(1, 0, 2)[None]
        # Matching is normalized; values retain their style-relevant amplitude.
        with torch.autocast(device_type=maps.device.type, enabled=False):
            logits = torch.matmul(F.layer_norm(query.float(), (d,)),
                                  F.layer_norm(key.float(), (d,)).transpose(-1, -2)) / d**0.5
            read = logits.softmax(-1) @ value.float()
        read = read.transpose(1, 2).reshape(b, k, 9, self.hidden)
        read = (read * keep[:, :, None, None]).sum(1) / keep.sum(1)[:, None, None]
        residual = self.out(read.to(self.out.weight.dtype)).transpose(1, 2).reshape(b, c, 3, 3)
        return mean + residual.to(mean.dtype)


def attach_reader(model):
    # Adding the module must not change training episode/noise streams.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(3407 * 1009 + 9)
        model.ref_reader = RefGlobal9Reader()


def gather_maps(es_cache, samples, device):
    rows = [torch.stack([es_cache.spatial_tensor(split, font, cp) for cp in refs])
            for split, font, refs in zip(samples['split'], samples['font_stem'], samples['ref_chars'])]
    k = max(len(row) for row in rows)
    return torch.stack([F.pad(row, (0, 0, 0, 0, 0, 0, 0, k - len(row))) for row in rows]).to(device)


def apply_reader(model, maps, samples, cfg_mask=None):
    if not hasattr(model, 'ref_reader'):
        return maps
    lengths = torch.tensor([len(r) for r in samples['ref_chars']], device=maps.device)
    keep = torch.arange(maps.shape[1], device=maps.device)[None] < lengths[:, None]
    result = model.ref_reader(maps, keep)
    if cfg_mask is not None:
        result = result.masked_fill(cfg_mask[:, None, None, None], 0)
    return result


def enable_local_count_norm(unet):
    # Only style cross-attention in up blocks receives a global9+local mask.
    for block in unet.up_blocks:
        for module in block.modules():
            if module.__class__.__name__ == 'CrossAttention':
                module.local_count_norm = True
