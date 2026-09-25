"""H: target-aligned local appearance completion, shared by train and inference."""
import math
import types
import torch
from torch import nn
import torch.nn.functional as F


def lr_factor(update, total=10000):
    # 500 warmup, stable to 5k, cosine to 10% at 10k. update is one-based.
    if update <= 500:
        return update / 500
    if update <= total // 2:
        return 1.0
    progress = min(1., (update - total // 2) / (total - total // 2))
    return .1 + .9 * (1 + math.cos(math.pi * progress)) / 2


class AppearanceMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(256, 256)
        self.k = nn.Linear(256, 256)
        self.v = nn.Linear(256, 256)
        self.mlp = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
        self.out = nn.Linear(256, 1024)
        self.readout = nn.Linear(256, 128)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 12), torch.linspace(-1, 1, 12), indexing='ij')
        freq = 2 ** torch.linspace(0, 5, 64)
        pos = torch.cat([torch.sin(x.flatten()[:, None] * freq), torch.cos(x.flatten()[:, None] * freq),
                         torch.sin(y.flatten()[:, None] * freq), torch.cos(y.flatten()[:, None] * freq)], -1)
        self.register_buffer('position', pos)

    def forward(self, refs, query, keep):
        # refs [B,K,144,256]; query frozen Ec [B,256,12,12].
        b, k, s, c = refs.shape
        if c != 256 or s != 144 or keep.shape != (b, k) or not keep.any(1).all():
            raise ValueError('invalid reference tensor or empty reference set')
        # Clear padding before any projection, including NaN-valued padding.
        refs = refs.masked_fill(~keep[:, :, None, None], 0)
        q = self.q(query.flatten(2).transpose(1, 2)) + self.position
        q = q.reshape(b, 144, 4, 64).permute(0, 2, 1, 3)[:, None]
        key = self.k(refs).reshape(b, k, s, 4, 64).permute(0, 1, 3, 2, 4)
        value = self.v(refs).reshape(b, k, s, 4, 64).permute(0, 1, 3, 2, 4)
        with torch.autocast(device_type=refs.device.type, enabled=False):
            logits = F.layer_norm(q.float(), (64,)) @ F.layer_norm(key.float(), (64,)).transpose(-1, -2) / 8
            aligned = (logits.softmax(-1) @ value.float()).permute(0, 1, 3, 2, 4).reshape(b, k, 144, 256)
            memory = (aligned * keep[:, :, None, None]).sum(1) / keep.sum(1)[:, None, None]
        memory = memory + self.mlp(memory)
        return self.out(memory), self.readout(memory), memory


def attention_once(module, hidden, context=None, mask=None, count=0):
    context = hidden if context is None else context
    q = module.reshape_heads_to_batch_dim(module.to_q(hidden))
    k = module.reshape_heads_to_batch_dim(module.to_k(context))
    v = module.reshape_heads_to_batch_dim(module.to_v(context))
    valid = None if mask is None else mask[:, None, :].bool().repeat_interleave(module.heads, 0)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        scores = q.float() @ k.float().transpose(-1, -2) * module.scale
        if count:
            if mask is None or (context.shape[1] - 9) % count:
                raise ValueError('N requires global9 + T_per_ref * K valid local tokens')
            n = mask[:, 9:].sum(-1).float().div(count).clamp_min(1)
            bias = torch.zeros(mask.shape, device=hidden.device, dtype=torch.float32)
            bias[:, 9:] = -n.log()[:, None]
            scores = scores + bias.repeat_interleave(module.heads, 0)[:, None]
        if valid is not None:
            if not valid.any(-1).all():
                raise ValueError('all attention positions masked')
            scores = scores.masked_fill(~valid, float('-inf'))
        out = scores.softmax(-1) @ v.float()
    out = module.reshape_batch_dim_to_heads(out.to(q.dtype))
    return module.to_out(out)


def attention_forward(self, hidden_states, context=None, mask=None):
    result = attention_once(self, hidden_states, context, mask, getattr(self, '_h_count', 0))
    old = getattr(self, '_h_old', None)
    gate = getattr(self, '_h_gate', 1.)
    if context is not None and old is not None and gate < 1:
        result = gate * result + (1 - gate) * attention_once(self, hidden_states, old)
    return result


class HModel(nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        self.base = base
        self.arm = arm
        self.memory_arm = arm in ('H3', 'H4', 'H-D+', 'H-D-')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3407 * 1009 + 144)
            if self.memory_arm:
                self.reader = AppearanceMemory()
            elif arm in ('H1', 'H2'):
                self.local = nn.Linear(256, 1024)
        self.register_buffer('teacher_mean', torch.zeros(128))
        self.register_buffer('teacher_std', torch.ones(128))
        self.up_attn = []  # references, not a second ModuleList/state_dict copy
        for name, module in base.unet.named_modules():
            if module.__class__.__name__ == 'CrossAttention':
                module.forward = types.MethodType(attention_forward, module)
                if name.startswith('up_blocks.') and name.endswith('attn2'):
                    if module.to_k.in_features != 1024:
                        raise ValueError(name)
                    self.up_attn.append(module)
        if not self.up_attn:
            raise ValueError('no up-path style consumers found')

    def conditions(self, style, refs, query, keep, cfg_mask):
        b = style.shape[0]
        old = style.flatten(2).transpose(1, 2).masked_fill(cfg_mask[:, None, None], 0)
        pred = None
        if self.memory_arm:
            seq, pred, _ = self.reader(refs, query, keep)
            seq = seq.masked_fill(cfg_mask[:, None, None], 0)
            mask = torch.ones(seq.shape[:2], device=seq.device, dtype=torch.bool)
        elif self.arm in ('H1', 'H2'):
            if self.arm == 'H1':
                r = refs.reshape(b * refs.shape[1], 12, 12, 256).permute(0, 3, 1, 2)
                refs = F.adaptive_avg_pool2d(r, 4).flatten(2).transpose(1, 2).reshape(b, -1, 16, 256)
            local = self.local(refs.flatten(1, 2))
            seq = torch.cat([old, local], 1)
            lm = keep[:, :, None].expand(-1, -1, refs.shape[2]).flatten(1)
            lm = lm & ~cfg_mask[:, None]
            mask = torch.cat([torch.ones((b, 9), device=seq.device, dtype=torch.bool), lm], 1)
        else:
            seq, mask = old, None
        return old, seq, mask, pred

    def denoise(self, noisy, timestep, style, content, structure, context, gate=1.):
        old, seq, mask, _ = context
        for attn in self.up_attn:
            attn._h_old = old if self.arm != 'H0' and gate < 1 else None
            attn._h_gate = gate
            attn._h_count = 16 if self.arm == 'H1' else 144 if self.arm == 'H2' else 0
        return self.base(x_t=noisy, timesteps=timestep, content_images=None,
                         content_encoder_downsample_size=3, style_features=style,
                         content_features=content, structure_features=structure,
                         style_seq_tokens=seq, style_seq_mask=mask)

    def forward(self, noisy, timestep, style, refs, query, keep, content, structure, cfg_mask, gate):
        context = self.conditions(style, refs, query, keep, cfg_mask)
        style = style.masked_fill(cfg_mask[:, None, None, None], 0)
        noise, offset = self.denoise(noisy, timestep, style, content, structure, context, gate)
        pred = context[3]
        # H4 keeps readout in the graph even with zero supervision.
        zero = noise.sum() * 0 if pred is None else pred.sum() * 0
        return noise, offset + zero, pred

    def train_state(self):
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith(('base.style_encoder.', 'base.content_encoder.'))}

    def load_train_state(self, state):
        missing, extra = self.load_state_dict(state, strict=False)
        assert not extra and all(k.startswith(('base.style_encoder.', 'base.content_encoder.')) for k in missing)
