"""I: online target-conditioned appearance memory and dynamic structural set routing."""
import copy
import types
import torch
from torch import nn
import torch.nn.functional as F
from scripts.hrfont_h import attention_once


class LocalMemory(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.blocks = copy.deepcopy(encoder.blocks[:3]).requires_grad_(True)
        self.shallow = nn.Conv2d(128, 256, 1)
        self.q = nn.Linear(256, 256)
        self.k = nn.Linear(256, 256)
        self.v = nn.Linear(256, 256)
        self.slots = nn.Parameter(torch.randn(144, 256) * .02)
        self.mlp = nn.Sequential(nn.LayerNorm(256), nn.Linear(256,512), nn.GELU(), nn.Linear(512,256))
        self.out = nn.Linear(256,1024)
        self.readout = nn.Linear(256,128)

    def forward(self, refs, query, keep):
        b, n = refs.shape[:2]
        assert keep.shape == (b,n) and keep.any(1).all()
        # Packed valid images: no wasted encoder passes for padded shots.
        x = refs[keep]
        shallow = None
        for i, blocks in enumerate(self.blocks):
            for block in blocks:
                x = block(x)
            if i == 1:
                shallow = self.shallow(x)
        tokens = torch.cat([shallow.flatten(2).transpose(1,2), x.flatten(2).transpose(1,2)],1)
        s = tokens.shape[1]  # 24^2 + 12^2 = 720 per reference
        padded = tokens.new_zeros(b,n,s,256)
        padded[keep] = tokens
        memory = padded.flatten(1,2)
        q = (self.q(query.flatten(2).transpose(1,2)) + self.slots).reshape(b,144,4,64).transpose(1,2)
        k = self.k(memory).reshape(b,n*s,4,64).transpose(1,2)
        v = self.v(memory).reshape(b,n*s,4,64).transpose(1,2)
        mask = keep[:,:,None].expand(-1,-1,s).flatten(1)
        with torch.autocast(device_type=refs.device.type, enabled=False):
            score = F.layer_norm(q.float(),(64,)) @ F.layer_norm(k.float(),(64,)).transpose(-1,-2) / 8
            score = score.masked_fill(~mask[:,None,None,:], -torch.inf)
            local = (score.softmax(-1) @ v.float()).transpose(1,2).reshape(b,144,256)
        local = local + self.mlp(local)
        return self.out(local), self.readout(local)


class SetOffset(nn.Module):
    """Per-layer, per-position, per-denoising-step donor simplex; no donor-ID parameters."""
    def __init__(self, original):
        super().__init__()
        self.original = original
        c = original.style_proj_in.in_channels
        self.query = nn.Conv2d(original.content_proj_in.in_channels,32,1)
        self.key = nn.Conv2d(c,32,3,padding=1)
        self.content = nn.Conv2d(c,32,1)
        self.ref = nn.Linear(1024,32)
        self.time = nn.Sequential(nn.Linear(2,32),nn.SiLU(),nn.Linear(32,32))
        self.strength = nn.Parameter(torch.tensor(.1))

    def forward(self, hidden, payload):
        donors, alpha, neutral, active, ref, timestep = payload
        b,m,c,h,w = donors.shape
        assert hidden.shape[-2:] == (h,w)
        keys = self.key(donors.flatten(0,1)).reshape(b,m,32,h,w)
        tt = timestep.float().reshape(b,1)/1000
        q = self.query(hidden) + self.content(neutral)
        q = q + (self.ref(ref) + self.time(torch.cat([tt,tt.square()],1)))[:,:,None,None]
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            logits = (F.normalize(q.float(),dim=1)[:,None] * F.normalize(keys.float(),dim=2)).sum(2)
            logits = logits * self.strength.float() + alpha.float().clamp_min(1e-12).log()[:,:,None,None]
            logits = logits.masked_fill(alpha[:,:,None,None] <= 0, -torch.inf)
            weights = logits.softmax(1)
            mixed = (weights[:,:,None] * donors.float()).sum(1)
        # Drop AFTER routing, so source/CFG drop cannot leave affine-offset leakage.
        offset = self.original(hidden, mixed.to(hidden.dtype))
        return offset * active[:,None,None,None]


def residual_attention(self, hidden_states, context=None, mask=None):
    old = attention_once(self,hidden_states,context,mask)
    if context is None or self._i_local is None:
        return old
    local = attention_once(self,hidden_states,self._i_local)
    return old + self._i_gain * self._i_ramp * local * self._i_active[:,None,None]


class IModel(nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        self.base, self.arm = base, arm
        self.reader = LocalMemory(base.style_encoder)
        self.local_gain = nn.Parameter(torch.tensor(.01))
        self.register_buffer('teacher_mean',torch.zeros(128))
        self.register_buffer('teacher_std',torch.ones(128))
        self.up_attn = []
        for name, module in base.unet.named_modules():
            if module.__class__.__name__ == 'CrossAttention' and name.startswith('up_blocks.') and name.endswith('attn2'):
                module.forward = types.MethodType(residual_attention,module)
                module._i_local = None
                self.up_attn.append(module)
        for block in base.unet.up_blocks:
            if hasattr(block,'sc_interpreter_offsets'):
                block.sc_interpreter_offsets = nn.ModuleList([SetOffset(x) for x in block.sc_interpreter_offsets])
        assert self.up_attn

    def conditions(self,style,refs,query,keep,cfg_mask):
        local, pred = self.reader(refs,query,keep)
        active = ~cfg_mask
        old = style.flatten(2).transpose(1,2).masked_fill(cfg_mask[:,None,None],0)
        return old, local.masked_fill(cfg_mask[:,None,None],0), active, pred

    def denoise(self,noisy,timestep,style,content,structure,context,gate=1.):
        old,local,active,_ = context
        # bypass Module.__setattr__ for shared parameter references, not duplicate registration
        for attn in self.up_attn:
            attn._i_local, attn._i_active, attn._i_ramp = local, active, gate
            object.__setattr__(attn,'_i_gain',self.local_gain)
        ref = old.mean(1)
        payload = [(d,a,c,enabled,ref,timestep) for d,a,c,enabled in structure]
        return self.base(x_t=noisy,timesteps=timestep,content_images=None,
            content_encoder_downsample_size=3,style_features=style,content_features=content,
            structure_features=payload,style_seq_tokens=old,style_seq_mask=None)

    def forward(self,noisy,timestep,style,refs,query,keep,content,structure,cfg_mask,gate):
        context = self.conditions(style,refs,query,keep,cfg_mask)
        pred, offset = self.denoise(noisy,timestep,style.masked_fill(cfg_mask[:,None,None,None],0),
                                    content,structure,context,gate)
        return pred, offset + context[3].sum()*0, context[3]

    def train_state(self):
        return {k:v for k,v in self.state_dict().items() if not k.startswith(('base.style_encoder.','base.content_encoder.'))}

    def load_train_state(self,state):
        missing,extra=self.load_state_dict(state,strict=False)
        assert not extra and all(k.startswith(('base.style_encoder.','base.content_encoder.')) for k in missing)
