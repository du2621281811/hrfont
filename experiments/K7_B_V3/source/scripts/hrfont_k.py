"""Original K specification: I LocalMemory/SetOffset, non-vanishing TC use."""
import types
import torch
from torch import nn
from scripts.hrfont_i import IModel
from scripts.hrfont_h import attention_once


def forced_attention(self, hidden_states, context=None, mask=None):
    global_value = attention_once(self, hidden_states, context, mask)
    if context is None or self._k_local is None or self._k_beta == 0:
        return global_value
    local = attention_once(self, hidden_states, self._k_local)
    with torch.no_grad():
        g = global_value.float().square().mean((1, 2), keepdim=True).add(1e-6).sqrt()
        l = local.float().square().mean((1, 2), keepdim=True).add(1e-6).sqrt()
        scale = (g/l).clamp(.25, 4.)
        self._k_scale = scale.flatten().detach()
    return global_value + self._k_beta * scale.to(local.dtype) * local * self._k_active[:, None, None]


class OfficialOffset(nn.Module):
    """Matched K2 replacement: original RSI on first actual reference Ec."""
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, hidden, payload):
        donors, _, _, active, _, _ = payload
        return self.original(hidden, donors[:, 0]) * active[:, None, None, None]


class KModel(IModel):
    def __init__(self, base, arm='K1'):
        if arm not in ('K0', 'K1', 'K2-TC-', 'K2-Delta-'):
            raise ValueError(arm)
        super().__init__(base, arm)
        del self.local_gain
        self.use_tc = arm not in ('K0', 'K2-TC-')
        if not self.use_tc:
            self.reader = None
        for attn in self.up_attn:
            attn.forward = types.MethodType(forced_attention, attn)
            attn._k_local = None
            attn._k_beta = 0.
            attn._k_scale = None
        for block in base.unet.up_blocks:
            if not hasattr(block, 'sc_interpreter_offsets'):
                continue
            if arm == 'K0':
                block.rsi_enabled = False
            if arm == 'K2-Delta-':
                block.sc_interpreter_offsets = nn.ModuleList([
                    OfficialOffset(m.original) for m in block.sc_interpreter_offsets])
            # Gate the actual structural residual, not only the offset input.
            for zero in block.zero_convs:
                zero._k_active = None
                zero.register_forward_hook(self._gate_structure)

    @staticmethod
    def _gate_structure(module, inputs, output):
        active = module._k_active
        return output if active is None else output * active[:, None, None, None]

    def conditions(self, style, refs, query, keep, cfg_mask):
        old = style.flatten(2).transpose(1, 2).masked_fill(cfg_mask[:, None, None], 0)
        if self.reader is None:
            return old, None, ~cfg_mask, None
        local, readout = self.reader(refs, query, keep)
        return old, local.masked_fill(cfg_mask[:, None, None], 0), ~cfg_mask, readout

    def denoise(self, noisy, timestep, style, content, structure, context, gate=1.):
        old, local, active, _ = context
        for attn in self.up_attn:
            attn._k_local, attn._k_active, attn._k_beta = local, active, .8 * gate
        ref = old.mean(1)
        payload = [(d, a, c, enabled, ref, timestep) for d, a, c, enabled in structure]
        for block in self.base.unet.up_blocks:
            if hasattr(block, 'zero_convs'):
                enabled = structure[-block.upblock_index-2][3]
                for zero in block.zero_convs:
                    zero._k_active = enabled
        return self.base(x_t=noisy, timesteps=timestep, content_images=None,
                         content_encoder_downsample_size=3, style_features=style,
                         content_features=content, structure_features=payload,
                         style_seq_tokens=old, style_seq_mask=None)

    def forward(self, noisy, timestep, style, refs, query, keep, content, structure, cfg_mask, gate):
        context = self.conditions(style, refs, query, keep, cfg_mask)
        pred, offset = self.denoise(noisy, timestep, style.masked_fill(cfg_mask[:, None, None, None], 0),
                                    content, structure, context, gate)
        return pred, offset, context[3]
