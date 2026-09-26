"""I3/I4 building blocks. Not a training entry point or a launch-ready claim."""
import torch
from torch import nn
import torch.nn.functional as F


class InkReadout(nn.Module):
    """Training-only readout attached to the actual 144x1024 generator tokens."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1024, 64)

    @staticmethod
    def unpatchify(patches):
        if patches.ndim != 3 or patches.shape[1:] != (144, 64):
            raise ValueError('Expected B x 144 x 64 patches')
        b = patches.shape[0]
        return patches.reshape(b, 12, 12, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, 1, 96, 96)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (144, 1024):
            raise ValueError('Expected B x 144 x 1024 actual generator tokens')
        return self.unpatchify(self.proj(tokens).float().sigmoid())


def gray(image):
    if image.ndim != 4 or image.shape[1] not in (1, 3):
        raise ValueError('Expected Bx1xHxW or Bx3xHxW')
    image = image.float()
    if image.shape[1] == 1:
        return image
    weights = image.new_tensor([.299, .587, .114])[None, :, None, None]
    return (image * weights).sum(1, keepdim=True)


def sobel(image):
    kx = image.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
    kernels = torch.stack((kx, kx.T))[:, None]
    # Replication avoids inventing a dark boundary around the white canvas.
    return F.conv2d(F.pad(image, (1, 1, 1, 1), mode='replicate'), kernels)


def detail_distance(prediction, target):
    """Per-example FP32 loss; target-derived masks and targets are detached.

    Defaults: normalized Sobel threshold .02, one-pixel (3x3) edge band,
    M=1+2*ink+2*band in [1,5], scales96/48 weighted1/.5.
    """
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        pred, gt = gray(prediction), gray(target.detach())
        if pred.shape != gt.shape or pred.shape[-2:] != (96, 96):
            raise ValueError('Expected aligned native96 image batches')
        if not torch.isfinite(gt).all() or gt.min() < -1e-6 or gt.max() > 1+1e-6:
            raise ValueError('Target must be finite grayscale/RGB in [0,1]')
        result = pred.new_zeros(pred.shape[0])
        for size, factor in ((96, 1.), (48, .5)):
            p = pred if size == 96 else F.avg_pool2d(pred, 2)
            y = gt if size == 96 else F.avg_pool2d(gt, 2)
            yg = sobel(y)
            band = F.max_pool2d((yg.abs().amax(1, keepdim=True) > .02).float(), 3, 1, 1)
            weight = (1 + 2*(1-y) + 2*band).clamp(1, 5)
            denom = weight.sum((1, 2, 3))
            pixel = (weight*(p-y).abs()).sum((1, 2, 3))/denom
            edge = (weight*(sobel(p)-yg).abs()).sum((1, 2, 3))/(2*denom)
            result = result + factor*(pixel + .5*edge)
        return result


def raw_x0(noisy, epsilon, alpha_bar, timesteps):
    """Unclipped epsilon-parameterized x0 in image [0,1] coordinates."""
    with torch.autocast(device_type=noisy.device.type, enabled=False):
        alpha = alpha_bar.to(noisy.device, torch.float32)[timesteps.long()].reshape(-1, 1, 1, 1)
        if not ((alpha > 0) & (alpha <= 1)).all():
            raise ValueError('Invalid alpha_bar')
        clean = (noisy.float()-(1-alpha).sqrt()*epsilon.float())/alpha.sqrt()
        return clean/2+.5, alpha.flatten()


def visual_losses(tokens_image, prediction, target, conditional, alpha, step):
    """Already weighted extra objective; fixed task-batch denominator, no inverse SNR renormalization."""
    if conditional.dtype != torch.bool or conditional.shape != (target.shape[0],):
        raise ValueError('Expected per-example boolean conditional mask')
    if alpha.shape != conditional.shape:
        raise ValueError('Expected per-example alpha_bar')
    mask = conditional.float()
    tc = (detail_distance(tokens_image, target)*mask).mean()
    render = (detail_distance(prediction, target)*mask*alpha.float()).mean()
    ramp = min(1., max(0., step/1000.))
    return ramp*(.02*tc+.05*render), tc, render


def sampling_distribution(base_weights, group_ids, difficulty_weights, step):
    """Preserve each script's old mass while raising validated difficult-font exposure.

    Difficulty scoring/manifest preparation is deliberately separate and not
    implemented by this function. It accepts only the frozen audited weights.
    """
    base = torch.as_tensor(base_weights, dtype=torch.float64)
    groups = torch.as_tensor(group_ids, device=base.device)
    weights = torch.as_tensor(difficulty_weights, device=base.device, dtype=torch.float64)
    if base.ndim != 1 or base.numel() == 0 or base.shape != groups.shape or base.shape != weights.shape:
        raise ValueError('Mismatched sampling arrays')
    if not torch.isfinite(base).all() or (base < 0).any() or base.sum() <= 0:
        raise ValueError('Invalid base weights')
    if not torch.isfinite(weights).all() or not ((weights >= 1) & (weights <= 2)).all():
        raise ValueError('Difficulty weights must be in [1,2]')
    base = base/base.sum()
    hard = base*weights
    for group in groups.unique():
        mask = groups == group
        mass = base[mask].sum()
        if mass > 0:
            hard[mask] *= mass/hard[mask].sum()
    fraction = .5*min(1., max(0., (step-1000)/1000.))
    return (1-fraction)*base+fraction*hard
