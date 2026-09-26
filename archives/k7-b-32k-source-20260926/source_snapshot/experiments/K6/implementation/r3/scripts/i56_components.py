"""I5/I6 region-balanced detail objective; native96, detached GT masks, FP32."""
import math
import torch
import torch.nn.functional as F
from scripts.i34_components import gray, sobel, raw_x0


def region_masks(target):
    """Disjoint ink / nearby white / distant background, including enclosed white.

    The integral horizontal/vertical spans retain interiors of outlined glyphs,
    rather than classifying a large empty outline interior as distant background.
    These are supervision masks only; antialiased target pixels are unchanged.
    """
    ink = (target.detach() < .95).float()
    dilated = F.max_pool2d(ink, 5, 1, 2)
    horizontal = (ink.cumsum(-1) > 0) & (ink.flip(-1).cumsum(-1).flip(-1) > 0)
    vertical = (ink.cumsum(-2) > 0) & (ink.flip(-2).cumsum(-2).flip(-2) > 0)
    enclosed = (horizontal & vertical).float()
    near = torch.maximum(dilated, enclosed) * (1 - ink)
    distant = 1 - ink - near
    edge = F.max_pool2d((sobel(target.detach()).abs().amax(1, keepdim=True) > .02).float(), 3, 1, 1)
    return ink, near, distant, edge


def region_detail_distance(prediction, target):
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        p, y = gray(prediction), gray(target.detach())
        if p.shape != y.shape or p.shape[-2:] != (96, 96):
            raise ValueError('Expected aligned native96 images')
        if not torch.isfinite(y).all() or y.min() < -1e-6 or y.max() > 1 + 1e-6:
            raise ValueError('Expected finite target in [0,1]')
        result = p.new_zeros(p.shape[0])
        for size, factor in ((96, 1.), (48, .5)):
            pp, yy = (p, y) if size == 96 else (F.avg_pool2d(p, 2), F.avg_pool2d(y, 2))
            ink, near, distant, edge = region_masks(yy)
            pixel, mass = pp.new_zeros(pp.shape[0]), pp.new_zeros(pp.shape[0])
            error = (pp - yy).abs()
            for mask, weight in ((ink, .5), (near, .4), (distant, .1)):
                denom = mask.sum((1, 2, 3))
                active = (denom > 0).float()
                pixel += weight * active * (error * mask).sum((1, 2, 3)) / denom.clamp_min(1)
                mass += weight * active
            pixel = pixel / mass.clamp_min(1e-8)
            edges = ((sobel(pp) - sobel(yy)).abs() * edge).sum((1, 2, 3))
            edges = edges / (2 * edge.sum((1, 2, 3))).clamp_min(1)
            result += factor * (pixel + .5 * edges)
        return result


def visual_losses(tokens_image, prediction, target, conditional, alpha, step, render_weight):
    if conditional.dtype != torch.bool or conditional.shape != (target.shape[0],):
        raise ValueError('Expected per-example boolean conditional mask')
    if alpha.shape != conditional.shape or not torch.isfinite(alpha).all() or not ((alpha > 0) & (alpha <= 1)).all():
        raise ValueError('Invalid alpha_bar')
    if not math.isfinite(render_weight) or render_weight <= 0:
        raise ValueError('Invalid frozen render weight')
    tc = (region_detail_distance(tokens_image, target) * conditional.float()).mean()
    render = (region_detail_distance(prediction, target) * conditional.float() * alpha.float()).mean()
    ramp = min(1., max(0., step / 1000.))
    return ramp * (.02 * tc + render_weight * render), tc, render
