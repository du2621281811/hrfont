"""Frozen, reviewed real-detail exposure; script masses and per-font caps preserved."""
import collections
import json
import torch
from pathlib import Path
from scripts.hrfont_feature_cache import sha256_file


def capped_distribution(base, group_ids, font_ids, selected, extra_fraction=.3, cap=3.):
    """Mix at most 30% detail, cap each font at3x, return excess to base.

    Within a script/font the original uniform target-character distribution is
    preserved. Unreviewed fonts stay eligible in the base component.
    """
    base = torch.as_tensor(base, dtype=torch.float64)
    groups, fonts = torch.as_tensor(group_ids), torch.as_tensor(font_ids)
    selected = torch.as_tensor(selected, dtype=torch.bool)
    if base.ndim != 1 or any(x.shape != base.shape for x in (groups, fonts, selected)):
        raise ValueError('Inconsistent sampling arrays')
    if not torch.isfinite(base).all() or (base <= 0).any() or not 0 <= extra_fraction <= 1 or cap < 1:
        raise ValueError('Invalid sampling weights')
    base = base / base.sum()
    result = base.clone()
    for group in groups.unique():
        gm = groups == group
        ids = fonts[gm].unique()
        mass = base[gm].sum()
        fontmass = torch.stack([base[gm & (fonts == f)].sum() for f in ids]) / mass
        flags = torch.tensor([bool(selected[gm & (fonts == f)].all()) for f in ids])
        for f in ids:
            vals = selected[gm & (fonts == f)]
            if bool(vals.any()) != bool(vals.all()):
                raise ValueError('Detail label must be font/script-level, not cherry-picked characters')
        if not flags.any():
            continue
        p = (1 - extra_fraction) * fontmass
        room = (cap * fontmass - p).clamp_min(0) * flags
        remaining = float(extra_fraction)
        # Water fill uniformly across confirmed fonts, respecting their caps.
        for _ in range(len(ids) + 1):
            active = room > 1e-14
            if not active.any() or remaining < 1e-14:
                break
            add = torch.minimum(room, torch.full_like(room, remaining / int(active.sum()))) * active
            p += add
            room -= add
            remaining -= float(add.sum())
        # If few complex fonts, distribute unused detail budget back to base.
        for _ in range(len(ids) + 1):
            if remaining < 1e-14:
                break
            room = (cap * fontmass - p).clamp_min(0)
            active = room > 1e-14
            allocation = fontmass * active
            if not active.any():
                raise ValueError('No room for returned base mass')
            add = torch.minimum(room, remaining * allocation / allocation.sum())
            p += add
            remaining -= float(add.sum())
        if remaining > 1e-10:
            raise ValueError('Mass redistribution failed')
        for f, prob, old in zip(ids, p, fontmass):
            fm = gm & (fonts == f)
            result[fm] = base[fm] * prob / old
    if not torch.allclose(result.sum(), base.sum(), atol=1e-10) or (result > cap * base + 1e-12).any():
        raise ValueError('Probability mass/cap violation')
    return result


def load_sampling_manifest(path, dataset, args):
    path = Path(path)
    m = json.loads(path.read_text())
    if m.get('status') != 'REVIEWED' or m.get('version') != 'real-detail-panel-v1':
        raise ValueError('Expected reviewed real-detail manifest')
    if dataset.phase != 'train':
        raise ValueError('Training-only sampling')
    for name, digest in m['sources_sha256'].items():
        if sha256_file(Path(args.v0913_clean_map) / name) != digest:
            raise ValueError('Clean source changed: ' + name)
    names = sorted({Path(p).parent.name for p in dataset.target_images})
    if names != m['train_fonts']:
        raise ValueError('Training-font membership changed')
    fi = {f: i for i, f in enumerate(names)}
    groups, fonts, selected, keys = [], [], [], []
    for p in dataset.target_images:
        p = Path(p); font = p.parent.name; cp = p.stem[len(font) + 1:]
        script = m['cp_group'][cp]; key = font + '|' + script
        groups.append(('latin', 'kana', 'bopomofo').index(script))
        fonts.append(fi[font]); selected.append(key in m['confirmed_detail']); keys.append(key)
    base = torch.tensor(dataset.sample_weights, dtype=torch.float64)
    target = capped_distribution(base, groups, fonts, selected)
    return target, selected, keys, sha256_file(path)


def sampling_distribution(base, target, step):
    base = torch.as_tensor(base, dtype=torch.float64)
    base = base / base.sum()
    ramp = min(1., max(0., step / 1000.))
    return (1 - ramp) * base + ramp * target


def episode_indices(base, target, flags, step, attempt, overfit=False):
    weights = sampling_distribution(base, target, 1000 if overfit else step)
    if overfit:
        weights = weights * torch.as_tensor(flags, dtype=torch.float64)
        if weights.sum() <= 0:
            raise ValueError('Overfit diagnostic requires confirmed detail fonts')
    return torch.multinomial(weights, 64, replacement=True,
        generator=torch.Generator().manual_seed(3407+(0 if overfit else attempt))).tolist()
