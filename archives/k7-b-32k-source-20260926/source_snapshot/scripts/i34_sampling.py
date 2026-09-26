"""Consume a frozen, reviewed difficulty manifest; no online score changes."""
import json
from pathlib import Path
import torch
from scripts.hrfont_feature_cache import sha256_file


def load_sampling_manifest(path, dataset, args):
    path = Path(path)
    m = json.loads(path.read_text())
    if m['status'] != 'REVIEWED':
        raise ValueError('Difficulty manifest requires real panel review')
    assert m['encoder_sha256']==sha256_file(Path(args.warm_start_from)/'style_encoder.pth')
    clean = Path(args.v0913_clean_map)
    for name, digest in m['sources_sha256'].items():
        if sha256_file(clean/name) != digest:
            raise ValueError(f'Clean map changed: {name}')
    cp_group = m['cp_group']
    groups, factors = [], []
    for p in dataset.target_images:
        p = Path(p); font = p.parent.name; cp = p.stem[len(font)+1:]
        group = cp_group[cp]
        row = m['font_groups'][font+'|'+group]
        groups.append(('latin','kana','bopomofo').index(group))
        factors.append(row['weight'])
    assert dataset.phase == 'train'
    return torch.tensor(groups), torch.tensor(factors), sha256_file(path)
