"""Stateless global64 quota episodes and the frozen original K objective."""
import collections
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from scripts.i56_components import region_detail_distance
from scripts.i34_components import gray, raw_x0

SCRIPT = {'western': 32, 'kana': 24, 'bopomofo': 8}
COMPLEX = {'western': 19, 'kana': 14, 'bopomofo': 5}
PAIRS = {'western': 4, 'kana': 3, 'bopomofo': 1}


class KSampler:
    version = 'K-original-global64-attempt-keyed-v1'

    def __init__(self, dataset, detail_manifest, seed=3407):
        assert dataset.phase == 'train'
        m = json.loads(Path(detail_manifest).read_text())
        assert m['status'] == 'REVIEWED'
        self.seed = seed
        self.entries, self.pools, self.pair_pools = [], {}, {}
        self.font_chars = collections.defaultdict(lambda: collections.defaultdict(list))
        bycp = collections.defaultdict(list)
        for i, path in enumerate(dataset.target_images):
            p = Path(path); font = p.parent.name; cp = p.stem[len(font)+1:]
            rawgroup = m['cp_group'][cp]
            group = 'western' if rawgroup == 'latin' else rawgroup
            flag = m['confirmed_detail'].get(font+'|'+rawgroup, {}).get('decision') == 'confirmed_detail'
            self.entries.append((font, cp, group, flag))
            self.font_chars[(group, flag)][font].append(i)
            bycp[(group, cp, flag)].append(i)
        for group in SCRIPT:
            for flag in (False, True):
                assert self.font_chars[(group, flag)]
                self.pools[(group, flag)] = sorted(self.font_chars[(group, flag)])
            chars = sorted({cp for g,cp,flag in bycp if g==group and flag} &
                           {cp for g,cp,flag in bycp if g==group and not flag})
            assert chars
            self.pair_pools[group] = (chars, bycp)

    def batch(self, attempt):
        rng = random.Random(self.seed + int(attempt)*1000003)
        ids, pairs = [], []
        def draw(group, flag):
            f = rng.choice(self.pools[(group, flag)])
            return rng.choice(self.font_chars[(group, flag)][f])
        for group, count in SCRIPT.items():
            chars, pool = self.pair_pools[group]
            for _ in range(PAIRS[group]):
                cp = rng.choice(chars)
                pair = (rng.choice(pool[(group, cp, True)]), rng.choice(pool[(group, cp, False)]))
                assert self.entries[pair[0]][0] != self.entries[pair[1]][0]
                ids.extend(pair); pairs.append(pair)
            ids.extend(draw(group, True) for _ in range(COMPLEX[group]-PAIRS[group]))
            ids.extend(draw(group, False) for _ in range(count-COMPLEX[group]-PAIRS[group]))
        rng.shuffle(ids)
        report = self.report(ids)
        assert report['script'] == SCRIPT and report['complex'] == COMPLEX and len(ids)==64
        report['pairs'] = dict(PAIRS)
        return ids, report

    def report(self, ids):
        script, complex_, fonts = collections.Counter(), collections.Counter(), collections.Counter()
        for i in ids:
            font, cp, group, flag = self.entries[i]
            script[group] += 1; complex_[group] += int(flag); fonts[font+'|'+group] += 1
        return dict(script=dict(script), complex=dict(complex_), fonts=dict(fonts))


def masked_mean(error, mask):
    return (error*mask).sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)


def detail_distances(prediction, content, target):
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        p, c, y = gray(prediction), gray(content.detach()), gray(target.detach())
        values = {k:p.new_zeros(p.shape[0]) for k in ('D_change','D_add','D_remove','D_high')}
        for size, factor in ((96,1.),(48,.5)):
            pp,cc,yy = (p,c,y) if size==96 else tuple(F.avg_pool2d(z,2) for z in (p,c,y))
            add = F.max_pool2d(((cc-yy)>.10).float(),3,1,1)
            remove = F.max_pool2d(((yy-cc)>.10).float(),3,1,1)
            error = ((pp-yy).square()+1e-6).sqrt()
            da,dr = masked_mean(error,add), masked_mean(error,remove)
            va,vr = add.sum((1,2,3))>0, remove.sum((1,2,3))>0
            change = (da*va+dr*vr)/(va.float()+vr.float()).clamp_min(1)
            high = lambda x: x-F.avg_pool2d(x,3,1,1)
            hm = F.max_pool2d(((yy<.95)|(cc<.95)).float(),5,1,2)
            values['D_change'] += factor*change
            values['D_add'] += factor*da
            values['D_remove'] += factor*dr
            values['D_high'] += factor*masked_mean((high(pp)-high(yy)).abs(),hm)
        values['D_region'] = region_detail_distance(prediction,target)
        values['D_out'] = values['D_region']+values['D_change']+.25*values['D_high']
        return values


def extra_losses(appearance, teacher, prediction, content, target, conditional, alpha, update):
    d = detail_distances(prediction,content,target)
    detail = (conditional.float()*alpha.float().sqrt()*d['D_out']).mean()
    comp = F.smooth_l1_loss(appearance.float(),teacher.detach().float()) if appearance is not None else detail*0
    ramp = min(1.,update/1000.)
    return ramp*(.01*comp+.05*detail), comp, detail, d
