"""Predeclared4k validation-only checkpoint; no test-based tuning."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scripts.i56_components import region_detail_distance


def scores(directory):
    d=json.loads((directory/'metrics.json').read_text())
    vals=[]
    for r in d['rows']:
        with Image.open(directory/r['png']) as im:p=np.asarray(im.convert('L'),dtype=np.float32)/255
        with Image.open(directory/f'GT__{r["font"]}__{r["cp"]}.png') as im:y=np.asarray(im.convert('L'),dtype=np.float32)/255
        vals.append(float(region_detail_distance(torch.from_numpy(p)[None,None],torch.from_numpy(y)[None,None])))
    return d,float(np.mean(vals))


def assess(out, baseline):
    current,cd=scores(out);old,bd=scores(baseline)
    key=lambda r:(r['font'],r['cp'],r['group'],r['k'],tuple(r['refs']),r['seed'])
    assert sorted(map(key,current['rows']))==sorted(map(key,old['rows']))
    avg=lambda rows,k:float(np.mean([r[k] for r in rows]))
    l1=avg(current['rows'],'l1');old_l1=avg(old['rows'],'l1')
    ssim=avg(current['rows'],'ssim');old_ssim=avg(old['rows'],'ssim')
    passed=cd<bd and l1<=old_l1*1.02 and ssim>=old_ssim-.005
    return dict(status='CONTINUE' if passed else 'REVIEW_REQUIRED',step=4000,
        baseline=str(baseline),detail=cd,baseline_detail=bd,l1=l1,baseline_l1=old_l1,
        ssim=ssim,baseline_ssim=old_ssim,
        rule='Fixed val192: regional detail improves vs I3@4k, L1 regression<=2%, SSIM drop<=.005. Otherwise save and pause for review.',
        note='Pragmatic early review trigger, not a statistical significance or style-quality claim')
