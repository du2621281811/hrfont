"""Fail-closed acceptance: eight-rank updates, resume, actual generation, TC use."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from i56_runtime import ROOT,CODE,atomic_json,sha256_file


def main():
    prep=ROOT/'artifacts/i56_20260916';run=ROOT/'runs/I5-PREFLIGHT-V0916-S3407'
    done=json.loads((run/'DONE.json').read_text())
    assert done['smoke'] and done['step']==304 and done['arm']=='I5'
    logs=[json.loads(s) for s in (run/'train_log.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(1,305)), 'Resume must continue without replay'
    assert json.loads((run/'config.json').read_text())['resume'] is not None
    ranks=[]
    for rank in range(8):
        r=json.loads((run/f'rank{rank}.jsonl').read_text().splitlines()[-1])
        assert r['step']==r['attempt']==304 and r['skips']==0 and r['ddp_spread']==0
        assert r['aux_loss_rank']==0
        for name in ('loss','encoder_grad','token_projection_grad','ink_readout_grad'):
            assert math.isfinite(r[name]) and r[name]>0,(rank,name)
        ranks.append(r)
    before=json.loads((prep/'before/metrics.json').read_text());after=json.loads((prep/'after/metrics.json').read_text())
    assert [(r['font'],r['cp'],r['refs']) for r in before['rows']]==[(r['font'],r['cp'],r['refs']) for r in after['rows']]
    tc_ratio=after['mean_tc']/before['mean_tc'];render_ratio=after['mean_pred']/before['mean_pred']
    wins=sum(a['pred_detail']<b['pred_detail'] for a,b in zip(after['rows'],before['rows']))
    effect=float(np.median([r['local_intervention_l1'] for r in after['rows']]))
    checks=dict(tc_learns=tc_ratio<.95,actual_generation_improves=render_ratio<1.0 and wins>=8,
        local_path_used=effect>1e-5,finite_images=all(r['finite'] for r in after['rows']),
        noncollapsed_images=all(.001<r['pred_ink_fraction']<.9 for r in after['rows']))
    for path in prep.glob('*/review.png'):
        with Image.open(path) as im:im.load()
    evidence=dict(status='PASSED' if all(checks.values()) else 'REVIEW_REQUIRED',checks=checks,
        tc_ratio=tc_ratio,actual_generation_ratio=render_ratio,improved_episodes=wins,total_episodes=16,
        local_intervention_median_l1=effect,ranks=ranks,
        first10_loss_median=float(np.median([r['loss'] for r in logs[:10]])),
        last10_loss_median=float(np.median([r['loss'] for r in logs[-10:]])),
        code_sha256={str(p.relative_to(CODE)):sha256_file(p) for p in sorted((CODE/'scripts').glob('*.py'))},
        detail_sha256=sha256_file(prep/'detail_manifest.json'),calibration_sha256=sha256_file(prep/'calibration.json'),
        warning='Technical/learnability gate on seen training examples; not generalization validation')
    atomic_json(prep/'PREFLIGHT_RESULT.json',evidence)
    print(json.dumps(evidence),flush=True)
    if not all(checks.values()):raise RuntimeError('I5 preflight requires review; do not launch full training')


if __name__=='__main__':main()
