"""CPU-only GT redraw-demand strata; calibrate on training fonts, never predictions.

This is a geometric analysis proxy, not a human complexity label. No training
configuration, sampler, model, or existing difficulty manifest is modified.
"""
import argparse
import collections
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gray(p):
    with Image.open(p) as im:
        assert im.size == (96, 96)
        return np.asarray(im.convert('L'), dtype=np.float32) / 255.


@lru_cache(maxsize=1200)
def content(p):
    c = gray(p)
    return c, c - uniform_filter(c, size=3, mode='reflect')


def measure(job):
    split, font, cp, script, target, source = job
    y = gray(target)
    c, ch = content(source)
    mask = (y < .95) | (c < .95)
    yh = y - uniform_filter(y, size=3, mode='reflect')
    return dict(split=split, font=font, cp=cp, script=script,
                redraw=float(np.abs(y-c)[mask].mean()),
                detail=float(np.abs(yh-ch)[mask].mean()), target_sha256=sha(target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/root/projects/hrfont'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    data = Path('/root/data1/hrfont_dataset_v2_20260917/v2')
    clean = Path('/root/data1/hrfont_dataset_v2_20260917/manifests/v2')
    oldtrain = set(json.loads((a.root/'manifests/split_v3_228_16_16.json').read_text())['stems']['train'])
    jobs, audit = [], {'pairs': {}, 'reference_codepoints': {}}
    sources = {}
    for split in ('train', 'val', 'test'):
        path = clean/f'pairs_{split}.tsv'
        rows = list(csv.DictReader(path.open(), delimiter='\t'))
        sources[str(path)] = sha(path)
        audit['pairs'][split] = dict(collections.Counter(r['script_group'] for r in rows))
        for r in rows:
            font, cp = r['font'], r['cp']
            script = 'western' if r['script_group']=='latin' else r['script_group']
            target = data/split/'TargetImage'/font/f'{font}+{cp}.png'
            source = data/split/'ContentImage'/f'{cp}.png'
            jobs.append((split, font, cp, script, str(target), str(source)))
        cps = {p.stem.split('+')[-1] for p in (data/split/'StyleImage').glob('*/*.png')}
        is_han = lambda c: 0x3400 <= int(c[1:],16) <= 0x9FFF or 0x20000 <= int(c[1:],16) <= 0x323AF
        audit['reference_codepoints'][split] = dict(unique=len(cps), han=sum(map(is_han,cps)), non_han=sorted(c for c in cps if not is_han(c)))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(measure, jobs))
    print('MEASURED', len(rows), flush=True)
    # Control character identity before aggregating a font/script difficulty.
    # Empirical training percentiles use mid-ranks for ties; holdout values use
    # these same training reference distributions, without refitting thresholds.
    bycp = collections.defaultdict(list)
    for r in rows:
        if r['split']=='train' and r['font'] in oldtrain: bycp[r['cp']].append(r)
    distributions = {cp: {m: np.sort([r[m] for r in rr]) for m in ('redraw','detail')} for cp,rr in bycp.items()}
    for r in rows:
        if len(bycp[r['cp']]) < 16:
            r['calibrated'] = None
            continue
        ranks = []
        for metric in ('redraw','detail'):
            ref = distributions[r['cp']][metric]
            ranks.append(float((np.searchsorted(ref,r[metric],'left') + np.searchsorted(ref,r[metric],'right')) / (2*len(ref))))
        r['calibrated'] = float(np.mean(ranks))
    byfont = collections.defaultdict(list)
    for r in rows: byfont[(r['split'],r['font'],r['script'])].append(r)
    groups = []
    for (split,font,script), rr in sorted(byfont.items()):
        vals = [r['calibrated'] for r in rr if r['calibrated'] is not None]
        groups.append(dict(split=split,font=font,script=script,
                           difficulty_score=float(np.median(vals)) if len(vals)>=8 else None,
                           comparable_chars=len(vals),total_chars=len(rr)))
    thresholds = {}
    for script in sorted({g['script'] for g in groups}):
        vals = [g['difficulty_score'] for g in groups if g['split']=='train' and g['font'] in oldtrain and g['script']==script and g['difficulty_score'] is not None]
        low, high = np.quantile(vals,[1/3,2/3])
        thresholds[script] = dict(low=float(low),high=float(high),training_fonts=len(vals))
    for g in groups:
        score = g['difficulty_score']; t = thresholds[g['script']]
        g['difficulty'] = 'unknown' if score is None else 'easy' if score<t['low'] else 'hard' if score>t['high'] else 'medium'
    for name, rr in [('gt_character_demand.csv',rows),('difficulty_font_script.csv',groups)]:
        with (a.out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n'); w.writeheader(); w.writerows(rr)
    sources.update({p:sha(p) for p in sorted({j[-1] for j in jobs})})
    frozen = json.loads(Path('/root/data1/hrfont_k_stratified_20260917/DIFFICULTY_MANIFEST.json').read_text())
    assert thresholds == frozen['thresholds'], 'Frozen 0913 difficulty calibration changed'
    meta = dict(version='GT-redraw-demand-v1-v2-data-frozen0913-calibration', definition='mean of two training-font same-character empirical percentiles: ink-union normalized gray absolute difference, and 3x3 high-pass absolute difference; median across all clean characters per font/script; training-font within-script tertiles',
                scope='geometric redraw demand, not human visual complexity or cross-script inference difficulty',
                min_training_fonts_per_character=16,min_characters_per_font_script=8,
                threshold_fit_split='0913 train only',thresholds=thresholds,sources_sha256=sources,
                source_script_sha256=sha(__file__),data_audit=audit,
                files_sha256={p.name:sha(p) for p in a.out.glob('*.csv')},
                model_outputs_used=False, evaluator_used=False,
                defined_after_K1_results=True, training_or_model_selection_changed=False)
    (a.out/'DIFFICULTY_MANIFEST.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),groups=len(groups),thresholds=thresholds,audit=audit),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
