"""Read-only aggregation of historical predictions under the frozen clean mask.

Removed pairs are review exclusions, not independently verified per-glyph errors.
This script measures evaluation-mask effects, not effects of clean retraining.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / 'manifests/v0913_clean.json').read_text())
    with (root / 'manifests/v0913_clean/pairs_eval47.tsv').open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    mask = {(r['font'], r['cp']): r for r in rows}
    assert len(mask) == 752 and sum(int(r['keep']) for r in rows) == 704
    metrics = json.loads((root / 'reports/f03_test16_strat/metrics_items.json').read_text())
    grouped = defaultdict(list)
    for item in metrics:
        key = (item['font'], item['cp'])
        if key in mask:
            grouped[item['method']].append(item)
    result = {'scope': 'historical-checkpoint mask-only reaggregation; no retraining',
              'training': {}, 'evaluation': {}, 'methods': {}, 'comparisons': {}}
    for group, spec in contract['script_groups'].items():
        old = 228 * spec['n_chars']
        kept = contract['pair_counts']['train'][group]
        result['training'][group] = dict(old_pairs=old, kept_pairs=kept,
            removed_pairs=old-kept, removed_fraction=(old-kept)/old,
            old_uniform_sampling_share=spec['n_chars']/295,
            new_sampling_share=contract['train_sample']['group_p'][group])
    old = 228 * 295
    kept = contract['pair_counts']['train']['all']
    result['training']['all'] = dict(old_pairs=old, kept_pairs=kept,
        removed_pairs=old-kept, removed_fraction=(old-kept)/old)
    result['evaluation'] = dict(old_pairs=len(mask), kept_pairs=704,
        removed_pairs=48, removed_fraction=48/752,
        removed_by_group=dict(Counter(r['script_group'] for r in rows if r['keep']=='0')))
    for method, items in sorted(grouped.items()):
        keys = [(x['font'], x['cp']) for x in items]
        assert len(keys) == len(set(keys)), f'duplicate metric rows: {method}'
        assert set(keys) == set(mask), f'incomplete eval47: {method}'
        kept_items = [x for x in items if mask[(x['font'], x['cp'])]['keep']=='1']
        subsets = {'old752': items, 'valid704': kept_items,
                   'removed48': [x for x in items if mask[(x['font'], x['cp'])]['keep']=='0']}
        for group in contract['script_groups']:
            subsets[group] = [x for x in kept_items if mask[(x['font'], x['cp'])]['script_group']==group]
        result['methods'][method] = {}
        for name, subset in subsets.items():
            summary = {'n': len(subset)}
            for metric in ('L1', 'SSIM', 'LPIPS'):
                values = [x[metric] for x in subset if x.get(metric) is not None]
                summary[metric] = fmean(values) if values else None
                summary[metric+'_n'] = len(values)
            result['methods'][method][name] = summary
    for a, b in [('F2P_40000','F2_40000'), ('F2VEC_40000','F2_40000'),
                 ('F2_80000','F1_80000'), ('F2_80000','F0_100k')]:
        if a in result['methods'] and b in result['methods']:
            result['comparisons'][a+' minus '+b] = {
                subset: {m: result['methods'][a][subset][m]-result['methods'][b][subset][m]
                         for m in ('L1','SSIM','LPIPS')}
                for subset in ('old752','valid704','latin','kana','bopomofo')}
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
    else:
        print(rendered, end='')


if __name__ == '__main__':
    main()
