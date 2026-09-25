"""Reproduce review evidence from Git metadata, without training or changing E12."""
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rankdata(values):
    positions = defaultdict(list)
    for rank, i in enumerate(sorted(range(len(values)), key=lambda i: values[i])):
        positions[values[i]].append(rank + 1)
    return [sum(positions[v]) / len(positions[v]) for v in values]


def corr(x, y):
    mx, my = sum(x)/len(x), sum(y)/len(y)
    a, b = [v-mx for v in x], [v-my for v in y]
    return sum(u*v for u,v in zip(a,b)) / math.sqrt(sum(u*u for u in a)*sum(v*v for v in b))


def main():
    cs = json.loads((ROOT/'manifests/charset_cn2west_v2_planned.json').read_text())
    cfg = json.loads((ROOT/'reports/e12_b/run_meta/phi/config.canonical.json').read_text())
    cn = list(dict.fromkeys(cs['style_ref8_subset'] + cs['style_han_338'][:96]))
    actual_cn = [c for c in cfg['data']['chinese_chars'] if c in cn]
    fonts = list(csv.DictReader((ROOT/'manifests/v0913_clean/fonts.tsv').open(), delimiter='\t'))
    # Read actual key names rather than relying on column labels across revisions.
    names = [list(r.values())[0] for r in fonts if list(r.values())[1]=='train' and list(r.values())[3]=='1']
    units = sorted(names); random.Random(3407).shuffle(units)
    splits = dict(train=sorted(units[:156]), val=sorted(units[156:189]), test=sorted(units[189:]))
    for split in ('val', 'test'):
        reported = json.loads((ROOT/f'reports/e12_b/run_meta/membership/{split}_metrics.json').read_text())
        assert splits[split] == sorted(reported['families']), 'reconstruction differs from recorded split'
    valkeys = [(f,c) for f in splits['val'] for c in actual_cn][:128]
    # This is metadata reconstruction, not a claim that the unshipped cache was inspected.
    expected_false_pairs = (64*63/2) * (len(actual_cn)-1)/(156*len(actual_cn)-1)
    by_prefix = defaultdict(list)
    for split, stems in splits.items():
        for stem in stems:
            if stem.startswith('FZYouHK_'):
                by_prefix['FZYouHK'].append(dict(font=stem, split=split))
    items = json.loads((ROOT/'reports/e12_paper/human_board/items.json').read_text())['items']
    ratings = defaultdict(dict)
    with (ROOT/'reports/e12_paper/human_ratings_partial.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row.get('score','').strip(): ratings[row['rater']][row['item_id']] = float(row['score'])
    published = json.loads((ROOT/'reports/e12_paper/human_spearman_summary.json').read_text())
    x, y = [], []
    for it in items:
        vals = [ratings[r][it['item_id']] for r in published['passing_raters'] if it['item_id'] in ratings[r]]
        if vals and not it.get('attention_check') and it.get('style_cosine') is not None:
            x.append(sum(vals)/len(vals)); y.append(it['style_cosine'])
    conc = disc = tie_x = tie_y = 0
    for i in range(len(x)):
        for j in range(i):
            a,b = x[i]-x[j],y[i]-y[j]
            if a == b == 0: continue
            if a == 0: tie_x += 1
            elif b == 0: tie_y += 1
            elif a*b > 0: conc += 1
            else: disc += 1
    result = dict(basis='Git code/config/charset reconstruction; execution E12 cache unavailable on V100 host',
                  configured_cn=len(cfg['data']['chinese_chars']), cached_cn=len(cn), used_cn=len(actual_cn),
                  used_cn_chars=actual_cn, val_first128_font_counts=dict(Counter(f for f,c in valkeys)),
                  expected_same_font_off_diagonal_unordered_pairs_per_batch=expected_false_pairs,
                  candidate_related_stems=by_prefix, human_n=len(x), human_ratings_counts=dict(Counter(x)),
                  published_spearman=published['spearman_human_vs_cosine'],
                  corrected_spearman_average_ranks=corr(rankdata(x),rankdata(y)),
                  published_kendall=published['kendall_tau_human_vs_cosine'],
                  corrected_kendall_tau_b=(conc-disc)/math.sqrt((conc+disc+tie_x)*(conc+disc+tie_y)))
    out=ROOT/'reports/review_20260915/e12_audit.json'; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
