"""Reproducible matched tables and font-paired intervals from frozen E12c scores."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def key(r):
    return (r['split'],r['font'],r['cp'],r['generation_group'],r['generation_shot'],
            tuple(r['generation_refs']),r['generation_seed'])


def macro(rows, metric='logit'):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['font']].append(r[metric])
    return statistics.mean(statistics.mean(v) for v in groups.values())


def paired(a,b,name):
    aa={key(r):r for r in a};bb={key(r):r for r in b};assert aa.keys()==bb.keys()
    by=collections.defaultdict(list)
    for k,r in aa.items():by[r['font']].append(r['logit']-bb[k]['logit'])
    d=np.asarray([statistics.mean(v) for _,v in sorted(by.items())])
    boot=np.random.default_rng(20260916).choice(d,size=(10000,len(d)),replace=True).mean(1)
    low,high=np.quantile(boot,[.025,.975])
    return dict(comparison=name,font_paired_logit_delta=float(d.mean()),ci95_low=float(low),
        ci95_high=float(high),fonts=len(d),font_wins=int((d>0).sum()),
        note='Paired font bootstrap, descriptive; small sample and multiple comparisons, no corrected significance claim')


def dump(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();p=a.directory
    with gzip.open(p/'scores.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
    assert len(rows)==48376
    tests={arm:[r for r in rows if r['model']==arm and r['protocol']=='matched4888' and r['split']=='test'] for arm in ('I0','I1','I2','I3','I4')}
    for group in tests.values():assert len(group)==3008 and {key(r) for r in group}=={key(r) for r in tests['I0']}
    vals={arm:[r for r in rows if r['model']==arm and r['protocol']=='val192' and r['step']==4000] for arm in ('I1','I2','I3','I4','I5')}
    required={key(r) for r in vals['I5']}
    vals['I0']=[r for r in rows if r['model']=='I0' and r['protocol']=='formal4096' and key(r) in required]
    for group in vals.values():assert len(group)==192 and {key(r) for r in group}==required
    table=[]
    for arm,rs in tests.items():
        r=dict(model=arm,**{f'k{k}':macro([r for r in rs if r['generation_shot']==k]) for k in (1,2,4,8)},
            mean=macro(rs),cosine_diagnostic=macro(rs,'cosine_diagnostic'),gt_diagnostic=macro(rs,'gt_logit_diagnostic'),images=len(rs))
        table.append(r)
    dump(p/'test_final_table.csv',table)
    table4=[]
    for arm in sorted(vals):
        rs=vals[arm];table4.append(dict(model=arm,step='I0 parent' if arm=='I0' else '4000',
            **{f'k{k}':macro([r for r in rs if r['generation_shot']==k]) for k in (1,4,8)},
            mean=macro(rs),cosine_diagnostic=macro(rs,'cosine_diagnostic'),images=len(rs)))
    dump(p/'val4k_table.csv',table4)
    scripts=[]
    for arm,rs in tests.items():
        scripts.append(dict(model=arm,**{g:macro([r for r in rs if r['script']==g]) for g in ('latin','kana','bopomofo')}))
    dump(p/'test_script_table.csv',scripts)
    intervals=[paired(tests[x],tests[y],f'test_final:{x}-{y}') for x,y in (('I4','I2'),('I2','I0'),('I3','I1'))]
    intervals += [paired(vals['I5'],vals[x],f'val4k:I5-{x}') for x in ('I0','I3','I4')]
    dump(p/'paired_intervals.csv',intervals)
    with (p/'TABLES.md').open('x') as f:
        f.write('# Frozen E12c fixed-REF8 results\n\nHigher logit means higher learned font compatibility, not probability or universal visual quality. Font-macro averaging; main evaluator refs always8.\n\n')
        for title,records in [('Final matched test16 ×47chars ×4shots',table),('Matched val192: four Latin chars, I0 fixed parent versus I1–I5@4k',table4),('Test script breakdown, all generation shots',scripts),('Paired font bootstrap intervals',intervals)]:
            fields=list(records[0]);f.write('## '+title+'\n\n|'+'|'.join(fields)+'|\n|'+'|'.join(['---']*len(fields))+'|\n')
            for r in records:f.write('|'+'|'.join(f'{r[k]:.6f}' if isinstance(r[k],float) else str(r[k]) for k in fields)+'|\n')
            f.write('\n')
    outputs=['test_final_table.csv','val4k_table.csv','test_script_table.csv','paired_intervals.csv','TABLES.md']
    with (p/'ANALYSIS_PROVENANCE.json').open('x') as f:
        json.dump(dict(input_scores_sha256=hashlib.sha256((p/'scores.jsonl.gz').read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            files_sha256={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in outputs}),f,indent=2)
    print(json.dumps(dict(test_final=table,val4k=table4,test_scripts=scripts,intervals=intervals),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
