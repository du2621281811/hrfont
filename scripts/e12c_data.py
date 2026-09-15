"""E12c instance-disjoint clean training-data protocol, with optional known families."""
import argparse
import collections
import json
import random
from pathlib import Path
from new_data_inventory import PREP, sha, write_new

SCRIPTS=('latin','kana','bopomofo')
REF8=[f'u{ord(c):04X}' for c in '永和书风骨韵天地']


def prepare(known_families=None):
    src=PREP/'inventory.json'; inv=json.loads(src.read_text())
    family=json.loads(Path(known_families).read_text()) if known_families else {}
    assert all(isinstance(v,str) and v.strip() for v in family.values())
    denied_families={family[f] for f in inv['excluded_main_eval_fonts'] if f in family}
    fonts=[f for f in inv['fonts'] if family.get(f) not in denied_families]
    units=collections.defaultdict(list)
    for f in fonts: units['family:'+family[f] if f in family else 'instance:'+f].append(f)
    keys=sorted(units); random.Random(3407).shuffle(keys)
    n=len(keys); assert n>=40
    split={f:('train' if i<int(.8*n) else 'val' if i<int(.9*n) else 'test')
           for i,key in enumerate(keys) for f in units[key]}
    rows=[r for r in inv['records'] if r['font'] in split]
    lookup={(r['font'],r['role'],r['cp']):i for i,r in enumerate(rows)}
    byfont={f:{g:[] for g in ('han',)+SCRIPTS} for f in split}
    for i,r in enumerate(rows): byfont[r['font']][r['script']].append(i)
    for f in split:
        assert len(byfont[f]['han'])>=8
        assert all((f,'style',c) in lookup for c in REF8)
        assert any(len(byfont[f][g])>=2 for g in SCRIPTS)
    manifests={}
    for sp in ('val','test'):
        pool=sorted(f for f in split if split[f]==sp)
        gallery={f:[lookup[f,'style',c] for c in REF8] for f in pool}
        queries=[]; episodes=[]
        for f in pool:
            for group in SCRIPTS:
                candidates=byfont[f][group]
                if not candidates: continue
                chosen=random.Random('3407:'+f+group).sample(candidates,min(4,len(candidates)))
                for idx in chosen:
                    cp=rows[idx]['cp']
                    negatives=[g for g in pool if g!=f and not (f in family and g in family and family[f]==family[g])
                               and (g,'target',cp) in lookup]
                    assert negatives,(sp,f,cp)
                    other=random.Random('3407:'+f+cp).choice(negatives)
                    queries.append(dict(index=idx,font=f,script=group))
                    for k in (1,2,4,8):
                        episodes.append(dict(pos=idx,neg=lookup[other,'target',cp],refs=gallery[f][:k],
                                             font=f,negative_font=other,cp=cp,script=group,k=k))
        assert {q['script'] for q in queries}==set(SCRIPTS)
        assert {(e['script'],e['k']) for e in episodes}=={(g,k) for g in SCRIPTS for k in (1,2,4,8)}
        manifests[sp]=dict(gallery=gallery,queries=queries,episodes=episodes)
    for sp in ('train','val','test'):
        assert {g for f in split if split[f]==sp for g in SCRIPTS if byfont[f][g]}==set(SCRIPTS)
    obj=dict(version='E12C-LITE-INSTANCE-SPLIT-v1',inventory_sha256=sha(src),split=split,rows=rows,
        byfont=byfont,known_families=family,known_family_source_sha256=sha(known_families) if known_families else None,
        unknown_family_fonts=[f for f in split if f not in family],source_sha256=inv['sources_sha256'],
        excluded_main_eval_fonts=inv['excluded_main_eval_fonts'],validation=manifests,
        preprocessing='native96 RGB [0,1], ImageNet mean/std; no resize/crop/augmentation',
        selection=dict(A='font then script macro R@1',B='script x shot macro ROC-AUC'),
        split_counts=dict(collections.Counter(split.values())))
    write_new(PREP/'e12c_manifest.json',obj)
    print(json.dumps(dict(split_counts=obj['split_counts'],rows=len(rows),known_family_fonts=len(split)-len(obj['unknown_family_fonts']))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--known-families');a=ap.parse_args();prepare(a.known_families)
