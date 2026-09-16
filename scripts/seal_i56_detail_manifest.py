"""Seal explicitly reviewed training panels. Labels are authored, never inferred here."""
import argparse
import collections
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.i56_sampling import capped_distribution
from scripts.hrfont_feature_cache import sha256_file as sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panels',type=Path,required=True)
    ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();p=json.loads((a.panels/'panels.json').read_text()); labels=json.loads(a.labels.read_text())
    assert labels['reviewer']=='Codex visual inspection' and labels['status']=='REVIEWED'
    for name,digest in p['panel_sha256'].items():assert sha(a.panels/name)==digest
    rows={r['key']:r for r in p['rows']}
    assert set(labels['decisions'])==set(rows), 'Every inspected panel needs a decision'
    selected={}
    for key,decision in labels['decisions'].items():
        assert decision['decision'] in ('confirmed_detail','base_only') and decision['reason']
        row=rows[key];assert len(row['cn'])>=8 and len(row['target'])>=8
        if decision['decision']=='confirmed_detail':
            assert decision['tags'] and decision['cross_script_visible']
            selected[key]=dict(**decision,cn_chars=[r['cp'] for r in row['cn']],target_chars=[r['cp'] for r in row['target']])
    out=dict(status='REVIEWED',version='real-detail-panel-v1',reviewer=labels['reviewer'],
        confirmed_detail=selected,train_fonts=p['train_fonts'],cp_group=p['cp_group'],
        sources_sha256=p['sources_sha256'],inventory_sha256=p['inventory_sha256'],
        panels_sha256=sha(a.panels/'panels.json'),labels_sha256=sha(a.labels),
        policy=dict(base=.7,detail=.3,font_cap=3.,ramp_steps=1000,unreviewed='base exposure retained',
            duplicate_policy='No additional equivalence merge: no duplicate selected groups confirmed in reviewed panels'),
        coverage='Conservative confirmed subset from36 reviewed font/script panels, not a census of541 groups')
    root=Path('/root/projects/hrfont/manifests/v0913_clean')
    import csv
    pairs=list(csv.DictReader((root/'pairs_train.tsv').open(),delimiter='\t'))
    specs=json.loads((root/'sample_weights.json').read_text())
    fonts={f:i for i,f in enumerate(p['train_fonts'])}
    base=[specs['pair_weight'][r['script_group']] for r in pairs]
    groups=[('latin','kana','bopomofo').index(r['script_group']) for r in pairs]
    ids=[fonts[r['font']] for r in pairs]; flags=[r['font']+'|'+r['script_group'] in selected for r in pairs]
    probs=capped_distribution(base,groups,ids,flags);b=torch.tensor(base,dtype=torch.float64);b/=b.sum()
    summary={}
    for group,idx in zip(('latin','kana','bopomofo'),range(3)):
        mask=torch.tensor(groups)==idx;detail=mask&torch.tensor(flags)
        summary[group]=dict(script_mass=float(probs[mask].sum()),
            confirmed_fonts=len({r['font'] for r,f in zip(pairs,flags) if f and r['script_group']==group}),
            base_detail_fraction=float(b[detail].sum()/b[mask].sum()),
            final_detail_fraction=float(probs[detail].sum()/probs[mask].sum()))
    out['probability_summary']=summary
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
