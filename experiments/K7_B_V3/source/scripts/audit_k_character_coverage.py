"""Read-only replay of frozen attempt-keyed K samplers; no torch/GPU imports."""
import ast
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import random
import types
import argparse


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    root=Path('/root/projects/hrfont')
    pairpath=root/'manifests/v0913_clean/pairs_train.tsv'
    rows=list(csv.DictReader(pairpath.open(),delimiter='\t'))
    assert len({(r['font'],r['cp']) for r in rows})==len(rows)
    rows.sort(key=lambda r:(r['font'],r['cp']))
    data=root/'data/fontdiffuser-p253-t295-s338-cn2west-v2/train/TargetImage'
    ds=types.SimpleNamespace(phase='train',target_images=[str(data/r['font']/f"{r['font']}+{r['cp']}.png") for r in rows])
    detail=root/'artifacts/i56_20260916/detail_manifest.json'
    summary={'checked_at':datetime.datetime.now().astimezone().isoformat(),
             'method':'Replay actual frozen KSampler AST without importing torch; reconstruct FontDataset clean target ordering; compare replayed per-font/script counts exactly against saved exposure checkpoint; no skipped updates allowed',
             'pairs_sha256':sha(pairpath),'detail_sha256':sha(detail),'runs':{}}
    for arm,name,code in [('K1','K1-ORIGINAL-V0917-S3407','hrfont_k_20260917'),('K3','K3-PURENOISE-V0917-S3407','hrfont_k3_20260917')]:
        out=root/'runs'/name;c=json.loads((out/'config.json').read_text());h=json.loads((out/'heartbeat.json').read_text())
        assert h['skips']==0 and h['attempt']==h['step']
        source=Path('/root/projects')/code/'scripts/k_components.py'
        identity=json.loads((source.parents[1]/'K_CODE_IDENTITY.json').read_text())
        assert sha(source)==identity['files']['scripts/k_components.py']
        assert sha(detail)==c['detail_sha256']
        tree=ast.parse(source.read_text());nodes=[]
        for n in tree.body:
            if isinstance(n,ast.ClassDef) and n.name=='KSampler':nodes.append(n)
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['SCRIPT','COMPLEX','PAIRS'] for t in n.targets):nodes.append(n)
        ns=dict(collections=collections,json=json,Path=Path,random=random)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
        sampler=ns['KSampler'](ds,detail);assert sampler.version==c['sampler']
        saved_paths=sorted(out.glob('exposure_*.json'),key=lambda p:int(p.stem.split('_')[-1]))
        checkpoints={int(p.stem.split('_')[-1]):json.loads(p.read_text()) for p in saved_paths}
        bycp=collections.Counter();byfontcp=collections.Counter();exposure=collections.Counter();first_seen={};verified=[]
        for attempt in range(h['step']):
            ids,quota=sampler.batch(attempt);exposure.update(quota['fonts'])
            for i in ids:
                font,cp,script,flag=sampler.entries[i]
                bycp[(script,cp)]+=1;byfontcp[(font,script,cp)]+=1;first_seen.setdefault((script,cp),attempt+1)
            if attempt+1 in checkpoints:
                saved=checkpoints[attempt+1]
                assert dict(exposure)==saved['counts'],(name,attempt+1,'exposure mismatch')
                verified.append(attempt+1)
        groups={}
        for script in ns['SCRIPT']:
            pairs={(font,cp) for font,cp,g,flag in sampler.entries if g==script}
            cps=sorted({cp for font,cp in pairs});counts=[bycp[(script,cp)] for cp in cps]
            covered=sum(byfontcp[(font,script,cp)]>0 for font,cp in pairs)
            groups[script]=dict(unique_characters=len(cps),seen_characters=sum(v>0 for v in counts),
                unseen_characters=[cp for cp,v in zip(cps,counts) if not v],
                char_exposures_min=min(counts),char_exposures_max=max(counts),
                all_chars_first_seen_step=max(first_seen[(script,cp)] for cp in cps) if min(counts)>0 else None,
                total_draws=sum(counts),eligible_font_character_pairs=len(pairs),seen_font_character_pairs=covered,
                font_character_coverage=covered/len(pairs))
        summary['runs'][arm]=dict(run_id=name,step=h['step'],heartbeat_time=h['time'],attempt=h['attempt'],
            skips=h['skips'],ddp_spread=h.get('ddp_spread'),sampler_sha256=sha(source),
            code_commit=identity['commit'],verified_exposure_checkpoints=verified,groups=groups,
            scope='main sampler only; K3 auxiliary pairs excluded')
        charrows=[dict(script=s,cp=cp,char=chr(int(cp[1:],16)),draws=n,first_seen_step=first_seen[(s,cp)]) for (s,cp),n in sorted(bycp.items())]
        pairrows=[dict(font=f,script=s,cp=cp,char=chr(int(cp[1:],16)),draws=byfontcp[(f,s,cp)]) for f,cp,s,flag in sampler.entries]
        for suffix,rr in [('characters',charrows),('font_characters',pairrows)]:
            with (a.out/f'{arm}_{suffix}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
    western=sorted({r['cp'] for r in rows if r['script_group']=='latin'})
    expected={f'u{ord(c):04X}' for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'}
    assert expected<=set(western)
    font_counts=collections.Counter(r['font'] for r in rows if r['script_group']=='latin')
    assert set(font_counts.values())=={89}
    summary['western_inventory']=dict(fonts=len(font_counts),unique_characters=len(western),
        uppercase=26,lowercase=26,digits=10,extended_latin=27,
        characters=''.join(chr(int(cp[1:],16)) for cp in western),all_62_basic_present_in_every_font=True,
        punctuation_present=False,non_Latin_alphabets_covered=False)
    (a.out/'COVERAGE_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
