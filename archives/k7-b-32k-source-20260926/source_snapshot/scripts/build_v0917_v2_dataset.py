#!/usr/bin/env python3
"""Build reviewable v0917 and v2 without changing the frozen v0913 dataset.

Heavy files live outside Git. Run render, then assemble, then verify. No GPU use.
The original scripts whitelist and missing-character metadata are authoritative.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont, features
import PIL

SPLITS = ('train', 'val', 'test')
FIELDS = ('split', 'font', 'char', 'cp', 'script', 'script_group', 'bucket')
GROUP = dict(ascii_digits='latin', ascii_letters='latin', latin_ext_letters='latin',
             hiragana='kana', katakana='kana', bopomofo='bopomofo')
ZIP_SHA256 = '25de9f1759df0b6ae86cb85464aefa4f6707400ff6ebb42331cf8c00394c6b14'


def digest(p, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read(p):
    return json.loads(Path(p).read_text())


def tsv(p):
    with Path(p).open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_tsv(p, rows, fields):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fields, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def cp(ch):
    return f'u{ord(ch):04X}'


def family_key(stem):
    # Explicit weight suffixes only. Keep the exact stem and evidence in the audit.
    return re.sub(r'[-_](?:EL|UL|EB|UB|DB|B|H|L|M|R|T|Cu|Zhong|Te|Xian|Da|Xi|Zhun|Italic|50[0-9][RMBH]|51[0-9][RMBH])$', '', stem)


def draw_glyph(font, ch):
    img = Image.new('L', (96, 96), 255); d = ImageDraw.Draw(img)
    b = d.textbbox((0, 0), ch, font=font); w, h = b[2]-b[0], b[3]-b[1]
    if w > 84 or h > 84:
        raise ValueError(f'protocol-A bbox overflow: {ch!r} {w}x{h}')
    d.text(((96-w)//2-b[0], (96-h)//2-b[1]), ch, font=font, fill=0)
    return img.convert('RGB'), (w, h)


def render_one(job):
    rec, root_s, charset = job; root = Path(root_s); stem = rec['clean']
    out = root/'rendered'/stem; done = out/'record.json'
    if done.exists():
        return read(done)
    path = root/'source/ttf'/rec['font_file']
    f = TTFont(path, lazy=True); cmap = f.getBestCmap()
    def present(ch):
        return ord(ch) in cmap and f.getGlyphID(cmap[ord(ch)]) != 0
    names = {}; name_decode_warnings = []
    for n in (1, 2, 6, 16, 17):
        values = set()
        for x in f['name'].names:
            if x.nameID != n: continue
            try: values.add(x.toUnicode())
            except UnicodeDecodeError:
                name_decode_warnings.append(dict(name_id=n, platform=x.platformID, encoding=x.platEncID))
        names[str(n)] = sorted(values)
    target = []
    for bucket, chars in charset['target'].items():
        if bucket not in rec['coverage']:
            continue
        missing = ''.join(c for c in chars if not present(c))
        if set(missing) != set(rec['missing_chars'].get(bucket, '')):
            raise ValueError(f'{stem}/{bucket}: cmap missing {missing!r} != declared {rec["missing_chars"]}')
        if sum(present(c) for c in chars) != rec['coverage'][bucket]['hit']:
            raise ValueError(f'{stem}/{bucket}: coverage mismatch')
        target.extend((c, bucket) for c in chars if present(c))
    style = [c for c in charset['style_han_338'] if present(c)]
    missing_style = [c for c in charset['style_han_338'] if not present(c)]
    if len(style) < 8:
        raise ValueError(f'{stem}: fewer than 8 valid Chinese reference characters')
    f.close()
    allchars = [c for c, _ in target] + style
    scratch = ImageDraw.Draw(Image.new('L', (1, 1)))
    def fits(size):
        font = ImageFont.truetype(str(path), size)
        boxes = [scratch.textbbox((0, 0), c, font=font) for c in allchars]
        return max(max(b[2]-b[0], b[3]-b[1]) for b in boxes) <= 84
    lo, hi = 1, 128
    while fits(hi):
        hi *= 2
        if hi > 4096:
            raise ValueError(f'{stem}: invalid font size search')
    if not fits(1):
        raise ValueError(f'{stem}: no valid font size')
    while lo + 1 < hi:
        mid = (lo+hi)//2
        if fits(mid): lo = mid
        else: hi = mid
    font = ImageFont.truetype(str(path), lo)
    # U+FFFF is unmapped; Pillow renders this font's own .notdef, no fallback font.
    nd = Image.new('L', (96,96), 255); dr = ImageDraw.Draw(nd)
    b = dr.textbbox((0,0), '\uffff', font=font)
    dr.text(((96-(b[2]-b[0]))//2-b[0], (96-(b[3]-b[1]))//2-b[1]), '\uffff',font=font,fill=0)
    nd_hash = hashlib.sha256(nd.tobytes()).hexdigest()
    rows, rejects, image_records = [], [], []
    for kind, entries in [('target', target), ('style', [(c,'han') for c in style])]:
        dest = out/kind; dest.mkdir(parents=True,exist_ok=True)
        for ch, bucket in entries:
            im, (w,h) = draw_glyph(font,ch); a = np.asarray(im.convert('L'))
            mask = a < 250; pix = hashlib.sha256(a.tobytes()).hexdigest()
            why = 'empty' if not mask.any() else 'notdef' if pix == nd_hash else None
            if mask.any() and (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()):
                why = 'touches_canvas_edge'
            if why:
                rejects.append(dict(font=stem,char=ch,cp=cp(ch),kind=kind,reason=why)); continue
            p = dest/f'{stem}+{cp(ch)}.png'; im.save(p)
            ys,xs = np.nonzero(mask)
            image_records.append(dict(font=stem,kind=kind,char=ch,cp=cp(ch),pixel_sha256=pix,
                                      sha256=digest(p),ink_ratio=float(mask.mean()),
                                      bbox_ratio=float((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1)/9216),
                                      text_width=w,text_height=h))
            if kind == 'target':
                rows.append(dict(font=stem,char=ch,cp=cp(ch),script=bucket,script_group=GROUP[bucket],bucket='v0917_selected'))
    style_valid = [x['cp'] for x in image_records if x['kind']=='style']
    assert len(style_valid) >= 8
    fingerprints = {}
    for label, chars in [('ascii62','0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'),
                         ('han8',charset['style_ref8_subset'])]:
        m = {x['char']:x['pixel_sha256'] for x in image_records if x['kind']==('style' if label=='han8' else 'target')}
        if all(c in m for c in chars):
            fingerprints[label] = hashlib.sha256(''.join(m[c] for c in chars).encode()).hexdigest()
    result = dict(**rec, ttf_sha256=digest(path), names=names, name_decode_warnings=name_decode_warnings,
                  size=lo, style_missing_cmap=missing_style,
                  style_cps=style_valid,target_rows=rows,render_rejections=rejects,
                  fingerprints=fingerprints,images=image_records,
                  render_protocol='A96-inner84-font-fixed-no-resize-eligible-target-plus-present-Han338')
    dump(done,result); return result


def render(args):
    root=args.root; contracts=args.contracts
    src=root/'source/v0917_west_style_ttf_232.zip'
    assert digest(src)==ZIP_SHA256, 'ZIP hash mismatch'
    official=read(contracts/'manifests/v0917_west_style/fonts_all_232.json')['train']
    contract=read(contracts/'manifests/v0917_west_style.json')
    with zipfile.ZipFile(src) as z:
        prefix='v0917_west_style_ttf_232/'
        pack=z.read(prefix+'manifest_fonts.json')
        pack_rows={r['clean']:r for r in json.loads(pack)['fonts']}
        files=[x for x in z.namelist() if x.lower().endswith('.ttf')]
        assert len(files)==len(official)==232
        for r in official:
            assert Path(r['font_file']).name == r['font_file'] and r['font_file'].lower().endswith('.ttf')
            expected={b for s in r['scripts'] for b in contract['script_to_buckets'][s]}
            assert expected==set(r['coverage']), r['clean']
            assert set(r['missing_chars'])<=expected
            for k in ('clean','scripts','font_file','font_sha1_12','font_bytes','missing_chars','coverage'):
                assert r[k]==pack_rows[r['clean']][k],(r['clean'],k)
            data=z.read(prefix+'fonts/'+r['font_file'])
            assert len(data)==r['font_bytes']
            assert hashlib.sha1(data).hexdigest()[:12]==r['font_sha1_12']
            p=root/'source/ttf'/r['font_file'];p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists(): p.write_bytes(data)
            assert hashlib.sha256(data).hexdigest()==digest(p)
    charset=read(contracts/'manifests/charset_cn2west_v2_planned.json')
    jobs=[(r,str(root),charset) for r in official]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result=[]
        for i,r in enumerate(pool.map(render_one,jobs),1):
            result.append({k:v for k,v in r.items() if k not in ('images','target_rows')})
            if i%10==0 or i==232:print(f'rendered {i}/232',flush=True)
    dump(root/'render_summary.json',dict(fonts=result,zip_sha256=ZIP_SHA256,pillow=PIL.__version__,
         freetype=features.version_module('freetype2'),target_count=sum(len(read(root/'rendered'/r['clean']/'record.json')['target_rows']) for r in official)))


class Union:
    def __init__(self, names): self.p={n:n for n in names}
    def find(self,x):
        if self.p[x]!=x:self.p[x]=self.find(self.p[x])
        return self.p[x]
    def join(self,a,b): self.p[self.find(max(a,b))]=self.find(min(a,b))


def old_fingerprints(args, old_split, charset):
    result={}
    for split,fonts in old_split.items():
        for stem in fonts:
            fps={}
            for label, chars, kind in [('ascii62','0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz','TargetImage'),
                                      ('han8',charset['style_ref8_subset'],'StyleImage')]:
                vals=[]
                for c in chars:
                    p=args.old_root/split/kind/stem/f'{stem}+{cp(c)}.png'
                    if not p.is_file():break
                    with Image.open(p) as im: vals.append(hashlib.sha256(im.convert('L').tobytes()).hexdigest())
                if len(vals)==len(chars):fps[label]=hashlib.sha256(''.join(vals).encode()).hexdigest()
            result[stem]=fps
    return result


def choose_split(records, old_split, old_fp, seed):
    old={f:s for s,fs in old_split.items() for f in fs}; new={r['clean']:r for r in records}
    assert not set(old)&set(new), 'stem collision requires explicit merge policy'
    allnames=sorted(set(old)|set(new));u=Union(allnames);edges=[]
    def join_groups(table,why):
        for key, names in table.items():
            names=sorted(set(names))
            for b in names[1:]:
                u.join(names[0],b);edges.append(dict(a=names[0],b=b,reason=why,key=key))
    groups=defaultdict(list)
    for n in allnames:groups[family_key(n).casefold()].append(n)
    join_groups(groups,'explicit_weight_suffix')
    for field in ('family','ttf_sha256'):
        groups=defaultdict(list)
        for n,r in new.items():groups[r[field]].append(n)
        join_groups(groups,field)
    groups=defaultdict(list)
    for n,r in new.items():
        # Typographic family names (16), when available, group actual weight variants.
        for name in r['names']['16']:groups[name.casefold()].append(n)
    join_groups(groups,'ttf_typographic_family_name16')
    groups=defaultdict(list)
    for n,fps in {**old_fp,**{n:r['fingerprints'] for n,r in new.items()}}.items():
        # Requiring all 62 ASCII plus 8 CN avoids treating shared stock Latin as full-font identity.
        if 'ascii62' in fps and 'han8' in fps:groups[fps['ascii62']+fps['han8']].append(n)
    join_groups(groups,'exact_ascii62_plus_han8_pixels')
    comps=defaultdict(list)
    for n in allnames:comps[u.find(n)].append(n)
    fixed={}; free=[]; audit=[]; conflicts=[]
    for names in comps.values():
        added=sorted(set(names)&set(new)); anchors=sorted({old[n] for n in names if n in old})
        if len(anchors)>1:
            conflicts.append(dict(members=names,old_splits=anchors,new_members=added))
            if added:raise ValueError(f'new component bridges frozen old splits: {names}')
        if not added:continue
        cid=hashlib.sha256('|'.join(sorted(names)).encode()).hexdigest()[:16]
        if anchors:
            for n in added:fixed[n]=anchors[0]
        else:free.append((cid,added))
        audit.append(dict(group_id=cid,members=names,new_members=added,old_anchors=anchors))
    target={'train':200,'val':16,'test':16}; remaining={s:target[s]-Counter(fixed.values())[s] for s in SPLITS}
    assert min(remaining.values())>=0, remaining
    # Pick held-out whole groups via deterministic subset DP; minimize deviations in
    # six script counts and incomplete-font counts. No model or image quality enters.
    totals=np.zeros(7)
    def vector(names):
        return np.array([sum(b in new[n]['coverage'] for n in names) for b in GROUP]+[sum(bool(new[n]['missing_chars']) for n in names)],dtype=float)
    for n in new:totals+=vector([n])
    allocation=dict(fixed)
    for split in ('test','val'):
        desired=totals*target[split]/len(new)
        base=vector([n for n,s in fixed.items() if s==split])
        best=None
        for attempt in range(400):
            ordered=sorted(free,key=lambda x:hashlib.sha256(f'{seed}:{split}:{attempt}:{x[0]}'.encode()).hexdigest())
            states={0:[]}
            for cid,names in ordered:
                for count,sel in sorted(list(states.items()),reverse=True):
                    nxt=count+len(names)
                    if nxt<=remaining[split] and nxt not in states:states[nxt]=sel+[(cid,names)]
            if remaining[split] not in states:continue
            selected=states[remaining[split]]; v=base+vector([n for _,ns in selected for n in ns])
            score=float((((v-desired)/np.maximum(desired,1))**2).sum())
            candidate=(score,attempt,selected)
            if best is None or candidate[:2]<best[:2]:best=candidate
        if best is None:raise ValueError(f'cannot fill {split} without splitting families')
        selected_ids={cid for cid,_ in best[2]}
        for _,names in best[2]:
            for n in names:allocation[n]=split
        free=[x for x in free if x[0] not in selected_ids]
    for _,names in free:
        for n in names:allocation[n]='train'
    assert dict(Counter(allocation.values()))==target
    return allocation,dict(seed=seed,target=target,groups=audit,edges=edges,legacy_cross_split_components=conflicts,
                           policy='freeze old assignments; anchor related supplements; split unanchored groups with metadata-only stratification')


def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink():
        assert dst.resolve()==src.resolve(),dst
    else:dst.symlink_to(src.resolve())


def link_directory(src,dst):
    """Reuse a fully eligible immutable directory; finish prior partial views safely."""
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink():
        assert dst.resolve()==src.resolve(),dst
    elif dst.exists():
        expected={p.name for p in src.iterdir()}
        existing={p.name for p in dst.iterdir()}
        assert existing<=expected,dst
        for name in expected-existing:link(src/name,dst/name)
    else:dst.symlink_to(src.resolve(),target_is_directory=True)


def make_manifests(dest, pairs, stems, origins, font_records, dataset_id):
    dest.mkdir(parents=True,exist_ok=True)
    counts={}; donors={g:[] for g in ('latin','kana','bopomofo')}; donors_cp=defaultdict(list)
    for split in SPLITS:
        rows=sorted(pairs[split],key=lambda x:(x['font'],x['cp'],x['script']))
        write_tsv(dest/f'pairs_{split}.tsv',rows,FIELDS)
        c=Counter(x['script_group'] for x in rows)
        counts[split]=dict(fonts=len(stems[split]),effective_fonts=len({x['font'] for x in rows}),pairs=len(rows),**dict(c))
        if split=='train':
            for g in donors:donors[g]=sorted({x['font'] for x in rows if x['script_group']==g})
            for row in rows:donors_cp[row['cp']].append(row['font'])
    dump(dest/'split.json',dict(dataset_id=dataset_id,seed=3407,stems=stems))
    dump(dest/'donor_train.json',donors)
    dump(dest/'donor_train_by_cp.json',{c:sorted(set(fs)) for c,fs in sorted(donors_cp.items())})
    p={'latin':.5,'kana':.38,'bopomofo':.12}
    dump(dest/'sample_weights.json',dict(group_p=p,pair_weight={g:p[g]/counts['train'][g] for g in p},
         note='Compatibility weights only; no new training has been started. Use per-codepoint donors for incomplete/script-selective fonts.'))
    rows=[]
    for split in SPLITS:
        active={r['font'] for r in pairs[split]}
        for f in stems[split]:
            rec=font_records[f]
            rows.append(dict(stem=f,split=split,source=origins[f],usable=int(f in active),
                             bucket='v0917_selected' if origins[f]=='0917' else rec.get('bucket',''),
                             family=rec.get('family',family_key(f)),display_name=rec.get('disp',f),
                             missing_chars=json.dumps(rec.get('missing_chars',{}),ensure_ascii=False)))
    write_tsv(dest/'fonts.tsv',rows,('stem','split','source','usable','bucket','family','display_name','missing_chars'))
    pools={f:font_records[f]['style_cps'] for s in SPLITS for f in stems[s]}
    dump(dest/'style_pool.json',pools)
    active={r['font'] for s in SPLITS for r in pairs[s]}
    common=set.intersection(*(set(pools[f]) for f in active))
    canonical='永和书风骨韵天地'
    preferred='永和骨天地山水人'+canonical
    common8=[]
    for ch in preferred:
        if cp(ch) in common and ch not in common8:common8.append(ch)
        if len(common8)==8:break
    assert len(common8)==8
    incompatible={f:''.join(ch for ch in canonical if cp(ch) not in pools[f]) for f in sorted(active)}
    dump(dest/'reference_compatibility.json',dict(original_ref8=canonical,
         original_ref8_incompatible_fonts={f:v for f,v in incompatible.items() if v},
         optional_shared_ref8=''.join(common8),shared_reference_cps=sorted(common),
         policy='Use only actual same-font codepoints in style_pool.json. No font fallback or simplified/traditional substitution. The optional common ref8 is not applied to existing experiments.'))
    idx=dict(dataset_id=dataset_id,status='built_for_user_review',counts=counts,
             files={p.name:digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='INDEX.json'},
             required_pair_filter=True,required_donor_filter='donor_train_by_cp.json',
             note='The old 5 excluded train fonts remain excluded. Do not enumerate original dirty PNG roots as training targets.')
    dump(dest/'INDEX.json',idx);return idx


def assemble(args):
    root=args.root; contracts=args.contracts
    old_index=read(contracts/'manifests/v0913_clean/INDEX.json')
    for n,h in old_index['files'].items():assert digest(contracts/'manifests/v0913_clean'/n)==h,n
    old_split=read(contracts/'manifests/split_v3_228_16_16.json')['stems']
    old_pairs={s:tsv(contracts/'manifests/v0913_clean'/f'pairs_{s}.tsv') for s in SPLITS}
    old_fonts={r['stem']:r for r in tsv(contracts/'manifests/v0913_clean/fonts.tsv')}
    charset=read(contracts/'manifests/charset_cn2west_v2_planned.json')
    recs=[read(p) for p in sorted((root/'rendered').glob('*/record.json'))];assert len(recs)==232
    oldfp=old_fingerprints(args,old_split,charset)
    split,audit=choose_split(recs,old_split,oldfp,3407)
    dump(root/'split_audit.json',audit);dump(root/'old_fingerprints.json',oldfp)
    new_pairs={s:[] for s in SPLITS};new_stems={s:[] for s in SPLITS}
    newdir=root/'v0917';merged=root/'v2'
    if (root/'ASSEMBLED.json').exists():raise ValueError('dataset already assembled; run verify instead')
    source_records=[]
    for r in recs:
        f=r['clean'];s=split[f];new_stems[s].append(f)
        for kind,folder in [('target','TargetImage'),('style','StyleImage')]:
            src=root/'rendered'/f/kind
            link_directory(src,newdir/s/folder/f)
            link_directory(src,merged/s/folder/f)
        for row in r['target_rows']:
            new_pairs[s].append(dict(row,split=s))
    # Preserve every old eligible target and every existing old style/content image.
    # The five excluded old fonts stay in the roster with no target rows.
    for s in SPLITS:
        per_font=defaultdict(list)
        for row in old_pairs[s]:per_font[row['font']].append(row)
        for font,rs in per_font.items():
            srcdir=args.old_root/s/'TargetImage'/font
            dstdir=merged/s/'TargetImage'/font
            expected={f'{font}+{r["cp"]}.png' for r in rs}
            if expected=={p.name for p in srcdir.iterdir()}:
                link_directory(srcdir,dstdir)
            else:
                dstdir.mkdir(parents=True,exist_ok=True)
                existing={p.name for p in dstdir.iterdir()}
                assert existing<=expected,dstdir
                for name in expected-existing:link(srcdir/name,dstdir/name)
        for row in old_pairs[s]:
            rel=Path(s)/'TargetImage'/row['font']/f'{row["font"]}+{row["cp"]}.png'
            src=args.old_root/rel;assert src.is_file(),src
            h=digest(src)
            source_records.append(dict(path=str(rel),sha256=h,kind='target'))
        for f in old_split[s]:
            (merged/s/'TargetImage'/f).mkdir(parents=True,exist_ok=True)
            link_directory(args.old_root/s/'StyleImage'/f,merged/s/'StyleImage'/f)
            for src in sorted((args.old_root/s/'StyleImage'/f).glob('*.png')):
                rel=src.relative_to(args.old_root)
                source_records.append(dict(path=str(rel),sha256=digest(src),kind='style'))
        for src in sorted((args.old_root/s/'ContentImage').glob('*.png')):
            rel=src.relative_to(args.old_root);link(src,merged/rel);link(src,newdir/rel)
            source_records.append(dict(path=str(rel),sha256=digest(src),kind='content'))
    allrecs={**old_fonts,**{r['clean']:r for r in recs}}
    for s in SPLITS:
        for f in old_split[s]:
            allrecs[f]['style_cps']=sorted(p.stem.split('+')[-1] for p in (args.old_root/s/'StyleImage'/f).glob('*.png'))
    origins={f:'0913' for s in SPLITS for f in old_split[s]};origins.update({r['clean']:'0917' for r in recs})
    ni=make_manifests(root/'manifests/v0917',new_pairs,new_stems,origins,allrecs,'v0917_split')
    merged_pairs={s:old_pairs[s]+new_pairs[s] for s in SPLITS}
    stems={s:sorted(old_split[s]+new_stems[s]) for s in SPLITS}
    vi=make_manifests(root/'manifests/v2',merged_pairs,stems,origins,allrecs,'v2')
    write_tsv(root/'old_source_sha256.tsv',source_records,('path','sha256','kind'))
    rejects=[x for r in recs for x in r['render_rejections']]
    write_tsv(root/'new_render_rejections.tsv',rejects,('font','char','cp','kind','reason'))
    missing=[]
    for r in recs:
        for bucket,chars in r['missing_chars'].items():
            missing.extend(dict(font=r['clean'],split=split[r['clean']],char=c,cp=cp(c),bucket=bucket,reason='documented_missing') for c in chars)
    write_tsv(root/'documented_missing.tsv',missing,('font','split','char','cp','bucket','reason'))
    result=dict(v0917=ni,v2=vi,old_frozen_counts=old_index['counts'],new_render_rejections=rejects,
                documented_missing_pairs=len(missing),old_pngs_hashed=len(source_records),
                source_commit='4ececb4300bba7aed703d1505f8021afbf7a1537',zip_sha256=ZIP_SHA256)
    dump(root/'ASSEMBLED.json',result)
    for d,id_ in [(newdir,'v0917_split'),(merged,'v2')]:
        dump(d/'DATASET_ROOT.json',dict(dataset_id=id_,manifests=str(root/'manifests'/('v0917' if id_=='v0917_split' else 'v2')),
             immutable_sources=[str(args.old_root),str(root/'rendered')],status='awaiting_user_visual_review'))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def verify_new_view_links(root):
    """Check new-font views point to that exact font, including resumed file links."""
    root=Path(root).resolve()
    allocation=read(root/'manifests/v0917/split.json')['stems'];checked=0
    for split,fonts in allocation.items():
        for font in fonts:
            for folder,kind in [('TargetImage','target'),('StyleImage','style')]:
                src=root/'rendered'/font/kind
                for dataset in ('v0917','v2'):
                    dst=root/dataset/split/folder/font
                    if dst.is_symlink():
                        assert os.readlink(dst)==str(src),dst
                        checked+=1
                    else:
                        for p in dst.iterdir():
                            assert p.is_symlink() and os.readlink(p)==str(src/p.name),p
                            checked+=1
    return dict(status='PASS',checked_links=checked,new_font_views_point_to_their_own_gt=True)


def verify(args):
    root=args.root;c=args.contracts
    original_project=args.old_root.parent.parent
    old_index=read(c/'manifests/v0913_clean/INDEX.json')
    for name,h in old_index['files'].items():
        assert digest(original_project/'manifests/v0913_clean'/name)==h, name
    assert digest(original_project/'manifests/split_v3_228_16_16.json')==digest(c/'manifests/split_v3_228_16_16.json')
    old_split=read(c/'manifests/split_v3_228_16_16.json')['stems']
    old_pairs={s:tsv(c/'manifests/v0913_clean'/f'pairs_{s}.tsv') for s in SPLITS}
    indexes={d:read(root/'manifests'/d/'INDEX.json') for d in ('v0917','v2')}
    def key(r):return tuple(r[k] for k in FIELDS)
    for d,idx in indexes.items():
        for name,h in idx['files'].items():assert digest(root/'manifests'/d/name)==h
        sp=read(root/'manifests'/d/'split.json')['stems'];assert len(set(sum(sp.values(),[])))==sum(map(len,sp.values()))
        for s in SPLITS:
            rows=tsv(root/'manifests'/d/f'pairs_{s}.tsv');assert len({(r['font'],r['cp']) for r in rows})==len(rows)
            physical={(f.parent.name,f.stem.split('+')[-1]) for f in (root/d/s/'TargetImage').glob('*/*.png')}
            assert physical=={(r['font'],r['cp']) for r in rows}
            for r in rows:
                p=root/d/s/'TargetImage'/r['font']/f'{r["font"]}+{r["cp"]}.png';assert p.is_file(),p
                assert (root/d/s/'ContentImage'/f'{r["cp"]}.png').is_file()
            if d=='v2':
                assert set(old_split[s])<=set(sp[s])
                assert {key(r) for r in rows if r['font'] in old_split[s]}=={key(r) for r in old_pairs[s]}
                new=tsv(root/'manifests/v0917'/f'pairs_{s}.tsv')
                assert {key(r) for r in rows}=={key(r) for r in old_pairs[s]+new}
    old_hashes=tsv(root/'old_source_sha256.tsv')
    for r in old_hashes:
        src=args.old_root/r['path'];dst=root/'v2'/r['path']
        assert src.resolve()==dst.resolve() and digest(src)==r['sha256'],src
    forbidden=tsv(root/'documented_missing.tsv')+tsv(root/'new_render_rejections.tsv')
    official={r['clean']:r for r in read(c/'manifests/v0917_west_style/fonts_all_232.json')['train']}
    found={r['clean']:read(root/'rendered'/r['clean']/'record.json') for r in official.values()}
    for s in SPLITS:
        rows=tsv(root/'manifests/v0917'/f'pairs_{s}.tsv');ks={(r['font'],r['cp']) for r in rows}
        assert not ks&{(r['font'],r['cp']) for r in forbidden if r.get('kind','target')=='target'}
        for r in rows:assert r['script'] in official[r['font']]['coverage']
    for d in ('v0917','v2'):
        expected=defaultdict(set)
        for r in tsv(root/'manifests'/d/'pairs_train.tsv'):expected[r['cp']].add(r['font'])
        donors=read(root/'manifests'/d/'donor_train_by_cp.json')
        assert {k:set(v) for k,v in donors.items()}==dict(expected)
        pools=read(root/'manifests'/d/'style_pool.json')
        refs=read(root/'manifests'/d/'reference_compatibility.json')
        stems=read(root/'manifests'/d/'split.json')['stems']
        for split,fonts in stems.items():
            for font in fonts:
                physical={p.stem.split('+')[-1] for p in (root/d/split/'StyleImage'/font).glob('*.png')}
                assert set(pools[font])==physical
        assert all(set(cp(ch) for ch in refs['optional_shared_ref8'])<=set(pools[f]) for f in pools)
    # Recompute deterministic split; no filesystem iteration ordering may affect it.
    allocated,audit=choose_split(list(found.values()),old_split,read(root/'old_fingerprints.json'),3407)
    actual=read(root/'manifests/v0917/split.json')['stems']
    assert all(allocated[f]==s for s,fs in actual.items() for f in fs)
    image_count=0
    for f,r in found.items():
        for x in r['images']:
            p=root/'rendered'/f/x['kind']/f'{f}+{x["cp"]}.png'
            assert digest(p)==x['sha256']
            with Image.open(p) as im:
                assert im.mode=='RGB' and im.size==(96,96)
            image_count+=1
    view_links=verify_new_view_links(root)
    result=dict(status='PASS',old_font_assignments_unchanged=True,old_pair_rows_unchanged=True,
         old_target_style_content_bytes_unchanged=True,old_files_verified=len(old_hashes),
         new_images_verified=image_count,new_fonts=232,deterministic_split_verified=True,
         forbidden_missing_or_rejected_pairs_in_targets=0,unselected_script_pairs_in_targets=0,
         old_five_excluded_fonts_still_excluded=True,known_new_group_cross_split_leakage=0,new_view_links=view_links,
         legacy_cross_split_components=audit['legacy_cross_split_components'],
         limitations=['Old raw TTFs are unavailable; historical lineage is not exhaustively verifiable.',
                     'Exact pixel and metadata checks do not prove semantic glyph correctness; visual review remains required.'])
    dump(root/'VERIFICATION.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['render','assemble','verify'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--contracts',type=Path,required=True)
    p.add_argument('--old-root',type=Path,default=Path('/root/projects/hrfont/data/fontdiffuser-p253-t295-s338-cn2west-v2'))
    p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    a.root=a.root.resolve();a.contracts=a.contracts.resolve();a.old_root=a.old_root.resolve()
    globals()[a.stage](a)


if __name__=='__main__':main()
