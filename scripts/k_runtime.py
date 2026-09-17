"""Isolated K runtime, reusing the frozen I architecture and clean data contracts."""
import hashlib
import json
import os
import random
from pathlib import Path
import torch
from h_runtime import (CODE, ROOT, CACHE, PARENT0, T, args_for, dataset as old_dataset,
                       batch_to, seed_episode, atomic_json, sha256_file)
from i_runtime import DataContext as IDataContext
from scripts.hrfont_k import KModel


def model_for(arm, device, output):
    args = args_for(PARENT0)
    T._verify_caches(args, PARENT0)
    torch.manual_seed(3407)
    base = T.FontDiffuserModel(unet=T.build_unet(args), style_encoder=T.build_style_encoder(args),
                              content_encoder=T.build_content_encoder(args))
    T._load_parent(base,PARENT0,output)
    base.style_encoder.requires_grad_(False).eval()
    base.content_encoder.requires_grad_(False).eval()
    model = KModel(base,arm).to(device)
    stats_path = CACHE/'teacher_stats.pt'
    manifest = json.loads((CACHE/'manifest.json').read_text())
    assert manifest['teacher_stats_sha256']==sha256_file(stats_path)
    stats = torch.load(stats_path,map_location=device,weights_only=True)
    model.teacher_mean.copy_(stats['mean']); model.teacher_std.copy_(stats['std'])
    return model,args


def dataset(args, split):
    return old_dataset(args,split)


def train_sample(ds,index):
    sample=ds[index]
    pool=sorted(set(ds.style_by_font_char[sample['font_stem']])-{sample['char_cp']})
    refs=random.sample(pool,random.randint(1,8))
    sample['ref_chars']=refs
    sample['ref_image_paths']=[str(ds.style_by_font_char[sample['font_stem']][cp]) for cp in refs]
    assert sample['split']=='train' and sample['char_cp'] not in refs
    return sample


class DataContext(IDataContext):
    @torch.no_grad()
    def conditions(self,samples,cfg,source,no_delta=False,official=False,need_refs=True):
        style,queries,*_=T._style_conditions(self.es,samples,self.device)
        lengths=[len(r) for r in samples['ref_chars']]
        keep=torch.arange(max(lengths))[None]<torch.tensor(lengths)[:,None]
        if need_refs:
            images=list(self.pool.map(self.image,sum(samples['ref_image_paths'],[])))
            refs=torch.zeros(len(lengths),max(lengths),3,96,96)
            refs[keep]=torch.stack(images)
            refs=refs.pin_memory().to(self.device,non_blocking=True)
        else:
            refs=None
        neutral=self.ec.features_many('content',[('',cp) for cp in samples['char_cp']])
        content=[torch.cat([neutral[('',cp)][s] for cp in samples['char_cp']]).to(self.device) for s in range(5)]
        query=content[-1].clone()
        selections=[]
        if not no_delta and not official:
            for font,cp,chars,q in zip(samples['font_stem'],samples['char_cp'],samples['ref_chars'],queries):
                # The same reference-conditioned alpha on CPU avoids thousands of CUDA scalar syncs.
                idx,alpha,_=T.compute_alpha(q.cpu(),self.library.prototypes(chars,'cpu'),
                    self.library.font_index.get(font),self.alpha_cfg,extra_exclude=self.exclude(cp))
                assert idx and all(self.fonts[i]!=font for i in idx)
                selections.append(([self.fonts[i] for i in idx],alpha))
            donors=self.ec.features_many('target',[(f,cp) for (fonts,_),cp in zip(selections,samples['char_cp']) for f in fonts])
        elif official:
            selections=[([f],torch.ones(1)) for f in samples['font_stem']]
            donors=self.ec.features_many('style',[(f,r[0]) for f,r in zip(samples['font_stem'],samples['ref_chars'])])
        width=max((len(x[0]) for x in selections),default=1)
        alpha=torch.zeros(len(lengths),width)
        for i,(_,a) in enumerate(selections): alpha[i,:len(a)]=a
        if no_delta: alpha.fill_(1)
        alpha=alpha.to(self.device)
        active=(~(cfg|source)).float() if not no_delta else torch.zeros_like(cfg,dtype=torch.float32)
        structure=[]
        for s,c in enumerate(content):
            # Only these two levels enter the existing RSI up blocks.
            if s not in (1,2) or no_delta:
                d=c.new_zeros(len(lengths),width,*c.shape[1:])
            else:
                rows=[]
                for i,((fonts,_),cp) in enumerate(zip(selections,samples['char_cp'])):
                    keys=[(f,samples['ref_chars'][i][0] if official else cp) for f in fonts]
                    row=torch.cat([donors[key][s] for key in keys])
                    if not official: row=row-neutral[('',cp)][s]
                    if len(fonts)<width:
                        row=torch.cat([row,torch.zeros(width-len(fonts),*row.shape[1:])])
                    rows.append(row)
                d=torch.stack(rows).pin_memory().to(self.device,non_blocking=True)
            structure.append((d,alpha,c,active))
        return style,refs,query,keep.to(self.device),[
            c.masked_fill(cfg[:,None,None,None],0) for c in content],structure


def code_identity():
    metadata=CODE/'K_CODE_IDENTITY.json'
    if metadata.exists():
        result=json.loads(metadata.read_text())
        for name,digest in result['files'].items():
            assert sha256_file(CODE/name)==digest, 'Source drift: '+name
        return result
    paths=list((CODE/'scripts').glob('*.py'))+list((CODE/'code/variants/cn2west_f123_rsi/FontDiffuser').rglob('*.py'))
    return {'commit':'development','files':{str(p.relative_to(CODE)):sha256_file(p) for p in paths}}
