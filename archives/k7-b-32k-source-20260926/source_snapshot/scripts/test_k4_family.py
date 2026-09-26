"""Regression and exhaustive legal-query coverage for family-filtered alpha."""
import collections,json
import torch
from k4_runtime import *
from h_runtime import args_for


def scope_regression():
    policy=FamilyPolicy(ASSETS/'weight_groups.json',list(json.loads((ASSETS/'weight_groups.json').read_text())['family_by_font']))
    for a,b in [('FZLuoMTJW-L','FZLuoMTJW-R'),('FZSiNTJW-H','FZSiNTJW-UL'),('FZYouHJW_508R','FZYouHJW_513B')]:
        assert policy.family(a)==policy.family(b),(a,b)
    for a,b in [('FZLTHProGBK_H','FZLTHProJW_H'),('FZSongYJW','FZSongYuanK'),('fzsj_1275635','fzsj_2017088'),('FZYouHJW_508R','FZYouHK_508R'),('FZLuoMTJW-L','FZLuoMXTJW-M')]:
        valid=policy.exclude(a,torch.ones(len(policy.fonts),dtype=torch.bool))
        assert valid[policy.fonts.index(b)] and not valid[policy.fonts.index(a)],(a,b)
    return 'PASS'


def regression():
    lib=Library.__new__(Library)
    lib.fonts=['FZDeSHJW_507R','FZDeSHJW_508R','FZBGDT','FZXLB']
    lib.font_index={f:i for i,f in enumerate(lib.fonts)}
    lib.char_index={'ref1':0,'ref2':1}
    # Same-family donors have the highest similarities; they must disappear
    # before both top-k and the softmax normalization.
    lib.table=torch.tensor([[[1.,0.],[1.,0.]],[[1.,.01],[1.,.01]],[[.4,.6],[.4,.6]],[[.8,.2],[.8,.2]]])
    lib.present=torch.ones(4,2,dtype=torch.bool);lib.present[3,1]=False
    lib.bycp={'u0041':torch.ones(4,dtype=torch.bool),'u0042':torch.tensor([True,True,False,True])}
    lib.family_policy=FamilyPolicy(ASSETS/'weight_groups.json',lib.fonts)
    q=torch.tensor([[1.,0.],[1.,0.]])
    checked=[]
    for font in ['FZDeSHJW_507R','FZDeSHJW_515H']:
        for mode in ['topk','soft','threshold']:
            selected,alpha=lib.select('u0041',['ref1','ref2'],q,font,T.DeltaConfig(mode=mode,k_top=1,k_max=1))
            assert selected==['FZBGDT'] and torch.equal(alpha,torch.ones(1)),(font,mode,selected,alpha)
            checked.append([font,mode])
    for font,cp,error in [('unknown','u0041',ValueError),('FZDeSHJW_515H','u0042',RuntimeError)]:
        try:lib.select(cp,['ref1','ref2'],q,font,T.DeltaConfig())
        except error:pass
        else:raise AssertionError((font,cp,'did not fail closed'))
    assert lib.family_policy.snapshot()['violations']==0
    return checked


def real_bank():
    torch.set_num_threads(1)
    pools=json.loads((DATA/'manifests/v2/style_pool.json').read_text())
    poolsets={f:set(cs) for f,cs in pools.items()};references={f:ref8(cs) for f,cs in pools.items()}
    fam=json.loads((ASSETS/'weight_groups.json').read_text())['family_by_font']
    es=MergedEs();summary={}
    for name in ['v2','v0917']:
        dm=json.loads((DATA/'manifests'/name/'donor_train_by_cp.json').read_text())
        lib=Library(es,dm,pools);args=configure(args_for(PARENT0),name,name)
        cfg=T.DeltaConfig(tau=args.delta_tau,eps_alpha=args.delta_eps_alpha,k_max=args.delta_k_max,k_top=args.delta_k_top,mode=args.delta_mode,rng_seed=args.seed)
        # Exhaustive combinatorial check: every legal (font,cp,shot) in train,
        # val, and test; train arbitrary references additionally covered by the
        # complete-reference worst-case bound in contracts().
        queries=0;minimum=100000;representatives={};eligible={};dmsets={c:set(fs) for c,fs in dm.items()}
        splits=['train','val','test'] if name=='v2' else ['train']
        for split in splits:
            for r in tsv(DATA/'manifests'/name/f'pairs_{split}.tsv'):
                f,cp=r['font'],r['cp'];refs=references[f];group=fam[f]
                for k in [1,2,4,8]:
                    if (f,k) not in eligible:
                        needed=set(refs[:k]);eligible[f,k]={d for d in lib.fonts if fam[d]!=group and needed<=poolsets[d]}
                    candidates=dmsets[cp]&eligible[f,k]
                    assert candidates,(name,split,f,cp,k)
                    queries+=1;minimum=min(minimum,len(candidates))
                representatives.setdefault((split,f,r['script_group']),dict(split=split,font=f,cp=cp,refs=refs))
        actual=0;maximum_family_mass=0.;examples=[]
        for r in representatives.values():
            for k in [1,2,4,8]:
                refs=r['refs'][:k]
                q=torch.stack([torch.nn.functional.normalize(es.pooled_tensor(r['split'],r['font'],c),dim=0) for c in refs])
                selected,alpha=lib.select(r['cp'],refs,q,r['font'],cfg)
                mass=sum(float(w) for d,w in zip(selected,alpha) if fam[d]==fam[r['font']])
                assert mass==0.;maximum_family_mass=max(maximum_family_mass,mass);actual+=1
                if r['font'].startswith(('FZDeSH','FZAiS','FZSiNT','FZYouSJJW')) and len(examples)<60:
                    examples.append(dict(**r,shot=k,selected=selected,alpha=alpha.tolist()))
        summary[name]=dict(legal_queries_checked=queries,minimum_eligible=minimum,actual_alpha_queries=actual,
                           maximum_same_family_alpha_mass=maximum_family_mass,guard=lib.family_policy.snapshot(),examples=examples)
        print(name,queries,actual,minimum,flush=True);del lib
    return summary


if __name__=='__main__':
    atomic_json(STORE/'control/FAMILY_POLICY_PASSED.json',dict(status='PASS',scope=scope_regression(),regression=regression(),banks=real_bank(),family_sha256=sha256_file(ASSETS/'weight_groups.json')))
