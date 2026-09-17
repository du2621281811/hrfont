import collections,json,pathlib,random,sys,time
import torch
import torch.nn.functional as F
sys.path.insert(0,'scripts')
from k4_runtime import *
from h_runtime import args_for
torch.set_num_threads(1);torch.cuda.set_device(0)
OUT=Path('/root/data1/hrfont_k4_family_20260918/control');OUT.mkdir(parents=True,exist_ok=True)
family=json.loads((OUT/'family_groups.json').read_text())['family_by_font']
es=MergedEs();pools=json.loads((DATA/'manifests/v2/style_pool.json').read_text());results={};examples=[]
for name,attempts in [('v2',40),('v0917',16)]:
    args=configure(args_for(PARENT0),name,name);ds=dataset(args,'train');sampler=KSampler(ds,ASSETS/'detail_manifest.json')
    dm=json.loads((DATA/'manifests'/name/'donor_train_by_cp.json').read_text());lib=Library(es,dm,pools)
    cfg=T.DeltaConfig(tau=args.delta_tau,eps_alpha=args.delta_eps_alpha,k_max=args.delta_k_max,k_top=args.delta_k_top,mode=args.delta_mode,rng_seed=args.seed)
    count=collections.Counter();fonts=set();pairs=collections.Counter()
    for attempt in range(attempts):
        ids,_=sampler.batch(attempt)
        for rank in range(8):
            seed_episode(3407+attempt*1000+rank);jobs=[]
            for i in ids[rank*8:rank*8+8]:
                f,cp,script,flag=sampler.entries[i];pool=sorted(pools[f]);random.sample(pool,random.randint(1,8))
                refs=random.sample(sorted(set(pool)-{cp}),random.randint(1,8));jobs.append((f,cp,refs))
            source=torch.rand(8,device='cuda')<.05;drop=torch.rand(8,device='cuda')<.02;active=(~(source|drop)).cpu().tolist()
            for (f,cp,refs),enabled in zip(jobs,active):
                q=torch.stack([F.normalize(es.pooled_tensor('train',f,c),dim=0) for c in refs])
                selected,alpha=lib.select(cp,refs,q,f,cfg)
                hits=[(d,float(w)) for d,w in zip(selected,alpha) if family[d]==family[f] and w>0]
                count['queries']+=1;count['active_queries']+=enabled;count['queries_with_family_donor']+=bool(hits);count['active_queries_with_family_donor']+=bool(hits) and enabled
                if hits:
                    fonts.add(f)
                    for d,w in hits:pairs[(f,d)]+=1
                    if len(examples)<100:examples.append(dict(dataset=name,update=attempt+1,rank=rank,font=f,cp=cp,refs=refs,active=enabled,same_family_donors=hits,selected=selected,alpha=alpha.tolist()))
        if attempt%10==0:print(name,attempt,dict(count),flush=True)
    results[name]=dict(count,affected_query_fonts=len(fonts),family_pairs=[dict(query=f,donor=d,occurrences=n) for (f,d),n in pairs.most_common()],interpretation='Replay of first 40 actually executed C updates; Python refs and CUDA source/CFG masks use original seeds.' if name=='v2' else 'Counterfactual audit of old planned A recipe; A had not started.')
    del lib
atomic_json(OUT/'OLD_ALPHA_AUDIT.json',dict(execution_code_identity=code_identity(),results=results,examples=examples,stop=json.loads(Path('/root/projects/hrfont/runs/K4-C-K1RECIPE-V2-S3407/STOPPED.json').read_text()),sampling_boundary='Counts cover stated replay windows, not all 6265 updates; positive weights on active examples establish actual family-donor use.'))
print('AUDIT_COMPLETE',json.dumps({k:{a:b for a,b in v.items() if a!='family_pairs'} for k,v in results.items()}),flush=True)
